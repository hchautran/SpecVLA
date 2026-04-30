"""
Autoresearch DFlash-drafter training script. Single-GPU, single-file.

Trains a DFlash-style block-diffusion drafter for the autoregressive pi0-fast
VLA on the LIBERO dataset. The target (pi0-fast) is loaded frozen via
prepare.py; the drafter is the only thing that gets gradients.

Objective (per batch):
  1. Run target (no grad) → prefix hidden states + true FAST action tokens.
  2. Pick a random block of size B inside the action sequence; mask all
     positions except position 0 (which is given by the target during real
     speculative decoding).
  3. Drafter forward, conditioned on selected layers of target hidden states
     concatenated and projected. Predict logits at every block position via
     the target's lm_head.
  4. CE loss against true tokens at the masked positions.

Eval (after training): mean teacher-forced acceptance length, computed by
prepare.evaluate_acceptance_length.

Usage: uv run train.py
"""

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import argparse
import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import wandb

from prepare import (
    TIME_BUDGET,
    TARGET_HIDDEN_SIZE, TARGET_NUM_LAYERS,
    EVAL_BATCHES, EVAL_BATCH_SIZE, EVAL_SEED,
    load_target_policy, target_components,
    make_dataset, make_dataloader,
    target_forward_with_hidden_states, fast_hidden_at_action_positions,
    evaluate_acceptance_length,
)

from dflash.model import DFlashDraftModel, extract_context_feature
from transformers import Qwen3Config


# ---------------------------------------------------------------------------
# Drafter hyperparameters (everything below this line is fair game)
# ---------------------------------------------------------------------------

@dataclass
class DrafterConfig:
    block_size:        int  = 8
    num_hidden_layers: int  = 3
    num_attn_heads:    int  = 16
    num_kv_heads:      int  = 4
    intermediate_size: int  = 8192
    rms_norm_eps:      float = 1e-6
    rope_theta:        float = 1_000_000.0
    attention_dropout: float = 0.0
    loss_decay_gamma:  float = 4.0  # DFlash paper Eq. 4: w_k = exp(-(k-1)/γ)


@dataclass
class TrainConfig:
    batch_size:        int    = 4
    num_workers:       int    = 4
    lr:                float  = 3e-4
    weight_decay:      float  = 0.01
    grad_clip:         float  = 1.0
    warmup_steps:      int    = 100
    log_every:         int    = 25


# ---------------------------------------------------------------------------
# Drafter construction
# ---------------------------------------------------------------------------

def make_drafter(target_policy, drafter_cfg: DrafterConfig) -> DFlashDraftModel:
    target = target_components(target_policy)
    pg_tok = target["paligemma_tokenizer"]
    vocab_size = target["embed_tokens"].weight.shape[0]

    # PaliGemma has no dedicated mask token — repurpose <pad>. Any reserved
    # never-emitted id works; the drafter just needs a stable embedding for
    # "this position is to be denoised."
    mask_token_id = pg_tok.pad_token_id

    cfg = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=TARGET_HIDDEN_SIZE,
        intermediate_size=drafter_cfg.intermediate_size,
        num_hidden_layers=drafter_cfg.num_hidden_layers,
        num_attention_heads=drafter_cfg.num_attn_heads,
        num_key_value_heads=drafter_cfg.num_kv_heads,
        head_dim=TARGET_HIDDEN_SIZE // drafter_cfg.num_attn_heads,
        rms_norm_eps=drafter_cfg.rms_norm_eps,
        rope_theta=drafter_cfg.rope_theta,
        attention_dropout=drafter_cfg.attention_dropout,
        attention_bias=False,
        max_position_embeddings=drafter_cfg.block_size * 4,
        sliding_window=None,
        layer_types=["full_attention"] * drafter_cfg.num_hidden_layers,
        attn_implementation="eager",
    )
    cfg.num_target_layers = TARGET_NUM_LAYERS
    cfg.block_size = drafter_cfg.block_size
    cfg.dflash_config = {"mask_token_id": mask_token_id}

    return DFlashDraftModel(cfg)


# ---------------------------------------------------------------------------
# Block sampling and one training step
# ---------------------------------------------------------------------------

def sample_masked_block(true_tokens, true_mask, block_size, mask_token_id):
    """Pick one random block per batch row; mask positions 1..B-1.

    Returns (block_ids, block_targets, block_valid, starts) where block_ids
    is what the drafter sees (position 0 is the true token, rest are mask)
    and block_targets / block_valid are aligned ground-truth + validity.
    """
    B, T = true_tokens.shape
    device = true_tokens.device
    max_start = max(1, T - block_size)
    starts = torch.randint(0, max_start, (B,), device=device)

    block_ids     = torch.full((B, block_size), mask_token_id, dtype=torch.long, device=device)
    block_targets = torch.empty((B, block_size), dtype=torch.long, device=device)
    block_valid   = torch.empty((B, block_size), dtype=torch.bool, device=device)
    for b in range(B):
        s = starts[b].item()
        block_targets[b] = true_tokens[b, s:s + block_size]
        block_valid[b]   = true_mask[b, s:s + block_size]
        block_ids[b, 0]  = true_tokens[b, s]
    return block_ids, block_targets, block_valid, starts


def train_step(drafter, target_policy, target_comp, batch, preprocessor, drafter_cfg):
    embed_tokens = target_comp["embed_tokens"]
    lm_head      = target_comp["lm_head"]

    batch = preprocessor(batch)
    target_out = target_forward_with_hidden_states(target_policy, batch)
    fast_hiddens = fast_hidden_at_action_positions(
        target_out["hidden_states"], target_out["fast_offset"]
    )
    true_tokens = target_out["fast_token_ids"]
    true_mask   = target_out["fast_token_mask"]
    if true_tokens.shape[1] < drafter_cfg.block_size + 1:
        return None

    block_ids, block_targets, block_valid, starts = sample_masked_block(
        true_tokens, true_mask, drafter_cfg.block_size, drafter.mask_token_id
    )

    ctx = extract_context_feature(fast_hiddens, drafter.target_layer_ids)
    B = block_ids.shape[0]
    ctx_blocks = torch.stack(
        [ctx[b, starts[b]:starts[b] + drafter_cfg.block_size] for b in range(B)], dim=0
    )

    noise_emb = embed_tokens(block_ids)
    block_positions = torch.arange(drafter_cfg.block_size, device=block_ids.device)
    position_ids = block_positions.repeat(2)[None].expand(B, -1)

    h = drafter(
        target_hidden=ctx_blocks,
        noise_embedding=noise_emb,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        is_causal=False,
    )
    logits = lm_head(h)  # (B, block_size, vocab)

    # Loss alignment matches the fixed eval in prepare.evaluate_acceptance_length:
    # logits at position k predict the token at block position k+1 (next-token
    # within the block). Train/eval used to disagree (at-position vs next-token),
    # which floored accept_len at 1.0.
    pred_logits  = logits[:, :-1].reshape(-1, logits.size(-1))
    pred_targets = block_targets[:, 1:].reshape(-1)
    pred_valid   = block_valid[:, 1:].reshape(-1).float()

    # Position-wise loss decay (DFlash Eq. 4): early-block errors are more
    # costly because they invalidate everything after them under spec-decode's
    # prefix-acceptance rule. k is 1-indexed within the block.
    block_positions = torch.arange(1, drafter_cfg.block_size, device=block_ids.device, dtype=torch.float32)
    pos_weights = torch.exp(-(block_positions - 1) / drafter_cfg.loss_decay_gamma)
    pos_weights = pos_weights[None, :].expand(B, -1).reshape(-1)

    loss_per_tok = F.cross_entropy(pred_logits, pred_targets, reduction="none")
    weight = pred_valid * pos_weights
    loss = (loss_per_tok * weight).sum() / weight.sum().clamp(min=1)
    return loss


# ---------------------------------------------------------------------------
# Eval loss (held-out)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_loss(drafter, target_policy, target_comp, preprocessor, drafter_cfg):
    """Mean masked-block CE loss over EVAL_BATCHES of the val split."""
    was_training = drafter.training
    drafter.eval()
    val_loader = make_dataloader(
        make_dataset("val"),
        batch_size=EVAL_BATCH_SIZE,
        num_workers=2,
        shuffle=False,
    )
    val_iter = iter(val_loader)
    losses = []
    rng_state = torch.get_rng_state()
    torch.manual_seed(EVAL_SEED)
    try:
        for _ in range(EVAL_BATCHES):
            try:
                batch = next(val_iter)
            except StopIteration:
                break
            loss = train_step(drafter, target_policy, target_comp, batch, preprocessor, drafter_cfg)
            if loss is not None:
                losses.append(loss.item())
    finally:
        torch.set_rng_state(rng_state)
        if was_training:
            drafter.train()
    return sum(losses) / max(1, len(losses))


# ---------------------------------------------------------------------------
# Smoke check
# ---------------------------------------------------------------------------

def _smoke_sanity_check(target_policy, preprocessor, batch):
    """Verify wiring on a single batch before committing to a full run."""
    print("Smoke: sanity-checking target forward...")
    processed = preprocessor(batch)
    out = target_forward_with_hidden_states(target_policy, processed)

    hs = out["hidden_states"]
    n_layers_returned = len(hs) - 1  # minus the embedding output at index 0
    print(f"  hidden_states layers returned: {n_layers_returned} (expected {TARGET_NUM_LAYERS})")
    assert n_layers_returned == TARGET_NUM_LAYERS, (
        f"target language model returned {n_layers_returned} hidden state layers, "
        f"expected {TARGET_NUM_LAYERS}. output_hidden_states may be silently dropped."
    )

    fast_ids   = out["fast_token_ids"]
    fast_mask  = out["fast_token_mask"]
    fast_logits = out["fast_logits"]
    print(f"  fast_token_ids shape:  {tuple(fast_ids.shape)}")
    print(f"  fast_token_mask shape: {tuple(fast_mask.shape)}")
    print(f"  fast_logits shape:     {tuple(fast_logits.shape)}")
    assert fast_ids.shape == fast_mask.shape
    assert fast_logits.shape[:2] == fast_ids.shape

    seq_len = hs[0].shape[1]
    fast_offset = out["fast_offset"]
    assert 0 <= fast_offset < seq_len, f"bad fast_offset {fast_offset} vs seq_len {seq_len}"
    assert fast_offset + fast_ids.shape[1] == seq_len, (
        "fast tokens should end at end of sequence"
    )
    print("  target forward OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="One-batch wiring check: train 2 steps, eval 1 batch, print sanity asserts.")
    parser.add_argument("--wandb-project", default="autoresearch-dflash")
    parser.add_argument("--wandb-mode", default="online",
                        choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    drafter_cfg = DrafterConfig()
    train_cfg   = TrainConfig()
    device      = torch.device("cuda")

    if args.smoke:
        import prepare
        prepare.EVAL_BATCHES = 1
        train_cfg.num_workers = 0  # keep tracebacks readable

    wandb.init(
        project=args.wandb_project,
        mode="disabled" if args.smoke else args.wandb_mode,
        config={**drafter_cfg.__dict__, **train_cfg.__dict__, "time_budget": TIME_BUDGET},
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print("Loading frozen target (pi0-fast)...")
    train_ds = make_dataset("train")
    target_policy, preprocessor, postprocessor = load_target_policy(dataset=train_ds)
    target_comp = target_components(target_policy)

    print("Building drafter...")
    drafter = make_drafter(target_policy, drafter_cfg).to(device, dtype=torch.bfloat16)
    drafter.train()
    n_params = sum(p.numel() for p in drafter.parameters())
    print(f"Drafter params: {n_params/1e6:.1f}M")

    print("Building dataloader...")
    train_loader = make_dataloader(
        train_ds,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        [p for p in drafter.parameters() if p.requires_grad],
        lr=train_cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=train_cfg.weight_decay,
    )

    def lr_at(step, train_t0):
        if step < train_cfg.warmup_steps:
            return step / max(1, train_cfg.warmup_steps)
        if train_t0 is None:
            return 1.0
        progress = min(1.0, (time.time() - train_t0) / TIME_BUDGET)
        # Cosine decay from 1.0 -> 0.1 over the wall-clock training budget.
        return 0.1 + 0.45 * (1.0 + math.cos(math.pi * progress))

    if args.smoke:
        print("Smoke: running 2 train steps + 1 eval batch...")
    else:
        print(f"Training (budget = {TIME_BUDGET}s)...")
    overall_t0 = time.time()
    train_t0 = None
    step = 0
    losses = []
    data_iter = iter(train_loader)

    while True:
        if args.smoke and step >= 2:
            break
        if not args.smoke and train_t0 is not None and time.time() - train_t0 >= TIME_BUDGET:
            break
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        if args.smoke and step == 0:
            _smoke_sanity_check(target_policy, preprocessor, batch)

        for g in optimizer.param_groups:
            g["lr"] = train_cfg.lr * lr_at(step, train_t0)

        loss = train_step(drafter, target_policy, target_comp, batch, preprocessor, drafter_cfg)
        if loss is None:
            continue

        if args.smoke:
            assert torch.isfinite(loss), f"smoke: non-finite loss {loss}"

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(drafter.parameters(), train_cfg.grad_clip)
        optimizer.step()

        # Start the wall-clock budget *after* the first step so warmup compile/
        # cudagraph time doesn't eat into the budget.
        if train_t0 is None:
            train_t0 = time.time()

        losses.append(loss.item())
        step += 1
        wandb.log(
            {"train/loss": loss.item(), "train/lr": optimizer.param_groups[0]["lr"]},
            step=step,
        )
        if step % train_cfg.log_every == 0:
            recent = sum(losses[-train_cfg.log_every:]) / min(len(losses), train_cfg.log_every)
            elapsed = time.time() - train_t0
            print(f"step {step:5d} | loss {recent:.4f} | lr {optimizer.param_groups[0]['lr']:.2e} | t {elapsed:.0f}s")

    train_seconds = time.time() - train_t0
    print("Evaluating held-out loss...")
    eval_t0 = time.time()
    eval_loss = evaluate_loss(drafter, target_policy, target_comp, preprocessor, drafter_cfg)
    print(f"  eval_loss: {eval_loss:.4f}")
    print("Evaluating acceptance length...")
    accept_len = evaluate_acceptance_length(drafter, target_policy, preprocessor)
    eval_seconds = time.time() - eval_t0

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    total_seconds = time.time() - overall_t0

    wandb.log(
        {"eval/loss": eval_loss, "eval/accept_len": accept_len,
         "eval/peak_vram_mb": peak_vram_mb, "eval/train_seconds": train_seconds},
        step=step,
    )
    wandb.summary["accept_len"] = accept_len
    wandb.summary["eval_loss"]  = eval_loss
    wandb.finish()

    print()
    print("---")
    print(f"accept_len:       {accept_len:.4f}")
    print(f"training_seconds: {train_seconds:.1f}")
    print(f"eval_seconds:     {eval_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {n_params/1e6:.1f}")
    print(f"block_size:       {drafter_cfg.block_size}")
    print(f"draft_layers:     {drafter_cfg.num_hidden_layers}")


if __name__ == "__main__":
    main()
