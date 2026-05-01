"""
Benchmark spec decoding (drafter + pi0-fast) vs naive autoregressive pi0-fast
on LIBERO. Reports per-action-chunk wall-clock for both decoders, the resulting
speedup, mean acceptance length under real (non-teacher-forced) decoding, and
optionally end-to-end LIBERO success rate.

At pi0-fast's default `temperature=0.0`, both decoders produce bit-identical
action chunks, so success rate is decoder-independent and only needs to be
measured once.

Usage:
    uv run train.py             # writes ~/.cache/autoresearch/drafter.pt
    uv run bench.py             # per-chunk wall-clock on val split
    uv run bench.py --libero    # also runs the LIBERO success-rate sweep
"""

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import argparse
import math
import time

import torch

import prepare
from prepare import (
    CACHE_DIR,
    EVAL_BATCH_SIZE, EVAL_SEED,
    load_target_policy,
    make_dataset, make_dataloader,
    evaluate_libero_success_rate,
)
from train import DrafterConfig, make_drafter

from dflash.model import extract_context_feature
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK


DEFAULT_CKPT = os.path.join(CACHE_DIR, "drafter.pt")


# ---------------------------------------------------------------------------
# Drafter loading
# ---------------------------------------------------------------------------

def load_drafter(target_policy, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg_dict = ckpt["drafter_cfg"]
    cfg = DrafterConfig(**{k: v for k, v in cfg_dict.items()
                           if k in DrafterConfig.__dataclass_fields__})
    drafter = make_drafter(target_policy, cfg).to(device, dtype=torch.bfloat16)
    drafter.load_state_dict(ckpt["state_dict"])
    drafter.eval()
    saved_accept_len = float(ckpt.get("accept_len", float("nan")))
    return drafter, cfg, saved_accept_len


# ---------------------------------------------------------------------------
# Naive AR baseline — pi0-fast's own KV-cache decoder
# ---------------------------------------------------------------------------

@torch.no_grad()
def naive_ar_generate(policy, batch, max_decoding_steps):
    images, img_masks = policy._preprocess_images(batch)
    return policy.model.sample_actions_fast_kv_cache(
        images, img_masks,
        batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK],
        max_decoding_steps=max_decoding_steps,
        temperature=0.0,
    )


# ---------------------------------------------------------------------------
# Speculative decoding for pi0-fast
#
# Mirrors `sample_actions_fast_kv_cache` for the prefill (multimodal prefix
# [images | language | bos]), then runs `block_size` warmup AR steps to fill a
# rolling buffer of action-position target hidden states, then enters a
# draft-and-verify loop modeled on `dflash.model.dflash_generate`. Greedy
# (temperature=0) → output tokens are identical to the AR path modulo
# numerical noise from the parallel block forward.
# ---------------------------------------------------------------------------

def _block_attn_mask(inner, current_pad_mask, block_size, dtype):
    """4D mask: each new query attends to all valid past KV + causal-within-block."""
    B = current_pad_mask.shape[0]
    past_len = current_pad_mask.shape[1]
    device = current_pad_mask.device
    past_part = current_pad_mask[:, None, :].expand(B, block_size, past_len)
    block_causal = torch.tril(torch.ones(
        block_size, block_size, dtype=torch.bool, device=device,
    ))[None].expand(B, block_size, block_size)
    att_2d = torch.cat([past_part, block_causal], dim=-1)
    return inner._prepare_attention_masks_4d(att_2d, dtype=dtype)


@torch.no_grad()
def spec_generate(policy, drafter, batch, max_decoding_steps):
    """Returns (action_token_ids, accept_lengths). Batch size 1 supported."""
    inner = policy.model
    pg = inner.paligemma_with_expert.paligemma
    lm = pg.model.language_model
    lm_head = pg.lm_head
    embed_tokens = lm.embed_tokens
    hidden_size = embed_tokens.weight.shape[1]
    bf16 = lm.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
    embed_dtype = torch.bfloat16 if bf16 else torch.float32

    block_size = drafter.block_size
    mask_token_id = drafter.mask_token_id
    target_layer_ids = drafter.target_layer_ids

    images, img_masks = policy._preprocess_images(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks  = batch[OBS_LANGUAGE_ATTENTION_MASK]
    bsize, device = tokens.shape[0], tokens.device

    # ---- Prefill prefix [images | language | bos] ----
    bos = torch.full((bsize, 1), inner._paligemma_tokenizer.bos_token_id,
                     dtype=torch.long, device=device)
    tokens_in = torch.cat([tokens, bos], dim=1)
    masks_in  = torch.cat([masks, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1)
    prefix_embs, prefix_pad_masks, prefix_att_masks, _, _ = inner.embed_prefix_fast(
        images, img_masks, tokens_in, masks_in,
        fast_action_tokens=None, fast_action_masks=None,
    )
    if bf16:
        prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    att_4d = inner._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)
    prefix_out = lm.forward(
        inputs_embeds=prefix_embs, attention_mask=att_4d, position_ids=position_ids,
        past_key_values=None, use_cache=True, output_hidden_states=False,
    )
    past_kv = prefix_out.past_key_values
    next_token = lm_head(prefix_out.last_hidden_state[:, -1:, :])[:, -1].argmax(dim=-1, keepdim=True)

    generated = torch.zeros((bsize, max_decoding_steps), dtype=torch.long, device=device)
    generated[:, 0] = next_token.squeeze(-1)
    current_pad_mask = prefix_pad_masks
    target_hiddens_buf = []
    accept_lengths = []
    t = 1

    def _ar_step(next_token_in, pad_mask_in, past_kv_in):
        next_emb = embed_tokens(next_token_in) * math.sqrt(hidden_size)
        if bf16:
            next_emb = next_emb.to(dtype=torch.bfloat16)
        new_col = torch.ones((bsize, 1), dtype=torch.bool, device=device)
        pad_mask_out = torch.cat([pad_mask_in, new_col], dim=1)
        cur_pos = (pad_mask_out.sum(dim=1, keepdim=True) - 1).long()
        step_att = inner._prepare_attention_masks_4d(pad_mask_out.unsqueeze(1), dtype=embed_dtype)
        step_out = lm.forward(
            inputs_embeds=next_emb, attention_mask=step_att, position_ids=cur_pos,
            past_key_values=past_kv_in, use_cache=True, output_hidden_states=True,
        )
        ctx_step = extract_context_feature(step_out.hidden_states, target_layer_ids)
        new_logits = lm_head(step_out.last_hidden_state)
        new_token_ = new_logits[:, -1].argmax(dim=-1, keepdim=True)
        return new_token_, ctx_step, pad_mask_out, step_out.past_key_values

    # ---- Warmup: AR for `block_size` steps to fill the rolling ctx buffer ----
    warmup_steps = min(block_size, max_decoding_steps - t)
    for _ in range(warmup_steps):
        next_token, ctx_step, current_pad_mask, past_kv = _ar_step(
            next_token, current_pad_mask, past_kv,
        )
        target_hiddens_buf.append(ctx_step)
        if t < max_decoding_steps:
            generated[:, t] = next_token.squeeze(-1)
        t += 1

    if len(target_hiddens_buf) < block_size:
        return generated[:, :t], accept_lengths

    # ---- Spec phase ----
    while t < max_decoding_steps:
        # Tail of chunk too short for a full spec block — fall back to AR.
        if max_decoding_steps - t + 1 < block_size:
            next_token, ctx_step, current_pad_mask, past_kv = _ar_step(
                next_token, current_pad_mask, past_kv,
            )
            target_hiddens_buf.append(ctx_step)
            generated[:, t] = next_token.squeeze(-1)
            t += 1
            continue

        # Build masked block: [seed, mask, mask, ..., mask].
        block_ids = torch.cat([
            next_token,
            torch.full((bsize, block_size - 1), mask_token_id, dtype=torch.long, device=device),
        ], dim=1)

        # ---- Drafter (no KV cache; matches train/eval call) ----
        ctx = torch.cat(target_hiddens_buf[-block_size:], dim=1)
        noise_emb = embed_tokens(block_ids)
        block_positions = torch.arange(block_size, device=device)
        d_pos = block_positions.repeat(2)[None].expand(bsize, -1)
        draft_h = drafter(
            target_hidden=ctx, noise_embedding=noise_emb,
            position_ids=d_pos, past_key_values=None, use_cache=False, is_causal=False,
        )
        draft_logits = lm_head(draft_h)
        block_ids[:, 1:] = draft_logits[:, :-1].argmax(dim=-1)

        # ---- Target verify (parallel forward over the whole block) ----
        block_emb = embed_tokens(block_ids) * math.sqrt(hidden_size)
        if bf16:
            block_emb = block_emb.to(dtype=torch.bfloat16)
        att_4d_blk = _block_attn_mask(inner, current_pad_mask, block_size, embed_dtype)
        last_pos = (current_pad_mask.sum(dim=1, keepdim=True) - 1).long()
        block_pos = last_pos + 1 + torch.arange(block_size, device=device)[None, :]
        target_out = lm.forward(
            inputs_embeds=block_emb, attention_mask=att_4d_blk, position_ids=block_pos,
            past_key_values=past_kv, use_cache=True, output_hidden_states=True,
        )
        past_kv = target_out.past_key_values
        target_logits = lm_head(target_out.last_hidden_state)
        posterior = target_logits.argmax(dim=-1)

        # Greedy verify: accepted prefix length (target's prediction at block
        # position k matches drafter's proposal at block position k+1).
        matches = (block_ids[:, 1:] == posterior[:, :-1]).int()
        accept_len = matches.cumprod(dim=1).sum(dim=1)[0].item()

        # ---- Commit accept_len + 1 new tokens (accepted + 1 bonus) ----
        accepted = block_ids[:, 1:1 + accept_len]
        bonus = posterior[:, accept_len:accept_len + 1]
        new_tokens = torch.cat([accepted, bonus], dim=1)  # (B, accept_len + 1)
        new_count = min(new_tokens.shape[1], max_decoding_steps - t)
        generated[:, t:t + new_count] = new_tokens[:, :new_count]

        # Crop target cache to keep prefix + accepted block positions [0..accept_len].
        # Block position 0 = the seed; positions 1..accept_len = accepted proposals.
        # The bonus's KV is *not* in cache (it'll be re-forwarded next iter as the
        # next seed). Matches `dflash_generate`'s `past_key_values_target.crop(start)`.
        past_len = current_pad_mask.shape[1]
        keep = past_len + accept_len + 1
        past_kv.crop(keep)
        current_pad_mask = torch.cat([current_pad_mask, torch.ones(
            (bsize, accept_len + 1), dtype=torch.bool, device=device,
        )], dim=1)

        # Append target hiddens for the newly forwarded valid block positions
        # 0..accept_len. Block pos 0 is the seed (its hidden was NOT in the
        # buffer yet — buffer's last entry is the previous iter's bonus, but
        # that bonus hadn't been forwarded as input until now).
        target_hidden_full = extract_context_feature(target_out.hidden_states, target_layer_ids)
        for k in range(accept_len + 1):
            target_hiddens_buf.append(target_hidden_full[:, k:k + 1, :])

        next_token = bonus
        t += new_count
        accept_lengths.append(accept_len + 1)

    return generated[:, :t], accept_lengths


# ---------------------------------------------------------------------------
# Per-chunk wall-clock benchmark
# ---------------------------------------------------------------------------

def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_chunks(label, fn, batches):
    fn(batches[0])  # warmup (compile / cudnn benchmark)
    _cuda_sync()
    times = []
    for batch in batches:
        _cuda_sync()
        t0 = time.perf_counter()
        fn(batch)
        _cuda_sync()
        times.append(time.perf_counter() - t0)
    mean = sum(times) / len(times)
    print(f"  {label:24s}  mean={mean * 1000:7.1f} ms/chunk  "
          f"min={min(times) * 1000:7.1f}  max={max(times) * 1000:7.1f}  n={len(times)}")
    return mean, times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--num-batches", type=int, default=8,
                        help="Per-chunk benchmark batches (val split)")
    parser.add_argument("--max-decoding-steps", type=int, default=None,
                        help="Override pi0-fast's max_decoding_steps")
    parser.add_argument("--libero", action="store_true",
                        help="Also run LIBERO success-rate sweep (slow)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"No drafter checkpoint at {args.checkpoint}; run train.py first.")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print("Loading frozen pi0-fast target...")
    train_ds = make_dataset("train")
    policy, preproc, postproc = load_target_policy(dataset=train_ds)

    print(f"Loading drafter from {args.checkpoint}...")
    drafter, drafter_cfg, saved_accept_len = load_drafter(policy, args.checkpoint, device)
    print(f"  block_size={drafter.block_size}  layers={drafter_cfg.num_hidden_layers}  "
          f"params={sum(p.numel() for p in drafter.parameters()) / 1e6:.1f}M  "
          f"teacher_forced_accept_len={saved_accept_len:.3f}")

    max_decoding_steps = args.max_decoding_steps or policy.config.max_decoding_steps

    # ---- Pull val batches ----
    val_loader = make_dataloader(
        make_dataset("val"),
        batch_size=EVAL_BATCH_SIZE, num_workers=2, shuffle=False,
    )
    val_iter = iter(val_loader)
    batches = []
    torch.manual_seed(EVAL_SEED)
    for _ in range(args.num_batches):
        try:
            b = next(val_iter)
        except StopIteration:
            break
        batches.append(preproc(b))

    # ---- Per-chunk wall-clock ----
    print(f"\nBenchmark on {len(batches)} chunks "
          f"(max_decoding_steps={max_decoding_steps}, temperature=0.0):")
    naive_mean, _ = _time_chunks(
        "naive AR (pi0-fast)",
        lambda b: naive_ar_generate(policy, b, max_decoding_steps),
        batches,
    )
    spec_mean, _ = _time_chunks(
        "spec (drafter+target)",
        lambda b: spec_generate(policy, drafter, b, max_decoding_steps)[0],
        batches,
    )

    # ---- Acceptance length under real (non-teacher-forced) decoding ----
    accept_iters = []
    for b in batches:
        _, accs = spec_generate(policy, drafter, b, max_decoding_steps)
        accept_iters.extend(accs)
    real_accept = sum(accept_iters) / max(1, len(accept_iters))

    # ---- LIBERO success rate (greedy decode → equivalent across decoders) ----
    libero_succ = None
    if args.libero:
        print("\nLIBERO success-rate sweep "
              f"(suite={prepare.LIBERO_EVAL_SUITES}, "
              f"episodes_per_task={prepare.LIBERO_EVAL_EPISODES_PER_TASK})...")
        libero_succ = evaluate_libero_success_rate(policy, preproc, postproc)

    # ---- Summary ----
    print()
    print("---")
    print(f"naive_ms_per_chunk:       {naive_mean * 1000:.1f}")
    print(f"spec_ms_per_chunk:        {spec_mean * 1000:.1f}")
    print(f"speedup:                  {naive_mean / spec_mean:.2f}x")
    print(f"real_accept_len:          {real_accept:.3f}  "
          f"(over {len(accept_iters)} blocks, max = block_size = {drafter.block_size})")
    print(f"teacher_forced_accept_len: {saved_accept_len:.3f}  (training-time metric)")
    if libero_succ is not None:
        print(f"libero_success_rate:      {libero_succ:.3f}")


if __name__ == "__main__":
    main()
