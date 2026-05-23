"""
One-time data and model preparation for autoresearch experiments.

Task: train a DFlash-style block-diffusion **drafter** for the autoregressive
pi0-fast VLA. The target (pi0-fast) is frozen; the drafter is trained to
predict blocks of FAST action tokens in parallel, conditioned on the target's
intermediate hidden states. The fixed metric is mean acceptance length under
teacher-forced verification.

Mirrors the role of the original LM-training prepare.py: fixed constants,
one-time download of artifacts, runtime utilities (dataloader, target forward,
eval) imported by train.py. Not modified by the agent.

Downloads:
- lerobot/libero                  (LIBERO demonstrations)
- lerobot/pi0fast_base            (target VLA, PaliGemma-Gemma backbone)
- lerobot/fast-action-tokenizer   (FAST action tokenizer)
- google/paligemma-3b-pt-224      (PaliGemma text tokenizer / image processor)

Usage:
    python prepare.py                  # full prep (data + model + tokenizers)
    python prepare.py --skip-data
    python prepare.py --skip-model

Caches under ~/.cache/autoresearch/. Sets HF_HOME and MUJOCO_GL there so HF
artifacts and the LIBERO sim (headless EGL) are pinned to the same place.

Assumes the vendored ./lerobot package is installed (e.g. `uv pip install
-e ./lerobot[pi,libero]`) and the vendored ./dflash package is on the path
(`uv pip install -e ./dflash[transformers]`). Single CUDA GPU is assumed.
"""

import os

# ---------------------------------------------------------------------------
# Cache paths and environment (must be set before any HF / MuJoCo import)
# ---------------------------------------------------------------------------

CACHE_DIR     = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR      = os.path.join(CACHE_DIR, "data")
MODELS_DIR    = os.path.join(CACHE_DIR, "models")
HF_CACHE_DIR  = os.path.join(CACHE_DIR, "huggingface")

os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(HF_CACHE_DIR, "hub"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import argparse
import time
from contextlib import nullcontext

import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 900                # training time budget in seconds (15 minutes)

# Hub artifacts — match docs/source/pi0fast.mdx LIBERO recipe
DATASET_REPO_ID    = "lerobot/libero"
PRETRAINED_POLICY  = "lerobot/pi0fast-base"
ACTION_TOKENIZER   = "lerobot/fast-action-tokenizer"
TEXT_TOKENIZER     = "google/paligemma-3b-pt-224"

# Pi0-Fast target settings (match the published LIBERO finetune)
CHUNK_SIZE         = 10
N_ACTION_STEPS     = 10
MAX_ACTION_TOKENS  = 256
EMPTY_CAMERAS      = 1
DTYPE              = "bfloat16"
GRADIENT_CKPT      = False        # target is frozen; no need
CONTROL_MODE       = "relative"

# Gemma-2B backbone shape (used to size the drafter)
TARGET_HIDDEN_SIZE     = 2048
TARGET_NUM_LAYERS      = 18

# Acceptance-length eval (the fixed metric — DO NOT change in train.py)
EVAL_BATCHES           = 16       # held-out batches drawn from val split
EVAL_BATCH_SIZE        = 2
EVAL_SEED              = 0

# Optional end-to-end LIBERO eval (opt-in; not the fixed metric)
LIBERO_EVAL_SUITES              = ("libero_object",)
LIBERO_EVAL_EPISODES_PER_TASK   = 1
LIBERO_EVAL_BATCH_SIZE          = 1
LIBERO_EVAL_MAX_PARALLEL_TASKS  = 1


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _ensure_dirs():
    for d in (CACHE_DIR, DATA_DIR, MODELS_DIR, HF_CACHE_DIR):
        os.makedirs(d, exist_ok=True)


def download_dataset():
    from huggingface_hub import snapshot_download
    print(f"Data: downloading {DATASET_REPO_ID}...")
    t0 = time.time()
    path = snapshot_download(
        repo_id=DATASET_REPO_ID, repo_type="dataset",
        cache_dir=os.path.join(HF_CACHE_DIR, "hub"),
    )
    print(f"Data: ready in {time.time() - t0:.1f}s at {path}")


def download_model():
    from huggingface_hub import snapshot_download
    for repo_id in (PRETRAINED_POLICY, ACTION_TOKENIZER, TEXT_TOKENIZER):
        print(f"Model: downloading {repo_id}...")
        t0 = time.time()
        path = snapshot_download(
            repo_id=repo_id, repo_type="model",
            cache_dir=os.path.join(HF_CACHE_DIR, "hub"),
        )
        print(f"Model: {repo_id} ready in {time.time() - t0:.1f}s at {path}")


# ---------------------------------------------------------------------------
# Dataset / dataloader
# ---------------------------------------------------------------------------

def _resolve_delta_timestamps(ds_meta):
    from lerobot.utils.constants import ACTION
    return {ACTION: [i / ds_meta.fps for i in range(CHUNK_SIZE)]}


def make_dataset(split: str = "train"):
    """Build a LeRobotDataset over lerobot/libero with action chunks of length CHUNK_SIZE.

    `split` controls which episodes are used: "train" / "val" deterministically
    partition by episode index (90/10) so eval is held-out.
    """
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    ds_meta = LeRobotDatasetMetadata(DATASET_REPO_ID)
    delta_timestamps = _resolve_delta_timestamps(ds_meta)

    n_eps = ds_meta.total_episodes
    val_step = max(1, n_eps // 10)
    val_eps = list(range(0, n_eps, val_step))
    if split == "val":
        episodes = val_eps
    elif split == "train":
        val_set = set(val_eps)
        episodes = [i for i in range(n_eps) if i not in val_set]
    else:
        raise ValueError(f"Unknown split {split!r}")

    return LeRobotDataset(
        DATASET_REPO_ID,
        delta_timestamps=delta_timestamps,
        episodes=episodes,
        return_uint8=True,
    )


def make_dataloader(dataset, batch_size, num_workers=4, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


# ---------------------------------------------------------------------------
# Target policy (frozen pi0-fast)
# ---------------------------------------------------------------------------

def _make_policy_config():
    from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
    return PI0FastConfig(
        chunk_size=CHUNK_SIZE,
        n_action_steps=N_ACTION_STEPS,
        max_action_tokens=MAX_ACTION_TOKENS,
        empty_cameras=EMPTY_CAMERAS,
        dtype=DTYPE,
        gradient_checkpointing=GRADIENT_CKPT,
        action_tokenizer_name=ACTION_TOKENIZER,
        text_tokenizer_name=TEXT_TOKENIZER,
        device="cuda" if torch.cuda.is_available() else "cpu",
        pretrained_path=PRETRAINED_POLICY,
    )


def load_target_policy(dataset=None):
    """Load lerobot/pi0fast_base as a frozen target.

    Returns (policy, preprocessor, postprocessor). The policy's parameters
    have requires_grad=False and the policy is in eval mode.
    """
    from lerobot.policies import make_policy, make_pre_post_processors

    cfg = _make_policy_config()
    if dataset is None:
        dataset = make_dataset("train")

    policy = make_policy(cfg=cfg, ds_meta=dataset.meta)

    device = str(cfg.device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=PRETRAINED_POLICY,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**cfg.input_features, **cfg.output_features},
                "norm_map": cfg.normalization_mapping,
            },
            "action_tokenizer_processor": {"action_tokenizer_name": ACTION_TOKENIZER},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": cfg.output_features,
                "norm_map": cfg.normalization_mapping,
            },
        },
    )

    for p in policy.parameters():
        p.requires_grad_(False)
    policy.eval()
    return policy, preprocessor, postprocessor


def target_components(policy):
    """Convenience accessors that the drafter shares with the target."""
    inner = policy.model  # PI0FastPytorch
    pg = inner.paligemma_with_expert.paligemma
    return {
        "language_model": pg.model.language_model,   # Gemma-2B trunk
        "embed_tokens":   pg.model.language_model.embed_tokens,
        "lm_head":        pg.lm_head,
        "paligemma_tokenizer": inner._paligemma_tokenizer,
    }


# ---------------------------------------------------------------------------
# Target forward with all-layer hidden states
#
# We re-use pi0-fast's `embed_prefix_fast` to build the [images | language |
# fast_tokens] embedding sequence, then call the Gemma trunk directly with
# `output_hidden_states=True`. The trunk returns hidden states for each of
# the 18 layers + the embedding layer — DFlash uses these as conditioning.
# ---------------------------------------------------------------------------

@torch.no_grad()
def target_forward_with_hidden_states(policy, batch):
    """Run the frozen target on a preprocessed batch and return:

    Returns:
        hidden_states_list: tuple of (num_layers + 1) tensors, each
            (B, seq_len, hidden_size). Index 0 is the embedding output;
            index L is the output of the L-th transformer layer.
        fast_logits: (B, T_fast, vocab_size) — target's own next-token logits
            at action positions, useful as a teacher signal.
        fast_token_ids: (B, T_fast) — ground-truth FAST action token labels.
        fast_token_mask: (B, T_fast) — valid (non-padding) action positions.
        fast_offset: int — index into seq_len where FAST tokens start.
    """
    from lerobot.utils.constants import (
        ACTION_TOKENS, ACTION_TOKEN_MASK,
        OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK,
    )

    inner = policy.model  # PI0FastPytorch
    images, img_masks = policy._preprocess_images(batch)
    fast_tokens = batch[ACTION_TOKENS]
    fast_masks  = batch[ACTION_TOKEN_MASK]
    tokens      = batch[OBS_LANGUAGE_TOKENS]
    masks       = batch[OBS_LANGUAGE_ATTENTION_MASK]

    prefix_embs, pad_masks, att_masks, _, num_fast_embs = inner.embed_prefix_fast(
        images, img_masks, tokens, masks,
        fast_action_tokens=fast_tokens,
        fast_action_masks=fast_masks,
    )
    lm = inner.paligemma_with_expert.paligemma.model.language_model
    if lm.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
        prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    att_2d_4d = inner._prepare_attention_masks_4d(att_masks, dtype=prefix_embs.dtype)

    out = lm.forward(
        inputs_embeds=prefix_embs,
        attention_mask=att_2d_4d,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        output_hidden_states=True,
    )
    hidden_states_list = out.hidden_states
    fast_offset = prefix_embs.shape[1] - num_fast_embs

    lm_head = inner.paligemma_with_expert.paligemma.lm_head
    fast_hidden = out.last_hidden_state[:, fast_offset:, :]
    fast_logits = lm_head(fast_hidden)

    return {
        "hidden_states":   hidden_states_list,
        "fast_logits":     fast_logits,
        "fast_token_ids":  fast_tokens,
        "fast_token_mask": fast_masks,
        "fast_offset":     fast_offset,
    }


def fast_hidden_at_action_positions(hidden_states_list, fast_offset):
    """Slice hidden_states_list to keep only the action-token positions."""
    return tuple(h[:, fast_offset:, :] for h in hidden_states_list)


# ---------------------------------------------------------------------------
# Evaluation — teacher-forced acceptance length (FIXED METRIC)
#
# For each (batch, starting position t) over a held-out val split:
#   - Take a length-`block_size` block of true action tokens y[t..t+B-1].
#   - Drafter input: noise_embedding for [y[t], <mask>, <mask>, ..., <mask>]
#     (position t given by target as in real spec decoding; positions
#     t+1..t+B-1 masked).
#   - Drafter output (argmax) at positions t+1..t+B-1 are compared to true
#     tokens; accepted length = number of consecutive matches starting at
#     t+1, plus the bonus token from the target. This matches the verifier
#     in dflash.model.dflash_generate.
# Mean over (t, batch) is the metric. Higher is better.
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_acceptance_length(drafter, policy, preprocessor):
    """Mean teacher-forced acceptance length (in tokens). Fixed metric."""
    from torch.utils.data import DataLoader

    drafter.eval()
    target_comp = target_components(policy)
    embed_tokens = target_comp["embed_tokens"]
    lm_head      = target_comp["lm_head"]

    val_loader = make_dataloader(
        make_dataset("val"),
        batch_size=EVAL_BATCH_SIZE,
        num_workers=2,
        shuffle=False,
    )
    val_iter = iter(val_loader)

    block_size = drafter.block_size
    mask_token_id = drafter.mask_token_id
    target_layer_ids = drafter.target_layer_ids

    total_accepted = 0.0
    total_blocks   = 0

    torch.manual_seed(EVAL_SEED)
    for _ in range(EVAL_BATCHES):
        try:
            batch = next(val_iter)
        except StopIteration:
            break
        batch = preprocessor(batch)

        target_out = target_forward_with_hidden_states(policy, batch)
        fast_hiddens = fast_hidden_at_action_positions(
            target_out["hidden_states"], target_out["fast_offset"]
        )
        true_tokens = target_out["fast_token_ids"]
        true_mask   = target_out["fast_token_mask"]
        B, T = true_tokens.shape
        if T < block_size + 1:
            continue

        # Sample one block per batch element at a random valid start position.
        max_start = T - block_size
        starts = torch.randint(0, max_start, (B,), device=true_tokens.device)

        # Build masked block input: position 0 is true token; rest are <mask>.
        block_ids = torch.full((B, block_size), mask_token_id,
                               dtype=torch.long, device=true_tokens.device)
        for b in range(B):
            block_ids[b, 0] = true_tokens[b, starts[b]]

        # Concat target hidden across selected layers and slice the block window.
        from dflash.model import extract_context_feature  # noqa: WPS433
        ctx = extract_context_feature(fast_hiddens, target_layer_ids)
        ctx_blocks = torch.stack(
            [ctx[b, starts[b]:starts[b] + block_size] for b in range(B)], dim=0
        )

        noise_emb = embed_tokens(block_ids)
        block_positions = torch.arange(block_size, device=block_ids.device)
        position_ids = block_positions.repeat(2)[None].expand(B, -1)
        h = drafter(
            target_hidden=ctx_blocks,
            noise_embedding=noise_emb,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            is_causal=False,
        )
        logits = lm_head(h)
        pred = logits.argmax(dim=-1)  # (B, block_size)

        # Verify against true tokens t+1..t+B-1; accepted length follows the
        # cumprod-of-matches scheme used in dflash_generate.
        true_block = torch.stack(
            [true_tokens[b, starts[b]:starts[b] + block_size] for b in range(B)], dim=0
        )
        valid_block = torch.stack(
            [true_mask[b, starts[b]:starts[b] + block_size] for b in range(B)], dim=0
        )
        match = (pred[:, :-1] == true_block[:, 1:]).int()
        accepted = (match.cumprod(dim=1).sum(dim=1) + 1).float()  # +1 for the target's bonus
        total_accepted += (accepted * valid_block[:, 0].float()).sum().item()
        total_blocks   += valid_block[:, 0].float().sum().item()

    return total_accepted / max(1, total_blocks)


# ---------------------------------------------------------------------------
# Optional end-to-end LIBERO eval (NOT the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_libero_success_rate(policy, preprocessor, postprocessor, *, use_amp=True):
    """End-to-end success rate on a small LIBERO suite. Slow; opt-in."""
    from lerobot.envs import make_env
    from lerobot.envs.factory import make_env_config, make_env_pre_post_processors
    from lerobot.scripts.lerobot_eval import close_envs, eval_policy_all

    env_cfg = make_env_config(
        "libero",
        task=",".join(LIBERO_EVAL_SUITES),
        control_mode=CONTROL_MODE,
    )
    envs = make_env(env_cfg, n_envs=LIBERO_EVAL_BATCH_SIZE, use_async_envs=False)
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    device = next(policy.parameters()).device
    amp_ctx = torch.autocast(device_type=device.type) if use_amp else nullcontext()

    policy.eval()
    try:
        with amp_ctx:
            info = eval_policy_all(
                envs=envs,
                policy=policy,
                env_preprocessor=env_pre,
                env_postprocessor=env_post,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=LIBERO_EVAL_EPISODES_PER_TASK,
                max_episodes_rendered=0,
                videos_dir=None,
                start_seed=EVAL_SEED,
                max_parallel_tasks=LIBERO_EVAL_MAX_PARALLEL_TASKS,
            )
    finally:
        close_envs(envs)

    return float(info["overall"]["pc_success"]) / 100.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and target for DFlash drafter training")
    parser.add_argument("--skip-data",  action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()

    _ensure_dirs()
    print(f"Cache directory: {CACHE_DIR}")
    print(f"HF_HOME:         {os.environ['HF_HOME']}")
    print()

    if not args.skip_data:
        download_dataset()
        print()
    if not args.skip_model:
        download_model()
        print()

    print("Done! Ready to train.")
