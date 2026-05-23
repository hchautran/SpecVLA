"""
LIBERO chunk decode demo: side-by-side timing of naive AR pi0-fast vs
DFlash-drafter speculative decoding. Loads the trained drafter from
~/.cache/autoresearch/drafter.pt (produced by train.py) and pi0-fast-base
as the frozen target, pulls a few held-out LIBERO val chunks, runs both
decoders, and prints per-chunk timings + a summary table.

Greedy (temperature=0.0) so the two decoders should produce identical
action token sequences modulo numerical noise — the demo verifies this.

Usage:
    uv run train.py    # produces drafter.pt
    uv run demo.py     # 4 chunks by default; pass --num-chunks for more
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import argparse
import math
import time

import torch

from prepare import (
    CACHE_DIR, EVAL_SEED,
    load_target_policy,
    make_dataset, make_dataloader,
)
from bench import load_drafter, naive_ar_generate, spec_generate
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK


DEFAULT_CKPT = os.path.join(CACHE_DIR, "drafter.pt")


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_call(fn, *args, **kwargs):
    _cuda_sync()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    _cuda_sync()
    return out, (time.perf_counter() - t0) * 1000.0


def decode_instruction(tokens, mask, tokenizer):
    valid = tokens[mask.bool()]
    text = tokenizer.decode(valid.tolist(), skip_special_tokens=True)
    return text.strip().replace("\n", " ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--num-chunks", type=int, default=4)
    parser.add_argument("--max-decoding-steps", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"No drafter checkpoint at {args.checkpoint}; run train.py first.")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print("=" * 72)
    print(" DFlash drafter speedup demo on LIBERO")
    print("=" * 72)
    print()
    print("Loading frozen pi0-fast target...")
    train_ds = make_dataset("train")
    policy, preproc, _ = load_target_policy(dataset=train_ds)
    target_params = sum(p.numel() for p in policy.parameters()) / 1e9
    paligemma_tok = policy.model._paligemma_tokenizer

    print(f"Loading drafter from {args.checkpoint}...")
    drafter, drafter_cfg, saved_accept = load_drafter(policy, args.checkpoint, device)
    drafter_params = sum(p.numel() for p in drafter.parameters()) / 1e6

    max_decoding_steps = args.max_decoding_steps or policy.config.max_decoding_steps
    print()
    print(f"  Target:   pi0-fast-base, {target_params:.2f}B params (frozen)")
    print(f"  Drafter:  {drafter_params:.1f}M params, {drafter_cfg.num_hidden_layers}L, "
          f"block_size={drafter.block_size}, intermediate={drafter_cfg.intermediate_size}")
    print(f"  Decode:   greedy (temperature=0.0), max_steps={max_decoding_steps}")
    print(f"  Drafter teacher-forced accept_len at train end: {saved_accept:.3f}")
    print()

    # Pull val chunks deterministically. Batch size 1 because bench.spec_generate
    # is batch-size-1 (its accept_len decision is taken from sample 0).
    val_loader = make_dataloader(
        make_dataset("val"),
        batch_size=1, num_workers=2, shuffle=False,
    )
    val_iter = iter(val_loader)
    batches = []
    torch.manual_seed(EVAL_SEED)
    for _ in range(args.num_chunks):
        try:
            batch = next(val_iter)
        except StopIteration:
            break
        batches.append(preproc(batch))

    # Warmup once to amortize compile/cudnn.
    print("Warming up CUDA kernels...")
    _ = naive_ar_generate(policy, batches[0], max_decoding_steps)
    _ = spec_generate(policy, drafter, batches[0], max_decoding_steps)
    _cuda_sync()
    print()

    # Per-chunk side-by-side.
    print(f"{'#':>2}  {'task':40s}  {'naive ms':>9}  {'spec ms':>9}  "
          f"{'speedup':>8}  {'accept':>7}  {'match':>5}")
    print("-" * 88)

    naive_total, spec_total = 0.0, 0.0
    speedups, accept_means = [], []
    matches_total = 0
    for i, batch in enumerate(batches):
        # Show one task per chunk (first batch element).
        instr = decode_instruction(
            batch[OBS_LANGUAGE_TOKENS][0],
            batch[OBS_LANGUAGE_ATTENTION_MASK][0],
            paligemma_tok,
        )
        instr_short = (instr[:38] + "..") if len(instr) > 40 else instr

        naive_out, naive_ms = time_call(naive_ar_generate, policy, batch, max_decoding_steps)
        (spec_out, accs), spec_ms = time_call(spec_generate, policy, drafter, batch, max_decoding_steps)

        # Greedy => outputs should match (modulo numerical noise from parallel block forward).
        n_match = min(naive_out.shape[1], spec_out.shape[1])
        match = (naive_out[:, :n_match] == spec_out[:, :n_match]).all().item()
        match_str = "yes" if match else "no"

        speedup = naive_ms / spec_ms
        accept_mean = (sum(accs) / len(accs)) if accs else 0.0
        naive_total += naive_ms
        spec_total += spec_ms
        speedups.append(speedup)
        accept_means.append(accept_mean)
        matches_total += int(match)

        print(f"{i+1:>2}  {instr_short:40s}  {naive_ms:>7.0f} ms  {spec_ms:>7.0f} ms  "
              f"{speedup:>6.2f}x  {accept_mean:>5.2f}/8  {match_str:>5}")

    n = len(batches)
    print("-" * 88)
    avg_naive = naive_total / n
    avg_spec  = spec_total / n
    avg_speed = sum(speedups) / n
    avg_accept = sum(accept_means) / n
    print(f"    {'mean':40s}  {avg_naive:>7.0f} ms  {avg_spec:>7.0f} ms  "
          f"{avg_speed:>6.2f}x  {avg_accept:>5.2f}/8  {matches_total}/{n}")
    print()
    print("=" * 72)
    print(f" {avg_speed:.2f}x speedup  |  {avg_accept:.2f} of {drafter.block_size} tokens "
          f"accepted per draft round  |  outputs match: {matches_total}/{n}")
    print("=" * 72)


if __name__ == "__main__":
    main()
