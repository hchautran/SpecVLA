"""
save_videos.py — Per-task LIBERO MP4s with naive-vs-spec timing overlay.

For each of the top-K distinct tasks found in val, this script picks one
episode, loads all its observation frames (the camera videos that were
recorded from the LIBERO MuJoCo sim during data collection), times naive
AR vs spec decoding on chunks of that episode, and writes an MP4 with
the timing overlay.

This isn't a live policy rollout (pi0-fast-base isn't LIBERO-finetuned,
so it can't generate valid actions in sim). It IS real LIBERO simulation
video — the frames come from the original sim runs that produced the
dataset — annotated with the wall-clock latency of both decoders so you
can see how much faster spec-decoding is per chunk.

Usage:
    uv run train.py            # produces drafter.pt
    uv run save_videos.py      # default: top 4 tasks, 60 frames each, fps=10
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import argparse
import time
from collections import defaultdict

import numpy as np
import torch
import imageio.v2 as imageio
import cv2

from prepare import (
    CACHE_DIR, EVAL_SEED,
    load_target_policy,
    make_dataset, make_dataloader,
)
from bench import load_drafter, naive_ar_generate, spec_generate


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


def overlay(frame, lines, color=(255, 255, 255)):
    """Draw text lines on an HWC uint8 image (in place)."""
    out = frame.copy()
    h, w = out.shape[:2]
    # Translucent strip for readability.
    strip_h = 22 * len(lines) + 12
    strip = out[:strip_h].copy()
    cv2.rectangle(strip, (0, 0), (w, strip_h), (0, 0, 0), -1)
    out[:strip_h] = (0.55 * strip + 0.45 * out[:strip_h]).astype(np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


def safe_filename(s: str, max_len: int = 60) -> str:
    s = "".join(c if (c.isalnum() or c in " -_") else "_" for c in s).strip()
    s = "_".join(s.split())
    return s[:max_len] or "task"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--top-k", type=int, default=4,
                        help="Number of distinct tasks to render")
    parser.add_argument("--frames-per-video", type=int, default=60,
                        help="How many frames of the chosen episode to render")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--out-dir", default="videos_out")
    parser.add_argument("--scan-chunks", type=int, default=80,
                        help="How many shuffled chunks to scan to find diverse tasks")
    parser.add_argument("--split", default="train", choices=["train", "val"],
                        help="Which split to sample from (val has only 2 tasks)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"No drafter checkpoint at {args.checkpoint}; run train.py first.")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading frozen pi0-fast target...")
    train_ds = make_dataset("train")
    policy, preproc, _ = load_target_policy(dataset=train_ds)

    print(f"Loading drafter from {args.checkpoint}...")
    drafter, drafter_cfg, _ = load_drafter(policy, args.checkpoint, device)
    max_decoding_steps = policy.config.max_decoding_steps
    drafter_m = sum(p.numel() for p in drafter.parameters()) / 1e6

    val_ds = make_dataset(args.split)
    val_loader = make_dataloader(val_ds, batch_size=1, num_workers=2, shuffle=True)

    # Step 1: scan shuffled chunks until we have one episode_index per task.
    print(f"Scanning up to {args.scan_chunks} shuffled val chunks for distinct tasks...")
    torch.manual_seed(EVAL_SEED)
    task_to_episode = {}
    for i, raw in enumerate(val_loader):
        if i >= args.scan_chunks:
            break
        task = raw["task"][0] if isinstance(raw["task"], list) else str(raw["task"])
        ep = int(raw["episode_index"][0])
        if task not in task_to_episode:
            task_to_episode[task] = ep
        if len(task_to_episode) >= args.top_k:
            break
    print(f"Found {len(task_to_episode)} distinct tasks:")
    for t, ep in task_to_episode.items():
        print(f"   ep={ep}  task={t!r}")

    # Step 2: for each task, load that episode's frames and time each chunk.
    # Episodes are contiguous in the dataset; iterate sequentially to gather frames.
    chunk_size = policy.config.chunk_size
    for task, ep in task_to_episode.items():
        # Find all frame indices for this episode.
        # LeRobotDataset has hf_dataset with an 'episode_index' column we can mask on,
        # but the simplest robust path is just to scan the underlying dataset.
        print(f"\nGathering frames for episode {ep} (task={task[:50]!r})...")
        frames_for_ep = []
        # find indices where episode_index == ep in the dataset
        ep_idx_col = val_ds.hf_dataset["episode_index"]
        idxs = [i for i, v in enumerate(ep_idx_col) if int(v) == ep]
        idxs = idxs[: args.frames_per_video]
        if not idxs:
            print(f"  no frames found for episode {ep}, skipping")
            continue

        # Pull samples one at a time so frame fetching is straightforward.
        for j, idx in enumerate(idxs):
            sample = val_ds[idx]
            img = sample["observation.images.image"]  # (3, 256, 256) uint8 tensor
            frames_for_ep.append(img.permute(1, 2, 0).numpy())  # → HWC

        # Time naive + spec on a couple of chunks from this episode (we treat each
        # CHUNK_SIZE frames as one decode call, in line with how LIBERO is run).
        # We use the first sample of the episode (containing tokenized text + the
        # action chunk) to get a representative timing.
        first_raw = val_ds[idxs[0]]
        # The dataloader produced batched dicts; reconstruct what preproc expects.
        batch = {}
        for k, v in first_raw.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0)
            else:
                batch[k] = [v]
        batch = preproc(batch)

        # Warmup once across tasks (cheap); then time naive + spec.
        _ = naive_ar_generate(policy, batch, max_decoding_steps)
        _ = spec_generate(policy, drafter, batch, max_decoding_steps)
        _, naive_ms = time_call(naive_ar_generate, policy, batch, max_decoding_steps)
        (_, accs), spec_ms = time_call(spec_generate, policy, drafter, batch, max_decoding_steps)
        speedup = naive_ms / spec_ms if spec_ms > 0 else 0.0
        accept_mean = (sum(accs) / len(accs)) if accs else 0.0
        print(f"  naive {naive_ms:.0f} ms, spec {spec_ms:.0f} ms ({speedup:.2f}x), "
              f"accept_mean {accept_mean:.2f}/{drafter.block_size}")

        # Compose annotated frames.
        lines = [
            f"Task: {task[:55]}",
            f"naive AR: {naive_ms:5.0f} ms     spec: {spec_ms:5.0f} ms     {speedup:.2f}x faster",
            f"drafter: {drafter_m:.0f}M params, {drafter_cfg.num_hidden_layers}L,  "
            f"accept {accept_mean:.2f}/{drafter.block_size}",
        ]
        annotated = [overlay(f, lines) for f in frames_for_ep]

        # Write MP4. Need BGR → RGB for imageio? imageio expects RGB; cv2 ops above
        # were on RGB-ordered uint8 arrays (from sample), so we're fine.
        path = out_dir / f"{safe_filename(task[:50])}.mp4"
        imageio.mimsave(path, annotated, fps=args.fps, codec="libx264")
        print(f"  wrote {path} ({len(annotated)} frames @ {args.fps} fps)")

    print()
    print(f"All videos written to {out_dir}/")


if __name__ == "__main__":
    main()
