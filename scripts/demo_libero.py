"""
demo_libero.py — Side-by-side LIBERO video: naive AR vs DFlash-spec.

Produces an MP4 with two panes that start playback at the same wall-clock
moment (t=0) and run concurrently for one full LIBERO task episode:
  LEFT  — "naive AR (pi0-fast)" at its measured decode latency
  RIGHT — "spec (drafter + pi0-fast)" at its measured decode latency

Both panes show the same MuJoCo camera footage recorded by the original
LIBERO sim. For each robot-action chunk a pane freezes on the current
frame for `decode_ms` of wall-clock (rendered as freeze frames with a
"decoding..." overlay + progress bar), then plays the chunk's 1 second
of motion. The right pane finishes the whole episode well before the
left does and freezes on its last frame — that gap is the speedup.

Notes:
- Robot motion is identical between panes (greedy decoding produces the
  same actions modulo float noise); only decode latency differs.
- We can't render a *live* policy rollout because pi0fast-base is the
  base checkpoint, not LIBERO-finetuned, and `predict_action_chunk`
  returns all-<bos> garbage even on val batches (verified). The
  dataset's recorded frames stand in for what successful execution
  looks like.
- A full episode at `--decode-scale 1.0` runs ~90s. Pass
  `--decode-scale 0.3` to compress freeze portions while preserving
  the speedup *ratio* if you want a shorter clip.

Usage:
    uv run train.py        # produces drafter.pt
    uv run demo_libero.py  # default: full episode of one randomly chosen task
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import argparse
import time

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


def overlay(frame, lines, color=(255, 255, 255), bg_alpha=0.55):
    """Draw text lines on an HWC uint8 image (returns a copy)."""
    out = frame.copy()
    h, w = out.shape[:2]
    line_h = 22
    strip_h = line_h * len(lines) + 12
    strip = np.zeros_like(out[:strip_h])
    out[:strip_h] = (bg_alpha * strip + (1 - bg_alpha) * out[:strip_h]).astype(np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (10, line_h + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


def progress_bar(frame, frac, color=(80, 200, 80)):
    """Draw a thin progress bar across the bottom of the frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    bar_h = 6
    cv2.rectangle(out, (0, h - bar_h), (w, h), (40, 40, 40), -1)
    cv2.rectangle(out, (0, h - bar_h), (int(w * frac), h), color, -1)
    return out


def build_pane_timeline(chunks, decode_ms, output_fps, label, accent_color):
    """Build a list of frames-per-output-fps showing one decoder's run.

    For each chunk of robot frames:
      1. `decode_n` freeze frames showing the previous frame with a
         decoding overlay + progress bar growing 0→1.
      2. `output_fps` (1 sec) playback frames stretched from the chunk's
         dataset frames (chunk_size=10 frames @ 10 fps original).
    """
    out_frames = []
    decode_n = max(1, int(round(decode_ms * output_fps / 1000.0)))
    for c, chunk in enumerate(chunks):
        # Freeze on the last frame of the previous chunk (or first frame of c).
        seed_frame = chunks[c - 1][-1] if c > 0 else chunk[0]
        for i in range(decode_n):
            frac = (i + 1) / decode_n
            f = overlay(seed_frame, [
                f"{label}",
                f"decoding chunk {c+1}...  {decode_ms:.0f} ms",
            ], color=accent_color)
            f = progress_bar(f, frac, color=accent_color)
            out_frames.append(f)
        # Playback: stretch chunk_size dataset frames to `output_fps` output frames.
        n_chunk = len(chunk)
        for i in range(output_fps):
            idx = min(n_chunk - 1, (i * n_chunk) // output_fps)
            f = overlay(chunk[idx], [
                f"{label}",
                f"executing chunk {c+1}",
            ], color=accent_color)
            out_frames.append(f)
    return out_frames


def composite_side_by_side(left, right, divider_w=8, divider_color=(40, 40, 40)):
    """Place left + right panes horizontally; freeze on last frame for the
    shorter side until the longer side finishes."""
    n = max(len(left), len(right))
    if n == 0:
        return []
    h_l, w_l = left[0].shape[:2]
    h_r, w_r = right[0].shape[:2]
    H = max(h_l, h_r)
    W = w_l + divider_w + w_r
    out = []
    for i in range(n):
        canvas = np.full((H, W, 3), divider_color, dtype=np.uint8)
        l = left[i] if i < len(left) else left[-1]
        r = right[i] if i < len(right) else right[-1]
        canvas[:h_l, :w_l] = l
        canvas[:h_r, w_l + divider_w:w_l + divider_w + w_r] = r
        out.append(canvas)
    return out


def banner(width: int, naive_ms: float, spec_ms: float, accept_mean: float,
           block_size: int, drafter_m: float, task: str, height: int = 64) -> np.ndarray:
    """Title strip drawn above the two panes."""
    bar = np.full((height, width, 3), 18, dtype=np.uint8)
    speedup = naive_ms / spec_ms if spec_ms > 0 else 0.0
    cv2.putText(bar, f"DFlash drafter speedup demo  -  {task[:60]}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar,
                f"naive AR: {naive_ms:.0f} ms / chunk    spec: {spec_ms:.0f} ms / chunk    "
                f"speedup: {speedup:.2f}x    drafter: {drafter_m:.0f}M, "
                f"accept {accept_mean:.2f}/{block_size}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return bar


def stack_with_banner(banner_strip: np.ndarray, frames: list) -> list:
    out = []
    for f in frames:
        h_b, w_b = banner_strip.shape[:2]
        h_f, w_f = f.shape[:2]
        # banner width should match frame width; if not, pad/crop banner.
        if w_b != w_f:
            banner_strip = cv2.resize(banner_strip, (w_f, h_b))
            w_b = w_f
        canvas = np.zeros((h_b + h_f, w_f, 3), dtype=np.uint8)
        canvas[:h_b] = banner_strip
        canvas[h_b:] = f
        out.append(canvas)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--num-chunks", type=int, default=0,
                        help="How many decode-and-play chunks to render (each = 1 sec of "
                             "motion). 0 = use the full episode.")
    parser.add_argument("--output-fps", type=int, default=30,
                        help="MP4 output framerate")
    parser.add_argument("--decode-scale", type=float, default=1.0,
                        help="Multiplier on real decode-ms when rendering "
                             "(use 0.5 to compress a long video, 2.0 to dramatize)")
    parser.add_argument("--out", default="videos_out/sidebyside.mp4")
    parser.add_argument("--scan-chunks", type=int, default=80,
                        help="How many shuffled chunks to scan to find a task episode")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"No drafter checkpoint at {args.checkpoint}; run train.py first.")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print("Loading frozen pi0-fast target...")
    train_ds = make_dataset("train")
    policy, preproc, _ = load_target_policy(dataset=train_ds)

    print(f"Loading drafter from {args.checkpoint}...")
    drafter, drafter_cfg, _saved = load_drafter(policy, args.checkpoint, device)
    drafter_m = sum(p.numel() for p in drafter.parameters()) / 1e6
    chunk_size = policy.config.chunk_size
    max_decoding_steps = policy.config.max_decoding_steps

    # Pick one episode (first task we find via shuffled scan).
    ds = make_dataset(args.split)
    loader = make_dataloader(ds, batch_size=1, num_workers=2, shuffle=True)
    torch.manual_seed(EVAL_SEED)
    chosen_ep, chosen_task = None, None
    for i, raw in enumerate(loader):
        if i >= args.scan_chunks:
            break
        chosen_ep = int(raw["episode_index"][0])
        chosen_task = raw["task"][0] if isinstance(raw["task"], list) else str(raw["task"])
        break
    if chosen_ep is None:
        raise SystemExit("Couldn't find any task in scan window")
    print(f"Chosen task (ep={chosen_ep}): {chosen_task!r}")

    # Pull all frames for that episode from the underlying hf_dataset.
    ep_idx_col = ds.hf_dataset["episode_index"]
    idxs = [i for i, v in enumerate(ep_idx_col) if int(v) == chosen_ep]
    full_episode_chunks = len(idxs) // chunk_size
    requested = args.num_chunks if args.num_chunks > 0 else full_episode_chunks
    n_frames_needed = min(len(idxs), requested * chunk_size)
    idxs = idxs[:n_frames_needed]
    print(f"Episode {chosen_ep}: {len(idxs)} frames available, "
          f"rendering {requested} chunks ({len(idxs) // chunk_size} full chunks).")
    flat_frames = []
    for idx in idxs:
        s = ds[idx]
        img = s["observation.images.image"]  # (3, 256, 256) uint8
        flat_frames.append(img.permute(1, 2, 0).numpy())  # → HWC RGB

    # Group into chunks of `chunk_size` frames each.
    chunks = [flat_frames[i:i + chunk_size]
              for i in range(0, len(flat_frames) - chunk_size + 1, chunk_size)]
    if requested > 0 and len(chunks) > requested:
        chunks = chunks[:requested]

    # Time naive + spec ONCE on a representative batch from this episode.
    print("Timing decoders on a representative batch...")
    first_raw = ds[idxs[0]]
    batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v])
             for k, v in first_raw.items()}
    batch = preproc(batch)
    # Warmup
    _ = naive_ar_generate(policy, batch, max_decoding_steps)
    _ = spec_generate(policy, drafter, batch, max_decoding_steps)
    # Measure (take median of 3 to dampen variance)
    naive_runs = []
    spec_runs = []
    accept_runs = []
    for _ in range(3):
        _, n_ms = time_call(naive_ar_generate, policy, batch, max_decoding_steps)
        naive_runs.append(n_ms)
        (_, accs), s_ms = time_call(spec_generate, policy, drafter, batch, max_decoding_steps)
        spec_runs.append(s_ms)
        accept_runs.append((sum(accs) / len(accs)) if accs else 0.0)
    naive_ms = float(np.median(naive_runs))
    spec_ms  = float(np.median(spec_runs))
    accept_mean = float(np.median(accept_runs))
    speedup = naive_ms / spec_ms if spec_ms > 0 else 0.0
    print(f"  naive: {naive_ms:.0f} ms, spec: {spec_ms:.0f} ms ({speedup:.2f}x), "
          f"accept {accept_mean:.2f}/{drafter.block_size}")

    # Apply visualization scale (lets the user shorten very long videos).
    naive_ms_render = naive_ms * args.decode_scale
    spec_ms_render  = spec_ms  * args.decode_scale

    # Build per-pane timelines.
    print("Building side-by-side video...")
    left  = build_pane_timeline(chunks, naive_ms_render, args.output_fps,
                                "naive AR  (pi0-fast)", accent_color=(160, 160, 160))
    right = build_pane_timeline(chunks, spec_ms_render,  args.output_fps,
                                "spec  (drafter + pi0-fast)", accent_color=(120, 200, 255))
    print(f"  left: {len(left)} frames ({len(left)/args.output_fps:.1f}s)")
    print(f"  right: {len(right)} frames ({len(right)/args.output_fps:.1f}s)")

    composite = composite_side_by_side(left, right)
    if not composite:
        raise SystemExit("No frames produced; check chunk count")

    # Add a banner above with summary stats.
    bar = banner(composite[0].shape[1], naive_ms, spec_ms, accept_mean,
                 drafter.block_size, drafter_m, chosen_task)
    composite = stack_with_banner(bar, composite)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {out_path} ({len(composite)} frames @ {args.output_fps} fps)...")
    imageio.mimsave(out_path, composite, fps=args.output_fps, codec="libx264")
    print(f"Done. Total duration: {len(composite)/args.output_fps:.1f}s")


if __name__ == "__main__":
    main()
