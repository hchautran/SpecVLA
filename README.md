# autoresearch — DFlash drafter for pi0-fast

![teaser](progress.png)

Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) repurposed
for a different task: train a small **DFlash-style block-diffusion drafter** that
proposes blocks of FAST action tokens in parallel for the autoregressive
**pi0-fast** VLA on the LIBERO benchmark, then have an autonomous AI agent
iterate on the drafter — architecture, masking strategy, optimizer, training
loop — while pi0-fast itself stays frozen.

The agent edits one file (`scripts/train.py`), trains for 1 epoch on the LIBERO action
sequences, evaluates teacher-forced acceptance length and end-to-end speedup
vs naive pi0-fast decoding, decides keep-or-discard, and repeats. You wake up
to a log of experiments and (hopefully) a faster drafter. The human edits
`program.md` (the agent's "skill"); the agent edits `train.py`.

## How it works

Five files matter:

- **`src/prepare.py`** — fixed constants, hub artifact downloads (LIBERO,
  pi0-fast-base, FAST tokenizer, PaliGemma), dataloader, frozen target loader,
  target forward with hidden states, and the **fixed teacher-forced
  acceptance-length metric**. Read-only.
- **`scripts/train.py`** — the file the agent edits. Drafter architecture
  (inlined, see `Drafter` class), block-masking strategy, optimizer, and
  1-epoch training loop. Saves a drafter checkpoint at the end. **Edited
  and iterated on by the agent.**
- **`scripts/bench.py`** — loads the saved drafter checkpoint and benchmarks
  speculative decoding (drafter + pi0-fast) vs naive autoregressive pi0-fast
  per chunk on the LIBERO val split. Reports the wall-clock speedup. Run
  after every `train.py`. Read-only for the agent.
- **`3rd-party/dflash/`** — vendored DFlash package. The drafter architecture
  is inlined into `train.py` (so the agent can modify it freely); only the
  `extract_context_feature` helper is still imported from here so train and
  eval agree on layer-id semantics.
- **`program.md`** — the agent's instructions. Point your agent at this and
  let it go. **Edited by the human.**

By design, each experiment runs for **1 full epoch** through the LIBERO train
split. The primary metric is **`accept_len`** — the mean teacher-forced
acceptance length per drafted block (range `[1.0, block_size]`, higher is
better). The secondary metric is **`speedup`** from `bench.py` — wall-clock
speed of (drafter + pi0-fast spec-decode) vs naive AR pi0-fast (higher is
better).

## Results

![latency per task](latency_per_task.png)

Per-chunk decode latency on LIBERO train tasks for the current drafter
(`max_decoding_steps=256`, greedy decode). Grey bars are naive AR pi0-fast,
blue bars are speculative decoding (drafter + pi0-fast); annotations show the
per-task speedup.

## Quick start

**Requirements:** A single NVIDIA GPU (tested on A100 40 GB), Python 3.10+.

Two install paths — pick one:

- **`uv`** (upstream-style, this section) — fastest if your environment is clean.
- **`conda` + `pip`** — see [INSTALLATION.md](INSTALLATION.md) for a tested recipe with the exact version pins this repo was verified against (use this if `uv sync` fights with your system, or if you need to put envs on a non-default disk).

```bash
# 1. Install uv (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install deps + vendored submodules
uv sync
git submodule update --init --recursive

# 3. Download HF artifacts (one-time, ~5 min, ~30 GB)
uv run src/prepare.py

# 4. Smoke check (2 train steps + 1 eval batch, ~30 s)
uv run scripts/train.py --smoke

# 5. Run a single training experiment (1 epoch, ~10–20 min)
uv run scripts/train.py

# 6. Benchmark spec decoding vs naive AR (~30 s)
uv run scripts/bench.py
```

If those all work, your setup is good and you can go into autonomous research mode.

## Running the agent

Spin up Claude / Codex / your-agent-of-choice in this repo (and disable
permission prompts), then prompt:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

`program.md` is the agent's skill. It will:

1. Branch off `master` to `autoresearch/<tag>`.
2. Read in-scope files, run a smoke check, init `results.tsv`.
3. Loop forever: edit `scripts/train.py` → commit → `scripts/train.py` →
   `scripts/bench.py` → log `accept_len` and `speedup` to `results.tsv` →
   keep-if-improved-else-revert → repeat.

## Project structure

```
src/
  prepare.py         — fixed constants, dataloader, frozen target, eval (do not modify)
scripts/
  train.py           — drafter architecture (inlined), training loop (agent modifies this)
  bench.py           — spec-decode vs naive AR speedup benchmark (do not modify)
  demo.py            — per-chunk decode demo (naive vs spec)
  demo_libero.py     — full LIBERO episode video demo
  save_videos.py     — render side-by-side decoder videos
3rd-party/
  dflash/            — vendored DFlash package (read-only)
  lerobot/           — vendored lerobot (read-only; required for pi0-fast & LIBERO)
program.md           — agent instructions
results.tsv          — append-only log of experiments (untracked; written by the agent)
pyproject.toml       — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. The drafter
  architecture is inlined as `Drafter` so attention patterns, depth, init,
  mask handling, etc. are all editable without leaving the file.
- **1-epoch budget.** Each experiment trains for one full pass through the
  LIBERO train split, then evaluates. Gives the agent a comparable budget
  across architectural changes — bigger drafters get less wall-clock per
  step but the same number of optimizer steps' worth of data.
- **Two metrics.** `accept_len` (teacher-forced, fixed eval) is primary —
  higher is better. `speedup` (spec-decode vs naive AR, end-to-end) is
  secondary — guards against drafters that are accurate but too heavy to
  actually win on wall-clock.
- **Self-contained.** Single GPU, single file (modulo vendored deps),
  two metrics.

## Platform support

Tested on A100 40 GB. Should work on any single NVIDIA GPU with ≥24 GB VRAM
(pi0-fast itself is ~3 B params and dominates memory). CPU / MPS / multi-GPU
are out of scope.

## License

MIT (matches upstream autoresearch).
