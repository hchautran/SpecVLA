# AGENT_GUIDE — autonomous research loop

How this repo is structured for an autonomous AI agent (Claude, Codex, etc.) to iterate on the drafter without supervision. For the project description, results, and human-reproduction recipe, see [README.md](README.md). For the procedural skill the agent runs, see [program.md](program.md).

## The setup

The agent edits one file (`scripts/train.py`), trains for one fixed-time-budget epoch on the LIBERO action sequences, evaluates teacher-forced acceptance length and end-to-end speedup vs naive pi0-fast decoding, decides keep-or-discard, and repeats. You wake up to a log of experiments and (hopefully) a faster drafter.

The human edits [`program.md`](program.md) (the agent's skill, defining what to do step by step). The agent edits `scripts/train.py` (the drafter implementation).

## Files the agent touches

| File | Role | Agent permission |
|---|---|---|
| `scripts/train.py` | Drafter architecture (inlined `Drafter` class), block-masking, optimizer, training loop. | **Write** — the only file the agent edits. |
| `src/prepare.py` | Fixed constants, hub artifacts, dataloader, frozen-target loader, fixed `evaluate_acceptance_length` metric. | Read-only. |
| `scripts/bench.py` | Spec-decode vs naive-AR per-chunk wall-clock benchmark. | Read-only. |
| `3rd-party/dflash/` | Vendored DFlash package. Only `extract_context_feature` is imported (so train/eval agree on layer-id semantics); everything else the agent might want is inlined into `train.py`. | Read-only. |
| `3rd-party/lerobot/` | Vendored lerobot (pi0-fast policy + LIBERO dataset). | Read-only. |
| `results.tsv` | Append-only experiment log, written by the agent. | Append-only; untracked in git. |
| `program.md` | The agent's skill / step-by-step procedure. | Read; human-edited. |

## Metrics

- **`accept_len`** — mean teacher-forced acceptance length per drafted block, range `[1.0, block_size]`. Higher is better. This is the **primary** metric, computed by `prepare.evaluate_acceptance_length` (which the agent cannot change). 1.0 means zero usable proposals; `block_size` means every drafted block is fully accepted.
- **`speedup`** — wall-clock per-chunk speedup of (drafter + pi0-fast spec-decode) vs naive AR pi0-fast on the val split, from `scripts/bench.py`. Higher is better. **Secondary** — guards against drafters that are accurate-but-too-heavy to win on wall-clock.

The keep/discard rule: advance the branch only if `accept_len` improved **and** `speedup` did not regress materially. Otherwise reset.

## Running the agent

Spin up Claude / Codex / your-agent-of-choice in this repo (and disable permission prompts), then prompt:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

`program.md` is the agent's skill. It will:

1. Branch off `master` to `autoresearch/<tag>`.
2. Read in-scope files, run a smoke check, init `results.tsv`.
3. Loop forever: edit `scripts/train.py` → commit → train → bench → log `accept_len` and `speedup` to `results.tsv` → keep-if-improved-else-revert → repeat.

The agent runs autonomously until you interrupt it. Each experiment takes ~16 minutes (15-min train + ~1-min eval + ~30-s bench), so a single overnight session typically produces ~30 experiments.

## Design choices

- **Single file to modify.** Architecture, masking, optimizer, init — all editable from one place. The drafter class is inlined into `train.py` (not imported from `dflash/`) so the agent can change forward signatures, attention patterns, etc. without leaving the file.
- **Fixed time budget.** Each experiment trains for a fixed wall-clock window (`prepare.TIME_BUDGET = 900s`). Gives every architectural change a comparable optimizer-step budget — bigger drafters get fewer steps but the same wall-clock.
- **Two metrics, one decision rule.** `accept_len` primary, `speedup` secondary, keep iff both not-worse. Encodes the actual deployment objective: a drafter that's accurate but slow doesn't help.
- **Self-contained.** Single GPU, two files of in-scope code, two metrics, one decision rule. The whole project fits in a context window.

## Platform support

Tested on A100 40 GB. Should work on any single NVIDIA GPU with ≥24 GB VRAM (pi0-fast itself is ~3 B params + activations and dominates memory). CPU / MPS / multi-GPU are out of scope.

## License

MIT (matches upstream autoresearch).
