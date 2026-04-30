# autoresearch: DFlash drafter for pi0-fast

This is an experiment to train a small **DFlash-style block-diffusion drafter**
that proposes blocks of FAST action tokens in parallel for the autoregressive
**pi0-fast** VLA on the LIBERO benchmark. The agent (you) iterates on the
drafter — architecture, masking strategy, optimizer, training loop — while
pi0-fast itself stays frozen.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, hub artifacts, dataset/dataloader, frozen target loader, target forward with hidden states, **and the fixed acceptance-length metric**. Do not modify.
   - `train.py` — the file you modify. Drafter construction, masking, optimizer, training loop.
   - `dflash/dflash/model.py` — the canonical `DFlashDraftModel` you import. Read for context but don't edit (it's a vendored package).
4. **Verify data exists**: Check that `~/.cache/autoresearch/huggingface/hub/` contains snapshots of `lerobot/libero`, `lerobot/pi0fast_base`, `lerobot/fast-action-tokenizer`, and `google/paligemma-3b-pt-224`. If not, tell the human to run `uv run prepare.py`.
5. **Smoke-check the wiring**: Run `uv run train.py --smoke > smoke.log 2>&1`. This trains 2 steps and runs 1 eval batch. Confirm the run prints `accept_len` and a finite training loss. If the smoke run crashes, fix the bug or escalate to the human before starting the real loop.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding setup/eval). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: drafter depth/width, attention head counts, block size, mask token id, masking schedule, optimizer (AdamW / Muon / Lion / etc.), LR schedule, batch size, weight decay, gradient clipping, what fraction of positions to mask, whether to use a noise schedule à la diffusion vs. fixed all-but-first masking, whether to add auxiliary KL-to-target losses, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, dataset/dataloader, target loader, and training constants (time budget, target shape constants, eval batches, etc.).
- Modify the target. `pi0-fast` is frozen. `prepare.load_target_policy` already calls `requires_grad_(False)`; don't unfreeze it.
- Modify `dflash/`. The vendored package is treated as read-only. If you want a different drafter forward signature, write a new module under `train.py` rather than editing `dflash/`.
- Modify the evaluation harness. `evaluate_acceptance_length` in `prepare.py` is the ground-truth metric.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.

**The goal is simple: get the highest `accept_len`.** Note this is the opposite of the original autoresearch task — **higher is better**, since acceptance length measures how many of the drafter's parallel proposals the verifier (target) accepts in a row. Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything below the target/eval line is fair game.

**VRAM** is a soft constraint. The frozen pi0-fast target alone is heavy (~3B params + activations), so the budget for the drafter and training batch is tight. Some increase is acceptable for meaningful `accept_len` gains, but if you OOM, lower batch size or drafter width before doing anything more invasive.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
accept_len:       1.8423
training_seconds: 300.1
eval_seconds:     12.8
total_seconds:    345.6
peak_vram_mb:     58000.0
num_steps:        420
num_params_M:     86.4
block_size:       4
draft_layers:     2
```

`accept_len` is in `[1.0, block_size]`: 1.0 means the drafter produces zero usable proposals (only the target's bonus token gets through), `block_size` means every drafted block is fully accepted. You can extract the key metric from the log file:

```
grep "^accept_len:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	accept_len	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. `accept_len` achieved (e.g. 1.842300) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 56.6 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	accept_len	memory_gb	status	description
a1b2c3d	1.842300	56.6	keep	baseline
b2c3d4e	1.910100	57.0	keep	block_size 4 -> 6
c3d4e5f	1.812000	56.5	discard	swap mask token to <unk>
d4e5f6g	0.000000	0.0	crash	8-layer drafter (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^accept_len:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If `accept_len` improved (**higher**), you "advance" the branch, keeping the git commit
9. If `accept_len` is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes of training + ~1 minute of eval (acceptance-length eval pulls 16 batches of size 2 through the frozen target and the drafter). If a run exceeds 12 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**Sanity checks for ideas that don't move `accept_len`**: Loss should be falling steadily over the 5-minute window — if it isn't, your LR / masking strategy is likely wrong, not the architecture. A drafter that overfits to one position in the block (e.g. only learns position 1) shows up as `accept_len` plateauing near 2.0; vary the masking schedule to fix it.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read the DFlash paper referenced in `dflash/README.md`, re-read the in-scope files for new angles, try combining previous near-misses, try more radical drafter architectures (different attention patterns, different conditioning, etc.). The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~6 minutes (train + eval) then you can run approx 10/hour, for a total of about 80 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
