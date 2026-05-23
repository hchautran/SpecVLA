# Installation

Use `uv` — this is the path that's actually verified to work end-to-end on this host (the conda recipe in earlier revisions of this file hit cascading version conflicts and is no longer recommended).

## Requirements

- Linux + a single NVIDIA GPU (tested on A100 40 GB, CUDA 12.8 driver).
- ~70 GB free on the disk where `uv` puts its cache and your `.venv` lives. If `/` is small, see the cache-redirect step.
- [`uv`](https://docs.astral.sh/uv/) installed. `curl -LsSf https://astral.sh/uv/install.sh | sh` puts it in `~/.local/bin`.

## Why this is more involved than `uv sync`

`pyproject.toml` declares only the leaf-level deps (`torch`, `kernels`, etc.); the heavy lifting — `transformers`, `lerobot`, `dflash` — is installed separately from the vendored submodules. Three constraints make the order matter:

1. **`uv.lock` does not include the submodules.** `uv sync` gives you `torch==2.9.1+cu128` and the small CPU deps. You install `dflash` and `lerobot` afterwards with `uv pip install`.
2. **`dflash` cannot be installed editable.** `3rd-party/dflash/pyproject.toml` declares no `[build-system]`, so PEP 660 editable mode fails with *"build backend is missing the 'build_editable' hook"*. Install non-editable; the project rules treat it as read-only anyway.
3. **`torchcodec` is fragile to install correctly.** Three issues stacked:
   - Wheel-version × torch-version ABI: torchcodec 0.10.x is built for torch 2.10, so it loads with `undefined symbol _ZN3c10...` against our torch 2.9.1.
   - The pip torchcodec wheel doesn't bundle FFmpeg, and dlopens the AV1 decoder (`libdav1d`) at decode time — without it the LIBERO AV1 video files fail with `Could not push packet to decoder: Invalid data found when processing input`.
   - The conda-forge `torchcodec` package links against conda's libtorch, which conflicts with the pip torch.

   **The fix that actually works: don't use torchcodec at all.** `lerobot` falls back to `pyav` automatically if `torchcodec` is uninstalled, and `pyav` 15.x handles AV1 fine.

## Step-by-step

```bash
# 0. (Optional) Redirect uv/pip caches and tmpdir off the root disk if `/` is small
export PIP_CACHE_DIR=/media/volume/Chau/.cache/pip
export TMPDIR=/media/volume/Chau/tmp
export UV_CACHE_DIR=/media/volume/Chau/.cache/uv
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$UV_CACHE_DIR"

# 1. Submodules
git submodule update --init --recursive

# 2. Project venv: torch (cu128) + kernels + numpy + pandas + ... from pyproject.toml / uv.lock
uv sync

# 3. dflash[transformers]  (non-editable — see notes above)
uv pip install "./3rd-party/dflash[transformers]"

# 4. lerobot[pi,libero]  (editable; pulls pi0-fast & LIBERO sim deps, bumps transformers to 5.3.0)
uv pip install -e "./3rd-party/lerobot[pi,libero]"

# 5. Drop torchcodec so lerobot uses pyav as the video backend
uv pip uninstall torchcodec
uv pip install "av>=14"   # already pulled in by lerobot[libero] — re-asserting for clarity
```

## Verifying

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
import torch, transformers, lerobot, av, prepare
from dflash.model import extract_context_feature
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('pyav', av.__version__)
print('prepare:', prepare.__file__, '| TIME_BUDGET:', prepare.TIME_BUDGET)
"
```

Expected last line: `cuda: True`.

## Smoke run

```bash
# (or:  uv run scripts/train.py --smoke)
.venv/bin/python scripts/prepare.py        # one-time HF artifact download (~30 GB)
.venv/bin/python scripts/train.py --smoke  # 2 train steps + 1 eval batch
```

A successful smoke prints a trailing block ending with `accept_len:`, `peak_vram_mb:`, etc. (`accept_len` will be 0.0 on a 2-step untrained run — that's expected; you're verifying the wiring, not the metric.)

## Pinned versions (reference)

Verified working state on this host:

| Package | Version | Notes |
|---|---|---|
| Python | 3.12.13 | from `.python-version`, managed by `uv` |
| torch | 2.9.1+cu128 | from `pyproject.toml`'s `[[tool.uv.index]] pytorch-cu128` |
| torchvision | 0.24.1+cu128 | matches torch 2.9 cu128 |
| transformers | 5.3.0 | from `uv.lock`; satisfies lerobot's pi0-fast (`get_image_features` returns a Tensor-with-pooler-output object in 5.x but a raw Tensor in 4.x) |
| huggingface_hub | 1.16.1 | pulled by transformers 5.3.0 |
| kernels | 0.14.x | works with hub 1.x |
| pyav (`av`) | 15.x | video decoder used by lerobot when torchcodec is absent |
| lerobot | 0.5.2 | editable, from `3rd-party/lerobot/` |
| dflash | 0.1.0 | non-editable, from `3rd-party/dflash/` |

## Things that don't work (recorded so you don't try them)

- **`pip install -e ./3rd-party/dflash`** — fails with PEP 660 error.
- **Conda env with pip-installed torch + conda-installed torchcodec** — libtorch ABI mismatch (`undefined symbol _ZN3c104cuda9SetDeviceEa`).
- **Pip torchcodec 0.10.x with our torch 2.9.1** — same ABI class of error (`_ZN3c1013MessageLogger6streamB5cxx11Ev`).
- **Pip torchcodec 0.9.x without `LD_PRELOAD=libdav1d.so.6`** — loads but fails at decode time on LIBERO's AV1 mp4s. Workable with LD_PRELOAD, but the no-torchcodec path is simpler.
