# Installation

Two supported setups — pick one:

- **conda + pip** (this guide) — recommended on shared hosts where `/` has limited space and the big volume is on a separate disk.
- **uv** — see [README.md](README.md#quick-start) for the upstream-style `uv sync` flow.

The conda recipe below is what's currently installed and verified to import on this machine.

## Requirements

- Linux + a single NVIDIA GPU (tested on A100 40 GB).
- CUDA 12.8 driver-compatible (the wheels here use the cu128 PyTorch index).
- Conda already configured to put envs/pkgs on a large volume (`conda config --show envs_dirs pkgs_dirs`).

## Why not just `pip install -e .` and `pip install -r requirements.txt`?

Three constraints push the install into a specific order:

1. **`torch==2.9.1` must come from the cu128 wheel index**, not PyPI (PyPI's `torch` wheels are CPU-only for some versions and would replace the cu128 build).
2. **`dflash`'s `pyproject.toml` declares no `[build-system]`**, so `pip install -e ./3rd-party/dflash` fails with *"build backend is missing the 'build_editable' hook"*. We install dflash **non-editable**; the project rules treat it as read-only anyway.
3. **`transformers==4.57.1`** (dflash's pin) requires `huggingface_hub<1.0`, but `lerobot` declares `huggingface_hub>=1.0`. The combo is unsatisfiable for pip's resolver, but runtime works fine with `huggingface_hub==0.36.x`. Pip emits a warning — ignore it.

Likewise, the `kernels` package must be pinned to `0.11.7`; newer versions (`0.14+`) require `huggingface_hub>=1.10` and fail at import time under the constraint above.

## Step-by-step

```bash
# 1. Submodules
git submodule update --init --recursive

# 2. Redirect pip cache + tmpdir away from the small root disk
export PIP_CACHE_DIR=/media/volume/Chau/.cache/pip
export TMPDIR=/media/volume/Chau/tmp
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

# 3. Fresh env on the big volume (Python 3.12 matches .python-version)
conda create -n autoresearch python=3.12 pip -y
conda activate autoresearch

# 4. torch 2.9.1 + torchvision 0.24.1 from the cu128 wheel index
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.9.1 torchvision==0.24.1

# 5. Rest of the project's runtime deps (kernels pinned to 0.11.7 — see notes above)
pip install \
    "kernels==0.11.7" "matplotlib>=3.10.8" "numpy>=2.2.6" \
    "pandas>=2.3.3" "pyarrow>=21.0.0" "requests>=2.32.0" \
    "rustbpe>=0.1.0" "tiktoken>=0.11.0" wandb

# 6. dflash (non-editable — see notes above)
pip install ./3rd-party/dflash[transformers]

# 7. lerobot (editable; pulls pi0-fast policy and LIBERO sim deps)
pip install -e "./3rd-party/lerobot[pi,libero]"

# 8. lerobot's install bumps torch back to 2.10 (CPU) and transformers to 5.x.
#    Force the cu128 build back and re-pin transformers.
pip install --index-url https://download.pytorch.org/whl/cu128 \
    --force-reinstall --no-deps torch==2.9.1 torchvision==0.24.1
pip install "transformers==4.57.1"
pip install "kernels==0.11.7"   # in case it was bumped by the above
```

## Verifying

```bash
python -c "
import sys; sys.path.insert(0, 'src')
import torch, transformers, huggingface_hub, lerobot
from dflash.model import extract_context_feature
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
import prepare
print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('hub', huggingface_hub.__version__)
print('prepare:', prepare.__file__, 'TIME_BUDGET:', prepare.TIME_BUDGET)
"
```

Expected output ends with `cuda: True`.

## Smoke run

After the imports clear:

```bash
python src/prepare.py            # one-time HF artifact download (~30 GB)
python scripts/train.py --smoke  # 2 train steps + 1 eval batch
```

If the smoke run prints a finite `accept_len`, the install is good.

## Pinned versions (reference)

The verified working state on this host:

| Package | Version |
|---|---|
| Python | 3.12.13 |
| torch | 2.9.1+cu128 |
| torchvision | 0.24.1+cu128 |
| transformers | 4.57.1 |
| huggingface_hub | 0.36.2 |
| kernels | 0.11.7 |
| lerobot | 0.5.2 (editable, from `3rd-party/lerobot/`) |
| dflash | 0.1.0 (non-editable, from `3rd-party/dflash/`) |

Pip will print resolver warnings about `huggingface_hub` and `kernels`. These are real version conflicts but cosmetic at runtime — leave them alone.
