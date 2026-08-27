#!/usr/bin/env bash
# The train venv on a rented machine — the WSL box's own recipe (docs/07
# 2026-08-26), verbatim: apt for the GL and rsync bits, uv, the lockfile's
# extras, then a torch/CUDA/mujoco check that names the card.
#
# Sent over ssh as `bash -s` by `tools/cloud-gpu.py bootstrap`, which
# exports the RQ_* variables first; the defaults below are its `Remote`.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
RQ_DIR="${RQ_DIR:-/workspace/robotiq}"
RQ_VENV="${RQ_VENV:-.venv-train}"
RQ_PYTHON="${RQ_PYTHON:-3.12.8}"
RQ_EXTRAS="${RQ_EXTRAS:---extra sim --extra viz --extra train}"
RQ_APT="${RQ_APT:-libegl1 libgl1 libglib2.0-0 rsync}"

# shellcheck disable=SC2086 # the package and extra lists are word lists
apt-get update -qq && apt-get install -y -qq --no-install-recommends $RQ_APT
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd "$RQ_DIR/pipeline"
# shellcheck disable=SC2086
UV_PROJECT_ENVIRONMENT="$RQ_VENV" uv sync --python "$RQ_PYTHON" $RQ_EXTRAS
"$RQ_VENV/bin/python" -c 'import torch, mujoco
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-",
      "mujoco", mujoco.__version__)'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
