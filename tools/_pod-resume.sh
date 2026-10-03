#!/usr/bin/env bash
# A fresh pod over the network volume, made ready in one line — the
# steps a bare container lacks (2026-09-03/04, docs/07): the GL and
# rsync apt bits, and the uv-managed Python BOTH venvs symlink to
# (`/root/.local/share/uv/python/cpython-<ver>`, the container disk a
# migrated pod kept and a fresh one does not). Idempotent; verifies
# both venvs import their heavy deps before returning 0.
#
#   ssh <door> 'bash -s' < tools/_pod-resume.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive PATH="/root/.local/bin:$PATH"
REPO="${TRAINNR_REPO:-/workspace/trainnr}"
PY="${TRAINNR_UV_PYTHON:-3.12.8}"   # the version both venvs' pyvenv.cfg name
if ! dpkg -s libegl1 libgl1 libglib2.0-0 rsync >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq libegl1 libgl1 libglib2.0-0 rsync >/tmp/trainnr-apt.log 2>&1
fi
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install "$PY" >/dev/null 2>&1 || true
cd "$REPO"
trainnr-mjlab/.venv/bin/python -c 'import torch, mujoco, warp; print("trainnr_mjlab venv ok, cuda", torch.cuda.is_available())'
trainnr/.venv-train/bin/python -c 'import torch, lerobot; print("train venv ok, cuda", torch.cuda.is_available())'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "pod ready: $REPO"
