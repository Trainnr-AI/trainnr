#!/usr/bin/env bash
# The train venv on a rented machine — the WSL machine's own recipe
# (2026-08-26), verbatim: apt for the GL and rsync bits, uv, the lockfile's
# extras, then a torch/CUDA/mujoco check that names the card.
#
# Sent over ssh as `bash -s` by `tools/cloud-gpu.py bootstrap`, which
# exports the TRAINNR_* variables first; the defaults below are its `Remote`.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
TRAINNR_DIR="${TRAINNR_DIR:-/workspace/trainnr}"
TRAINNR_VENV="${TRAINNR_VENV:-.venv-train}"
TRAINNR_PYTHON="${TRAINNR_PYTHON:-3.12.8}"
TRAINNR_EXTRAS="${TRAINNR_EXTRAS:---extra sim --extra viz --extra train}"
TRAINNR_APT="${TRAINNR_APT:-libegl1 libgl1 libglib2.0-0 rsync}"

# shellcheck disable=SC2086 # the package and extra lists are word lists
apt-get update -qq && apt-get install -y -qq --no-install-recommends $TRAINNR_APT
# uv's installer, pinned to a release and its SHA-256 rather than piped from
# a URL that can change (Scorecard's pinned-dependencies check, 2026-10-07).
# Move both lines together: the hash is of that version's installer.
UV_VERSION=0.12.23
UV_INSTALLER_SHA256=b8e6c43099ee9f9a550984d3ad56948457c689e7a99c090b35377234ac241491
if ! command -v uv >/dev/null; then
  curl -LsSf -o /tmp/uv-installer.sh "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-installer.sh"
  echo "$UV_INSTALLER_SHA256  /tmp/uv-installer.sh" | sha256sum -c - >/dev/null
  sh /tmp/uv-installer.sh && rm -f /tmp/uv-installer.sh
fi
export PATH="$HOME/.local/bin:$PATH"
cd "$TRAINNR_DIR/trainnr"
# shellcheck disable=SC2086
UV_PROJECT_ENVIRONMENT="$TRAINNR_VENV" uv sync --python "$TRAINNR_PYTHON" $TRAINNR_EXTRAS
"$TRAINNR_VENV/bin/python" -c 'import torch, mujoco
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-",
      "mujoco", mujoco.__version__)'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
