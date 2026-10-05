#!/usr/bin/env bash
# The per-axis mismatch matrices, run where the GPU is free (a local
# GPU workstation, 2026-09-05): the nine walk C1 checkpoints are tracked under
# docs/artifacts/walk-c1/<run>/train/ with their identities, so the
# matrix script can run straight on the repo tree and the certificates
# land beside the tracked ones, ready to commit. Then one fold per axis.
#
#   tools/walk-axes-local.sh [axes] [scales] [date]
#   e.g. tools/walk-axes-local.sh "kt friction R" "0.7 0.8 0.9 1.1 1.2 1.3" 2026-09-05
#
# Under WSL2, export the launch environment first (trainnr/wsl.env:
# MUJOCO_GL=egl, GALLIUM_DRIVER=d3d12, LD_LIBRARY_PATH for Warp).
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
axes="${1:-kt friction R}"
scales="${2:-0.7 0.8 0.9 1.1 1.2 1.3}"
date="${3:-$(date -u +%F)}"
root="$repo/docs/artifacts/walk-c1"
say() { echo "== $(date -u +%H:%M:%S) $*"; }

for run in point point#2 point#3 narrow narrow#2 narrow#3 wide wide#2 wide#3; do
  [ -f "$root/$run/train/model_7999.pt" ] || { echo "missing $root/$run/train/model_7999.pt — pull the checkpoints first"; exit 1; }
done
say "axes $axes over scales $scales on the nine tracked checkpoints"
ROBOT=microduck "$repo/tools/walk-mismatch-matrix.sh" "$root" "point narrow wide" "$scales" 0 "$axes"
for axis in $axes; do
  say "fold $axis"
  (cd "$repo" && uv run --project trainnr python tools/walk-matrix-fold.py "$root" --param "$axis" --date "$date")
done
say "AXES DONE — review docs/findings/walk-mismatch-matrix-*-$date.json, then commit docs/artifacts/walk-c1 and docs/findings"
