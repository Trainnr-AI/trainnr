#!/usr/bin/env bash
# One arm of the walk C1 study on one pod (docs/66 §8, 2026-09-04):
# train the G3 teacher recipe under a declared law-DR span, then judge
# it at the bundle's point fit (no law DR, pushes on) on 40 matched
# trials — so every arm is judged in the same world. Arms run one per
# pod in parallel; the study's finding folds their certificates.
#
#   tools/walk-c1-arm.sh <arm> <span> [iterations] [trials]
#   e.g. tools/walk-c1-arm.sh point 0 8000 40
#        tools/walk-c1-arm.sh narrow 0.10
#        tools/walk-c1-arm.sh wide 0.30
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
arm="${1:?arm name}"; span="${2:?law-DR span (0 = none)}"
iterations="${3:-8000}"; trials="${4:-40}"
root="$repo/runs/studies/walk-c1/$arm"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
mkdir -p "$root"
say() { echo "== $(date -u +%H:%M:%S) $*"; }

say "walk-c1 arm $arm: span $span, $iterations iterations"
cd "$repo/rq_mjlab"
.venv/bin/python -m rq_mjlab.walk_train --agent g3 --iterations "$iterations" \
  --dr-span "$span" --log-dir "$root/train" --no-recorder

ckpt="$(ls "$root"/train/model_*.pt | sort -t_ -k2 -n | tail -1)"
say "certify $arm at the fit: $ckpt ($trials trials, seed 1000)"
.venv/bin/python -m rq_mjlab.walk_verdict "$ckpt" --trials "$trials" --seed 1000 \
  --device cuda:0 --judge-at-fit --no-studio
say "certify $arm under its own DR too (the robustness view)"
.venv/bin/python -m rq_mjlab.walk_verdict "$ckpt" --trials "$trials" --seed 1000 \
  --device cuda:0 --no-studio
say "ARM DONE $arm"
