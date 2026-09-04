#!/usr/bin/env bash
# The walk's expert envelope (docs/e2e-research/68 §4.2 on the dynamic
# task): certify ONE policy at a grid of law-parameter scales around the
# fit — every law parameter pinned at fit × s, no draw, pushes on, 40
# matched trials per point — and read where it fails.
#
#   tools/walk-envelope.sh <checkpoint.pt> [scales]
#   e.g. tools/walk-envelope.sh runs/studies/walk-c1/narrow/train/model_7999.pt "0.5 0.7 0.85 1.0 1.15 1.3 1.5 2.0"
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
ckpt="${1:?checkpoint}"; scales="${2:-0.5 0.7 0.85 1.0 1.15 1.3 1.5 2.0}"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
say() { echo "== $(date -u +%H:%M:%S) $*"; }
cd "$repo/rq_mjlab"
for s in $scales; do
  say "envelope: judge $(basename "$ckpt") at fit x $s"
  .venv/bin/python -m rq_mjlab.walk_verdict "$repo/$ckpt" --trials 40 --seed 1000 \
    --device cuda:0 --judge-at-scale "$s" --no-studio
done
say "ENVELOPE DONE"
