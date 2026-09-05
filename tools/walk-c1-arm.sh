#!/usr/bin/env bash
# One arm of the walk C1 study on one pod (docs/66 §8, 2026-09-04):
# train the G3 teacher recipe under a declared law-DR span, then judge
# it at the bundle's point fit (no law DR, pushes on) on 40 matched
# trials — so every arm is judged in the same world. Arms run one per
# pod in parallel; the study's finding folds their certificates.
#
#   tools/walk-c1-arm.sh <arm> <span> [iterations] [trials] [seed] [replicate]
#   e.g. tools/walk-c1-arm.sh point 0 8000 40          # run 1: mjlab's seed 42
#        tools/walk-c1-arm.sh narrow 0.10 8000 40 43 2 # replicate 2 under seed 43
#        ROBOT=go1 tools/walk-c1-arm.sh wide 0.30 10000 40 43 1  # C1 on mjlab's Go1
# A replicate lands under runs/studies/walk-c1[-<robot>]/<arm>#<replicate>/
# so the fold pools it with its arm (tools/walk-c1-fold.py).
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
arm="${1:?arm name}"; span="${2:?actuator-DR span (0 = none)}"
iterations="${3:-8000}"; trials="${4:-40}"; seed="${5:-}"; replicate="${6:-}"
robot="${ROBOT:-microduck}"
bundle_flag=""; [ -n "${BUNDLE:-}" ] && bundle_flag="--bundle $BUNDLE"  # e.g. robots/actuator-bundles/xl330-refit.m6.bundle.json
study="walk-c1"; [ "$robot" = microduck ] || study="walk-c1-$robot"
root="$repo/runs/studies/$study/$arm${replicate:+#$replicate}"
seed_flag=""; [ -n "$seed" ] && seed_flag="--seed $seed"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
mkdir -p "$root"
say() { echo "== $(date -u +%H:%M:%S) $*"; }

say "walk-c1 arm $arm${replicate:+ replicate $replicate} on $robot: span $span, $iterations iterations${seed:+, seed $seed}"
cd "$repo/rq_mjlab"
.venv/bin/python -m rq_mjlab.walk_train --robot "$robot" --agent g3 --iterations "$iterations" \
  --dr-span "$span" --log-dir "$root/train" --no-recorder $seed_flag $bundle_flag

ckpt="$(ls "$root"/train/model_*.pt | sort -t_ -k2 -n | tail -1)"
say "certify $arm at the fit: $ckpt ($trials trials, seed 1000)"
.venv/bin/python -m rq_mjlab.walk_verdict "$ckpt" --robot "$robot" --trials "$trials" --seed 1000 \
  --device cuda:0 --judge-at-fit --no-studio $bundle_flag
say "certify $arm under the walk's default span too (±0.10 for every arm — not its own)"
.venv/bin/python -m rq_mjlab.walk_verdict "$ckpt" --robot "$robot" --trials "$trials" --seed 1000 \
  --device cuda:0 --no-studio $bundle_flag
say "ARM DONE $arm"
