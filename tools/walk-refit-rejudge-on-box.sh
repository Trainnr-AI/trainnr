#!/usr/bin/env bash
# Re-judge the refit arms' withdrawn certificates with the fixed DR seam
# (the log 2026-09-06, commit 62b999b: pins and declared spans are built
# from the bundle's params; before it the DR event took the refit
# bundle's INTERVAL bounds, so every pinned, drawn-±0.30 and ±0.10
# certificate of the six refit runs was judged in the wrong world —
# record walk-mismatch-matrix-refit-2026-09-06 WITHDRAWN, the ±0.10
# column of walk-c1-refit-2026-09-06 flagged). Runs where the GPU is
# free (the WSL box, 2026-09-07). The at-fit certificates stand and are
# kept. Every other certificate of the six runs is moved aside (to
# <aside>: evidence, not tracked; the originals are in git at 6bbc3a5)
# so the idempotent matrix script re-runs them; then the six ±0.10
# seconds (walk-c1-arm.sh's second certificate); then both folds
# rewrite their records in place — same id and study date, a `revised`
# note naming the re-judge — so the manuscript's citations keep
# resolving and the at-fit numbers do not move.
#
#   tools/walk-refit-rejudge-on-box.sh [rejudge_date] [aside_dir]
#
# Export the launch environment first (trainnr/wsl.env: MUJOCO_GL=egl,
# GALLIUM_DRIVER=d3d12, LD_LIBRARY_PATH for Warp), e.g. under
# tools/wsl-run.sh. Idempotent: a killed run resumes at the next
# missing certificate (rerun WITHOUT moving aside — set ASIDE=0).
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
rejudge="${1:-$(date -u +%F)}"
aside="${2:-${TMPDIR:-/tmp}/walk-refit-withdrawn-$rejudge}"
root="$repo/docs/artifacts/walk-c1"
study_date="2026-09-06"
bundle="robots/actuator-bundles/xl330-refit.m6.bundle.json"
scales="0.7 0.8 0.9 1.1 1.2 1.3"
runs=("point-refit#1" "point-refit#2" "point-refit#3" "identified#1" "identified#2" "identified#3")
say() { echo "== $(date -u +%H:%M:%S) $*"; }

for run in "${runs[@]}"; do
  [ -f "$root/$run/train/model_7999.pt" ] || { echo "missing $root/$run/train/model_7999.pt"; exit 1; }
done
if [ "${ASIDE:-1}" != 0 ]; then
  for run in "${runs[@]}"; do
    v="$root/$run/train/verdict"
    mkdir -p "$aside/$run"
    for f in "$v"/walk-verdict-at-x*-at-fit-cuda.json "$v"/records-at-x*-at-fit-cuda.jsonl \
             "$v"/walk-verdict-under-pm0.3-cuda.json "$v"/records-under-pm0.3-cuda.jsonl \
             "$v"/walk-verdict-cuda.json "$v"/records-cuda.jsonl; do
      if [ -e "$f" ]; then mv "$f" "$aside/$run/"; fi
    done
  done
  say "the withdrawn certificates of ${#runs[@]} runs moved aside to $aside"
fi

say "the matrix: ${#runs[@]} refit runs pinned at fit x {$scales} and drawn from ±0.30"
BUNDLE="$bundle" ROBOT=microduck "$repo/tools/walk-mismatch-matrix.sh" "$root" "point-refit identified" "$scales" 0.30

cd "$repo/trainnr_mjlab"
for run in "${runs[@]}"; do
  cert="$root/$run/train/verdict/walk-verdict-cuda.json"
  if [ -f "$cert" ]; then say "have the ±0.10 certificate for $run"; continue; fi
  say "certify $run under the walk's default span (±0.10, the second certificate)"
  MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -m trainnr_mjlab.walk_verdict "$root/$run/train/model_7999.pt" \
    --robot microduck --trials 40 --seed 1000 --device cuda:0 --no-studio --bundle "$repo/$bundle"
done

cd "$repo"
say "fold: the refit matrix, in place"
uv run --project trainnr python tools/walk-matrix-fold.py docs/artifacts/walk-c1 \
  --arms point-refit,identified --span 0.30 --date "$study_date" --id walk-mismatch-matrix-refit
say "fold: the refit arms, in place"
uv run --project trainnr python tools/walk-c1-fold.py docs/artifacts/walk-c1 \
  --arms point-refit,identified --id walk-c1-refit --date "$study_date"
say "REJUDGE DONE — add the revised notes (tools/walk-refit-rejudge-on-box.sh's docstring), review docs/findings, commit docs/artifacts/walk-c1 and docs/findings"
