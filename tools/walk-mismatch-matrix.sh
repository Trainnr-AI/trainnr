#!/usr/bin/env bash
# The mismatch matrix on the walk C1 policies (docs/70 §9, 2026-09-04):
# every trained policy — point / narrow ±0.10 / wide ±0.30, each with
# its replicates — judged (a) pinned at the fit scaled by s for each s
# in SCALES (no draw: "the actuator is s× what you measured"), and (b)
# under law DR drawn from ±0.30 (the folklore span as the judge). The
# at-fit and ±0.10 certificates already exist from tools/walk-c1-arm.sh.
# 40 matched trials, seed 1000, every certificate; ~1.5 min each.
#
#   tools/walk-mismatch-matrix.sh [study_root] [arms] [scales] [span]
#   e.g. tools/walk-mismatch-matrix.sh runs/studies/walk-c1 "point narrow wide" \
#          "0.7 0.8 0.9 1.1 1.2 1.3" 0.30
# Idempotent per certificate: a run whose certificate file exists is
# skipped, so a killed matrix resumes where it stopped. Fold with
# tools/walk-matrix-fold.py.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
root="${1:-runs/studies/walk-c1}"; [[ "$root" = /* ]] || root="$repo/$root"
arms="${2:-point narrow wide}"
scales="${3:-0.7 0.8 0.9 1.1 1.2 1.3}"
span="${4:-0.30}"
trials="${TRIALS:-40}"; seed="${SEED:-1000}"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
say() { echo "== $(date -u +%H:%M:%S) $*"; }

cd "$repo/rq_mjlab"
certify() { # <ckpt> <certificate file> <verdict args...>
  local ckpt="$1" cert="$2"; shift 2
  if [ -f "$(dirname "$ckpt")/verdict/$cert" ]; then say "have $cert for $ckpt"; return 0; fi
  .venv/bin/python -m rq_mjlab.walk_verdict "$ckpt" --trials "$trials" --seed "$seed" \
    --device cuda:0 --no-studio "$@"
}

total=0; done_n=0
for arm in $arms; do
  for run in "$root/$arm" "$root/$arm"#*; do
    [ -d "$run/train" ] || continue
    ckpt="$(ls "$run"/train/model_*.pt | sort -t_ -k2 -n | tail -1)"
    say "matrix: $(basename "$run") ($ckpt)"
    for s in $scales; do
      certify "$ckpt" "walk-verdict-at-x${s}-at-fit-cuda.json" --judge-at-fit --judge-at-scale "$s"
      done_n=$((done_n + 1))
    done
    certify "$ckpt" "walk-verdict-under-pm${span}-cuda.json" --judge-span "$span"
    done_n=$((done_n + 1))
    say "matrix: $(basename "$run") done ($done_n certificates so far)"
  done
done
say "MATRIX DONE ($done_n certificates)"
