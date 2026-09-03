#!/usr/bin/env bash
# The distill campaign (docs/66 D2, the paid half), on a rented card:
#   press N teacher demonstrations -> export -> lerobot-train a vision
#   student -> certify the student on 40 trials. Every stage's output
#   lands in ONE log the Studio's cloud feed tails; the final line is a
#   sentinel the feed turns into a card (a pod cannot stop itself
#   without the API key we never ship - docs/07 2026-09-01).
#
#   tools/campaign-distill.sh <run_root> <teacher.pt> [episodes] [steps] [worlds] [frame_every]
#   e.g. /workspace/robotiq/runs/campaign-2 runs/microduck-walk/<stamp>/model_7999.pt 120 30000 48 1
#
#   FROM=train tools/campaign-distill.sh ...   resumes at a stage (press |
#   export | train | certify): a campaign parked after its export
#   (2026-09-03) trains the next day from the dataset already on the
#   volume. BLANK=1 adds the blanked-camera certificate after the sighted
#   one (docs/07 2026-09-03, the plan's item 0).
#
# frame_every is the dataset's cadence in control ticks AND the student's
# stride at certificate time: they must be equal, and for this gait they
# must be 1. Measured 2026-09-02: the TEACHER itself, its actions held
# for 5 ticks, falls in 20-23 ticks (40/40); held for 2, 38/40 survive;
# every tick, 40/40. Campaign 1 pressed at 5 and certified a student
# that had learned its data well (normalized error 0.13) at 0/40.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
root="${1:?run_root}"; teacher="${2:?teacher checkpoint}"
episodes="${3:-240}"; steps="${4:-30000}"; worlds="${5:-48}"; frame_every="${6:-1}"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
mkdir -p "$root"
say() { echo "== $(date -u +%H:%M:%S) $*"; }
from="${FROM:-press}"; blank="${BLANK:-0}"
# Stages in order; `at <stage>` is true from the requested FROM stage on.
stages="press export train certify"
case " $stages " in *" $from "*) ;; *) echo "FROM must be one of: $stages" >&2; exit 2;; esac
reached=0
at() { [ "$reached" = 1 ] && return 0; [ "$1" = "$from" ] && reached=1; [ "$reached" = 1 ]; }

say "campaign: press $episodes episodes at 1/$frame_every ticks, train $steps steps (from $from)"
if at press; then
cd "$repo/rq_mjlab"
.venv/bin/python -m rq_mjlab.walk_press "$repo/$teacher" --out "$root/demos" \
  --episodes "$episodes" --worlds "$worlds" --seed 3000 --frame-every "$frame_every" --no-studio

fi

if at export; then
# Export onto the pod's LOCAL disk, then move the finished dataset to
# the run root in one pass: written straight onto the network volume,
# campaign 4's export sat at 40 % of one core for 14 minutes (I/O-bound
# on the volume, 2026-09-03); the writer fix (async threads) only pays
# on a local disk. EXPORT_SCRATCH overrides the staging directory.
scratch="${EXPORT_SCRATCH:-/tmp/rq-export}/$(basename "$root")"
rm -rf "$scratch"; mkdir -p "$scratch"
say "export (staged on $scratch)"
cd "$repo/pipeline"
.venv-train/bin/python -c "
from pathlib import Path
from rq_pipeline.collect.demo_export import export_batch
export_batch(Path('$root/demos'), Path('$scratch/dataset'), repo_id='rq-pipeline/microduck-walk-campaign', use_videos=False)
print('exported')"
say "move the dataset to $root/dataset"
rm -rf "$root/dataset" && mv "$scratch/dataset" "$root/dataset"

fi

if at train; then
say "train"
cd "$repo/pipeline"  # every stage owns its cwd: FROM=train skipped export's cd (2026-09-04)
.venv-train/bin/python -m lerobot.scripts.lerobot_train \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.chunk_size=20 --policy.n_action_steps=20 --policy.optimizer_lr=5e-5 \
  --dataset.repo_id=rq-pipeline/microduck-walk-campaign --dataset.root="$root/dataset" \
  --output_dir="$root/student" --job_name=walk-student \
  --steps="$steps" --batch_size=32 --num_workers=8 --log_freq=200 --save_freq=10000 \
  --wandb.enable=false

fi

at certify
say "certify the student (40 trials, cuda)"
cd "$repo/rq_mjlab"
.venv/bin/python -m rq_mjlab.walk_verdict "$repo/$teacher" --trials 40 --seed 1000 \
  --device cuda:0 --student "$root/student/checkpoints/last/pretrained_model" \
  --horizon 2 --stride "$frame_every" --no-studio
if [ "$blank" = 1 ]; then
say "certify the student with the camera BLANKED (plan item 0)"
.venv/bin/python -m rq_mjlab.walk_verdict "$repo/$teacher" --trials 40 --seed 1000 \
  --device cuda:0 --student "$root/student/checkpoints/last/pretrained_model" \
  --horizon 2 --stride "$frame_every" --no-studio --blank-camera
fi
say "CAMPAIGN DONE"
