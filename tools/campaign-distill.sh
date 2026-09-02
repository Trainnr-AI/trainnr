#!/usr/bin/env bash
# The distill campaign (docs/66 D2, the paid half), on a rented card:
#   press N teacher demonstrations -> export -> lerobot-train a vision
#   student -> certify the student on 40 trials. Every stage's output
#   lands in ONE log the Studio's cloud feed tails; the final line is a
#   sentinel the feed turns into a card (a pod cannot stop itself
#   without the API key we never ship - docs/07 2026-09-01).
#
#   tools/campaign-distill.sh <run_root> <teacher.pt> [episodes] [steps] [worlds]
#   e.g. /workspace/robotiq/runs/campaign-1 runs/microduck-walk/<stamp>/model_7999.pt 240 30000 48
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
root="${1:?run_root}"; teacher="${2:?teacher checkpoint}"
episodes="${3:-240}"; steps="${4:-30000}"; worlds="${5:-48}"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
mkdir -p "$root"
say() { echo "== $(date -u +%H:%M:%S) $*"; }

say "campaign: press $episodes episodes, train $steps steps"
cd "$repo/rq_mjlab"
.venv/bin/python -m rq_mjlab.walk_press "$repo/$teacher" --out "$root/demos" \
  --episodes "$episodes" --worlds "$worlds" --seed 3000 --frame-every 5 --no-studio

say "export"
cd "$repo/pipeline"
.venv-train/bin/python -c "
from pathlib import Path
from rq_pipeline.collect.demo_export import export_batch
export_batch(Path('$root/demos'), Path('$root/dataset'), repo_id='rq-pipeline/microduck-walk-campaign', use_videos=False)
print('exported')"

say "train"
.venv-train/bin/python -m lerobot.scripts.lerobot_train \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.chunk_size=20 --policy.n_action_steps=20 --policy.optimizer_lr=5e-5 \
  --dataset.repo_id=rq-pipeline/microduck-walk-campaign --dataset.root="$root/dataset" \
  --output_dir="$root/student" --job_name=walk-student \
  --steps="$steps" --batch_size=32 --num_workers=8 --log_freq=200 --save_freq=10000 \
  --wandb.enable=false

say "certify the student (40 trials, cuda)"
cd "$repo/rq_mjlab"
.venv/bin/python -m rq_mjlab.walk_verdict "$repo/$teacher" --trials 40 --seed 1000 \
  --device cuda:0 --student "$root/student/checkpoints/last/pretrained_model" \
  --horizon 2 --stride 5 --no-studio
say "CAMPAIGN DONE"
