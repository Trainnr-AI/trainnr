#!/usr/bin/env bash
# One DAgger round on the walk (docs/66 §6), on a pod: the current
# student drives N episodes over the bridge, the teacher labels every
# state, the referee keeps the passes; the kept episodes are exported
# and APPENDED to the student's dataset; a new student trains on the
# union; both students are certified on the same 40 trials.
#
#   tools/walk-dagger-round.sh <round_root> <teacher.pt> <student_pretrained_model> <base_dataset> [episodes] [steps]
#   e.g. runs/dagger/round-1 runs/microduck-walk/<stamp>/model_7999.pt runs/campaign-4/student/checkpoints/last/pretrained_model runs/campaign-4/dataset 120 30000
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
root="${1:?round_root}"; teacher="${2:?teacher.pt}"; student="${3:?student pretrained_model}"; base="${4:?base dataset}"
episodes="${5:-120}"; steps="${6:-30000}"
# The stages cd into different trees, so a relative root would land the
# demos under trainnr-mjlab/ and the export would look under trainnr/ — the
# 2026-09-04 round-1 relaunch lost 36 minutes to exactly that. Absolute.
[[ "$root" = /* ]] || root="$repo/$root"
# FROM=export|train|certify resumes a round whose earlier stages are done.
from="${FROM:-press}"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
mkdir -p "$root"
say() { echo "== $(date -u +%H:%M:%S) $*"; }
stage_after() { # true when $from is at or before the named stage
  local order="press export train certify" seen=0 s
  for s in $order; do [ "$s" = "$from" ] && seen=1; [ "$s" = "$1" ] && { [ "$seen" = 1 ] && return 0 || return 1; }; done; return 1
}

if stage_after press; then
say "dagger round: student $student drives $episodes episodes, teacher labels"
cd "$repo/trainnr_mjlab"
.venv/bin/python -m trainnr_mjlab.walk_press "$repo/$teacher" --out "$root/demos" \
  --episodes "$episodes" --worlds 48 --seed 5000 --frame-every 1 --no-studio \
  --student "$repo/$student" --horizon 2 --stride 1
fi

if stage_after export; then
say "export the relabeled batch (local disk), then merge with the base dataset"
scratch="${EXPORT_SCRATCH:-/tmp/trainnr-export}/dagger-$(basename "$root")"; rm -rf "$scratch"; mkdir -p "$scratch"
cd "$repo/trainnr"
.venv-train/bin/python -c "
from pathlib import Path
from trainnr.collect.demo_export import export_batch
export_batch(Path('$root/demos'), Path('$scratch/dataset'), repo_id='trainnr/walk-dagger', use_videos=False)
print('exported')"
.venv-train/bin/python -c "
from pathlib import Path
from trainnr.collect.dataset_ops import concat_datasets
concat_datasets([Path('$repo/$base'), Path('$scratch/dataset')], Path('$scratch/union'), repo_id='trainnr/walk-dagger-union')
print('union written')"
rm -rf "$root/dataset" && mv "$scratch/union" "$root/dataset"
fi

if stage_after train; then
say "train the next student on the union ($steps steps)"
.venv-train/bin/python -m lerobot.scripts.lerobot_train \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.chunk_size=20 --policy.n_action_steps=20 --policy.optimizer_lr=5e-5 \
  --dataset.repo_id=trainnr/walk-dagger-union --dataset.root="$root/dataset" \
  --output_dir="$root/student" --job_name=walk-dagger \
  --steps="$steps" --batch_size=32 --num_workers=8 --log_freq=200 --save_freq=10000 \
  --wandb.enable=false
fi

say "certify the new student (40 trials, seed 1000)"
cd "$repo/trainnr_mjlab"
.venv/bin/python -m trainnr_mjlab.walk_verdict "$repo/$teacher" --trials 40 --seed 1000 \
  --device cuda:0 --student "$root/student/checkpoints/last/pretrained_model" \
  --horizon 2 --stride 1 --no-studio
say "ROUND DONE"
