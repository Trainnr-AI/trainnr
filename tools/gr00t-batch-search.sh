#!/usr/bin/env bash
# GR00T N1.7 fine-tuning on one GPU, sized by measurement (docs/86 §5.5; the record
# docs/findings/gr00t-n17-on-a-24gb-card-2026-11-02.json):
#   gr00t-batch-search.sh setup               clone Isaac-GR00T at the measured commit, build its own
#                                             env, download the base model and its gated backbone
#   gr00t-batch-search.sh smoke [B] [CONFIG]  20 fine-tune steps on NVIDIA's SO-100 demo (batch 8, frozen)
#   gr00t-batch-search.sh maxbatch CONFIG     NVIDIA's method: double the batch until it fails, then bisect
# setup downloads (GitHub, Hugging Face) only when run; nothing here runs in trainnr's environment.
# CONFIGs (gr00t-fit.py beside this script adds the savers NVIDIA's launcher does not expose):
#   frozen       DiT action head frozen, NVIDIA's settings otherwise (fp32 weights, AdamW)
#   frozen-lean  frozen, plus the backbone in bf16 and the DiT's activations recomputed
#   full-lean    DiT trained too: bf16 backbone, recomputed DiT activations, AdamW state in 8 bits
# Each run is a fresh process; GPU memory is sampled every 0.5 s. Results in $OUT/results.txt.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
REPO="${GR00T_REPO:-$HOME/.cache/trainnr/Isaac-GR00T}"
COMMIT="c56f54128d92f743889b7e4cecfc1ceafd88739e"  # the Isaac-GR00T tip measured on 2026-11-02
OUT="${GR00T_RUNS:-${TRAINNR_HOME:-$HOME/trainnr}/gr00t-runs}"
MODEL="nvidia/GR00T-N1.7-3B"
# One GPU never uses DeepSpeed (experiment.py: only num_gpus > 1), but accelerate imports it when
# installed, and its import runs nvcc, which this machine (driver only) lacks. So setup removes it
# and every run is --no-sync, which would put it back.
# Under WSL the Windows driver spills a full card into system RAM instead of raising OOM (a 3090 Ti
# at 100 % and 150 W, 273 s a step): a run whose memory nears the card's size counts as a spill.
TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
WORKERS=${WORKERS:-2}  # four decoding workers took ~6 GB of RAM each and filled swap
mkdir -p "$OUT"
if [ "${1:-}" = setup ] && [ ! -d "$REPO" ]; then
  git clone --filter=blob:none https://github.com/NVIDIA/Isaac-GR00T.git "$REPO" || exit 1
  git -C "$REPO" checkout -q "$COMMIT" || exit 1
fi
cd "$REPO" || { echo "no Isaac-GR00T at $REPO: run setup first" >&2; exit 1; }

config() {
  case $1 in
    frozen) echo "--no-tune-diffusion-model" ;;
    frozen-lean) echo "--bf16-backbone --checkpoint-dit --no-tune-diffusion-model" ;;
    full-lean) echo "--bf16-backbone --checkpoint-dit --optim adamw_bnb_8bit" ;;
    *) echo "unknown config $1" >&2; exit 2 ;;
  esac
}

# Wait until the last run has released the card: a killed run frees it seconds later, and a run
# started before then reads as a spill.
IDLE=${IDLE:-3000}
wait_idle() {
  local i
  for i in $(seq 120); do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt "$IDLE" ] && return 0
    sleep 1
  done
  echo "the card is still above $IDLE MiB used; not starting" >&2; exit 1
}

# run <name> <config> <batch> <steps>; prints "name config batch status peak_mib step_ms".
run() {
  local name=$1 cfg=$2 batch=$3 steps=$4 log="$OUT/$1.log" mem="$OUT/$1.mem" mark="$OUT/$1.spill"
  local flags; flags=$(config "$cfg") || exit 2
  wait_idle
  rm -f "$mark"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 >"$mem" &
  local smi=$!
  # shellcheck disable=SC2086
  setsid uv run --no-sync python "$here/gr00t-fit.py" $flags \
    --base-model-path "$MODEL" \
    --dataset-path demo_data/cube_to_bowl_5 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/SO100/so100_config.py \
    --num-gpus 1 --output-dir "$OUT/ckpt-$name" \
    --max-steps "$steps" --save-steps 100000 \
    --global-batch-size "$batch" --dataloader-num-workers "$WORKERS" >"$log" 2>&1 &
  local job=$!
  # Stop a spilled run at once rather than let it crawl.
  while kill -0 $job 2>/dev/null; do
    if [ "$(sort -n "$mem" | tail -1)" -ge $((TOTAL - 600)) ] 2>/dev/null; then
      kill -TERM -- -$job; touch "$mark"; break  # the whole group: python and its data workers
    fi
    sleep 1
  done
  wait $job
  local rc=$?
  kill $smi 2>/dev/null; wait $smi 2>/dev/null
  rm -rf "$OUT/ckpt-$name"
  local status=pass
  grep -q -E "OutOfMemoryError|CUDA out of memory" "$log" && status=oom
  [ -e "$mark" ] && status=spill
  [ $rc -ne 0 ] && [ $status = pass ] && status="fail($rc)"
  local peak ms
  peak=$(sort -n "$mem" | tail -1)
  ms=$(grep -o "step_ms=[0-9.]*" "$log" | tail -1 | cut -d= -f2)
  echo "$name $cfg $batch $status ${peak:-?} ${ms:-?}" | tee -a "$OUT/results.txt"
  case $status in
    pass) return 0 ;;
    oom | spill) return 1 ;;
    *) echo "not a memory limit: see $log" >&2; exit 1 ;;  # a crash must not pass for a ceiling
  esac
}

case "${1:-}" in
  setup)
    echo "Isaac-GR00T at $(git rev-parse --short HEAD) (measured: ${COMMIT:0:7})"
    uv sync --python 3.12 || exit 1
    uv pip uninstall deepspeed || exit 1
    uv pip install bitsandbytes || exit 1
    uv run --no-sync hf download "$MODEL" || exit 1
    # The backbone is gated: accept its licence on huggingface.co and `hf auth login` first.
    uv run --no-sync hf download nvidia/Cosmos-Reason2-2B ;;
  smoke)
    cfg=${3:-frozen}
    echo "baseline_mib $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" | tee -a "$OUT/results.txt"
    run "smoke-$cfg-b${2:-8}" "$cfg" "${2:-8}" 20 ;;
  maxbatch)
    cfg=${2:?maxbatch needs a config: frozen, frozen-lean or full-lean}
    echo "baseline_mib $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" | tee -a "$OUT/results.txt"
    # Start at 4, not NVIDIA's 1: each run spends minutes decoding video before its first step.
    ok=0 b=${START:-4}
    while run "$cfg-b$b" "$cfg" "$b" 7; do ok=$b; b=$((b * 2)); done
    bad=$b
    while [ $((bad - ok)) -gt 1 ]; do
      mid=$(((ok + bad) / 2))
      if run "$cfg-b$mid" "$cfg" "$mid" 7; then ok=$mid; else bad=$mid; fi
    done
    echo "max_batch $cfg $ok (fails at $bad)" | tee -a "$OUT/results.txt" ;;
  *) echo "usage: $0 setup | smoke [B] [CONFIG] | maxbatch CONFIG" >&2; exit 2 ;;
esac
