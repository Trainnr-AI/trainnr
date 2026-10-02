#!/usr/bin/env bash
# The identified-interval arms of the walk C1, six pods in parallel
# (2026-09-06): three replicates of the point arm and three of the
# identified arm, both on the REFIT bundle (our fit of BAM's M6 on
# Rhoban's public logs with its bootstrap interval), so the comparison
# is at one point. Each pod: launch on the network volume, the resume
# script, sync HEAD, then tools/walk-c1-arm.sh under BUNDLE=.
#
#   tools/walk-c1-refit-pods.sh prepare   # launch + resume six pods (~10 min)
#   tools/walk-c1-refit-pods.sh run       # sync HEAD, start the six arms
#   tools/walk-c1-refit-pods.sh status    # the six stage lines
#   tools/walk-c1-refit-pods.sh stop      # stop the six pods (keeps the volume)
# Pod names are walk-refit-<arm>-<k>; doors are read from `machines`.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
cloud() { uv run --project "$repo/trainnr" python "$repo/tools/cloud-gpu.py" "$@"; }
GPU="NVIDIA RTX PRO 6000 Blackwell Server Edition"
VOLUME="${VOLUME:-q51i67dwu8}"; DC="${DC:-US-NC-2}"
BUNDLE="robots/actuator-bundles/xl330-refit.m6.bundle.json"
ITER="${ITER:-8000}"; TRIALS="${TRIALS:-40}"
# arm  span        replicate  seed
ARMS=(
  "point-refit 0 1 42"
  "point-refit 0 2 43"
  "point-refit 0 3 44"
  "identified identified 1 42"
  "identified identified 2 43"
  "identified identified 3 44"
)
name_of() { echo "walk-refit-$1-$3"; }
door_of() { cloud machines 2>/dev/null | awk -v n="$1" '$2==n && $3=="RUNNING" {print $NF}' | grep -o "root@[0-9.]*:[0-9]*" | head -1; }
say() { echo "== $(date -u +%H:%M:%S) $*"; }

case "${1:-}" in
prepare)
  for spec in "${ARMS[@]}"; do set -- $spec; n="$(name_of "$@")"
    if [ -n "$(door_of "$n")" ]; then say "$n already running"; continue; fi
    say "launch $n"; cloud launch --name "$n" --gpu "$GPU" --volume "$VOLUME" --datacenter "$DC" >/dev/null &
  done; wait
  say "waiting for doors"; sleep 60
  for spec in "${ARMS[@]}"; do set -- $spec; n="$(name_of "$@")"
    for _ in $(seq 1 30); do d="$(door_of "$n")"; [ -n "$d" ] && break; sleep 10; done
    [ -n "$d" ] || { echo "$n never got a door"; continue; }
    host="${d%:*}"; port="${d##*:}"
    say "resume $n at $d"
    ssh -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new -p "$port" "$host" 'bash -s' < "$repo/tools/_pod-resume.sh" > "/tmp/resume-$n.log" 2>&1 &
  done; wait; say "six pods ready (see /tmp/resume-*.log)"
  ;;
run)
  commit="$(git -C "$repo" rev-parse --short HEAD)"
  for spec in "${ARMS[@]}"; do set -- $spec; arm="$1"; span="$2"; rep="$3"; seed="$4"; n="$(name_of "$@")"
    d="$(door_of "$n")"; [ -n "$d" ] || { echo "$n has no door"; continue; }
    host="${d%:*}"; port="${d##*:}"
    say "sync $commit -> $n"
    git -C "$repo" archive --format=tar HEAD | ssh -o BatchMode=yes -o ConnectTimeout=30 -p "$port" "$host" "tar -x -C /workspace/robotiq && echo $commit > /workspace/robotiq/.trainnr-commit"
    say "start $arm#$rep (span $span, seed $seed) on $n"
    ssh -o BatchMode=yes -o ConnectTimeout=30 -p "$port" "$host" "cd /workspace/robotiq && mkdir -p runs/studies/walk-c1 && (BUNDLE=$BUNDLE nohup bash tools/walk-c1-arm.sh $arm $span $ITER $TRIALS $seed $rep > runs/studies/walk-c1/$arm-$rep.log 2>&1 < /dev/null &)"
  done; say "six arms started; logs runs/studies/walk-c1/<arm>-<k>.log on the volume"
  ;;
status)
  for spec in "${ARMS[@]}"; do set -- $spec; arm="$1"; rep="$3"; n="$(name_of "$@")"; d="$(door_of "$n")"
    [ -n "$d" ] || { echo "$n: no door"; continue; }
    host="${d%:*}"; port="${d##*:}"
    printf "%-24s " "$n"; ssh -o BatchMode=yes -o ConnectTimeout=20 -p "$port" "$host" "tr '\r' '\n' < /workspace/robotiq/runs/studies/walk-c1/$arm-$rep.log 2>/dev/null | grep -aE '^== |\[verdict\] survived|Traceback' | tail -1" || echo "(unreachable)"
  done
  ;;
stop)
  cloud machines 2>/dev/null | awk '$2 ~ /^walk-refit-/ && $3=="RUNNING" {print $1}' | while read -r id; do say "stop $id"; cloud stop "$id" >/dev/null; done
  ;;
*) echo "usage: $0 prepare|run|status|stop"; exit 2 ;;
esac
