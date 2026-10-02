#!/usr/bin/env bash
# Run a command under the WSL box's launch environment (trainnr/wsl.env):
#     cd trainnr && ../tools/wsl-run.sh .venv-train/bin/python ../tools/train-watch.py ...
# The sim venv can use `uv run --env-file wsl.env` directly; this exists for
# interpreters uv does not launch (the train venv) and for anything else.
set -euo pipefail
if [ "$#" -eq 0 ]; then
  echo "usage: tools/wsl-run.sh <command> [args...]" >&2
  exit 64
fi
env_file="$(cd "$(dirname "$0")/.." && pwd)/trainnr/wsl.env"
if [ ! -r "$env_file" ]; then
  echo "wsl-run.sh: no readable env file at $env_file" >&2
  exit 66
fi
set -a
# shellcheck disable=SC1090
. "$env_file"
set +a
exec "$@"
