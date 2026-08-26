#!/usr/bin/env bash
# Run a command under the WSL box's launch environment (pipeline/wsl.env):
#     cd pipeline && ../tools/wsl-run.sh .venv-train/bin/python ../tools/train-watch.py ...
# The sim venv can use `uv run --env-file wsl.env` directly; this exists for
# interpreters uv does not launch (the train venv) and for anything else.
set -euo pipefail
env_file="$(cd "$(dirname "$0")/.." && pwd)/pipeline/wsl.env"
set -a
# shellcheck disable=SC1090
. "$env_file"
set +a
exec "$@"
