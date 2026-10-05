#!/usr/bin/env sh
# The plugin's session-start hook (hooks/hooks.json), silent and in the
# background, so the first tool call does not wait:
#   - the Python environment the MCP server runs in (sim and mcp extras);
#   - the prebuilt Studio for this platform, once per version.
# A SessionStart hook's output goes into the conversation, so nothing is
# printed; a failure is left for the tool that needs the piece to report.
# Set TRAINNR_NO_PREFETCH=1 to skip both.
[ -n "${TRAINNR_NO_PREFETCH:-}" ] && exit 0
root="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
command -v uv >/dev/null 2>&1 || exit 0
# It runs from the plugin's own folder, never the one Claude Code opened:
# `python -m` puts the working folder first on the import path, and uv
# reads a uv.toml it finds there (security review, 2026-10-05). `-P` keeps
# the working folder off the path as well.
cd "$root" || exit 0
(
  uv sync --quiet --locked --directory "$root/trainnr" --extra sim --extra mcp >/dev/null 2>&1
  # A checkout that built the Studio needs no download.
  [ -x "$root/crates/trainnr-studio/target/release/trainnr-studio" ] && exit 0
  # uv supplies a current Python: macOS's command-line tools ship 3.9.
  PYTHONPATH="$root/trainnr" uv run --quiet --no-project --python 3.12 \
    python -P -m trainnr.studio_install --quiet >/dev/null 2>&1
) &
exit 0
