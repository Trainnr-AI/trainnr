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
(
  uv sync --quiet --directory "$root/trainnr" --extra sim --extra mcp >/dev/null 2>&1
  # A checkout that built the Studio needs no download.
  [ -x "$root/crates/trainnr-studio/target/release/trainnr-studio" ] && exit 0
  # uv supplies a current Python: macOS's command-line tools ship 3.9.
  PYTHONPATH="$root/trainnr" uv run --quiet --no-project --python 3.12 \
    python -m trainnr.studio_install --quiet >/dev/null 2>&1
) &
exit 0
