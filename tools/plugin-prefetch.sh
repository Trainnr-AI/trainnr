#!/usr/bin/env sh
# The plugin's session-start hook (hooks/hooks.json), silent and in the
# background, so the first tool call does not wait:
#   - the Python environment the MCP server runs in (sim and mcp extras);
#   - the prebuilt Studio for this platform, once per version.
# A SessionStart hook's output goes into the model's context, so it prints
# once, in the first session after the install, what the user can say
# first (a fresh-install audit found no word at all after the restart,
# 2026-10-08); a failure is left for the tool that needs the piece to
# report. Set TRAINNR_NO_PREFETCH=1 to skip the background setup.
root="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
# Claude Code keeps a plugin's data folder across updates; the marker says
# the first session has had its line.
data="${CLAUDE_PLUGIN_DATA:-$root}"
if [ ! -e "$data/.welcomed" ] && mkdir -p "$data" 2>/dev/null && : >"$data/.welcomed" 2>/dev/null; then
  printf '%s\n' "trainnr was just installed. It prepares its Python environment and downloads the Studio (the desktop app) in the background; the first tool call may wait for that. To begin, the user can say \"create a project called go2\" and follow the README's quickstart; launch_studio opens the Studio on the open project, and describe_studio says whether it runs."
fi
[ -n "${TRAINNR_NO_PREFETCH:-}" ] && exit 0
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
