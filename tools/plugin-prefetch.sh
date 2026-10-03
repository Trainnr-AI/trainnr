#!/usr/bin/env sh
# The plugin's session-start hook (hooks/hooks.json): fetch the prebuilt
# Studio for this platform in the background, once per version, so
# `launch_studio` opens the window without a download wait. Silent by
# design: a SessionStart hook's output goes into the conversation. A
# checkout that built the Studio, or a machine with no Python, is left
# alone; any failure is left for `launch_studio` to report.
root="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
py=$(command -v python3 || command -v python) || exit 0
[ -x "$root/crates/trainnr-studio/target/release/trainnr-studio" ] && exit 0
( PYTHONPATH="$root/trainnr" "$py" -m trainnr.studio_install --quiet >/dev/null 2>&1 & )
exit 0
