#!/usr/bin/env bash
# Point git at the versioned hooks in tools/hooks/.
#
# Run once per clone. Hooks in .git/hooks are not versioned and exist on
# exactly one machine, which is how a "we always check X" rule quietly
# becomes "one laptop checks X".
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath tools/hooks
echo "core.hooksPath -> tools/hooks"
echo "  pre-commit: fmt, clippy, no-unsafe, docs"
echo "  everything else: tools/verify.sh"
