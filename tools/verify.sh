#!/usr/bin/env bash
# Prove the whole system, end to end, in one command.
#
#   tools/verify.sh        everything; nothing here needs hardware
#
# Ordered cheapest-first, so a broken build fails in seconds rather than
# after the long Python suites. The firmware builds, the emulator HIL and
# the wire replays against the rig's crates left with the rig on
# 2026-10-02 (a private archive, with its own tools/verify.sh).
set -uo pipefail
cd "$(dirname "$0")/.."
source "$HOME/.cargo/env" 2>/dev/null || true

pass=0; fail=0
step() {                      # step "name" "command"
  printf "%-46s" "$1"
  if out=$(eval "$2" 2>&1); then
    echo "ok"; pass=$((pass+1))
  else
    echo "FAIL"; echo "$out" | tail -15; fail=$((fail+1))
  fi
}

PY="${PYTHON:-python3}"  # the repo-level gates' interpreter; override where python3 is not the name
# The tools run by name are pinned, as in CI (.github/workflows/gates.yml);
# ruff moves with the ruff-pre-commit rev in .pre-commit-config.yaml.
RUFF="ruff@0.16.10"
ZIZMOR="zizmor@1.30.1"
echo "=== trainnr end-to-end verification ==="
# the Studio is the one Rust crate left; it is its own workspace (own
# Cargo.lock, the rerun 0.36 pins), so every cargo step runs inside it.
step "formatting (trainnr-studio)"  "(cd crates/trainnr-studio && cargo fmt --check)"
step "clippy (trainnr-studio)"      "! (cd crates/trainnr-studio && cargo clippy -q --all-targets 2>&1) | grep -qE '^error'"
step "tests (trainnr-studio)"       "(cd crates/trainnr-studio && cargo test -q)"
step "docs describe real code" "\"$PY\" tools/check-docs.py"
step "package layers (docs/80)" "\"$PY\" tools/check-layers.py"
step "unsafe forbidden"        "\"$PY\" tools/check-unsafe-gates.py"
step "numbers traced to records" "\"$PY\" tools/check-numbers.py"
step "one version everywhere"  "\"$PY\" tools/release.py check"
# The required supply-chain and package jobs, locally: the workflows
# linted for security, the Studio's crates against deny.toml (needs
# cargo-deny: `cargo install --locked cargo-deny`), and both wheels built
# with LICENSE and NOTICE inside. The advisory audit (network, a moving
# database) is `tools/supply-chain.py --advisories`, not a gate here.
step "workflows linted (zizmor)" "uvx $ZIZMOR --offline --format plain .github/workflows"
step "the Studio's crates (cargo-deny)" "\"$PY\" tools/supply-chain.py --policy"
step "wheels carry LICENSE and NOTICE" "\"$PY\" tools/check-package.py"
# The MCP tools are public API: their names, arguments and descriptions
# match the snapshot in trainnr/tests/api, or the change is reviewed.
step "MCP API snapshot"        "\"$PY\" tools/api-snapshot.py"
# The Python package, under the same roof as the crate. These mirror
# the pre-commit hook — but verify.sh is the "prove EVERYTHING" command
# and until 2026-08-26 it proved everything except the Python half.
step "ruff format (trainnr)"   "(cd trainnr && uvx $RUFF format --check .)"
step "ruff lint (trainnr)"     "(cd trainnr && uvx $RUFF check .)"
# tools/ has its own .ruff.toml (extending trainnr's) and, until
# 2026-08-27, no gate that ran it — 63 findings had accrued.
step "ruff format (tools)"     "(cd trainnr && uvx $RUFF format --check ../tools)"
step "ruff lint (tools)"       "(cd trainnr && uvx $RUFF check ../tools)"
# trainnr_mjlab had NO Python gate until 2026-09-01 — its lint, types and
# tests ran only when somebody remembered. Now under the same roof.
step "ruff lint (trainnr_mjlab)"    "(cd trainnr-mjlab && uvx $RUFF check src tests)"
# Types, both packages: zero errors is the baseline; the config (and
# the untyped-C-extension skips) lives in each pyproject. Run INSIDE each
# project's environment (`uv run --with mypy`), never as an isolated
# `uvx mypy`: isolated, every installed package - mjlab, trainnr,
# torch, rerun - read as Any, the check went quiet on how we call them,
# and it missed an import of a function deleted a week earlier
# (2026-09-13).
# The extras are named so the check reads the same packages on every
# machine (an inexact sync read whatever the box last installed); mypy
# is pinned so a release of it does not move the gate.
MYPY="mypy==2.3.1"
step "mypy (trainnr)"          "(cd trainnr && uv run --extra sim --extra mcp --extra deploy --extra viz --with $MYPY mypy trainnr)"
step "mypy (trainnr_mjlab)"         "(cd trainnr-mjlab && uv run --extra viz --with $MYPY mypy src/trainnr_mjlab)"
step "python tests (trainnr)"  "(cd trainnr && uv run python -m unittest discover -s tests)"
# The recorder tests need the viz extra (rerun): named here, as the README
# names it, so a fresh clone runs them rather than skipping them (2026-09-27).
step "python tests (trainnr_mjlab)" "(cd trainnr-mjlab && uv run --extra viz python -m unittest discover -s tests -t .)"
# The USD door's own tests need Newton (the `usd` extra), which cannot
# share a venv with `mjx` (uv's conflicts), so the main suite skips them.
# Skipped, a wrong test sat unrun for a day (the stage-units line,
# 2026-09-25). They run here in their own venv, synced every time (a no-op
# when current), and a skip FAILS the step: the last line must be exactly
# "OK", not "OK (skipped=N)"; grep reads all of it, so a warning printed
# after it cannot break the pipe. The Isaac asset is fetched at its pinned
# commit when the cache (runs/assets) lacks it.
step "python tests (USD import, Newton)" \
     "(cd trainnr && UV_PROJECT_ENVIRONMENT=.venv-usd uv sync -q --extra usd --extra sim --extra gpu \
      && TRAINNR_FETCH_TEST_ASSETS=1 .venv-usd/bin/python -m unittest tests.test_usd_import tests.test_import_audit 2>&1 | tee /dev/stderr | grep -x OK >/dev/null)"
# Last, once the steps above have installed them: every environment
# here (trainnr, its usd venv, trainnr-mjlab) against the licence
# allow-list and its written exceptions (tools/supply-chain.py).
step "Python licences (installed envs)" "\"$PY\" tools/supply-chain.py --licences"

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
