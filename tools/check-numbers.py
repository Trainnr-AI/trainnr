"""Every k/n success figure quoted in the paper drafts and the goals
ledger must exist in a finding record — the traceability gate.

    python3 tools/check-numbers.py

A number in prose with no record behind it is exactly the failure the
2026-09-04 audit found five times ("1,080 pushed episodes", "~1 in 2
at gain 0.4", ...). The gate reads every `docs/findings/*.json`, collects
each k/n it states or derives (per-arm successes/trials, per-run counts
over their trials, replicate rows, any k/n in the claim or outcome
text), then scans the prose files for `k/n` tokens and refuses any
that no record supports. Ratios that are not success counts (dates,
"page 1/2", "11/12 docs") are excluded by the shapes listed below.
Stdlib only, so the commit gate runs it cold.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FINDINGS = REPO / "docs" / "findings"
# The prose whose numbers must trace: the paper drafts and the ledgers.
PROSE = (
    "docs/e2e-research/68-paper-section-4.md",
    "docs/e2e-research/69-paper-sections-1-3.md",
    "docs/e2e-research/70-paper-sections-5-9.md",
    "docs/70-goals-ledger.md",
    "docs/33-what-we-say.md",
    "docs/paper/reading.md",
    "docs/paper/manuscript.md",
)
# A trailing "." is a sentence's end, not a decimal: "90/120." must
# match (the matrix record's claim ends on one, 2026-09-04); ".5" must not.
RATIO = re.compile(r"(?<![\d.])(\d{1,3})/(\d{1,3})(?!\d|\.\d)")
# Ratios that are not success counts, by denominator: dates are never
# quoted as k/n here, but doc references ("docs 11/12") and pages are.
NOT_A_SCORE = {"1/2", "11/12", "2/3"}
# A slash-separated list of seeds or versions ("42/43/44", "13.0/13.2")
# is not a score: refuse to read a token that is followed by another /.
LISTED = re.compile(r"\d{1,3}/\d{1,3}/")
# A two-day date ("2026-09-12/13", a run that crossed midnight) carries a
# slash between day numbers, not a count over trials.
SPAN_DATE = re.compile(r"\d{4}-\d{2}-\d{2}/\d{1,2}")


def recorded_ratios() -> set[str]:  # noqa: PLR0912 - one pass over every record shape
    have: set[str] = set()
    for path in sorted(FINDINGS.glob("*.json")):
        text = path.read_text()
        for m in RATIO.finditer(text):
            have.add(f"{m.group(1)}/{m.group(2)}")
        record = json.loads(text)
        outcome = record.get("outcome", {})
        for arm in outcome.get("arms", {}).values():
            if not isinstance(arm, dict):
                continue
            if "successes" in arm and "trials" in arm:
                have.add(f"{arm['successes']}/{arm['trials']}")
                per_run_trials = arm["trials"] // max(int(arm.get("runs", 1)), 1)
                for k in arm.get("per_run") or []:
                    have.add(f"{k}/{per_run_trials}")
            if "kept" in arm and "max_attempt" in arm:
                have.add(f"{arm['kept']}/{arm['max_attempt']}")
        replicates = outcome.get("replicates", {})
        # A study's replicates are rows per arm; a bootstrap's is a count.
        if isinstance(replicates, dict):
            for rows in replicates.values():
                for row in rows:
                    have.add(f"{row['successes']}/{row['trials']}")
        for own in (outcome.get("under_span_0.10") or {}).values():
            if isinstance(own, dict):
                have.add(f"{own['successes']}/{own['trials']}")
                for k in own.get("per_run") or []:
                    have.add(f"{k}/40")
    return have


def main() -> int:
    have = recorded_ratios()
    problems: list[str] = []
    for rel in PROSE:
        text = (REPO / rel).read_text()
        for lineno, line in enumerate(text.splitlines(), 1):
            listed = {m.start() for m in LISTED.finditer(line)}
            dated = {
                pos
                for m in SPAN_DATE.finditer(line)
                for pos in range(m.start(), m.end())
            }
            for m in RATIO.finditer(line):
                token = f"{m.group(1)}/{m.group(2)}"
                if (
                    token in NOT_A_SCORE
                    or token in have
                    or m.start() in listed
                    or m.start() in dated
                ):
                    continue
                problems.append(f"{rel}:{lineno}: {token} has no finding record")
    for problem in problems:
        print(problem)
    print(f"{len(problems)} untraced number(s); {len(have)} ratios on record")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
