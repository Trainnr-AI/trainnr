"""A finding as a first-class record: the number, and everything that
lets a reader re-derive it.

The bar (operator, 2026-09-04): every finding carries a citation, a
provenance and a version — the repo commit that produced it, the exact
command line, the instrument stamp, the stamps of every dataset and
bundle it consumed, the certificate and datasheet files it rests on,
and, for an audit of someone else's code, the version or hash that was
read. Findings live in `docs/findings/<id>.json`, TRACKED, so the paper's
numbers are in the repository even when the runs that produced them are
on a rented machine's volume; `tools/findings.py` renders the ledger.

Stdlib only, like every record an auditor recomputes.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from trainnr.bundles.json_record import JsonRecord

FINDINGS_DIR = Path("docs") / "findings"
FINDING_FILE = "finding.json"


@dataclass(frozen=True)
class Finding(JsonRecord):
    """One claim with its provenance."""

    id: str  # kebab-case, the file's stem: "demo-count-curve-lift-2026-09-04"
    claim: str  # one sentence, the number in it
    date: str  # ISO day the measurement finished
    repo_commit: str  # git short hash of the tree that ran it
    argv: list[str]  # the command line, verbatim
    instrument: str  # the engine stamp the trials ran on
    outcome: dict[str, Any]  # the numbers: per-arm counts, intervals, effects
    inputs: dict[str, Any] = field(
        default_factory=dict
    )  # dataset/bundle stamps, spec hash
    artifacts: dict[str, str] = field(
        default_factory=dict
    )  # certificate/datasheet paths
    sources: dict[str, str] = field(
        default_factory=dict
    )  # audited externals: name -> version/hash
    protocol: str = ""  # the doc that defines the procedure
    caveats: tuple[str, ...] = ()  # what the number does NOT show

    def __post_init__(self) -> None:
        # JSON has no tuples: a record read back carries lists, and a
        # frozen dataclass compares them unequal to the tuple it wrote.
        object.__setattr__(self, "caveats", tuple(self.caveats))
        object.__setattr__(self, "argv", list(self.argv))

    def path(self, root: Path) -> Path:
        return Path(root) / FINDINGS_DIR / f"{self.id}.json"


# A tree shipped without .git (a pod's volume copy) carries its commit here.
COMMIT_FILE = ".trainnr-commit"


def repo_commit(root: Path) -> str:
    """The tree's short hash, with `-dirty` when the tree is not clean:
    a finding from an uncommitted tree says so. A tree without `.git`
    (the pod's volume copy is a `git archive`, 2026-09-04) reads the
    commit its sync wrote to `.trainnr-commit`, marked as such; a tree with
    neither is refused — a finding must name its code."""
    root = Path(root)
    if not (root / ".git").exists():
        stamp_file = root / COMMIT_FILE
        if not stamp_file.is_file():
            raise FileNotFoundError(
                f"{root} has no .git and no {COMMIT_FILE}: a finding cannot name "
                "the code that produced it (write the commit there when syncing)"
            )
        return f"{stamp_file.read_text().strip()}-archive"
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{head}-dirty" if dirty else head


def today() -> str:
    return date.today().isoformat()


def load_findings(root: Path) -> list[Finding]:
    """Every tracked finding, newest date first."""
    folder = Path(root) / FINDINGS_DIR
    if not folder.is_dir():
        return []
    found = [Finding.read(p) for p in sorted(folder.glob("*.json"))]
    return sorted(found, key=lambda f: (f.date, f.id), reverse=True)


def _flat(value: Any) -> str:
    return " ".join(str(value).split())


# What the records were written under, and what it is called now. The
# records are frozen evidence: their command lines and commits are kept
# as they ran.
FORMER_NAMES: tuple[tuple[str, str], ...] = (
    ("`rq_pipeline`, `pipeline/`", "the `trainnr` package, `trainnr/`"),
    ("`rq_mjlab`", "the `trainnr_mjlab` package, `trainnr-mjlab/`"),
    ("`--env.type=robotiq`", "`--env.type=trainnr`"),
    ("`/workspace/robotiq/…`", "a rented GPU pod's checkout, not a path in this tree"),
    ("`robotiq/<task>`", "the task ids `trainnr/<task>`"),
)
LOG_POINTER = "docs/07"  # the maintainers' log: not part of the public edition
SESSION_WORDS = ("scratchpad", "scratch ")
CLAIM_WIDTH = 110


def _not_public(f: Finding) -> str:
    return f"not public (maintainers' log, {f.date})"


def _commit(f: Finding) -> str:
    if LOG_POINTER in f.repo_commit or f.repo_commit.startswith("see "):
        return _not_public(f)
    return f"`{f.repo_commit}`"


def _protocol(f: Finding) -> str:
    text = f.protocol.strip()
    if not text or LOG_POINTER in text:
        return _not_public(f)
    return text


def _command(f: Finding) -> str:
    line = f"`{' '.join(f.argv)}`"
    if any(word in line for word in SESSION_WORDS):
        line += " (a session script, not reproducible from the tree)"
    return line


def _figure(path: str) -> str:
    # The page lives in docs/; a figure path is recorded from the repo root.
    return path[len("docs/") :] if path.startswith("docs/") else f"../{path}"


def _value_lines(key: str, value: Any, indent: int) -> list[str]:
    """A mapping rendered as key: value lines, nested one level per dict,
    never as a Python repr (the ledger's outcome fields, 2026-10-03)."""
    pad = "  " * indent
    if isinstance(value, dict):
        lines = [f"{pad}- {key}:"]
        for k, v in value.items():
            lines += _value_lines(str(k), v, indent + 1)
        return lines
    if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
        lines = [f"{pad}- {key}:"]
        for i, v in enumerate(value, 1):
            lines += _value_lines(str(i), v, indent + 1)
        return lines
    if isinstance(value, list):
        return [f"{pad}- {key}: {', '.join(_flat(v) for v in value)}"]
    return [f"{pad}- {key}: {_flat(value)}"]


def _claim_short(claim: str) -> str:
    text = _flat(claim).replace("|", "\\|")
    if len(text) <= CLAIM_WIDTH:
        return text
    cut = text[:CLAIM_WIDTH].rsplit(" ", 1)[0]
    return cut + "…"


def render_ledger(findings: list[Finding]) -> str:
    """The ledger page: an index, then one section per finding, every
    provenance field stated, nothing summarised away."""
    lines = [
        "# Findings ledger",
        "",
        "Every number here resolves to a commit, a command line, an",
        "instrument stamp and the artifacts it rests on. Generated by",
        "`tools/findings.py` from `docs/findings/*.json`; edit the records,",
        "not this page. The records are frozen evidence: a command line and",
        "a commit are kept as they ran, under the names of the time. What",
        "those names are called now:",
        "",
        "| then | now |",
        "|---|---|",
    ]
    lines += [f"| {then} | {now} | " for then, now in FORMER_NAMES]
    lines += [
        "",
        "A commit or protocol marked *not public* lives only in the",
        "maintainers' working log, which is not part of this edition; the",
        "record's numbers, command and artifacts stand on their own.",
        "`projects/…` paths are local project directories, untracked by",
        "design; where a copy ships it is named first.",
        "",
        f"## Index ({len(findings)} records, newest first)",
        "",
        "| record | date | claim |",
        "|---|---|---|",
    ]
    for f in findings:
        lines.append(f"| [{f.id}](#{f.id}) | {f.date} | {_claim_short(f.claim)} |")
    lines.append("")
    for f in findings:
        lines += [
            f"## {f.id}",
            "",
            f"**{f.claim}**",
            "",
            f"- date: {f.date} · commit: {_commit(f)}",
            f"- instrument: `{f.instrument}`",
            f"- command: {_command(f)}",
            f"- protocol: {_protocol(f)}",
        ]
        for label, mapping in (
            ("inputs", f.inputs),
            ("outcome", f.outcome),
            ("artifacts", f.artifacts),
            ("sources", f.sources),
        ):
            if mapping:
                lines.append(f"- {label}:")
                for k, v in mapping.items():
                    lines += _value_lines(str(k), v, 1)
        if f.caveats:
            lines.append("- caveats:")
            lines += [f"  - {c}" for c in f.caveats]
        if "figure.png" in f.artifacts:
            lines += ["", f"![{f.id}]({_figure(f.artifacts['figure.png'])})"]
        lines.append("")
    return "\n".join(lines)
