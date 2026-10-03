# The findings records

One JSON file per result, `docs/findings/<id>.json`. They are the source
of the ledger page (`docs/68-findings.md`) and of the numbers gate on the
paper: `tools/check-numbers.py` refuses any success ratio in
`docs/paper/manuscript.md` that is not in a record. The records are
frozen evidence: a command line and a commit are kept exactly as they
ran, under the names of the time.

## Schema

Written and read by `trainnr/trainnr/evaluate/findings.py` (standard
library only). Required fields:

| field | meaning |
|---|---|
| `id` | kebab-case, the file's stem, ending in the measurement's date |
| `claim` | one sentence with the number in it |
| `date` | the ISO day the measurement finished |
| `repo_commit` | the short hash of the tree that ran it; `-dirty` means the tree had uncommitted changes, `-archive` that it ran from a `git archive` copy on a rented machine, whose commit was written to `.trainnr-commit` |
| `argv` | the command line, verbatim |
| `instrument` | the engine stamp the trials ran on (simulator, backend, device) |
| `outcome` | the numbers: per-arm counts, intervals, effects |
| `protocol` | what procedure produced it, or the document that defines it |
| `caveats` | what the number does not show |

Optional fields: `inputs` (dataset and bundle stamps, the study's spec
hash), `artifacts` (certificate, datasheet and per-trial row paths),
`sources` (audited externals, name to version or hash), `revised` (dated
amendments; a withdrawn or re-judged number says so here, the original
stays).

## Reading the paths and names

- `rq_pipeline`, `pipeline/` are the `trainnr` package and `trainnr/`;
  `rq_mjlab` is `trainnr_mjlab` (`trainnr-mjlab/`); `--env.type=robotiq`
  is `--env.type=trainnr`; `robotiq/<task>` ids are `trainnr/<task>`.
  Commands are kept as they ran.
- `/workspace/robotiq/…` is a rented GPU pod's checkout, not a path in
  this tree.
- `projects/…` paths are local project directories, untracked by design.
  Where a copy ships under `docs/artifacts/`, the record names it first.
- A protocol or commit marked "not public (maintainers' log)" lives only
  in the maintainers' working log; the record's numbers, command and
  artifacts stand on their own.

## The tools

- `python3 tools/findings.py` renders `docs/68-findings.md`; `--check`
  fails when the page is stale (the commit hook's use).
- `python3 tools/check-numbers.py` traces every success ratio in the
  manuscript to a record.
- `python3 tools/finding-figure.py <id>` draws `docs/figures/<id>.{svg,pdf,png,csv}`
  from a record's outcome; the CSV beside each figure is the plotted data.
