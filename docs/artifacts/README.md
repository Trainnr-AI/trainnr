# The artifacts

The files the findings records rest on: per-trial rows, certificates,
datasheets, study specs and verdicts, and the biped's checkpoints. Each
record under `docs/findings/` names the artifacts it cites; this page is
the map the other way.

| directory | size | what is in it | records |
|---|---|---|---|
| `walk-c1/` | 78 MB | the biped walk: one directory per training arm (`train/model_7999.pt`, `identity.json`), the certificates and rows under each arm, the campaign logs | `walk-c1-2026-09-04`, `walk-mismatch-matrix-*`, `walk-c1-refit-2026-09-06` |
| `lift-studies/` | 47 MB | the SO-101 lift studies: `study.json` (the arms as data), `verdict.json`, per-arm datasheets, `*-records.jsonl` rows, `study.log` | `c1-competent-lift*`, `demo-count-lift*`, `visual-dr-lift*`, `lift-expert-envelope-2026-09-04`, `referee-keep-rates-on-the-cliff-2026-09-04` |
| `walk-verdicts/` | 0.8 MB | the biped teacher and students judged at the fit and with delays (`walk-verdict-*.json`, `records-*.jsonl`, `latency/`) | `walk-distillation-campaigns-2026-09-04`, `walk-cadence-controls-2026-09-02`, `walk-latency-budget-2026-09-05`, `walk-dagger-round-1-2026-09-04` |
| `go2/` | 0.8 MB | the quadruped: `certificates/` (every policy and rung, `certificate.json` each, the headline rows as `*-records-cuda.jsonl`), `deploy/` (the gates, pre-flight and attribution), `fits/` (the public chirp's identification), `scenes/` (the garden scene record) | `go2-*`, `gate-*-go2-c2-*`, `dds-gate-go2-c2-2026-09-13`, `course-gate-first-walker-2026-09-22`, `scene-walk-*` |
| `bam/xl330/` | 0.4 MB | the bench refit and the 100 bootstrap replicates of the servo's fit | `bam-xl330-refit-2026-09-05`, `bam-xl330-bootstrap-interval-2026-09-06` |
| `dagger/round-1/` | 0.4 MB | the DAgger round's datasheet and rows | `walk-dagger-round-1-2026-09-04` |
| `cliff-datasheets/` | 20 KB | the four lift datasheets at the hard test condition, byte-identical to the ones under `lift-studies/c1-competent-lift-cliff/`; kept because the record cites these paths | `c1-competent-lift-cliff-2026-09-03` |

## File kinds

- `certificate.json`, `walk-verdict-*.json`, `verdict.json`: a judged
  policy, with the count, the exact interval, the funnel and the hashes.
- `records-*.jsonl`, `*-records.jsonl`: one row per trial, the data every
  interval recomputes from.
- `datasheet.md`: a pressed batch's provenance (bases, draws, keep rate).
- `study.json`: a study's arms, declared as data; `study.log`: its run.
- `model_*.pt`: a checkpoint (the biped's only).
- `identity.json`: what a run trained on (bundle, task and instrument stamps).

## Names

The directories keep the names the runs used; the studies' words differ:

| directory name | the studies' name |
|---|---|
| `point` | the identified point, no randomization |
| `narrow` | ±10 % around the identified point |
| `wide` | ±30 % around the identified point |
| `identified#n` | the n-th replicate trained over the bootstrap intervals |
| `point-refit` | the point of the refit that lowers bench error |
| `#2`, `#3` | the second and third training replicate of an arm |
| lift `identified` | declared ±5 % |
| lift `guessed` | a guessed span, not from a measurement |

## Not shipped

The Go2 checkpoints and training logs, and the raw campaign logs of the
biped's walk-c1 arms (269 MB; their counts are in the records). Local
project directories (`projects/…`) are untracked; where a record cites
one and a copy exists here, the copy is named first.

## Sample projects

The sample projects `open_sample` opens are in the same dataset, under
`samples/`: each a tar.gz that `tools/make-sample.py` built from a real
project, deterministically and with nothing private in it, and that
`trainnr/trainnr/samples.py` pins by the dataset's commit, its size and
its SHA-256. Today: `samples/go2-walk-sample.tar.gz` (12 MB, 40 MB
unpacked), the Go2 walk through the whole loop, published at commit
`9f61659a2f912a0298c48e699fbdb08c6acc6203`. A new sample, or a new
build of one, is a new commit and a new pin.

## Large files

The model checkpoints (`*.pt`) and the study logs (`*.log`) are not in the
repository: they would add over 100 MB to every clone and plugin install.
They are in the Hugging Face dataset
[`trainnr/trainnr-artifacts`](https://huggingface.co/datasets/trainnr/trainnr-artifacts),
each at the path it has here, pinned below to the dataset's commit.
[`MOVED.tsv`](MOVED.tsv) lists every file with its path, size and SHA-256,
so each record that cites one still names exactly what it is. From the
repository root:

```sh
uvx --from huggingface_hub hf download trainnr/trainnr-artifacts \
  --repo-type dataset --revision 541ec05138492f8eebe0b6d376b1975efc1e41ba \
  --include "docs/*" --local-dir .
grep -v '^#' docs/artifacts/MOVED.tsv | awk -F'\t' '{print $3"  "$1}' | sha256sum -c
```
