# The documents

What is in `docs/`, read as a stranger would: which document is current,
which is a dated research read, and which is a historical note kept as
the record. Every document carries the date it speaks from. Numbers
missing from the sequence are the maintainers' working notes, which are
not part of this edition.

## Start here

| Read | What it is | Speaks from |
|---|---|---|
| [`../README.md`](../README.md) | What trainnr is, the eight-step loop, install, the layout | 2026-10 |
| [76 The loop](76-the-loop.md) | The design of the agent-driven loop: stamped artifacts, the state machine, the MCP tool families, the identify, deploy and drift seams, the Studio as the control surface | designed 2026-09-08, build notes to 2026-09-28 |
| [77 The Unitree loop](77-the-unitree-loop.md) | A Go2 from asset to sim-to-sim deployment through the tools: gates, attribution, pre-flight, drift, the fit-trained walk | 2026-09-10 to 2026-09-26 |
| [78 The scene loop](78-the-scene-loop.md) | A captured scene (phone video to splat and collision proxy) as a stamped artifact, and walking the Go2 on it | 2026-09-22 to 2026-09-25 |
| [35 The Studio](35-the-studio.md) | The desktop app: what it shows, how it talks to the project, measured performance lessons, the history of its design | 2026-09-08, updated 2026-10-03 |
| [68 Findings ledger](68-findings.md) | Every measured result with its commit, command, instrument, outcome and caveats; generated from [`findings/`](findings/) | 2026-08-27 to 2026-09-26 |
| [`paper/`](paper/README.md) | The manuscript, its reading copy, references and how to build the PDF | draft v3, 2026-09-29 |
| [Glossary](GLOSSARY.md) | The project's words: bundle, stamp, pinned, certificate, gate, basis, world, the campaign labels | 2026-10-03 |

## Design and decisions

| Document | What it covers | Status |
|---|---|---|
| [22 The pipeline as software](22-pipeline-architecture.md) | The founding module map and where ML and physics enter | historical, 2026-08-20; the current layout is the README and docs/80 §4 |
| [26 The synthetic STS3215 identifiability study](26-sts3215-synthetic-identifiability.md) | Which servo parameters identification recovers, and why position plus load beats velocity | current, 2026-08-25 |
| [28 Quickstart: identify](28-quickstart-identify.md) | Fit a robot's dynamics from its telemetry through the Python API; the MCP route is `onboard_robot`, `ingest_recording`, `identify_system` | current, 2026-08-25, route note 2026-10 |
| [31 The end-to-end test on ALOHA 2](31-aloha2-e2e.md) | The first run through every stage, kitting with ACT, the T0 to T6 ladder | historical, 2026-08-27; the loop is now the Go2 walk (77) |
| [32 The evaluation layer](32-evaluation-layer.md) | A gymnasium env and a LeRobot plugin outside, exact statistics, per-trial records, milestones and variations inside | plan 2026-08-26 with what was built in §9 and §10 |
| [34 The rented GPU](34-cloud-gpu.md) | The provider seam, the RunPod adapter, the runbook, measured costs | current, prices as of 2026-08-27 |
| [80 Names and repositories](80-trainnr-names-and-repos.md) | The names, the repository layout, the layering rules, the rename from the former identifiers | decided 2026-10-03 |
| [84 mjsim](84-mjsim.md) | The plan to cut the Studio's simulator window into its own module | planned, not started, 2026-10-03 |

## Research

[`e2e-research/`](e2e-research/README.md) holds the dated reads against
primary sources: the field survey of 2026-08, the prior-art reads of
2026-08-25, Isaac Lab Arena and the evaluation interfaces, Newton and the
solvers, the actuator identification work (BAM), mjlab and Unitree's
stack, scene capture and physics on gaussians, the USD import. Its index
says which reads are current and which are snapshots.

## Evidence

| Directory | What it holds |
|---|---|
| [`findings/`](findings/README.md) | One JSON record per measured result; the source of the ledger and of the numbers gate |
| [`artifacts/`](artifacts/README.md) | The per-trial rows, certificates, datasheets and checkpoints the records cite |
| [`figures/`](figures/README.md) | The figures drawn from the records, the paper's composites and stills |

## How the documents are kept

Every decision goes into a numbered document the day it is made, with
the date and the evidence. A document is never silently rewritten:
later facts are appended under their date, and a superseded document
keeps a banner that names what replaced it. `tools/check-docs.py` keeps
every link resolving; `tools/check-numbers.py` refuses a ratio in the
paper that no record carries.
