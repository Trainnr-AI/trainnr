# trainnr — a measurement instrument for robots

We fit the dynamics of *your* robot — motor by motor, unit by unit — and
report every parameter with a confidence interval and an honesty
verdict: **pinned**, or **NOT PINNED** with the reason. The output is a
hash-stamped robot bundle a simulator can load and a certificate can
cite. Where the field ships one copied guess, we ship a measurement — or
an explicit admission that the data cannot support one.

## The first artifact

**Ours** — from a real fit of this repo's own drivetrain
([`robots/rig-drivetrain/fits/`](robots/rig-drivetrain/fits/)):

```
scale_ref_damping        0.001    ± unbounded   [NOT PINNED]  FIXED anchor, not estimated
left_gear_per_damp    5.98e-05    ± 2.1e-06     [pinned]
right_gear_per_damp   6.08e-05    ± 2.1e-06     [pinned]

anchor: RATIO FIT: damping FIXED at 1e-3 as the scale reference, because
the torque scale is structurally unobservable at 50 Hz for this ~2 ms
motor (confirmed live 2026-08-24 when free bounds sent every parameter
NOT PINNED).

CROSS-RUN VERDICT: right_gear estimates span 14.9% across three runs
against ~3% per-run intervals — spread EXCEEDS intervals; trust the
spread.  (fits/SPREAD.json — the verdict is an artifact, not prose.)
```

**The field** — the SO-101 constants shipped by two companies, read in
their own trees (docs/e2e-research/30 §3.3–3.4): `kp=17.8, damping=0.60`
— one guess for six different joints, byte-identical across both repos;
the same vendor's own two repos disagree on the same arm by 56× in
stiffness. No intervals, no verdicts, no anchor, anywhere.

Ours says NOT PINNED where it cannot know. Theirs never says it
anywhere. **Possession of the robot is not identification of the robot.**

## What the instrument is

[`trainnr/`](trainnr/README.md) is the product
([architecture](docs/22-pipeline-architecture.md)): typed measurement
bundles with `name@hash` identity ([`robots/`](robots/rig-drivetrain/README.md));
identification wrapped thin over `mujoco.sysid` with intervals and
pinned/NOT-PINNED verdicts (`trainnr.robot.identify`); a fit-record
writer that **refuses** a fit without its recording hash and anchor
statement (`trainnr.robot.fit_record`); exact small-n statistics
with no dependencies, so a signed report is recomputable anywhere
(`trainnr.stats`); and an evaluation harness whose certificates gate
on the *lower* confidence bound (`trainnr.evaluate`).

Reproduce the flagship measurement in one command, no hardware — the
committed sweep recordings are the input, and the report is the product:

```sh
cd trainnr && uv sync --extra sim
uv run --extra sim python ../tools/fit-report.py ../robots/rig-drivetrain
```

To measure **your** robot from a CSV of its own log, start at
[`docs/28-quickstart-identify.md`](docs/28-quickstart-identify.md).

## Why you can trust the verdicts

The instrument was tested on itself, and the misses are on the record:

- The rehearsal converged to **2× truth with a tight interval** until
  the anchor was audited — which is why an anchor statement is now
  *refused-if-absent* on every fit record
  ([`robots/rig-drivetrain/README.md`](robots/rig-drivetrain/README.md)).
- A one-sample timestamp convention produced a biased fit with tight
  intervals (R13, [`docs/07-progress-log.md`](docs/07-progress-log.md));
  the same failure class was then caught **three more times** building
  the servo study — truth-recovery on synthetic data is now a standing
  gate for any new excitation harness
  ([`docs/26`](docs/26-sts3215-synthetic-identifiability.md)).
- The synthetic STS3215 study's cautionary finding: under structured
  corruption a fit can be **confidently wrong** — "pinned" with −53% to
  −88% errors — which is why fit records now spell out their pinning
  criterion and the residual-whiteness diagnostic is queued
  ([figure](data/sts3215-identifiability.html)).
- ⚠️ **There is no CI.** The gate (`tools/verify.sh`; it counts its own steps, and the
  Python suite alone is 220-odd tests) runs only when a human runs it, and one emulator step is RED —
  documented, not hidden: until it is fixed, "the full suite passes" is
  not a claim this repo can make
  ([details](docs/27-rig-tour.md)).

## What's next (the measured roadmap)

From the [strategy review](docs/25-strategy-review-2026-08-24.md):

1. **The bench protocol for a feedback arm is already specced,
   hardware-free**: the synthetic STS3215 study found position+load
   telemetry recovers all four parameters within ~6% at any rate down
   to 25 Hz — and that the derived-velocity register poisons the fit
   ([`docs/26`](docs/26-sts3215-synthetic-identifiability.md)).
2. **Paper 2** — the first sim↔real rank-correlation certificate for
   cheap arms; the real half is committed
   ([`data/armnetbench-v01-so101-counts.json`](data/armnetbench-v01-so101-counts.json)),
   the statistics are code
   ([`docs/23-research-agenda.md`](docs/23-research-agenda.md)).
3. **Paper 1** — the first graded robot bundle: parametric sysid of a
   low-cost servo arm plus the first published unit-to-unit spread.
4. Porting the whole loop to a new robot is a documented, honest recipe:
   [`docs/24-porting-the-rig.md`](docs/24-porting-the-rig.md).

## The shakedown vehicle

The $100 rover/arm rig — drive, chase, autonomous fetch laps — was the
shakedown vehicle that stress-tested this toolchain end to end; its
sessions live in [`recordings/`](recordings/README.md) and replay as
regression tests. The whole journey — staged roadmap, demos, learning
docs — is preserved in [`docs/27-rig-tour.md`](docs/27-rig-tour.md).

## Repository layout

```
trainnr/
├── trainnr/                # THE PRODUCT: bundles, identification, stats,
│   │                        #   evaluation, collection (Python, uv)
│   └── trainnr/{bundles,robot,stats,evaluate,collect,tasks,physics,envs}
├── robots/                  # measurement bundles: profile.json + model.xml
│   └── rig-drivetrain/      #   + fits/*.json + fits/SPREAD.json (real data)
├── data/                    # committed evidence (benchmarks, studies)
├── recordings/              # sessions that replay as regression tests
├── tools/                   # fit-report, sts-study/figure, verify.sh, viewers
├── crates/                  # Rust: wire protocol, recorder, sim-core, …
├── firmware/                # the rig's Pico firmware (no_std)
└── docs/                    # dated research, decisions, the progress log
```

## Knowledge base

Product and research, first:

- [`docs/35-the-studio.md`](docs/35-the-studio.md) — the app: a GPU-first, agentic, sim2real native platform, phased
- [`docs/29-the-platform.md`](docs/29-the-platform.md) — the destination: the two-sided platform, mapped and fenced
- [`docs/25-strategy-review-2026-08-24.md`](docs/25-strategy-review-2026-08-24.md) — the strategy: verdict, critical path, operator's list
- [`docs/22-pipeline-architecture.md`](docs/22-pipeline-architecture.md) — the codebase map and its contracts
- [`docs/23-research-agenda.md`](docs/23-research-agenda.md) — Papers 0–3, novelty adversarially pre-verified
- [`docs/26-sts3215-synthetic-identifiability.md`](docs/26-sts3215-synthetic-identifiability.md) — the servo study and its protocol verdicts
- [`docs/24-porting-the-rig.md`](docs/24-porting-the-rig.md) — the honest porting recipe
- [`docs/28-quickstart-identify.md`](docs/28-quickstart-identify.md) — measure your robot from a CSV
- [`docs/21-the-data-company.md`](docs/21-the-data-company.md) — the market read: the empty square is identification
- [`docs/20-video-to-vla-data.md`](docs/20-video-to-vla-data.md) — when a recording becomes training data
- [`docs/e2e-research/`](docs/e2e-research/README.md) — the fleet research (37 documents; start at [29-the-company](docs/e2e-research/29-the-company.md) and [30-the-pipeline](docs/e2e-research/30-the-pipeline.md))
- [`docs/07-progress-log.md`](docs/07-progress-log.md) — the dated log of everything done, decided, and gotten wrong

How this instrument was built (the apprenticeship record):

- [`docs/27-rig-tour.md`](docs/27-rig-tour.md) — the journey, the staged roadmap, and every demo
- [`docs/17-one-page.md`](docs/17-one-page.md) — the rig on one page
- [`docs/16-the-map.md`](docs/16-the-map.md) — every symbol, unit and formula
- [`docs/01`](docs/01-rust-robotics-stack.md)–[`docs/19`](docs/19-the-arm.md) — the dated research that got us here
- [`docs/learning/`](docs/learning/) — Rust walkthroughs, decoders, exercises
- [`docs/photos/`](docs/photos/README.md) — the dated photo record
