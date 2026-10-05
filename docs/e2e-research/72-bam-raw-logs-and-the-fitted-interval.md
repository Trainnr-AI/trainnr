# BAM's raw bench logs are public: the fitted interval the draft paper was missing

*Researched 2026-09-05 (primary sources, URLs and dates inline). The
question: can the "narrow" arm's declared ±10 % become a
FITTED interval without our own bench? Answer: yes for the XL330 and
five other servos, on a CPU, in hours.*

## 1. What is public

- **Rhoban/bam** (https://github.com/Rhoban/bam, Apache-2.0; v1.0.2
  2026-07-16 = commit aa17d1cd5a84938b79143239de09ae33e175b402, the
  one we vendored and refit with; the creation date, v1.0.0/v1.0.1
  dates and last-push date below are the research agent's read of the
  GitHub page on 2026-09-05, not re-verified: created 2024-03-26,
  v1.0.0/v1.0.1 2026-06-30, last push 2026-08-30). The repo holds code and the fitted params
  (`bam/params/<slug>/m1..m6.json`), no data.
- **Raw logs** are linked from the docs
  (https://bam.readthedocs.io/en/latest/usage/actuators.html) to a
  Hugging Face *bucket*, not a dataset repo — no versioning, no DOI,
  no stated licence: https://huggingface.co/buckets/Gregwar/bam_data
  (5 files, 44.3 MB, uploaded ~2026-07-20):
  `xl330_raw.zip` 7,076,946 B — sha256 `d308d126…09546`, 358 JSON
  recordings under `data_raw_2/` (45 MB unzipped);
  `feetech_sts3215_raw.zip` 5,097,654 B; `mx64_raw.tgz` 10,862,907 B;
  `mx106_raw.tgz` 13,836,387 B; `xl320_raw.tgz` ~7.4 MB. Waveshare
  ST3025 logs live in a third-party release
  (https://github.com/i1Cps/duck_mini_pro_headless/releases/download/st3025-bam-data-v1/waveshare_st3025_raw.zip).
  No raw data for eRob; the XC430 is not covered by BAM at all.
- **Format**: one JSON per ~6 s recording — header `{mass, arm-mass,
  length, kp, vin, motor, trajectory}` and `entries[]` of
  `{timestamp, position, speed, load, input_volts, temp,
  goal_position, torque_enable}` at ~86 Hz. The XL330 sweep: kp ∈
  {50, 100, 150, 200, 300}, mass ∈ {0.04, 0.059, 0.117, 0.159} kg,
  length ∈ {0.11, 0.14, 0.17} m, six trajectories.
  `python -m bam.process --raw … --logdir … --dt 0.005` resamples.
- **Fitting**: the `bam.fit` module, Optuna `CmaEsSampler(restart_strategy=
  "bipop")`, objective = position MAE of a vectorised Euler rollout
  over all logs; `python -m bam.fit --actuator xl330 --model m6
  --logdir <processed> --output <json> [--trials N] [--workers N]
  [--validation_kp 8]`. No seed is set; no spread is reported. The
  paper (arXiv:2410.08650 v4, 2025-11-04; ICRA 2025) says the fit
  "was repeated three times, consistently converging to the same
  scores and parameters in under 5 minutes and approximately 4000
  iterations" — optimiser repeatability on MX-64/106 and eRob, not a
  data-resampling interval; the XL330 and STS3215 are not in the
  paper.

## 2. What we found published (2026-09-05 search; entries not re-verified are marked)

No public XL330 or STS3215 fit carries parameter intervals: SO-ARM100's
simulation constants are point values "adapted from Open Duck Mini";
pollen-robotics/microduck_rl randomises around BAM's M6 with
undocumented ranges; Robonine's STS3215 test stand (HardwareX,
2026-03-28, Zenodo 10.5281/zenodo.19261714) measures backlash only.
No 2024–2026 paper found does bootstrap or Bayesian BAM-style servo
identification (nearest: Kovalev et al., arXiv:2604.10351, point
estimates via differentiable simulation).

## 3. What our bundle carries, and lacks

`robots/actuators/xl330/PROVENANCE.json` records the source repo,
version 1.0.2, the raw-data URL, licence, and vendored date
(2026-08-28); the wrapped bundles `robots/actuator-bundles/xl330.m*`
carry the stamp and the floor/bound checks and **no `uncertainty`
section**. The consumer already exists:
`trainnr/trainnr/robot/actuator_bundle.py` reads
`uncertainty: {param: {low, high}}` and names the basis
`identified-interval`. Missing for citation: the zip's sha256, BAM's
git SHA, the optimiser configuration.

## 4. The plan: a bootstrap refit, CPU only (executed; results in §5–§6)

Resample the 358 recordings by `(kp, mass, length)` block so the
design stays balanced, process each replicate with `bam.process`,
fit M6 with `bam.fit --trials ~5000`, repeat ~100 times: at under five
minutes a fit, about eight CPU-hours, parallel across cores. Record
per replicate the MAE and the parameter vector; the interval is the
2.5–97.5 percentile per parameter. Write it into the bundle's
`uncertainty` section plus a `metrics` section (zip sha256, BAM SHA
`aa17d1c` = v1.0.2, trials, sampler, replicate count). Then the
walk's randomization-width study ("walk C1": the microduck walk
trained at point, declared ±10 %, declared ±30 % and the identified
interval; the draft paper's §5.4 and §5.6) gains the arm the thesis names —
trained under the *identified* interval.

Honest caveats to print with it: the interval captures log-sampling
variability on Rhoban's single unit and bench (their rig's
`q_offset` and `command_delay` included), not unit-to-unit or
temperature spread; the data licence is unstated, so ask the author
(Grégoire Passault) before redistributing the logs themselves — we
redistribute only the fitted numbers.

If the refit is not run, the sentence for the paper is: "BAM publishes
point estimates only; its raw XL330 logs (358 recordings, HF bucket
Gregwar/bam_data, 2026-07-20) permit a bootstrap refit, which we did
not run."

## 5. First result (2026-09-05): the shipped point is not the logs' best fit

On this laptop, `bam.process` over the 358 logs and one `bam.fit`
of M6 at 5000 trials took 370 s and plateaued at a position MAE of
0.0202 rad after ~3700 trials. Scored on the same logs with `bam.fit
--eval`, the shipped v1.0.2 M6 parameters give 0.0272 rad. The refit
reproduces the load-bearing parameters within 5 % — kt 0.348 vs
0.366, R 2.67 vs 2.81, armature 0.00180 vs 0.00181,
load_friction_motor 0.275 vs 0.267 — and moves freely where the
objective does not care: the two terms the shipped fit holds at its
1e-5 floors, `alpha` (0.50, the lower search bound, vs 8.68), the rig
offset `q_offset` (0.005 vs 0.027). Record `bam-xl330-refit-2026-09-05`;
params and scores under `docs/artifacts/bam/xl330/`. The bootstrap
(100 replicates, 5000 trials each) followed (§6): tight on kt, R and
armature and wide on the floor terms and `alpha` — the honest shape of "the identification's uncertainty" for
this servo on this bench. Open question for the study: the walk
policies were trained at the SHIPPED point; an identified-interval
arm centred on the refit needs its own point arm beside it, or the
interval's relative width applied around the shipped point, stated as
such.

## 6. The interval (2026-09-06, 100 replicates × 5000 trials, 8 cores, ~75 min)

| parameter | 2.5 % | 97.5 % | median | shipped v1.0.2 |
|---|---|---|---|---|
| kt | 0.3416 | 0.3662 | 0.3530 | 0.3660 |
| R | 2.545 | 3.036 | 2.778 | 2.811 |
| armature | 0.001707 | 0.001972 | 0.001844 | 0.001808 |
| friction_viscous | 0.00538 | 0.00853 | 0.00692 | 0.00536 |
| load_friction_motor | 0.0028 | 0.327 | 0.268 | 0.267 |
| load_friction_external_stribeck | 0.0049 | 0.384 | 0.155 | 0.081 |
| dtheta_stribeck | 0.071 | 4.97 | 0.836 | 2.89 |
| alpha | 0.500 | 9.66 | 4.25 | 8.68 |

(the full table, every replicate's vector and the bootstrap's metrics
are in the bundle `robots/actuator-bundles/xl330-refit.m6.bundle.json`,
stamp `xl330-m6@e57c25635c89`; the summary with the zip hash and BAM
commit rides in its `metrics`.) The shape: the motor constant is known
to ±3.5 %, resistance to ±9 %, armature to ±7 %, viscous friction to
±23 %; the friction *split* between motor and external sides, the
Stribeck knee and the `alpha` exponent are not identified by this
bench — their intervals are their search ranges. The shipped fit's
kt sits at the interval's upper edge.

**Why the identified arm draws vectors, not boxes.** Those unidentified
terms trade off against each other in the fit: a replicate with high
motor friction has low external friction. Drawing each from its
marginal box independently would combine values no replicate had. The
bundle therefore also carries all 100 replicate vectors (`samples`),
and the DR event picks one whole vector per world (`trainnr_mjlab.dr`,
basis "identified-set"). The marginal `uncertainty` box stays on the
bundle for readers and for consumers that only understand boxes.

**Run 2026-09-05 UTC:** three point arms and three identified arms on the
refit bundle (`tools/walk-c1-refit-pods.sh`), certified at the refit's
fit; then their matrix cells (record `walk-c1-refit` and
`walk-mismatch-matrix-refit`). The draft paper's §5.6 (published separately) is those certificates.

## 7. BAM's declared search bounds (read from the clone at aa17d1c, 2026-09-06)

Rhoban/bam's model module (its Parameter takes initial, min, max): `alpha` (1.35, 0.5,
10.0); `dtheta_stribeck` (0.2, 0.01, 5.0); `load_friction_motor_quad`
and `load_friction_external_quad` (0.0, 0.0, 0.01); the Stribeck load
terms (0.05, 0.0, 1.0); `friction_base` and `friction_stribeck`
(0.05, 0.0, max_friction_base); `friction_viscous` (0.1, 0.0,
max_viscous_friction). Its Dynamixel actuator module (dynamixel.py under bam/actuators in their repository), the XL330 block:
`kt` (0.7, 0.25, 1.5), `armature` (0.0005, 0.0001, 0.01). So the
refit's alpha 0.5009 is at the declared floor, the bootstrap's alpha
interval [0.50, 9.66] spans nearly the whole declared range, and the
quadratic terms' 97.5 % values (0.0099) sit at their ceiling. These
bounds are recorded on `docs/artifacts/bam/xl330/bootstrap.json` and
the bootstrap record; the bundle's `checks.near_search_bound` knows
only the observed rails from the published fits, so it did not flag
alpha.
