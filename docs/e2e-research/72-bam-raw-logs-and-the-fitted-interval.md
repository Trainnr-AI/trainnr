# BAM's raw bench logs are public: the fitted interval paper 1 was missing

*Researched 2026-09-05 (one agent, primary sources, URLs and dates
inline). The question: can the "narrow" arm's declared ±10 % become a
FITTED interval without our own bench? Answer: yes for the XL330 and
five other servos, on a CPU, in hours.*

## 1. What is public

- **Rhoban/bam** (https://github.com/Rhoban/bam, Apache-2.0; created
  2024-03-26; releases v1.0.0/v1.0.1 2026-06-30, v1.0.2 2026-07-16;
  last push 2026-08-30). The repo holds code and the fitted params
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

## 2. What nobody else publishes

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
`pipeline/rq_pipeline/robot/actuator_bundle.py` reads
`uncertainty: {param: {low, high}}` and names the basis
`identified-interval`. Missing for citation: the zip's sha256, BAM's
git SHA, the optimiser configuration.

## 4. The plan: a bootstrap refit, CPU only

Resample the 358 recordings by `(kp, mass, length)` block so the
design stays balanced, process each replicate with `bam.process`,
fit M6 with `bam.fit --trials ~5000`, repeat ~100 times: at under five
minutes a fit, about eight CPU-hours, parallel across cores. Record
per replicate the MAE and the parameter vector; the interval is the
2.5–97.5 percentile per parameter. Write it into the bundle's
`uncertainty` section with a new `fit` section (zip sha256, BAM SHA
`aa17d1c` = v1.0.2, trials, sampler, replicate count). Then the walk
C1 gains the arm the thesis names — trained under the *identified*
interval — beside point, declared ±10 % and ±30 %.

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
