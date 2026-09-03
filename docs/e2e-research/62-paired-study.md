# The paired study (C1): does identified-region DR beat guessed DR, measurably?

*2026-08-31. The protocol for docs/00 Phase C1 — "the number the
datasheet earns its claims with". Smoke ran end to end the same day
(0/4 vs 0/4, UNRESOLVED, as §3 predicts). **The sized run ran the same
night on a rented B200** (32 demos/arm on `lift-study`, ACT 10k steps
batch 64, judged at the pinned truth on 40 matched trials/side — the
§3 power table's own N): **guessed 40/40, identified 40/40, CI
[0.912, 1.0] each, p = 1.0 → INSENSITIVE at α = 0.05, δ = 0.15.** A
real verdict, published as §4 demands: at THIS truth distance —
inside the ±30 % folklore span — with THIS open-loop-robust expert
and 10k-step ACT at ceiling, guessed DR is measurably sufficient.
§5's first risk materialized exactly as written. **Run 2, same night,
truth OUTSIDE the span both ways (damping ×1.50, gain ×0.60 — chosen
by sweep, expert 8/8 there)**: guessed 23/40 [0.409, 0.730],
identified 19/40 [0.315, 0.639], p = 0.50 → UNRESOLVED; paired
discordance 18/40, near-symmetric (11 guessed-only vs 7
identified-only). The load-bearing observation: the identified arm
scored 47.5 % AT ITS OWN TRAINING CENTER — the recipe (ACT 10k steps,
32 demos) caps at ~half at this difficulty, and when competence caps,
no DR basis can show through it. Two runs, two honest nulls: at an
easy truth both arms hit the ceiling; at a hard truth both hit the
recipe's ceiling. **Protocol amendment for run 3: a competence gate —
each arm is first judged under its own training center, and the
cross-judgment is informative only if the identified arm passes high
there.** Remaining knobs: more demos/steps at hard truths (raise the
ceiling), a brittler-but-learnable task, the C2 flagship's RL
locomotion (a different recipe with a real dynamics-sensitivity
profile), or the real robot's region. This page publishes the nulls
either way — the instrument reports what is, not what sells.*


> **Caveat added 2026-09-04.** Every lift-study evaluation before commit
> `ecb04ac` was judged through a harness that played each policy action
> for ONE control tick while the datasets were pressed at `frame_every=5`
> (10 Hz) — every chunk five times too fast (the walk's cadence bug in a
> second costume; docs/07 2026-09-04). The two nulls stand as measured:
> the easy-truth 40/40 vs 40/40 shows the task survived even a 5×-fast
> replay, and the hard-truth ~47% cap is a lower bound on what the
> recipe can do. Both are re-run under the held harness in the
> `c1-competent-lift` study (docs/studies/), which also adds the point arm.

## 0. The claim under test

Our datasheets say synthetic data is drawn from an identified
confidence region, not a guess, and imply that this matters for the
trained policy. docs/23's evidence line (sim-predicts-real r = 0.924
with careful correspondence vs ≈ 0.60 without) is suggestive and
uncontrolled; nobody in the surveyed ecosystems has run the controlled
version (59 §5: Arena's generation path has "no dynamics
randomization"; mjlab trains on hand-set ranges). C1 runs it.

**Hypothesis (directional):** a policy trained on demonstrations
pressed under dynamics drawn from a tight identified interval around
the truth succeeds at least as often — evaluated AT the truth — as the
same architecture trained on demonstrations pressed under the folklore
span around nominal, and the gap grows with the distance between
nominal and truth.

## 1. The design: synthetic truth, so the experiment is controlled

The real study wants a physical robot: identify it, train two ways,
deploy both. Before hardware (docs/38's first-touch plan), the
controlled version substitutes a SYNTHETIC truth — a dynamics vector
we choose and then hide from both arms of the study:

- **Truth**: `damping ×1.18, gain ×0.85` of nominal (off-nominal but
  inside the expert's measured competence — the lift expert keeps
  10/10 at ±30 % with the both-terms gain rule, docs/07 2026-08-26).
- **Arm GUESSED**: per-episode draws from `U(1 ∓ 0.30)` on both
  parameters, centred on NOMINAL — the folklore span, exactly what
  our own kitting generator and every surveyed framework does.
- **Arm IDENTIFIED**: per-episode draws from `U(truth · (1 ∓ 0.05))`
  — the stand-in for a fit record's interval. The real study replaces
  this line with `declared_ranges(bundle)` over a measured bundle and
  NOTHING else changes; the basis string on every manifest is the
  difference between "we declared this" and "we measured this".
- Everything else equal by construction: same task (`lift`), same
  open-loop scripted expert, same episode count, same converter, same
  trainer config, same eval protocol. The two datasets differ ONLY in
  the dynamics dict + basis line of their manifests — auditable from
  the datasheets alone.

Why lift: its expert is open-loop (a step-scheduled waypoint script),
so generation is deterministic given the draws; its referee reads
privileged state; it has the ArmnetBench cameras for the imitation
policy; and `tests/_instruments.py` pins its expert rate per engine
build.

## 2. The phases

1. **generate** (exists: `tools/paired-study.py generate`, on
   `pipeline/rq_pipeline/collect/scripted_demos.py`): press N episodes per arm, referee-
   gated, frames + full sidecars + datasheet per arm; `study.json`
   records truth, conditions, seeds, and this protocol's path.
2. **convert** (exists: `pipeline/rq_pipeline/collect/lerobot_export.py`): both arms to
   LeRobot datasets, provenance carried.
3. **train** (next): the same trainer config twice (LeRobot ACT at
   smoke scale locally; real scale on the rented card). Nothing about
   the condition leaks into the config.
4. **evaluate** (next): both checkpoints through the gymnasium env
   under the TRUTH dynamics on MATCHED trials (the paired protocol —
   policy A's trial k and policy B's trial k start identically), on
   BOTH instruments (CPU MuJoCo + MJX-Warp), folded by
   `pipeline/rq_pipeline/evaluate/records.py`, compared by `pipeline/rq_pipeline/stats/effects.py` — exact
   intervals, paired comparison, a verdict under declared alpha/delta.
   Also evaluated AT nominal, to show the guessed arm is not simply
   broken — the claim is about transfer to truth, not competence.

## 3. Sizing, honestly

The smoke run (tonight) is 8 episodes/arm and smoke-scale training:
it proves the protocol executes end-to-end and the accounting is
paired; its verdict will be UNRESOLVED by construction (docs/32's
power arithmetic — distinguishing rates at these sample sizes needs
dozens of paired trials). The real run's sizing comes from
`pipeline/rq_pipeline/stats/power.py` before any GPU hour is spent —
exact power of the SAME `fisher_exact` the verdict uses, stdlib-pure,
pinned by `pipeline/tests/test_stats_power.py`. Measured 2026-08-31
(alpha 0.05, power 0.8, per-side paired trials; the test's actual
size at n=40 is 0.03 — exact tests are conservative):

| guessed rate at truth | identified rate at truth | N per side |
|---|---|---|
| 0.3 | 0.7 | 29 |
| 0.4 | 0.8 | 27 |
| 0.5 | 0.9 | 23 |
| 0.6 | 0.9 | 36 |
| 0.7 | 0.95 | 39 |

Reading: if the truth-distance is enough to cost the guessed arm ~0.4
of success rate, ~25–30 paired trials per side resolve it; a subtler
0.3 gap needs ~36–39. **The budget the sizing implies**: evaluation is
cheap (a 5 s episode even on CPU; 2 × 40 trials is minutes) — the cost
is the two REAL trainings, and T5's precedent (docs/34: ACT 10k steps
at batch 64 on a rented B200 in well under an hour) puts the whole
study around **2–3 GPU-hours ≈ $10–20** plus demo generation, which
is CPU. The decision the operator holds is that number. That sizing
note — "we computed N before running" — is itself a positioning line
no surveyed framework can write.

## 4. What the result means either way

- Identified wins at truth: the datasheet's basis line is worth
  money; C-phase outreach cites the number.
- No detectable difference at N: the folklore span is good enough for
  THIS task/expert/architecture at THIS truth distance — publish
  that too; the harness exists to measure, not to flatter the wedge
  (the honesty rows in docs/33 bind us).
- Guessed wins: almost certainly a bug in the pairing or leakage;
  the paired records make it auditable episode by episode.

## 5. Risks

- The open-loop expert's robustness (10/10 at ±30 %) may flatten the
  training-data difference: both arms' demos succeed, so the policies
  may see similar state distributions. If the smoke suggests this,
  the real run moves the truth further out or uses the closed-loop
  stack expert — the protocol survives, the knob is declared.
- Smoke-scale training noise dwarfs everything: expected; the smoke
  validates plumbing, never the hypothesis.
- One seed per arm at generation: pairing is at EVALUATION (matched
  trials), which is where the statistics need it; generation pairing
  would couple the arms' draw sequences for no inferential gain.
