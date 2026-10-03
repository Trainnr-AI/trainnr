# Isaac Lab Arena: variations and sensitivity analysis, read code-first

*Source: the `isaaclab_arena` repository, read 2026-08-26 from a repomix
bundle made in session (132,303 lines; commit not recorded). The bundle is
not shipped; every `L<number>` below is a line of that bundle. Our-side line
numbers are from the 2026-08-26 tree: `EpisodeProtocol` is now
`trainnr/trainnr/protocol.py`, the aloha2 task module is now the package `trainnr/trainnr/tasks/aloha2/`, `bundles/` is
`trainnr/trainnr/bundles/`.*

*Fifth pass, 2026-08-26. One agent, one field, one primary source: the
full `isaaclab_arena` repository as a repomix bundle (132,303 lines).
Every claim cites `path:line` into that bundle with a quoted fragment;
nothing is filled from memory. Mapped against our harness
(`trainnr/evaluate/harness.py`, now `trainnr/trainnr/protocol.py`), our
certificate (`trainnr/stats/`), and our domain randomisation
(`tools/kitting-demos.py`, `tools/show-many.py`). Field: how Arena
parameterises an environment, sweeps it, and turns per-episode results
into "which factors break the policy".*

---

Legend: **CODE** = the source file; **DOCS** = Arena's `.rst` pages.

## 1. What Arena does

### 1.1 The variation: one knob, one sampler, one hook

A *variation* is Arena's unit of environment parameterisation: a named
object attached to a scene asset (a light, a camera, a rigid object)
that owns a *sampler* (a distribution to draw from) and a *hook* (the
code that writes the drawn value into the scene). CODE,
`isaaclab_arena/variations/variation_base.py:90770`: "A `VariationBase`
pairs a target asset with a sampler and a hook that realises one tweak
to the scene. Variations attach to any Asset and start disabled." Its
config has two base fields (`:90801-90805`): `enabled: bool = False`
and `sampler_cfg: SamplerBaseCfg`; applying a config rebuilds the live
sampler (`:90888`, `self._sampler = cfg.sampler_cfg.build()`).

Two samplers exist. **Uniform** over a box
(`variations/uniform_sampler.py:90722-90726`): `low: list[float]`,
`high: list[float]`, "Length determines the sampler's
shape_per_sample"; draws are `low + (high - low) * U(0, 1)` (`:90758`).
**Choice** over labels (`variations/choice_sampler.py:90013`):
`torch.randint(low=0, high=len(choices), ...)`. Both draw from torch's
*global* RNG — no sampler takes a seed or a trial index. Every sampler
carries *listeners* (`variations/sampler_base.py:90686`, "called as
`listener(sample, env_ids)` for every sample drawn"); that is how draws
get recorded (§1.4) without the hook knowing.

### 1.2 Build-time versus run-time

DOCS, `docs/pages/concepts/variations/variations.rst:7112-7119`:
build-time changes "Before the environment is built" and applies to
"Every parallel environment and episode in that build" (background
image, lighting); run-time changes "When an environment resets" and
applies to "One episode in one parallel environment" (camera
extrinsics, intrinsics). CODE, `variation_base.py:90894-90921`:
`RunTimeVariationBase` returns an Isaac Lab reset event term
(`build_event_cfg() -> tuple[str, EventTermCfg]`);
`BuildTimeVariationBase` implements `_realize_at_build_time()`,
"Called once per env build, while the variation is enabled."

Sweep consequence (`variations.rst:7121-7123`): "To collect several
values of a build-time variation, the environment must be rebuilt
several times." The run config carries `num_rebuilds: int = 1`,
"Number of fresh environment constructions over which metrics are
aggregated" (`isaaclab_arena/evaluation/arena_run.py:33455-33456`);
the episode budget is split "as evenly as possible across rebuilds"
(`evaluation/run_execution.py:34889-34902`), each rebuild offsets the
builder seed by its index (`:34810`, `cfg.environment_builder.seed +=
rebuild_index`) and writes its own `episode_results_rebuild{i}.jsonl`
(`:34783`).

### 1.3 The shipped catalogue (CODE, with defaults)

| Variation | Timing | Sampler default | Hook |
|---|---|---|---|
| `CameraExtrinsicsVariation` | run | uniform ±0.005 m XYZ in the camera's ROS optical frame (`variations/camera_extrinsics_variation.py:89672-89678`) | nominal + delta, nominal snapshotted on first call "so offsets don't compound across resets" (`:89733-89735`) |
| `CameraIntrinsicsVariation` | run | uniform ±0.1 fractional on (fx, fy) (`variations/camera_intrinsics_variation.py:89837-89842`) | scales USD aperture by 1/(1+d); forces the rig untiled because "a per-env intrinsic edit would leak across all tiles" (`:89853-89855`) |
| `HDRImageVariation` | build | choice over registered HDR names; `hdr_names` empty = all (`variations/hdr_image_variation.py:90112-90116`) | `self._light.add_hdr(hdr_cls())` (`:90162`) |
| `LightIntensityVariation` | build | uniform [100, 2000] (`variations/light_intensity_variation.py:90427`) | `set_intensity` |
| `LightColorVariation` | build | uniform RGB in [0,1]³ (`variations/light_color_variation.py:90249-90251`) | `set_color` |
| `LightColorTemperatureVariation` | build | uniform [1000, 10000] K (`variations/light_color_temperature_variation.py:90192`) | `set_color_temperature` |
| `LightDirectionVariation` | build | uniform azimuth [−π, π], elevation [0°, 80°]; dims the dome to 500 "so shadows are visible" (`variations/light_direction_variation.py:90336-90352`) | quaternion from (azimuth, elevation) |
| `ObjectMassVariation` | run | uniform [0.05, 2.0] kg absolute; `recompute_inertia=True` scales inertia by the mass ratio (`variations/object_mass_variation.py:90496-90500`, `:90639-90642`) | `set_masses_index` per env; masses below 1e-6 kg are rejected, not clamped (`:90487-90489`) |

Docs/code drift: the docs table (`variations.rst:7257-7280`) lists
`CameraIntrinsicsBuildTimeVariation` / `CameraIntrinsicsRunTimeVariation`
and omits `ObjectMassVariation`; the source has one run-time
`CameraIntrinsicsVariation` and does ship mass. Nothing varies
textures, friction, actuator dynamics or object pose — placement is a
separate relation-solver subsystem, not a variation.

### 1.4 Discovery, overrides, and the per-episode record

**Discovery.** `--list_variations` prints every attachable variation as
copyable override paths (`variations.rst:7130-7160`): `Enable:
light.hdr_image.enabled=true (default: False)` then `Fields:
light.hdr_image.hdr_names = []`.

**Overrides.** The key is `{asset_name}.{variation_name}.{field}`.
CODE, `variations/variations_hydra.py:91109-91113`:
`"cracker_box.color.enabled=true"`,
`"cracker_box.color.sampler.low=[0.2,0.2,0.0]"`. A typed dataclass
schema is built dynamically — one field per asset, one per variation,
"typed as the variation's own Cfg class" (`:91158-91164`) — and Hydra
composes overrides into it; an unknown path raises with the valid
paths listed (`:91209-91220`). In an experiment YAML the same paths sit
under `variations:` (`droid_pnp_camera_sensitivity_openpi_experiment.yaml:98656-98659`:
`...camera_extrinsics_wrist_camera.sampler_cfg.low: [-0.03, -0.03, -0.03]`).

**The record.** A `VariationRecorder` subscribes before any draw
(`environments/arena_env_builder.py:31977-31980`, "Attach the variation
recorder before any sampling, so it observes both build-time samples
... and run-time samples"). Draws are keyed by `EnvEpisodeKey(env_id,
episode_idx)` (`variations/variation_recorder.py:90942-90947`); a
build-time draw "applies to every episode of every env" (`:90979`). At
each reset one JSON line per finished episode is appended
(`recording/episode_recorder_manager.py:38180-38210`), merging core
fields `env_id, episode_in_env, seed, success, episode_length,
language_instruction, timestamp` (`recording/common_terms.py:38059-38067`)
with a `variations` block `{asset.variation: value}` (`:38070-38082`).
A real row (`episode_results_camera_displacement.jsonl:113210`):

```
{"job_name": "policy_runner", "env_id": 9, "episode_in_env": 0, "seed": 42,
 "success": true, "episode_length": 204, ...,
 "variations": {"light.hdr_image": "home_office_robolab",
   "droid_abs_joint_pos.camera_extrinsics_wrist_camera": [-0.0128, -0.0285, -0.0171]}}
```

This JSONL is the contract between evaluation and analysis.

### 1.5 The sensitivity package

**Schema is discovered from data, not authored.** CODE,
`analysis/sensitivity/episode_results_reader.py:21193-21199`: "A
number becomes a continuous factor, a numeric vector becomes one
continuous factor per component (named key[i]), and a string becomes
a categorical factor over its observed labels." Bools are labels
(`:21251-21253`); a continuous range is the observed `(min, max)`
(`:21364-21368`); a single-valued factor is dropped (`:21365-21367`);
all-constant raises (`:21386-21389`); rows must carry identical
factors (`:21315-21333`). A categorical whose top choice is ≥1.5× its
rarest warns (`_IMBALANCE_WARN_RATIO = 1.5`, `:21184`): "Its posterior
reflects this sampling frequency, not only its effect on the outcome"
(`:21405-21408`).

**Dataset.** `SensitivityDataset` (`analysis/sensitivity/dataset.py:21074-21081`)
holds `theta` "(num_episodes, num_factors) ... continuous factors in
the leading columns, then one integer-coded column per categorical
factor" and `x` "(num_episodes, num_outcomes)"; `FactorSpec(name,
type, range, choices)` (`:21052-21063`). `default_observation()` is 1.0
for every outcome and asserts outcomes are binary (`:21146-21154`).

**Estimator.** `SensitivityAnalyzer` (`analysis/sensitivity/analyzer.py:20935-20952`)
"Fits a neural posterior over all factors, conditioned on all
outcomes" with the `sbi` library: `MNPE` if any factor is categorical,
else `NPE` (`:20971-20977`); `BoxUniform` prior on normalised [0,1] /
[0, k−1] (`:20979-20987`); `fit()` = `append_simulations(theta, x)` +
`train(...)` (`:21010-21013`); `sample_posterior()` draws 5,000 at the
observation (`:21016-21028`). The reported quantity is **p(factors |
success = 1)** — no effect size, no ranking, no interval on the
posterior. Arena's case for it (DOCS,
`concept_sensitivity_analysis.rst:7691-7702`): "Factors interact ...
Factors confound each other ... The per-factor rate is a projection of
the joint posterior — derivable from it, but not the other way around."

**Report.** `analysis/sensitivity/generate_report.py:21453-21496`: seed
torch, read JSONL, fit, sample, `plot_marginals`. One panel per factor
(`analysis/sensitivity/plotting.py:21600-21603`): continuous = KDE of
the posterior, the uniform prior as a flat dashed line, a shaded 5–95%
band (`:21665-21683`); categorical = bar chart of rounded-code
frequencies (`:21699-21708`). CLI: `--episode_results`, `--outcome`,
`--factors`, `--observation`, `--seed` (`:21499-21551`). Reading rule
(`concept_sensitivity_analysis.rst:7808-7814`): "mass concentrated at
one end of its range means success favoured that end."

**Validation and scale.** Synthetic data with planted effects
(`tests/sensitivity_synthetic.py:51734-51741`: light +2.5, grasp
offset −2.5, oak best); tests assert the posterior mean crosses the
range midpoint in the planted direction
(`tests/test_sensitivity_analysis.py:80492-80499`) on 2,000–3,000
episodes (`:51765`, `:51779`). The shipped workflow fits on **10**
(`sensitivity_analysis.rst:12809`, "fitting NPE_C on 10 episodes"),
admittedly "not enough to judge the policy's robustness"
(`:12751-12752`). Declared scope (`concept_sensitivity_analysis.rst:7816-7835`):
binary outcomes; "Factors should be drawn from the prior the analyzer
assumes"; one policy/task per JSONL.

## 2. What we already have that is equivalent

| Arena | Ours | Where |
|---|---|---|
| Run-time variation of object pose at reset | `EpisodeProtocol.perturb(trial_index, home)`, deterministic per trial; ALOHA's cube walks spawn-box corners (`_corner_fraction(trial, inset=0.1)`) | `harness.py:45-67`, `trainnr/trainnr/tasks/aloha2/kitting.py` (lines as of 2026-08-26) |
| Paired trials | Built in: "policy A's trial 7 starts exactly where policy B's trial 7 starts"; perturb "receives the trial index (not an RNG)". Arena has no equivalent — its samplers hit the global RNG at reset | `harness.py:5-7, 49-51` |
| Dynamics variation | ±span scaling of joint damping and actuator gains around the bundle's identified values, per episode (`damping_scale`, `gain_scale`) and per world (`randomise()`) | `tools/kitting-demos.py:99-108`, `tools/show-many.py:126-144` |
| Per-episode record | `manifest.json` per kept demo: `seed, attempt, draws, dr_span, damping_scale, gain_scale, verdict` — Arena's row shape, but only for *kept demos*, never for evaluation trials | `tools/kitting-demos.py:168-185` |
| Outcome statistics | Exact Clopper-Pearson/Wilson, Spearman with Fisher-z and exact permutation p, `top_pick_probability`, pooling with Cochran's Q — stdlib only. Arena ships none | `trainnr/stats/intervals.py`, `ranking.py`, `pooling.py` |
| The certificate | `Certificate` binds bundle hashes, per-policy intervals, rank interval, gate bit | `trainnr/evaluate/certificate.py:50-96` |

What we lack, precisely: (a) a *declared, enumerable* variation space —
our perturbations are closures inside task files, not data; (b) a
per-trial record — `score_policies` returns only `SimScore(successes,
trials)` (`harness.py:135-143`), so which start produced which outcome
is gone when the loop ends; (c) any factor-to-outcome analysis.

## 3. Adopt / skip

*Status (2026-10-03): §3.1 is `trainnr/trainnr/evaluate/variations.py` and
`trainnr/trainnr/physics/variations.py`; §3.3's statistic is
`trainnr/trainnr/stats/effects.py`.*

### 3.1 ADOPT — a variation schema as data (new module `trainnr/evaluate/variations`)

```python
@dataclass(frozen=True)
class Variation:
    host: str                    # "cube", "right/shoulder", "top" (camera), "light0"
    name: str                    # "pose_xy", "damping_scale", "cam_offset", "mass"
    sampler: Uniform | Choice    # Uniform(low: tuple, high: tuple) | Choice(labels: tuple)
    timing: Literal["reset", "compile"]   # Arena's run-time / build-time
    enabled: bool = False
```

Keep Arena's three good decisions: the dotted key `host.name.field` as
override and record key; `enabled=False` by default plus a
`--list-variations` catalogue so the whole space is visible before a
trial is spent; and the snapshot-then-write hook rule ("nominal +
delta so offsets don't compound"). Change one: **the sampler takes the
trial index, never an RNG** — `draw(trial)` from a hash of
`(protocol_hash, host.name, trial)` — so every policy sees the
identical factor vector on trial *t*; the sensitivity comparison
*between* policies is then paired too, which Arena cannot do. For
`Choice`, assign labels round-robin by trial: exact balance, where
Arena can only warn at 1.5×.

MuJoCo collapses most of Arena's build-time class into reset-time:
`model.body_mass`, `model.dof_damping`, `model.actuator_gainprm`,
`model.cam_pos`, `model.light_*` are writable without recompiling.
Reserve `timing="compile"` for MjSpec-level changes (mesh/texture
swaps, geometry) and recompile — milliseconds, so no `num_rebuilds`
budget-splitting. First factor set, centred on identified values (the
show-many rule, never a guess): `damping_scale`, `gain_scale` within
the sysid interval, `cube.mass`, `cube.pose_xy`, `top.cam_offset`
(±3 cm, Arena's workflow range), `light.diffuse`. The dynamics factors
are the ones Arena lacks and the span paper (docs/paper/manuscript.md)
turns on.

### 3.2 ADOPT — the per-trial record

Extend `score_policies` to emit one JSONL row per (policy, trial)
beside `SimScore`: `{policy, trial, success, steps, source,
variations: {host.name: value}}`. Enforce Arena's reader contract at
write time: every row carries every enabled factor; bools are labels;
vectors named by component axis, not `[i]` (Arena's own TODO,
`episode_results_reader.py:21259-21262`). The certificate, the
sensitivity table and a Rerun replay all read this file; today we
throw it away.

### 3.3 ADOPT the question, SKIP the estimator — the statistic

Arena's question is right: *given the outcome, where were the
factors?* Its answer — a neural posterior via `sbi` — we skip. (1) For
a uniform, independent, balanced sweep the marginal p(θᵢ | success) is,
by Bayes, the per-bin success rate renormalised; Arena concedes "the
per-factor rate is a projection of the joint posterior". The joint
model earns its keep only against *confounding*, which a paired,
balanced factorial design removes by construction. (2) It reports no
uncertainty and no effect size, so "which factor" is read off a KDE by
eye — and at 4–20 trials per policy (10 in Arena's own workflow) a
neural density estimator fits noise. (3) `sbi` + torch breaks the
stdlib-only rule that keeps a certificate recomputable on an auditor's
laptop (`intervals.py:1-7`).

Adopt instead, stdlib only, as a new `trainnr/stats` module:

- **Main effect per factor.** Continuous: split at the range midpoint
  (Arena's own test threshold, `:80469-80472`); categorical: per
  label. Report success counts each side with Clopper-Pearson
  intervals, the rate difference, and an **exact Fisher test** p
  (hypergeometric from the `lgamma` we already use) — the "exact at
  small n" discipline of `exact_spearman_p`.
- **Verdict per factor**: SENSITIVE (difference interval excludes 0),
  INSENSITIVE (difference bounded inside ±δ, δ declared on the
  protocol), UNRESOLVED (neither — the honest answer at n = 4).
- **Paired across policies**: trial *t* carries the same factor vector
  for every policy, so two policies' per-factor effects difference as
  McNemar-style discordant counts.
- **Interactions** only as 2-factor tables when every cell meets the
  protocol's minimum count; otherwise say so rather than fit.

### 3.4 ADOPT the report shape; the deliverable is a table

Keep one-panel-per-factor and "posterior against a flat prior" — as
the binned success curve against the overall rate, with intervals —
for the viewer (Rerun, beside the run). The deliverable is a **table
in the certificate**: factor, range swept, n per side, rates with
intervals, difference, exact p, verdict. A customer reads "camera
offset: SENSITIVE (1/8 vs 7/8, p = 0.005)", not a density curve.

### 3.5 SKIP, with reasons

Hydra/OmegaConf schema composition (`variations_hydra.py:91127-91155`)
— ours are frozen dataclasses hashed into the protocol; a typed
override path is a 20-line function. Isaac Lab event-term machinery —
MuJoCo has no manager stack. `num_rebuilds` splitting — recompile is
free. Schema discovery *from* data and range inference from observed
min/max — the swept range is a protocol fact, hashed, never inferred;
keep discovery only as a fallback reader for someone else's JSONL.

## 4. Open questions

1. **Trial budget.** Arena validated on 2,000–3,000 synthetic episodes
   and shipped a 10-episode workflow. At what n does INSENSITIVE within
   ±δ become claimable? Tabulate the exact test's power first.
2. **Sweep design.** Independent uniform draws per factor (Arena) or a
   Latin hypercube by trial index (balanced in continuous factors too)?
   Either way the design goes into the protocol hash.
3. **The span paper's hook.** Sweeping `damping_scale`/`gain_scale` across the
   sysid *interval* turns dynamics uncertainty into a measured factor
   effect: a policy INSENSITIVE across the interval has a sim rank that
   holds wherever reality sits in it. Certificate field or separate report?
4. **Which factor first for ALOHA 2.** Camera offset is Arena's
   headline; our T1 gap was kinematic (fingertip height). Sweep the
   kinematic factor before the visual one?
5. **Vector naming.** Arena names `key[0]`; we should name by frame axis
   (`cam_offset.x_right`). Does the bundle's camera spec or the
   variation own the axis convention?
