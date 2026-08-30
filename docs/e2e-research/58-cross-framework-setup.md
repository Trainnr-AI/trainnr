# The cross-framework setup: three artifacts the whole ecosystem is missing, and who adopts each

*2026-08-31. The design the mjlab ([56](56-mjlab.md)), BAM + microduck
([57](57-bam-source-and-microduck.md)) reads add up to. Everything
cited is evidence from those reads; design decisions are marked as
such. Supersedes the narrower "rq_mjlab bridge" sketch discussed after
56 — the two new reads showed the bridge half-exists and the real gap
is one level up.*

## 0. The design constraint that names the shape

The ask is a setup **the mjlab creators themselves would want to use**.
Their demonstrated taste (56): uv, tyro, typed configs, tests + a
changelog entry per PR, native MuJoCo mechanisms over bolted-on
abstractions, minimal deps, artifacts over side channels, Apache-2.0.
Their demonstrated wants, from their own repo: actuator slots they
built and cannot fill with measured values (1.5.0's dcmotor extras used
by no robot), a recorder API with zero implementations, a nightly that
publishes charts and gates nothing, a recurring class of
silent-staleness bugs, and monthly mjwarp churn they ritualized a bot
for. Rhoban's wants (57): their mjlab kernel unstuck from the 1.3/3.7
wall, their validation MAE not thrown away, contributions checked by
more than an eyeball. Pollen-class users' wants: 57 §5 verbatim.

Three parties, each missing what another has, all Apache-2.0. So the
setup is NOT a robotiq framework that others import — it is **three
neutral, versioned artifacts plus the tools that produce, validate and
consume them**, each solving one party's named pain, each usable
without adopting anything else of ours. robotiq's business sits behind
them (identification as a service, certification, the Studio), not
inside them.

## 1. Artifact one: the certified actuator bundle

*(design)* A versioned envelope AROUND BAM's flat params dict — wrap,
never replace, so every existing `bam` loader keeps working on the
inner object:

- **verbatim inner params** (`{param: float}` + `model` + `actuator`);
- **provenance**: dataset hash, per-log bench conditions
  (mass/length/kp/vin/trajectory), BAM git SHA + version, optimizer
  config, date, operator;
- **metrics BAM computes and discards**: train MAE, held-out
  `--validation_kp` MAE, per-log MAE distribution, and rail-hit flags
  (57 §3: published fits ship with `alpha ≈ 1e1` and load terms at
  1e-13 — a certification signal nobody records);
- **uncertainty**: per-parameter intervals (bootstrap over logs is the
  cheapest honest option; our stats layer already speaks intervals);
- **the context the flat JSON lies about**: `vin` and `kp_fw` (today
  hard-coded class defaults their docs wrongly claim live in the JSON),
  actuator-class binding, armature, forcerange, the MJCF companions
  (`solref`/`solimp` for the no-noslip workaround), the rig-vs-target
  `command_delay` split their own docs warn about;
- **derived DR ranges** from the intervals — so "synthetic data centred
  on identified values" becomes a generated field, not a hand-typed
  ±10% beside fits-treated-as-exact (57 §5);
- `schema_version`, units per field, required-key validation that
  FAILS LOUDLY (their loader silently ignores unknown keys; their key
  sets differ per motor).

Producer: `rq-bundle wrap` over a `bam.fit` output directory;
`rq-bundle verify` recomputes metrics and checks integrity. Adoption
path: usable by anyone with a BAM fit today; a natural PR to Rhoban
(it preserves data they already compute); the artifact our own M1–M6
library re-publishes in.

## 2. Artifact two: the maintained mjlab consumer

*(design)* `rq_mjlab` — a dependent package (the plugin pattern their
registry already supports; no fork) that:

- **ports the ~30-line BAM kernel to CURRENT mjlab** and keeps it
  there. This is the single highest-leverage code contribution in the
  ecosystem right now: Rhoban's `bam.mjlab` is walled at mjlab 1.3 /
  mujoco-warp 3.7 / Python 3.12-only (57 §4), production users ride an
  unpinned git branch, and mjlab ships a breaking minor monthly. We
  adopt mjlab's own churn ritual (one upstream bump per PR,
  locked+unlocked CI) — the discipline exists, it just has no owner
  here. Depend on `bam` CORE only (numpy, no wall); the wall lives
  entirely in their `[mjlab]` extra we replace.
- **builds `ActuatorCfg`s from verified bundles only** — an unstamped
  or stale bundle is refused by name. Kills "which fit was this ONNX
  trained against" being unanswerable (57 §5).
- **registers the field-expansion event automatically** — microduck's
  no-op-event-carrying-a-decorator invariant, made unforgettable.
- **ships a no-op DR linter**: at env init, cross-check every `dr.*`
  term against fields the actuator overwrites per step and raise. Five
  silent no-ops in one production repo (57 §5) is the proof this
  belongs in the manager layer; built here first, offered upstream —
  it is exactly the class of staleness bug mjlab's own changelog keeps
  fixing one at a time.
- **`dr_from_bundle`**: DR ranges generated from parameter intervals.
- **`entity_from_bundle`**: our hash-stamped robot bundles → mjlab
  entities, identified passives flowing through their supported
  override path.
- **the first real `RecorderTerm`**: streaming rollouts to a Rerun
  viewer (the Studio's :9876, or any). Their recorder API has zero
  implementations and their FAQ wishes for training-time visualization
  (56 §5, §6); this fills both with thirty lines and no new deps for
  them.

## 3. Artifact three: the deployment manifest + the certificate

*(design)* The contract microduck keeps in French markdown and CLI
flags (57 §5: `--action-scale 0.8` vs trained `1.0`; kp silently
detuned by a default), made machine-readable and asserted at both
ends: obs layout and order, scales/clips/history, action scale,
default pose, control rate, kp, actuator bundle hash, task/env config
snapshot. mjlab's ONNX metadata envelope (56 §5) is the precedent and
carries half of this already — the manifest is its superset, and on
the robot side a runtime refuses to start when its flags contradict
the manifest.

Above the manifest sits what stays entirely ours: **certification**.
Policies trained anywhere (mjlab via `rq_mjlab`, LeRobot, openpi) come
back through the ONNX adapter into our evaluator — stamped tasks,
paired trials, milestone funnels, both instruments, intervals,
refusal semantics. Train anywhere; the certificate comes from us.
Bench validation graduates from thresholds "chosen by feel" (57 §5)
to our stats layer's declared alpha/delta.

## 4. Who adopts what, and why they would

| Party | Takes | Because it fixes their named pain |
|---|---|---|
| mjlab core | the DR no-op linter; the RecorderTerm; nightly gates on their own harness | their recurring staleness-bug class; an empty API; a chart that asserts nothing (56) |
| Rhoban/BAM | the bundle envelope (preserves the MAE they discard); the current-mjlab kernel | zero tests, no schema, a dead version wall (57 §3–4) |
| Pollen-class users | bundles + `rq_mjlab` + manifests | all of 57 §5 — they already built a worse version of each by hand |
| robotiq | owns the spec, the tools, the certification authority, the Studio view of all of it | the wedge: nobody else measures, and now the measurement has a portable artifact |

Nothing requires anyone to adopt anything else; every piece is
additive to their existing repos; everything is Apache-2.0. That is
the property that makes it adoptable by the people who built the
frameworks — it reads as ecosystem infrastructure, not a land grab.

## 5. What robotiq keeps proprietary-by-competence

Not by license — by being the only ones who can do it: the
identification service (bench + protocol + the fits themselves), the
certification layer (two instruments, paired protocol, referee), and
the Studio (the one place training telemetry, sim viewport, eval
records and the agentic pipeline share a timeline). The open artifacts
make these MORE valuable: every bundle in circulation is an argument
for the service that produces trustworthy ones.

## 6. Sequencing, with gates

1. **Bundle spec + `rq-bundle wrap/verify`** over our existing vendored
   BAM fits (no GPU needed; pure Python; can start now).
2. **Kernel port to mjlab 1.6** + the refusing ActuatorCfg + auto
   event registration (needs the GPU box → stacks on the standing
   rl-engineering-merge-before-WSL gate).
3. **DR linter + `dr_from_bundle` + `entity_from_bundle`.**
4. **RecorderTerm → Studio**; the `train` specialist in the panel
   learns the mjlab commands.
5. **Manifest + ONNX-certify adapter** (after the rl-engineering
   merge, eval-side).
6. **Upstream offers**: bundle PR to Rhoban; linter + RecorderTerm
   PRs/issues to mjlab. Timed after 1–4 exist and are tested — code
   first, then the conversation.

Validation target throughout: the microduck/XL330 case — a real robot,
real fits, a real production repo to diff against. If our setup can
express *their* system with fewer laptop paths and zero silent no-ops,
it works.

## 7. What changed since 56 §7

56 rejected "adoption as substrate" and proposed a bridge robotiq
would build alone. 57 showed the bridge half-exists (Rhoban's
`bam.mjlab`) and is production-used (Pollen). The design therefore
shifts one level up: from *building* the bridge to *owning the
artifact layer* the existing bridge lacks — provenance, uncertainty,
validation, currency with mjlab — and contributing the code where it
belongs. The train-there/certify-here split from 56 stands unchanged.
