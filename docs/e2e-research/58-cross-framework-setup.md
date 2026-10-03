# The cross-framework setup: three artifacts the whole ecosystem is missing, and who adopts each

*2026-08-31. THE consolidated design of the four-read arc: mjlab
([56](56-mjlab.md)), BAM + microduck
([57](57-bam-source-and-microduck.md)), and Isaac Lab Arena
([59](59-arena-composition-and-datagen.md), extending 39–46).
Everything cited is evidence from those reads; design decisions are
marked as such. Supersedes the narrower "trainnr_mjlab bridge" sketch
discussed after 56 — the later reads showed the bridge half-exists and
the real gap is one level up. §7–§8 (Arena, and the synthetic-data
segue) added the same day the Arena read landed.*

## 0. The design constraint that names the shape

The target is a setup **the mjlab creators themselves would want to use**.
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
setup is NOT a trainnr framework that others import — it is **three
neutral, versioned artifacts plus the tools that produce, validate and
consume them**, each solving one party's named pain, each usable
without adopting anything else of ours. trainnr's business sits behind
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

Producer: `tools/actuator-bundle.py wrap` over a `bam.fit` output directory;
`tools/actuator-bundle.py verify` recomputes metrics and checks integrity. Adoption
path: usable by anyone with a BAM fit today; a natural PR to Rhoban
(it preserves data they already compute); the artifact our own M1–M6
library re-publishes in. *(SHIPPED 2026-08-31 as
`trainnr/robot/actuator_bundle.py` + `tools/actuator-bundle.py` +
the 48 committed bundles under `robots/actuator-bundles/` — 6 rail
flags and 9 floor flags found in BAM's published fits on day one;
metrics/uncertainty sections await re-fitting from logs.)*

## 2. Artifact two: the maintained mjlab consumer

*(design)* `trainnr_mjlab` — a dependent package (the plugin pattern their
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

*(design)* The contract microduck keeps in prose plan documents and CLI
flags (57 §5: `--action-scale 0.8` vs trained `1.0`; kp silently
detuned by a default), made machine-readable and asserted at both
ends: obs layout and order, scales/clips/history, action scale,
default pose, control rate, kp, actuator bundle hash, task/env config
snapshot. mjlab's ONNX metadata envelope (56 §5) is the precedent and
carries half of this already — the manifest is its superset, and on
the robot side a runtime refuses to start when its flags contradict
the manifest.

Above the manifest sits what stays entirely ours: **certification**.
Policies trained anywhere (mjlab via `trainnr_mjlab`, LeRobot, openpi) come
back through the ONNX adapter into our evaluator — stamped tasks,
paired trials, milestone funnels, both instruments, intervals,
refusal semantics. Train anywhere; the certificate comes from us.
Bench validation graduates from thresholds "chosen by feel" (57 §5)
to our stats layer's declared alpha/delta.

## 4. Who adopts what, and why they would

| Party | Would take | Because it would fix their named pain |
|---|---|---|
| mjlab core | the DR no-op linter; the RecorderTerm; nightly gates on their own harness | their recurring staleness-bug class; an empty API; a chart that asserts nothing (56) |
| Rhoban/BAM | the bundle envelope (preserves the MAE they discard); the current-mjlab kernel | zero tests, no schema, a dead version wall (57 §3–4) |
| Pollen-class users | bundles + `trainnr_mjlab` + manifests | all of 57 §5 — they already built a worse version of each by hand |
| Arena/NVIDIA | certified bundles for their hand-entered gains; the manifest; certification behind their eval | their README asks for "sim-to-real validated evaluation methods"; `sysid` = 0 hits in 992 files ([59](59-arena-composition-and-datagen.md) §4) |
| trainnr | owns the spec, the tools, the certification authority, the Studio view of all of it | the wedge: nobody else measures, and now the measurement has a portable artifact |

Nothing requires anyone to adopt anything else; every piece is
additive to their existing repos; everything is Apache-2.0. That is
the property that makes it adoptable by the people who built the
frameworks — it reads as ecosystem infrastructure, not a land grab.

## 5. What trainnr would keep by competence, not by licence

Everything here is Apache-2.0; what stays distinctive is what only the
measuring party can do: the
identification service (bench + protocol + the fits themselves), the
certification layer (two instruments, paired protocol, referee), and
the Studio (the one place training telemetry, sim viewport, eval
records and the agentic pipeline share a timeline). The open artifacts
make these MORE valuable: every bundle in circulation is an argument
for the service that produces trustworthy ones.

## 6. Sequencing, with gates

*Status (2026-10-03): `trainnr-mjlab/` exists with the no-op linter; the
RecorderTerm is `trainnr-mjlab/src/trainnr_mjlab/recorder.py`; the
deployment manifest is `trainnr/trainnr/deploy/manifest.py`; the bundle
tool is `tools/actuator-bundle.py`; the upstream offers were drafted and
not sent.*

1. **Bundle spec + `tools/actuator-bundle.py wrap/verify`** over our existing vendored
   BAM fits (no GPU needed; pure Python; can start now).
2. **Kernel port to mjlab 1.6** + the refusing ActuatorCfg + auto
   event registration (needs the GPU; an earlier merge gate was dissolved
   2026-08-31, so this waited only on a GPU session).
3. **DR linter + `dr_from_bundle` + `entity_from_bundle`.**
4. **RecorderTerm → Studio**; the `train` specialist in the panel
   learns the mjlab commands.
5. **Manifest + ONNX-certify adapter** (eval-side).
6. **Upstream offers**: bundle PR to Rhoban; linter + RecorderTerm
   PRs/issues to mjlab. Timed after 1–4 exist and are tested — code
   first, then the conversation.

Validation target throughout: the microduck/XL330 case — a real robot,
real fits, a real production repo to diff against. If our setup can
express *their* system with fewer laptop paths and zero silent no-ops,
it works.

## 7. Arena joins the map: what the third pole adds ([59](59-arena-composition-and-datagen.md))

Arena is NVIDIA's config compiler over Isaac Lab — the eval-and-data
pole to mjlab's training pole. Four things it settles for this design:

- **The manifest has a proven shape.** `ArenaEnvGraphSpec` is a typed,
  simulator-import-free scene/task graph validated before any engine
  boots. Our cross-framework manifest (artifact three, §3) adopts that
  shape: pure-typed, validated cold, on a laptop — which Arena itself
  cannot do for anything else (one sim-free test file in 992; Linux
  x86-64 + RTX + EULA + a beta submodule to import a config).
- **The plugin mechanics are half-solved twice.** Arena has typed
  registries + a dotted external-class path but no entry points and no
  pip package ("vendor as an unmodified git submodule"); mjlab has
  import-side-effect registration and an unwritten plugin guide. Our
  packages use `importlib.metadata` entry points from day one — the
  boring, correct mechanism both ecosystems skipped.
- **The portable seams are now ecosystem-consensus.** Both poles
  independently arrived at the same boundary: out-of-process policy
  servers (websocket/zmq), ONNX with embedded metadata, LeRobot
  datasets out. That is where certification plugs in with zero
  framework adoption — Arena's remote policies and mjlab's exported
  ONNX both walk through our door unchanged.
- **The solver is converging under everyone.** Arena runs MuJoCo-Warp
  via Newton inside Isaac Lab (`MJWarpSolverCfg`); mjlab runs it
  natively; we instrumented it (49, 52). A solver-level identified
  actuator (per-step dof writes, 57 §2) therefore has a path into BOTH
  ecosystems — one measured artifact, two host frameworks. INFERENCE:
  the Newton path needs its own port and its own verification (51
  documented Newton's importer zeroing servo damping); do not claim it
  until measured.

And the thesis survives its third test: `sysid` has **zero hits in
Arena's 992 files**, a shipped arm carries `stiffness = 1745.32922e3`
with no source, their G1 fix for a gain-induced twitch was to copy
training values rather than measure hardware, and their README lists
"sim-to-real validated evaluation methods" under contributions they
WANT. Three ecosystems, one absent layer, now vendor-acknowledged.

## 8. The segue: synthetic data generation, using all of the above

The next phase (sim + synthetic data) starts from these reads, not
from zero. Borrow, with sources:

- **The success-gated generation contract** (Arena/Mimic, 59 §5):
  `generation_guarantee` + drop-failures + retry-to-N + abort-at-K.
  "N demos" must always mean N *successful* demos. Our scripted
  experts replace their teleop seeds — we can generate the source
  demonstrations Mimic requires humans for.
- **The augmentation unit**: per-subtask config (object ref,
  termination signal, offset range, nearest-neighbor source selection,
  bounded noise, interpolation) — the right granularity for
  demo-multiplication, and it maps onto our milestone vocabulary
  almost 1:1.
- **The two-file recording split**: bulk trajectories (HDF5/LeRobot)
  + an append-as-you-go per-episode JSONL of
  {seed, success, length, variations} — plus everything Arena's
  version omits and ours must refuse to run without: config
  content-hash, engine stamp, code SHA, and the identified-dynamics
  parameter vector actually sampled per episode.
- **DR breadth from mjlab** (56 §4): the ~50 typed randomizers with
  declared model fields; ranges generated from fit intervals
  (`dr_from_bundle`, §2 above) instead of Arena's flat action noise
  and microduck's hand-typed ±10%.
- **The differentiator, stated once**: every other data factory's
  realism knob is noise around a guess. Ours samples dynamics from the
  identified confidence region and *refuses* variants outside it — the
  dataset inherits the fit's provenance, and the eval layer refuses to
  compare datasets whose stamps disagree. Nothing in mjlab, Arena,
  BAM or microduck can currently say which dynamics produced a given
  demonstration; every record of ours will.

## 9. The dialect rule: whose conventions win where *(standing, 2026-08-31)*

The rule: the product must hold trainnr's coding standards
AND read as native in each host's dialect. The boundary principle —
**at every interface we speak THEIR language; inside the core we keep
ours** — resolved per artifact, from the idioms the reads measured:

- **`trainnr_mjlab` is written in mjlab's dialect** (56): `*Cfg`
  dataclasses, `entity_name` (their deliberate rename), the
  `edit_spec → initialize → compute` actuator lifecycle verbatim, the
  `param_names` fusion contract, tyro-compatible flags with explicit
  booleans, uv packaging, torch, Python ≥3.10, tests + a changelog
  entry per change — an mjlab maintainer should read it as mjlab code.
- **The bundle wraps BAM's dialect untouched** (57): their flat params
  dict rides verbatim under one key — no key ever renamed, their
  `model`/`actuator` keys sacred, `params/<motor>/` layout respected —
  with our additions as sibling namespaced keys (`provenance`,
  `metrics`, `uncertainty`, `context`, `schema_version`). Both a
  `python -m` module (their CLI style) and a console script.
- **Press outputs speak LeRobot exactly** (32/45): the v2.1 layout then,
  v3 now (README),
  `observation.state`/`action` feature naming — a dataset consumer
  must not be able to tell it wasn't written by their own recorder.
- **The manifest takes Arena's spec shape** (59): typed,
  validated-cold, `ArenaEnvGraphSpec`-like — but the CORE stays
  stdlib-pure (the auditor's-laptop rule): dataclasses + stdlib
  validation in core; pydantic and other host deps live only in
  adapter packages that already carry that host.
- **Skills follow Arena's shipped format** (59): `arena:SKILL.md` +
  `allowed-tools` + `arena:evaluations.md` — the shape both creator teams
  already merge into their repos.

Where dialects and standards conflict, OURS win on semantics, never
on surface: we emit files their loaders accept, but our loaders never
silently ignore unknown keys (BAM's does — that is a gap, not a
convention); no hardcoded constants regardless of host style; refusal
messages name the artifact and the reason everywhere. Surface is
theirs; discipline is ours.

## 10. What changed since 56 §7

56 rejected "adoption as substrate" and proposed a bridge trainnr
would build alone. 57 showed the bridge half-exists (Rhoban's
`bam.mjlab`) and is production-used (Pollen). 59 showed the third pole
has the same absent layer and asks for it by name. The design
therefore sits one level up: from *building* a bridge to *owning the
artifact layer* every bridge lacks — provenance, uncertainty,
validation, currency — with the code contributed where it belongs and
certification behind it. The train-there/certify-here split from 56
stands unchanged; Arena adds evaluate-anywhere/certify-here to it.
