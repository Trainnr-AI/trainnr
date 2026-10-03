# The data press: cross-framework synthetic data generation nobody has to adopt

*2026-08-31. The synthetic-data half of the cross-framework design
([58](58-cross-framework-setup.md) §8 expanded into a product plan).
Evidence base: the four-read arc (56, 57, 59), the field's measured
rankings ([22](22-data-generation.md)), the synthetic-data principles
recorded with the end-to-end test ([docs/22 T5](../22-pipeline-architecture.md)),
and — decisive for the plan's footing — what `trainnr/collect` already
does. Design decisions are marked as such.*

## 0. The zero-adoption principle, applied to data

Same rule as 58: every input is a file they already have, every output
is a format they already consume, and the engine runs standalone. A
user of mjlab, Arena, LeRobot or a bare Menagerie MJCF gets data
powers they are missing without importing anything of ours into their
framework — the press is a tool you point at your robot, not a
framework you move into.

## 1. The powers each creator is missing (measured)

- **mjlab has no data story at all** (56 §5): a recorder API with zero
  implementations, no dataset export, no scripted experts, no BC
  corpus. An mjlab user who wants imitation/VLA data for their robot
  has nothing — and 13 downstream projects means many such users.
- **Arena's factory needs humans and an RTX rig** (59 §5): the Mimic
  pipeline is real but teleop-seeded (a person in a headset per source
  demo), USD-only, EULA-gated, Linux-x86-64-only; its provenance is a
  seed, its only realism knob is isotropic pose noise, and there is no
  dynamics randomization in the data path. Their own published GR00T
  zero-shot numbers (0.0 on most objects) state the data hunger.
- **BAM's fits never reach data generation** (57): the one measured
  artifact in the ecosystem stops at the actuator; nothing samples
  training data from what was identified.
- **LeRobot-rig users collect by hand** ([21](21-data-collection.md)): ~16–48 demos/hour
  of human teleop under the 50-episode rules; the field's measured
  gains ([22](22-data-generation.md)) come mostly from curation and alignment because
  generation they can trust doesn't exist.

## 2. What already exists in-repo: the micro-press

`trainnr/collect/kitting_demos.py::generate_demos` is a
success-gated generator TODAY: draw dynamics, run the scripted expert,
keep only referee-passing episodes, and write each with a `Manifest`
recording the seed, the attempt number, **the actual draws**
(damping/gain scales), the DR span, the retries and the expert stamp.
`kitting_export.py::DatasetProvenance` carries bundle + expert +
every episode's manifest beside the dataset — its own docstring:
"traceable to the exact dynamics and draws, the same rule as the
certificate." `lerobot_export.py` writes the consensus format. One
task, two DR axes, fully principled. The plan below is this press,
generalized — not a new machine.

## 3. The press, full shape *(design)*

**Inputs** (each optional beyond the first):
1. An **MJCF robot/scene** — what every mjlab user, Menagerie robot
   and our bundles already have. (Arena's USD is the one wall; their
   own MJCF-adjacent path is Newton — see §6.)
2. A **certified actuator bundle** (58 artifact 1) — enables
   identified-dynamics sampling. Without it, the press still runs but
   the datasheet says so, loudly.
3. A **demonstration source**: a scripted expert (ours — the thing no
   other framework has, 56 §4), or N seed demos (LeRobot episodes, or
   Arena/Isaac-Lab HDF5 imported) for Mimic-style multiplication.
4. A **task referee** — success predicates + milestone funnel; for
   imported tasks, at minimum a success predicate, and the datasheet
   marks funnel-less data as such.

**The generation core**, on plain MuJoCo/MJX-Warp (the engine both
ecosystems are converging on, 59 §3 — and the only part of any of
their stacks that runs on a laptop):
- **Expert-driven**: the existing press loop, generalized past
  kitting: per-episode dynamics drawn from the bundle's identified
  confidence region (`dr_from_bundle`, 58 §2), refusal of draws
  outside it, referee-gated keep/discard, milestone funnel recorded
  per episode. The randomiser itself stays tested (the gain-that-was-a-setpoint
  lesson from the end-to-end test, [docs/22 T5](../22-pipeline-architecture.md),
  is a standing test pattern).
- **Multiplication from seeds** (for tasks without experts): the
  Mimic contract exactly as Arena runs it — per-subtask units (object
  ref, termination signal, offset range, nearest-neighbor source
  selection, bounded noise, interpolation; 59 §5), success-gated
  (`guarantee`, drop failures, retry to N, abort at K) — but with
  dynamics drawn per generated episode from the identified region,
  which their path cannot do, and with physics-verified success
  rather than pose-proximity.
- **Batch scale**: MJX-Warp parallel worlds for the draw sweep;
  CPU MuJoCo spot-checks a sample of kept episodes as the second
  instrument (the divergence habit from docs/e2e-research/49 applied to data).

**Outputs**:
- A **LeRobot dataset** (v2.1 when this was written; the press writes
  v3 since 2026-09, as the README says) — the format LeRobot trainers, GR00T
  finetuning and the HF hub already consume (Arena's own export
  target, 59 §5). Zero adoption on the training side.
- The **provenance sidecar**, generalized from `DatasetProvenance`:
  bundle hash, task stamp, expert stamp, engine stamp per episode,
  every episode's dynamics vector and funnel. The fold refuses to
  mix datasets whose stamps disagree (58 §8).
- A **datasheet**: an auto-generated report — kept/attempted rates,
  funnel distributions, draw histograms against the identified
  intervals, engine stamps, refusals with reasons. The document a
  dataset consumer reads before trusting; nobody in the arc ships
  one.

**Per-host adapters** (thin, optional, each a bridge not a burden):
- mjlab: the RecorderTerm (58 §2) captures RL rollouts into the same
  dataset+sidecar shape — mjlab's empty recorder API becomes their
  data story.
- Arena/Isaac: HDF5 importer for teleop seeds — their format in, our
  provenance out.
- LeRobot rigs: real teleop episodes enter as seeds with the same
  sidecar, so real and synthetic share one provenance schema — the
  co-training ratio question (how much real data a mix needs before
  synthetic data stops helping) becomes *measurable* per dataset
  instead of folklore.

## 4. What the field's measurements dictate *(evidence → ordering)*

[22](22-data-generation.md)'s ranking by measured real-robot gain puts curation and
alignment above everything generative, and world-model generation at
0.00–0.07 task success (Veo-Act). The press encodes that:
1. **Curation is built in, not bolted on** — the referee gate + funnel
   + refusal IS curation, applied at generation time; the +15–35
   point curation gains come free with the architecture.
2. **Dynamics before pixels**: identified-region dynamics DR is the
   press's realism engine; visual DR (mjlab's material/light functions
   exist if hosted there) ranks second.
3. **No generative video path.** Rejected with the field's own
   numbers; revisit only when someone publishes contact-correct
   generation.

## 5. Why the creators use it without "adopting" anything

| They have | They run | They get |
|---|---|---|
| an MJCF robot (mjlab user) | the press, standalone | a LeRobot dataset + datasheet — the data story mjlab doesn't have |
| teleop HDF5 + an RTX rig (Arena user) | the importer + multiplication | dynamics-randomized, provenance-stamped data their factory can't make — no USD, no EULA, on a laptop |
| a BAM fit (Rhoban-style user) | bundle wrap + the press | their fit finally producing training data, not just sim behavior |
| a LeRobot rig (Pollen-style user) | seeds in, press out | 10 human demos → N stamped synthetic ones, real+synthetic under one provenance schema |

The press never asks them to change their trainer, their format, or
their framework. That is the whole trick, twice now (58 §0).

## 6. Honest walls and open questions

- **Arena in-place generation is out of scope**: their assets are USD
  with no MJCF exit (59 §3). The bridge is their HDF5 out, not our
  code in. If Newton's USD→MuJoCo-Warp path matures, revisit — with
  51's importer findings as the checklist.
- **Experts don't generalize for free**: our scripted experts are
  per-task programs. The multiplication path exists precisely for
  tasks without them; the acceptance referee (task not accepted until
  the expert passes its own review) stays the quality floor for the
  expert path.
- **The sim-vs-real value of identified-region DR is our claim to
  prove, not assume**: the validation experiment is one paired study —
  same task, same trainer, dataset A (datasheet-guessed DR) vs
  dataset B (identified-region DR), evaluated under the paired
  protocol on both instruments. Until that number exists, the
  datasheet advertises provenance, not transfer gains. The
  microduck/XL330 stack is the natural external target (57 §5).
- **Video/camera features**: the press inherits the render path we
  have (offscreen MuJoCo); camera-heavy datasets at scale may want
  mjlab's Warp ray-traced cameras — a host-adapter question, not a
  core one.

## 7. Sequencing *(design)*

1. **Generalize the micro-press**: task-agnostic `generate_demos`
   (task registry + referee + expert in, today's kitting as case one),
   `DatasetProvenance` → the full sidecar schema. CPU-only, no gates.
   *(SHIPPED 2026-08-31: `trainnr/collect/press.py` — the loop +
   `EpisodeManifest` with task/instrument/expert stamps and the named
   dynamics vector; kitting is the first adapter, its legacy manifest
   and committed batches untouched.)*
2. **`dr_from_bundle` sampling + refusal** — lands with 58's bundle
   spec (also CPU-only). *(SHIPPED: `trainnr_mjlab/dr.py`
   `dr_from_bundle`, `trainnr/robot/actuator_bundle.py` `declared_ranges`.)*
3. **The datasheet generator** over existing records. *(SHIPPED:
   `trainnr/collect/datasheet.py`, rewritten over merged shards by
   `collect/shards.py`.)*
4. **Multiplication from seeds** (Mimic-contract, physics-verified) —
   the first piece that benefits from a GPU. *(SHIPPED as the
   `multiply_demos` tool.)*
5. **Host adapters** (mjlab RecorderTerm, Arena HDF5 import) with
   58's `trainnr_mjlab`. *(PARTLY: `trainnr_mjlab/recorder.py` is a
   Rerun recorder term, not a dataset recorder; no Arena import.)*
6. **The paired validation study** (§6) — the number the datasheet's
   claims rest on. *(RAN: [62](62-paired-study.md), then the
   c1-competent-lift records in docs/findings.)*

Steps 1–3 needed no GPU and no new deps: they were refactors and
reports over machinery the repo already trusted.
