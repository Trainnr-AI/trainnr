# End-to-end research: prototype → shippable robot fleet

Research date: **2026-08-08**. A field-by-field pass on what it would take to
build, deploy and operate a small commercial fleet of **mobile manipulators** —
data gathering, data generation, simulation, sim training, real training, the
app layer, deployment and telemetry.

**These are research notes, not a plan.** Nothing here has been built, bought or
committed to. Every claim carries a date and a source so the next pass can see
what has decayed, and [27-open-questions.md](27-open-questions.md) records what
this pass could *not* settle.

**Second pass: 2026-08-15.** Docs 20–27 re-swept field by field against primary
sources (search quota was again exhausted, so discovery ran on the arXiv/HF/
GitHub APIs — blog-only vendor news remains under-sampled). Each doc carries a
dated re-verification note; [29-the-company.md](29-the-company.md) records the
business thesis that pass was tested against.

**Third pass: 2026-08-16 → 2026-08-20 — the commercial and competitive
sweeps.** Thirteen companies examined at primary sources (vendor pages, SEC
filings, PyPI/download counts, and — for PhAIL and Lightwheel — their own
repositories read code-first). Results live in [30 §3](30-the-pipeline.md)
and [29 §5](29-the-company.md); the standing conclusion is that **nobody
measures the customer's robot**. An adversarial review of the whole corpus
followed; its findings and their fix status live in
[31-defects.md](31-defects.md).

**Fourth pass: 2026-08-25 — the full-loop sprint.** Seven fields
researched in parallel (one agent per field, primary sources) to
harden the surfaces the operator's platform brief added
([../30-the-full-loop.md](../30-the-full-loop.md)); results in docs
32–38, verdicts summarized in the brief's §6. One standing verdict
amended (physics splats, doc 23 cross-linked); one decided (MLOps
stack); the rest confirmed with refinements.

**Fifth pass: 2026-08-26 — Isaac Lab Arena, read code-first.** After
T5's training loop closed on the WSL card, six fields of NVIDIA's
Arena (the one shipping composable-evaluation system) were read in
parallel from a repomix bundle of its whole repository — docs and
source, 977 files — one agent per field, every claim cited to a bundle
line. Results in docs 39–44; verdicts and the order of adoption in
[../30-the-full-loop.md §7](../30-the-full-loop.md). The finding that
ties them: Arena has the coverage machinery and no honesty layer — no
interval anywhere in the code — and five of six reports converge on
one artifact we lack, the per-trial record.

**Sixth pass: 2026-08-26 — the evaluation layer, minimum lines.** The
operator's question "the most standard, ecosystem-fit, production-grade,
minimum-lines evaluation layer": LeRobot's contract read from the
installed package (45), six ecosystem eval interfaces from their
repositories (46), and a line-by-line audit of our own surface. The plan
is [../32-evaluation-layer.md](../32-evaluation-layer.md).

Read [19-the-system.md](19-the-system.md) first; it is the map.

| Doc | Field | The one thing to remember |
|---|---|---|
| [19](19-the-system.md) | The system end to end | Four tiers, and **Tier 0/1 is what this repo is for** |
| [20](20-policies-and-models.md) | Policies and models | Third-party benchmarks contradict author self-reports by **5×** — build your own harness |
| [21](21-data-collection.md) | Data collection | Leader-arm teleop, **~50 demos per task, then switch to corrections** |
| [22](22-data-generation.md) | Data generation | **Curation beats generation.** A $50 green screen beat diffusion augmentation |
| [23](23-simulation-and-real2sim.md) | Simulation and real→sim | Three channels; **the simulator's first job is evaluation, not training** |
| [24](24-compute-and-hardware.md) | Compute and hardware | No arm at this price runs 8 h/day; the WSL blocker narrowed to ManiSkill3 — **whether MuJoCo Warp runs under WSL is the open question** |
| [25](25-deployment-and-fleet-ops.md) | Deployment and fleet ops | MCAP, signed release manifests, and **calibration is device state** |
| [26](26-safety-and-regulation.md) | Safety and regulation | **Keep the ML out of the safety path** — it resolves two regimes at once |
| [27](27-open-questions.md) | Open questions | What the next pass should start with |
| [28](28-wifi-on-the-chip.md) | WiFi on the chip | cyw43 can't join WPA2-Enterprise, secure boot bypassable on A2 silicon — **the chip stays tethered; WiFi belongs on a Linux node** |
| [29](29-the-company.md) | The company thesis | Scene and engine are commodities; **the unclaimed layer is the customer's robot's own dynamics** |
| [30](30-the-pipeline.md) | The pipeline design | Twelve stages (⓪–⑪), three gates — **a number from an unvalidated simulator is not evidence** |
| [31](31-defects.md) | The defect register | Adversarial review of everything above, with fix status — **the corpus audits itself** |
| [32](32-recipe-engine-prior-art.md) | Auto-selection + data mixtures | Every pillar exists in isolation — **the certificate-judged recipe engine is unclaimed**; validation loss provably fails as a judge |
| [33](33-agentic-engineering.md) | Agent-written engineering | Copilots gate on humans, labs gate on solvability — **hash-stamped artifacts behind agent-untouchable gates is unclaimed**; referees must be tamper-evident |
| [34](34-physics-splats-2026.md) | Physics-carrying splats | Splats still never carry contact, but **the collision proxy now arrives through the splat pipeline** — scanner output is a splat+proxy pair |
| [35](35-modular-mojo-max.md) | Modular Mojo/MAX | Qualcomm owns it, Mojo is Apache-2.0, **zero VLAs ever served on MAX** — active-watch with defined flip triggers |
| [36](36-newton-status.md) | Newton status | One organism with MuJoCo, not a fork — and **MJX-Warp gives the Newton-era solver on the sysid'd mjModel directly** |
| [37](37-mlops-tooling.md) | MLOps tooling | **wandb is the only tracker robot learning uses** — adopt its API surface with the Trackio escape; the registry stays custom |
| [38](38-fleet-data-planes.md) | Fleet data planes | Every fleet converged on the three tiers; **the trigger-campaign retro-pull is the load-bearing AV pattern**; our calibration chain is ahead of practice |
| [39](39-arena-metrics-and-progress.md) | Arena: metrics & progress | `success_rate = np.mean`, no interval or paired test anywhere — **adopt the progress funnel** (ordered predicate chains, events, `success ≠ all_complete` flag), computed offline over the arrays we keep |
| [40](40-arena-experiments-and-runner.md) | Arena: experiments & runner | The per-episode JSONL row is the contract between evaluation and analysis — **adopt the record with the pairing key Arena lacks**, run-per-directory output, failed runs as data; skip OSMO/Hydra |
| [41](41-arena-variations-and-sensitivity.md) | Arena: variations & sensitivity | Samplers hit the global RNG, nothing is paired; sensitivity is a neural posterior on 10 episodes — **adopt the schema with `draw(trial)`, answer the question with a main-effects table in the certificate** |
| [42](42-arena-placement-and-relations.md) | Arena: placement & relations | Relations → Adam solver → validators → pools, with failed layouts stored by default — **adopt the validators on `mj_forward` contacts and `arm_ik`, raise on exhaustion**; skip the solver |
| [43](43-arena-policy-interface.md) | Arena: policy interface | One blocking `get_action`, horizons picked by eye — **executed horizon becomes a protocol field**, rig/model adapter split, a chunk-replay client over openpi's websocket for π0.5 |
| [44](44-arena-environment-and-agentic-generation.md) | Arena: env spec & agent layer | "Validation only proves a spec is admissible, not that it is the environment you asked for" — **the existence proof for §3.6**: load gates against the compiled model, critic loop capped at 3, hand-written predicates and physics |
| [45](45-lerobot-eval-contract.md) | LeRobot's eval contract (installed 0.6.1, code-first) | Four keys (`pixels`, `agent_pos`, `is_success`, `_max_episode_steps`), a `lerobot_env_*` package is auto-imported, `lerobot-train` evaluates through the same `make_env` — **expose our task as a LeRobot env, keep the harness as the judge**; `lerobot-eval` has no statistics |
| [46](46-ecosystem-eval-interfaces.md) | The ecosystem's eval interfaces (gymnasium, openpi, GR00T, LIBERO, SimplerEnv, EnvHub, robomimic) | One `gym.Env` with `{pixels, agent_pos, task}` and `info["success"]`+`info["is_success"]` drives all six — **no interval anywhere in any of them**, LeRobot's `eval_info.json` is the only written per-episode record |

## The five findings that changed the picture

1. **Simulation is a consumer of real data, not a postscript.** Reality reaches
   a simulator through three separate channels — the robot, the scene, the
   behaviour — and conflating them is why "just use a simulator" fails.
2. **Simulated evaluation appears to predict real policy quality better than a
   small real evaluation does** (r ≈ 0.92 versus r ≈ 0.60). That inverts the
   usual intuition about what a simulator is for.
3. **The published numbers disagree with each other by 5×** for the same model
   on the same class of hardware, and confidence intervals live in exactly one
   vendor post and one arXiv paper — in no leaderboard and no shipping
   framework (narrowed 2026-08-19).
4. **Curation and cheap augmentation beat everything generative.** World models
   generate plausible approach motion and cannot generate contact.
5. **Keeping the ML out of the safety path is worth €5–15k and 3–6 months**, and
   it is a description of the Tier 0 architecture this repository already has.

## Caveat on coverage

**WebSearch quota was exhausted before any research agent in the first two
passes issued a query.** Those passes ran on direct fetches of primary sources
— official repos, model cards, arXiv, vendor pages, standards bodies. Good
source quality, poor discovery coverage: **mid-2026 releases may be missing.**
The third pass reached further (SEC filings, hiring boards, repository
code-reads) but its company list is still a convenience sample, not a census —
[29 §5.1](29-the-company.md) carries that scope bound explicitly. See
[27 §0](27-open-questions.md).
- [47 — Newton, read docs-first](47-newton-docs-review.md): the engine's own docs reviewed locally (1.6.0.dev0); parity ledger (keyframes/sensors not imported), solver stable, MEASURED solver sweep — mjSOL_NEWTON + elliptic is the only configuration that passes kitting on both arm64 and x86_64 (§7.1).
- [48 — Solver landscape](48-solver-landscape.md): MuJoCo's solver family mapped to our scenes; our config is the documented anti-slip recipe; sysid fits are fits OF the discretization → the option block is bundle state.
- [49 — GPU path, probed](49-gpu-path-mjxwarp.md): MJX-Warp runs on the Mac (CPU), covers our contact regime fully, batches model params for DR; costs float32 + GPU non-determinism → CPU stays the metrology instrument.
- [50 — Newton delta + live probe](50-newton-delta-probe.md): v1.5.0 still latest (PyPI name is `newton`, not `newton-physics`); SolverMuJoCo runs on the Mac CPU (300-1000x slower than plain mujoco); our so101.xml imports with matching counts but the sensor block silently vanishes — verified by execution; Isaac Lab trigger unfired.
- [51 — SolverMuJoCo round-trip, measured](51-solvermujoco-roundtrip.md): sensors/keyframes/visual geoms gone, root mass rewritten, integrator flipped, servo kv ZEROED at import (undamped servos); deterministic-mode mechanism mapped, contact-path determinism unproven even by Newton's own suite.
- [52 — Warp determinism × mujoco_warp](52-warp-determinism-mjwarp.md): read at both sources — RUN_TO_RUN rewrites exactly the atomic patterns mjwarp's hot paths use (one result-deterministic CAS aside); implicitfast-on-Warp resolved (supported); deterministic GPU certificates are one WSL probe from settled, cost is workload-shaped and can be negative under contention.
- [53 — BAM actuator identification](53-bam-actuator-identification.md): a published, working identification pipeline for Feetech STS3215 among 6 other servos (ICRA 2025, Apache-2.0) — answers docs/27's open convergence question in part; a real mjlab/MJX-Warp version wall (mujoco-warp<3.8 vs our 3.11 pin); a backlash-modeling MJCF pattern directly relevant to our own measured STS3215 backlash.
- [54 — Zed + Rerun product architecture](54-zed-rerun-product-architecture.md): the literal premise (Rerun embedded in Zed) is blocked by Zed's own extension sandbox; the real path is ACP + MCP with Rerun as a companion window, and most of "bring your robot, sim, train, deploy" already exists in this repo — the gap is packaging, not architecture.
- [55 — Rerun's visualization catalog, mapped](55-rerun-viz-catalog.md): all 11 views and every archetype from the reference, against what the repo already logs; the split rule — native egui_plot panels for glanceable state (Rerun's own TimeSeriesView is egui_plot on the exact egui 0.36.1 studio-shell pins), the real Rerun viewer for anything with a scrubber, a query, or a camera; GaussianSplats3D, MCAP and StateTimeline flagged as already-first-class for later phases.
- [56 — mjlab, read at the source](56-mjlab.md): Berkeley's Isaac-Lab-on-MuJoCo-Warp framework (v1.6.0, 5 field agents over the full repo) — a clean actuator interface with zero measurement epistemics, task identity foreclosed by mutation-based curricula, eval-on-the-training-motion; adopt the NaN guard (→ divergence oracle on our seam), the event lifecycle + declared-fields DR vocabulary, the eval-per-commit harness structure; reject adoption — interop via identified-actuator artifacts instead.
