# The flagship (C2): microduck's capability, rebuilt so its flaws are impossible

*2026-09-01. The scoping doc for the second campaign, C2 (the
microduck walk; C1 was the paired lift study, [62](62-paired-study.md);
C3 and C4 below are the planned outreach and deployment-manifest
campaigns), redirected 2026-08-31: not a replica of microduck, but the cross-framework setup
plus everything microduck taught, built into an improved version that
serves the any-robot claim. Everything cited here is measured in
[57](57-bam-source-and-microduck.md) (their code, read line by line)
or built and tested in this repo. Status: gates G1–G4 (§3) all
CLOSED by 2026-09-02; the run and its certificates are under
`docs/artifacts/walk-verdicts/`.*

## 0. The one-sentence goal

Take the best real deployment in the surveyed ecosystem — Pollen's
microduck, walking in production on a BAM-identified actuator — and
express its whole training recipe through our stack such that every
flaw we measured in their repo is not fixed but IMPOSSIBLE, then show
the two side by side. The demo of the any-robot claim is someone
else's robot, from someone else's ecosystem, with someone else's fits,
coming out the other end measured, stamped, and rerunnable by a
stranger.

## 1. The mapping: their piece → ours → what becomes impossible

| Theirs (measured, 57 §5) | Ours (exists, tested) | What can no longer happen |
|---|---|---|
| Fitted params resolve via a private API in one script and `~/Rhoban/bam/params/xl330/m6_new.json` on a laptop in another; no hash, no date; four rival actuator classes named `old`/`new`/`antoine`/`marc` | The certified bundle store: their m6 fit wrapped verbatim with provenance + content-hash stamp (`robots/actuator-bundles/xl330.m6.bundle.json`), `verified_bundle` refusing tampering, the stamp carried on every cfg and record | "Which fit trained this policy" has one answer, checkable by anyone |
| Five silent no-op DR knobs (frictionloss + damping randomizers the actuator overwrites, an IMU field nothing reads, a mass no-op under 1.3, post-init config writes); a decorator-carrier event every env must remember | `trainnr_mjlab.linter` — DR terms writing overwritten fields REFUSE at construction; the expansion event is auto-registered and its absence raises | Turning a dial that does nothing; forgetting the event |
| Fitted-treated-as-exact beside hand-guessed ranges (±10 % friction scalar typed by hand; sag and delay ranges undocumented guesses their own testbench contradicts) | `dr_from_bundle` / `declared_ranges`: sampling regions come from the bundle's identified intervals or a caller-DECLARED span whose basis string says so; a point estimate with no span refuses | Randomizing from folklore while calling it measured; un-attributed ranges |
| Training telemetry: none (mjlab's recorder API is empty; their monitoring is the viewer's reward strip) | `RerunRecorder` → the Studio; the cloud feed tool (`tools/studio-cloud-feed.py`) for rented cards; the status card | Training you cannot watch or archive |
| Verification: "train → deploy → watch the video"; one-servo bench with thresholds chosen by feel; a stale hardcoded obs layout | The harness: seeded paired episodes on BOTH instruments (CPU MuJoCo + MJX-Warp, stamps on every record), exact intervals, declared thresholds, funnels; C1's protocol as the template | A verdict nobody can recompute; "rolls but face-plants 1 in 3" as a reporting standard |
| Deployment contract as folklore (`--action-scale 0.8` vs trained 1.0; a kp-ratio default that silently detunes 200→120; French plan markdown) | The deployment manifest + ONNX-certify adapter (C4, folds in here): the trained contract as a checked artifact the runtime refuses to violate | Shipping a policy under different physics than it trained with |

Respect what they got right and keep it: the obs normalizer baked into
the ONNX graph, a stable obs contract shared across tasks, NaN
terminations, the CoM-audit habit. The flagship adopts those, cited.

## 2. The slice

Their robot, their task family, our machinery, sim only (hardware
waits):

1. **The robot**: microduck's MJCF wrapped as a robot bundle
   (`robots/microduck/`, hash-stamped; acquisition = their public repo,
   fetched when the network allows or from the repomix packs). The XL330
   m6 fit is ALREADY in our store as a certified bundle.
2. **The env**: their velocity-tracking walk expressed as an mjlab task
   consumed through `trainnr_mjlab` — `BamActuatorCfg.from_bundle` (the
   identified law, per-world DR draws from `declared_ranges`), the
   linter green by construction, the RerunRecorder attached.
3. **Training**: rsl-rl through mjlab, a short local smoke first,
   real scale on a rented card with the Studio watching.
4. **The verdict**: a locomotion certificate in C1's shape — seeded
   paired episodes, tracking error + fall counts with exact intervals,
   both instruments, the bundle stamp and engine stamp on every row.
5. **The diff**: one table, their repo vs this run, every row of §1
   with its evidence — the artifact the planned upstream
   conversations (C3) lead with.

## 2.1 The transcription source, pinned (2026-09-01, G2 recon)

- Their walk recipe lives in
  `microduck_rl:src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`
  (949 lines at `d424a0c`) — a delta-stack over **mjlab 1.3.0's**
  velocity base (`pyproject` pin confirmed; three breaking minors
  behind our 1.6, so their package cannot import in our venv — the
  transcription route the scope assumed is the only route).
- Key numbers already extracted: commands `lin_vel_x (-0.4, 0.4)`,
  `lin_vel_y (-0.3, 0.3)`, `ang_vel_z (-1.0, 1.0)`; solver raised to
  `nconmax 200, iterations 30, ls_iterations 50` for the duck; their
  DR is the catalogued family (`randomize_delayed_actuator_gains`,
  `randomize_bam_friction`, `randomize_dof_field_scaled`,
  `randomize_base_orientation`, `push_by_setting_velocity`) plus the
  curricula (`standing_envs`, `pose_command_range`, `com_range`,
  ramped reward weights).
- The G2 build therefore reads THREE sources side by side: mjlab
  1.3's velocity base (their baseline), their delta file, and mjlab
  1.6's velocity base (our target) — and re-expresses the recipe on
  1.6 with our entity, the certified actuator, `dr_from_bundle`
  ranges, and the linter refusing what their five no-ops did
  silently. G1 and the entity brick are done (the tests beside
  `trainnr-mjlab/tests/test_microduck_entity.py`); the cfg is the next
  sitting's work.

## 2.2 The deltas, read (2026-09-01, both sources local)

mjlab 1.3.0's velocity base (the wheel's *velocity_env_cfg.py*, extracted) beside
their 949-line delta file. What the walk recipe actually is:

**Timing/actions** (base values kept): `decimation 4`,
`episode_length_s 20.0`; action = `JointPositionActionCfg` with
`scale = 1.0` set EXPLICITLY (the value their deploy flag
`--action-scale 0.8` contradicts — 57 §5's finding, confirmed at the
source).

**Rewards** (base term → their override; base weights in 1.3 noted
where they differ):
- `track_linear_velocity` w 2.0, std √0.1; `track_angular_velocity`
  w 2.0, std √0.5 (base stds differ).
- `upright` w 2.0 (base 1.0), body `trunk_base`, std √0.05.
- `pose` w 1.0 with their split stds (`std_standing` tight /
  `std_walking`), `walking_threshold 0.01`.
- `air_time` w 3.0 (base 0.0 "override per-robot"), thresholds
  0.125–0.300 s, command threshold 0.01.
- `foot_slip` w −0.1; `foot_clearance` target 0.02 m (raised from
  0.01, their comment: penalize dragging); `foot_swing_height`
  target 0.02 m; `action_rate_l2` w −0.1; `body_ang_vel` w −0.05;
  `angular_momentum` w −0.02; `dof_pos_limits` w −1.0 (base).
- ADDED: `self_collisions` w −1.0 (`mdp.self_collision_cost`).

**Commands**: `lin_vel_x (−0.4, 0.4)`, `lin_vel_y (−0.3, 0.3)`,
`ang_vel_z (−1.0, 1.0)` (from §2.1).

**Observations**: `base_lin_vel` DELETED from the actor (critic keeps
it — privileged), `height_scan` deleted from both (flat terrain).

**Terminations**: base plus `nan_state` (their NaN guard — keep,
cited as theirs).

**Events/DR** (the honesty frontier): `expand_bam_friction_fields`
(their decorator-carrier no-op → OUR auto-registered
`bam_expansion_event`), `reset_action_history`, `foot_friction
(0.7, 1.3)`, `reset_base` z (0.12, 0.13), `push_robot`,
`randomize_com` / `randomize_head_com` (audited ranges),
`randomize_motor_gains` (delayed kp/kd), `randomize_mass_inertia`,
`randomize_joint_friction` and `randomize_joint_damping` (TWO of the
five silent no-ops — fields the BAM actuator overwrites every step;
our linter REFUSES these by construction), `randomize_armature`,
`randomize_base_orientation`. The G2 build re-expresses each: passives
through mjlab's native events, law params through
`bam_param_dr_event` with a declared basis, and the two no-ops become
the linter's showcase refusal.

**Curricula** (from §2.1): `standing_envs`, `pose_command_range`,
`com_range`, ramped reward weights — 1.6 re-expression decided at
build time (mjlab 1.6 curricula mutate cfg in place; ours must stay
hashable — the one design tension the transcription must resolve;
  that mjlab cannot hash a task cuts both ways).

## 3. Gates, in order

- G1: microduck MJCF in a stamped bundle; scene compiles on both
  engines; census tests.
- G2: the env builds through trainnr_mjlab with zero linter findings and
  the m6 bundle's stamp on the cfg; a 2-minute smoke train on the WSL machine
  moves in the Studio.
- G3: the real training run (on rented compute) reaches a walking
  gait; recorder + feed archives the run. **CLOSED 2026-09-01**: 8,000
  iterations on a rented RTX PRO 6000 (1 h 30 m ≈ $3.15, 4,096 envs,
  0.70 s/iter, ~140k steps/s), mean reward 108→117.9, fell_over→0;
  live in the Studio through the feed's rsl-rl leg; run archived at
  `runs/microduck-walk/20260901-163412` (identity.json + model_7999.pt
  + tfevents + log kept locally; full checkpoints on the pod
  volume; the certificates are in `docs/artifacts/walk-verdicts/`). Judged on screen via `walk_play` (native viewer + recorder,
  identity-gated): ducks locomote under command and survive pushes —
  the gait is HOPPING-flavoured, consistent with the recipe (air_time
  weight 3.0 is the largest positive term, nothing rewards left-right
  alternation, ENABLE_SYMMETRY ships False with them, and 8k of their
  50k iterations); the policy also carries the head thrown fully back
  (nothing in the reward watches head posture; likely ballast). Both
  are §4 gait aesthetics — recorded, not retuned.
- G4: the certificate, on both instruments, committed with the run's
  stamps; the §1 diff table recorded with dates. **CLOSED
  2026-09-02**: `trainnr_mjlab.walk_verdict` — one seeded episode per
  trial under FULL DR and pushes, two declared milestones (survived;
  tracked = episode-mean planar velocity error closes at least half
  the standing-still gap, `err_ratio < 0.5` with a 0.1 m/s floored
  denominator), every row an `EpisodeRecord` with
  `microduck-walk@824eba27c48c` as source and the full
  mjlab+mujoco+warp+device instrument. The rows:

  | instrument | survived | tracked | success | CP95 | median err_ratio |
  |---|---|---|---|---|---|
  | `mjlab-1.6.0+mujoco-3.11.0+warp-1.17.0+cuda` | 40/40 | 33/40 | **33/40** | [0.672, 0.927] | 0.389 |
  | `mjlab-1.6.0+mujoco-3.11.0+warp-1.17.0+cpu` | 40/40 | 37/40 | **37/40** | [0.796, 0.984] | 0.406 |

  The instruments agree within their intervals (same seed; the draw
  streams differ per device, so trials are not paired across rows —
  each row is its own 40). Every miss on both instruments is the same
  shape: a gentle command (0.08–0.21 m/s) where the hop's own 6–11
  cm/s planar wobble does not shrink — the certificate NAMES the
  gait's weakness (station-keeping) instead of a demo video hiding
  it. Zero falls in 80 episodes under DR and pushes. Artifacts:
  `docs/artifacts/walk-verdicts/` (records-{cuda,cpu}.jsonl +
  walk-verdict-{cuda,cpu}.json, copied from the run's `verdict/`);
  criterion pinned by `trainnr-mjlab/tests/test_walk_verdict.py`. The plain-CPU-MuJoCo third instrument
  belongs to C4's deployment manifest, where the policy leaves torch.
- Each gate is a session-scale unit; G3 is the only one that costs
  money.

## 4. Out of scope, said now

Their exact reward weights and gait aesthetics (we compare processes,
not policies); hardware deployment; ROS/real-time
runtime work; beating their walk speed. The flagship's claim is the
PIPELINE — measured, stamped, rerunnable — not a better duck.

## 5. Open questions, as parked on 2026-09-01 (historical)

*All three were settled the same week: the MJCF is vendored at
`robots/microduck/` from their public tree with its licence; G3 ran
straight on a rented card after a local smoke; the deployment
manifest (C4) shipped separately.*

- Which repo the microduck MJCF is vendored from (their public tree
  vs the repomix pack), and its license ride-along.
- Whether G3 runs locally at reduced scale first (the mjlab smoke on
  an RTX 3090 Ti, B4 in the earlier build log) or goes straight to a
  rented card.
- Whether the C4 manifest work lands inside C2 (as §1's last row
  assumes) or ships separately first.
