# The flagship (C2): microduck's capability, rebuilt so its flaws are impossible

*2026-09-01. The scoping doc for docs/00 Phase C2, redirected by the
operator 2026-08-31: "I just dont want to replicate microduck, we have
to use our cross framework setup and use all the learning from
microduck to create an improved version that solves our goal and
mission to the any robot claim." Everything cited here is measured in
[57](57-bam-source-and-microduck.md) (their code, read line by line)
or built and tested in this repo. Status: SCOPE — no code yet.*

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
| Fitted params resolve via a private API in one script and `~/Rhoban/bam/params/xl330/m6_new.json` on a laptop in another; no hash, no date; four rival actuator classes named `old`/`new`/`antoine`/`marc` | The certified bundle store: their m6 fit wrapped verbatim with provenance + content-hash stamp (`robots/actuator-bundles/xl330.m6`), `verified_bundle` refusing tampering, the stamp carried on every cfg and record | "Which fit trained this policy" has one answer, checkable by anyone |
| Five silent no-op DR knobs (frictionloss + damping randomizers the actuator overwrites, an IMU field nothing reads, a mass no-op under 1.3, post-init config writes); a decorator-carrier event every env must remember | `rq_mjlab.linter` — DR terms writing overwritten fields REFUSE at construction; the expansion event is auto-registered and its absence raises | Turning a dial that does nothing; forgetting the event |
| Fitted-treated-as-exact beside hand-guessed ranges (±10 % friction scalar typed by hand; sag and delay ranges undocumented guesses their own testbench contradicts) | `dr_from_bundle` / `declared_ranges`: sampling regions come from the bundle's identified intervals or a caller-DECLARED span whose basis string says so; a point estimate with no span refuses | Randomizing from folklore while calling it measured; un-attributed ranges |
| Training telemetry: none (mjlab's recorder API is empty; their monitoring is the viewer's reward strip) | `RerunRecorder` → the Studio; `studio-cloud-feed` for rented cards; the status card | Training you cannot watch or archive |
| Verification: "train → deploy → watch the video"; one-servo bench with thresholds chosen by feel; a stale hardcoded obs layout | The harness: seeded paired episodes on BOTH instruments (CPU MuJoCo + MJX-Warp, stamps on every record), exact intervals, declared thresholds, funnels; C1's protocol as the template | A verdict nobody can recompute; "rolls but face-plants 1 in 3" as a reporting standard |
| Deployment contract as folklore (`--action-scale 0.8` vs trained 1.0; a kp-ratio default that silently detunes 200→120; French plan markdown) | The deployment manifest + ONNX-certify adapter (C4, folds in here): the trained contract as a checked artifact the runtime refuses to violate | Shipping a policy under different physics than it trained with |

Respect what they got right and keep it: the obs normalizer baked into
the ONNX graph, a stable obs contract shared across tasks, NaN
terminations, the CoM-audit habit. The flagship adopts those, cited.

## 2. The slice

Their robot, their task family, our machinery, sim only (hardware
waits on docs/38):

1. **The robot**: microduck's MJCF wrapped as a robot bundle
   (`robots/microduck/`, hash-stamped; acquisition = their public repo,
   fetched when the network allows or from the Mac's packs). The XL330
   m6 fit is ALREADY in our store as a certified bundle.
2. **The env**: their velocity-tracking walk expressed as an mjlab task
   consumed through `rq_mjlab` — `BamActuatorCfg.from_bundle` (the
   identified law, per-world DR draws from `declared_ranges`), the
   linter green by construction, the RerunRecorder attached.
3. **Training**: rsl-rl through mjlab, smoke on the box (short-local-
   runs rule), real scale on a rented card with the Studio watching.
4. **The verdict**: a locomotion certificate in C1's shape — seeded
   paired episodes, tracking error + fall counts with exact intervals,
   both instruments, the bundle stamp and engine stamp on every row.
5. **The diff**: one table, their repo vs this run, every row of §1
   with its evidence — the outreach artifact C3's mjlab/Rhoban
   conversations lead with.

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
  `rq_mjlab/tests/test_microduck_entity.py`); the cfg is the next
  sitting's work.

## 2.2 The deltas, read (2026-09-01, both sources local)

mjlab 1.3.0's `velocity_env_cfg.py` (the wheel, extracted) beside
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
hashable — the one design tension the transcription must resolve,
docs/33's "mjlab cannot hash a task" row cuts both ways).

## 3. Gates, in order

- G1: microduck MJCF in a stamped bundle; scene compiles on both
  engines; census tests.
- G2: the env builds through rq_mjlab with zero linter findings and
  the m6 bundle's stamp on the cfg; a 2-minute smoke train on the box
  moves in the Studio.
- G3: the real training run (paid, operator's go) reaches a walking
  gait; recorder + feed archives the run.
- G4: the certificate, on both instruments, committed with the run's
  stamps; the §1 diff table lands in docs/33 with dates.
- Each gate is a session-scale unit; G3 is the only one that costs
  money.

## 4. Out of scope, said now

Their exact reward weights and gait aesthetics (we compare processes,
not policies); hardware deployment (docs/38's arc); ROS/real-time
runtime work; beating their walk speed. The flagship's claim is the
PIPELINE — measured, stamped, rerunnable — not a better duck.

## 5. Open questions parked with the operator

- Which repo the microduck MJCF is vendored from (their public tree
  vs the Mac's repomix pack), and its license ride-along.
- Whether G3 runs on this box at reduced scale first (mjlab at 3090 Ti
  smoke proved in B4) or goes straight to a rented card.
- Whether the C4 manifest work lands inside C2 (as §1's last row
  assumes) or ships separately first.
