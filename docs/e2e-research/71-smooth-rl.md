# SmoothRL, read against our stack: online RL inside the asynchronous loop

Research date: **2026-09-05**. Primary source: Astribot Team, *SmoothRL:
Online Reinforcement Learning During Asynchronous Execution*,
arXiv:2608.29768v1 (30 Aug 2026), read in full from the PDF, with the
four papers it is built on read at their arXiv abstracts the same day:
RL Token [9] (2604.23073), Real-Time Chunking [1] (2506.07339),
training-time RTC [2] (2512.05964) and concurrent control [35]
(2004.06089). Every number below is theirs unless marked *ours*.

*Labels used here: "campaign 3" and "campaign 4" are the third and
fourth teacher→student distillation runs on the microduck walk (240
episodes at 60k steps; the same recipe re-pressed), as recorded in
docs/68-findings.md; E0–E2 are this document's three experiments, not
the manuscript's E1–E7.*

## 0. What the paper is, in one paragraph

A chunking policy (ACT, diffusion, a VLA) answers one observation with
a chunk of future actions. On a real robot the next chunk is inferred
*while* the current one executes, so each chunk is used only in part:
the frames that elapsed during its own inference were already supplied
by the previous chunk, a middle window is what the robot actually
executes, and the tail is thrown away when the next chunk arrives.
SmoothRL is an online actor-critic fine-tuning method that takes that
partial execution as a known constraint: the value gradient reaches the
policy only through the window that was executed, the critic still
sees the whole executed span, and training rollouts run under the same
timed loop as deployment. On three real tasks the fine-tuned policy
goes from 39 → 94 %, 8 → 83 % and 30 → 90 % success after 250 online
episodes, with 52 % less acceleration and 47 % less jerk on one rollout.

## 1. The mechanism, with every symbol defined here

**The timed loop.** Control runs at a fixed rate (30 Hz on their robot).
Inference is requested a fixed budget of `n` control frames before its
chunk is due (their `n = 6`, 200 ms); if it finishes early the system
waits, so handover always happens at the same frame. A chunk of `H`
frames (their `H = 32`) is then partitioned by frame index alone:

| region | frames | meaning |
|---|---|---|
| committed | `[0, n)` | elapsed during this chunk's own inference; the previous chunk already supplied these actions |
| execution | `[n, 2n)` | the only frames of this chunk the robot executes |
| discarded | `[2n, H)` | superseded by the next chunk before they run |

With a variable latency those boundaries would drift independently per
chunk (their Figure 2a); the budget freezes them (2b). The slack is spent
waiting rather than on fresher observations.

**The objective.** Write `ã[0,n)` for the actions the robot actually
executed in the committed region (issued by the previous chunk, fixed),
and `π_θ(s)[n,2n)` for the policy's execution window. The actor
maximizes `Q(s, ã[0,n), π_θ(s)[n,2n))` subject to the whole `[0,2n)`
span being executable (`E`: bounded per-frame velocity, acceleration,
jerk, continuous across frame `n`). Two rules follow:

1. *Gradient truncation.* `∇_a Q` is taken only with respect to
   `a[n,2n)`: the segment that receives the gradient is both the policy's
   output and the executed action, so the gradient is well defined. The
   discarded region never enters the objective.
2. *The critic sees the whole executed span.* `Q` takes `[s, ã[0,n),
   a[n,2n)]`. The committed region is state augmentation for a
   concurrent decision process (Xiao et al. 2020, ref 35: when you act
   while the previous action still runs, the in-flight action and the
   time to its completion belong in the state); the budget fixes the
   time to `n`, and chunk-stitching consistency makes the in-flight
   action exactly the recorded committed region.

The Bellman backup is chunk-level ("chunk-skip"): the chunk is the
atomic action, the reward `r` is the return accumulated over the `2n`
executed frames, the bootstrap state is `s_{t+2n}` and the discount is
`γ^{2n}`. That backup is unbiased only because the critic is conditioned
on the action sequence that produced the interval's reward.

**Smoothness.** Pretraining inherits smoothness from demonstrations;
once `Q` is maximized over the execution window nothing keeps it. So
continuity is put back as a constraint: the executable set `E`, relaxed
to a penalty `w_smooth · Σ_k w_k ‖Δ^k a‖²` for `k = 1, 2, 3` (velocity,
acceleration, jerk of the chunk). The actor also conditions on the
committed region so its window continues the trajectory in flight.

**The instantiation (theirs; the framework does not depend on it).**
Frozen base policy π0.5 fine-tuned per task; the RL Token skeleton [9]:
a compact readout `z_t` of the base's internal representation, a
3-layer 512-unit MLP actor taking `[z_t, s_t, reference chunk]` and
emitting a *bounded residual correction* (±0.05) added to the reference
chunk; a REDQ-style ensemble of LayerNorm MLP critics with TD3 target
noise and a pessimistic minimum over a random subset; actor loss =
`−Q + w_bc ‖a − a_target‖² + smoothness`, the BC anchor being the base's
reference chunk (TD3+BC style, ref 59) or the human's chunk when
intervened. Replay is seeded with 50 rollouts of the base alone under
the timed loop. Update-to-data ratio `G = 5`, actor every `D = 5` critic
steps, batch 256. Reward is sparse and human-labeled: an operator ends
the episode and marks success. Two intervention modes in the raw action
space: absolute (VR teleop replaces the chunk) and residual (a hand
controller adds deltas to the policy's chunk); both become ordinary
transitions plus BC targets.

## 2. What they measured, and what they did not

Platform: Astribot S1, 25 DoF, three 224×224 cameras, 31-D action at
30 Hz; RL touches the 20 arm dimensions. Base policies pretrained on
1,500 / 1,500 / 500 teleoperated demonstrations.

| task | base | 150 ep | 200 ep | 250 ep | episodes per checkpoint |
|---|---|---|---|---|---|
| dynamic tossing (bin 7×12 cm) | 39 % | 72 % | 83 % | 94 % | 18 (3 objects × 6 bins) |
| pen capping (≈5 mm clearance) | 8 % | 67 % | 75 % | 83 % | 12 |
| box opening (2–3 mm seam) | 30 % | 20 % | 40 % | 90 % | 10 |

Smoothness: RMS acceleration −52 % and jerk −47 % of the right
end-effector, measured over each chunk of *one* autonomous tossing
rollout (their Figure 1). Base-policy failures are systematic offsets
(a fixed leftward blade bias, over/undershoot by bin distance); after RL
the misses are a small spread around the target ("the last millimeter").
A side finding: on far bins, residual teleop succeeds ≈80 % where VR
chunk teleop succeeds ≈30 %.

What the evidence does not contain, read carefully:

- **One episode per configuration, one RL run per task, no intervals.**
  A checkpoint's success is 10 to 18 binary trials; the 20 % dip at
  150 episodes on box opening and the 94 % on tossing are single draws.
  At n = 18, 94 % (17/18) has an exact 95 % interval of roughly
  [0.73, 1.00]; the paper reports the point.
- **The central mechanism is never ablated.** No run compares
  truncated gradients against full-chunk gradients, or the timed loop
  against an untimed one. The truncation argument is a derivation, not
  a measurement.
- **Success is operator-judged**, and the sparse reward is the same
  operator's call; the evaluation configurations are the training
  configurations (their §4.4).
- **Nothing is stamped.** No policy, dataset or robot identity; the
  robot is a proprietary platform.

## 3. What it means for us

**We already run the loop they optimize for, without the latency.** Our
certificate drives a chunk policy through `ActionScheduler`
(`trainnr/trainnr/evaluate/scheduler.py`): a 20-frame ACT chunk,
`executed_horizon` of 2, re-asked every 2 ticks through the bridge. In
the simulator the environment waits for the answer, so today the
committed region is empty and the execution window starts at frame 0.
A real robot cannot wait: the moment our student runs on hardware with
tens of milliseconds of inference, its executed window shifts to
`[n, 2n)` and the mismatch the paper describes becomes ours. We can do
what their real robot cannot: emulate the budget `n` exactly in
simulation and *certify under it*. That is a missing column of every
certificate we have issued.

**Their evaluation is our positioning.** Our walk rows are 40 matched
trials per policy with exact intervals, repeated on a second instrument
(2026-09-02 and 2026-09-04, docs/68-findings.md); the SmoothRL rows
are single draws of 10–18 episodes from one run. The comparison
writes itself.

**Their smoothness number is the shape ours should take.** RMS
acceleration and jerk per chunk is a fine metric; one rollout is not a
measurement. The certificate loop already issues every action it
executes; the same three derivatives per trial, aggregated with an
interval, is a small change.

**The residual-RL story lands on our teacher–student gap.** The
campaign 4 student scores 27/40 against the teacher's 33/40, and the
rows say the gap is tracking, not falling (near-threshold `err_ratio`
misses). That is precisely "the last millimeter": a bounded
residual on top of a frozen base, trained by a value gradient with the
certificate's own criterion as the sparse reward, is the paper's recipe
applied where we already have a measured, stamped base. Our DAgger
round 1 is the only online-improvement loop we have run (+4 trials,
p 0.49, unresolved, 90 minutes a round). What SmoothRL-in-sim would
have that neither they nor our DAgger have: a free and exact reward, 48
parallel worlds (250 episodes in minutes, not hours), no human in the
loop, and a certificate to judge the result on the same 40 seeds.

**Their limitation 2 is a warning for us.** A bounded residual cannot
fix a systematic bias of the base; on the walk the base's misses are
small and late, so the regime fits, but the bound is a knob to report.

## 4. The experiments, cheapest first

| # | what | cost | what it decides |
|---|---|---|---|
| E0 | smoothness as a certificate column: RMS velocity, acceleration and jerk of the executed targets per trial, with an interval; the teacher and the campaign 3–4 students | an hour on an RTX 3090 Ti | whether our students are rougher than the teacher, with a number instead of a video |
| E1 | the latency-budget certificate: emulate an inference budget `n ∈ {0, 1, 2, 4, 8}` ticks in the scheduler (the previous chunk keeps executing while the new one is "inferred"); certify a student at each `n` (campaign 3's was the one certified, §4b) | 2 hours, local or on a pod | how much of a student's certificate survives real inference latency — the paper's motivation, measured with intervals; a new protocol field, hashed with the trials |
| E2 | SmoothRL-lite in sim: frozen ACT student + residual TD3 actor/critic on `[state, reference chunk]`, gradient truncation to `[n, 2n)`, BC anchor to the reference, the smoothness penalty, sparse reward = the certificate's criterion, rollouts under the timed loop in 48 worlds over the bridge; judged on the same 40 seeds as DAgger round 1 | 2–3 days to build; a pod-hour per run | whether value-gradient residual RL closes the tracking gap where DAgger's +4 did not, and whether truncation matters (we can ablate it, which they did not) |

E2's design (status 2026-10-03: the objective is transcribed in
`trainnr/trainnr/rl/smooth_rl.py`, pure numpy; no E2 result is
recorded): two processes over the
existing bridge, as in their Algorithm 1 - the rollout side (mjlab venv)
runs the timed loop and appends transitions `(s_t, ã[0,n), a[n,2n),
a_target, r, s_{t+2n}, ã'[0,n))` to a replay directory; the learner side
(train venv) trains the residual head and publishes weights the rollout
side reloads between episodes. Reward: the certificate's `survived ∧
tracked` at episode end, plus optionally the per-interval tracking error
as a dense shaping term reported separately. The ablation that the paper
lacks costs one flag: truncate or not.

## 5. Results, 2026-09-05: E0 and E1 on the walk (RTX 3090 Ti)

Record `docs/findings/walk-latency-budget-2026-09-05.json`, figure
`docs/figures/walk-latency-budget-2026-09-05.svg`; 40 matched trials
on seed 1000 per row, the campaign 3 student (`student-last@c39a1b54c06b`)
and its teacher (`model_7999`), one control tick = 20 ms.

| policy | condition | success | survived | tracked | median fall tick | median RMS jerk |
|---|---|---|---|---|---|---|
| teacher | synchronous | 32/40 [0.644, 0.909] | 40 | 32 | 1000 | 272,658 |
| teacher | actions delayed 1 tick | 0/40 | 17 | 0 | 790 | 309,576 |
| teacher | delayed 2 ticks | 0/40 | 0 | 0 | 26 | 349,006 |
| teacher | delayed 4 ticks | 0/40 | 0 | 0 | | |
| student | latency 0 (rows [0,2) per chunk) | 25/40 [0.458, 0.773] | 28 | 33 | 1000 | 267,036 |
| student | latency 1 | 0/40 | 0 | 3 | 61 | 290,578 |
| student | latency 2 | 0/40 | 0 | 4 | 72 | 253,614 |
| student | latency 4 | 0/40 | 0 | 2 | | |
| student | latency 8 | 0/40 | 0 | 4 | | |
| student | horizon 3, latency 0 (a third row per chunk) | 13/40 [0.186, 0.491] | 22 | 22 | 1000 | 255,217 |
| student | horizon 1, latency 1 (one row, one tick stale) | 0/40 | 0 | 5 | | |

**E0.** The student is as rough as its teacher: median jerk 267k against
273k in the action's units per second cubed (both re-target every tick
at 50 Hz). Under a budget the jerk rises 9 % as the duck goes down.
Smoothness is now a column on every certificate row, three engines.

*The teacher's 32/40 here against 33/40 in the walk-verdict record
(2026-09-04, commit e62b706), and the student's 25/40 against 24/40
there: same policies, same seed, two harnesses one commit apart (the
latency certificate, commit 6825ab1, drives the policy through the
scheduler's timed loop at budget 0; the walk verdict steps it
directly); one trial in forty moved, inside both intervals.*

**E1.** One tick of inference budget takes the student from 25/40 to
0/40 (every duck down, median fall at tick 61; effect −0.625, p 2e-10),
and so does every larger budget. The two controls separate the causes:
executing a third row per chunk with no staleness costs half the
successes (13/40), while one tick of staleness with a single row per
chunk costs all of them - staleness dominates. And the decisive
control is the teacher's: delayed one tick it tracks 0/40 (17 stand),
delayed two it falls 0/40. **Neither policy has a delay margin of even
20 ms; the student inherited its collapse from a teacher trained with
zero latency.** This is the mismatch the paper fine-tunes against,
measured with intervals on both sides of a distillation - and it says
the fix belongs on the teacher's training side first: a teacher
trained under a latency budget (the paper's "run the timed loop during
training", applied to the RL teacher) is the prerequisite for any
student that will run on hardware. Every walk certificate issued so
far was a synchronous one; a certificate should also be issued at the
deployment latency.

## 6. Building E2 without their code: the transcription protocol

SmoothRL ships no code (checked 2026-09-05: no repository linked from
the arXiv page or the project page; RL Token's page links none either).
"Exact and correct" therefore means faithful to the paper's own
specification, tested against the paper's own claims - the way the
actuator law was transcribed from BAM and pinned at both ends.

1. **Algorithm 1 is the reference.** It is complete pseudocode for the
   two processes (asynchronous rollout; off-policy updates over a shared
   replay). It gets transcribed line for line into one module, each
   line's number kept in a comment, with the state
   `(s_t, z_t, ã[0,n), a[n,2n), a_target, r, s', z', ā', ã'[0,n))` as
   one typed record.
2. **Every stated number is pinned to its section**, as a named field
   whose docstring cites the page: MLP 3 × 512 with LayerNorm (§3.4,
   §4.3), correction bound 0.05, update-to-data `G = 5`, actor delay
   `D = 5`, batch 256, the smoothness penalty over `Δ¹..Δ³` (Eq. 5),
   TD3 target noise and a pessimistic minimum over a random critic
   subset (REDQ), the BC anchor to the reference chunk, `γ^{2n}` in the
   backup, replay seeded by 50 base-only rollouts, `n = 6` frames at
   30 Hz, `H = 32`.
3. **Every unstated number is marked ours**, in one table in the module
   and here: `γ`, `τ`, the learning rate, the ensemble size `N` and
   subset `M`, the target-noise scale and clip, `w_bc`, `w_smooth`,
   `w_k`, the actor's activation, the "hold action" before the first
   chunk. They take standard TD3/REDQ values and are exposed as fields,
   never buried.
4. **The mechanism is tested, not trusted.** Three tests pin what the
   paper claims about its own objective: (a) the partition - for a
   budget `n` and horizon `H`, rows `[0,n)`, `[n,2n)`, `[2n,H)` land in
   the committed, execution and discarded regions, and the scheduler
   test already proves the executed rows are exactly `[n, ...)`; (b) the
   truncation - a finite-difference check that perturbing committed or
   discarded rows of the actor's chunk leaves `∂L_actor/∂θ` unchanged
   while perturbing execution rows changes it; (c) the backup - the
   reward summed over exactly `2n` frames, the bootstrap at `s_{t+2n}`,
   the discount `γ^{2n}`, and the critic's input carrying the executed
   committed region, checked on a hand-built two-step episode.
5. **The ablation the paper lacks costs one flag** (`truncate=False`
   sends the gradient through the whole `[0,2n)` span), so the central
   claim becomes a measured difference on our certificate, with an
   interval - which their single run per task cannot give.
6. **What we do not transcribe.** The RL token `z_t` (a readout of a
   VLA's internals) has no counterpart on an ACT student; the residual
   head conditions on the state and the reference chunk, and that
   departure is recorded as such. Human intervention has no role in
   simulation; the sparse reward is the certificate's criterion, exact
   and free, instead of an operator's call.

## 7. Sources

- Astribot Team, SmoothRL, arXiv:2608.29768v1, 2026-08-30 - §§1–5, Figures
  1–8, Table 1, Algorithm 1 (read in full).
- Xu et al., RL Token, arXiv:2604.23073 (the skeleton they adopt).
- Black et al., Real-Time Chunking, arXiv:2506.07339; Training-Time Action
  Conditioning for RTC, arXiv:2512.05964 (the chunk-stitching line the
  timed loop relies on).
- Xiao et al., Thinking While Moving, arXiv:2004.06089 (concurrent
  control: the in-flight action and the time to its completion belong in
  the state).
- Fujimoto & Gu, TD3+BC, arXiv:2106.06860 (the BC anchor).
