# SmoothRL, read against our stack: online RL inside the asynchronous loop

Research date: **2026-09-05**. Primary source: Astribot Team, *SmoothRL:
Online Reinforcement Learning During Asynchronous Execution*,
arXiv:2608.29768v1 (30 Aug 2026), read in full from the PDF, with the
four papers it is built on read at their arXiv abstracts the same day:
RL Token [9] (2604.23073), Real-Time Chunking [1] (2506.07339),
training-time RTC [2] (2512.05964) and concurrent control [35]
(2004.06089). Every number below is theirs unless marked *ours*.
Written on branch `smooth-rl-2026-09-05`.

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
(`pipeline/rq_pipeline/evaluate/scheduler.py`): a 20-frame ACT chunk,
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
(docs/07 2026-09-02, 2026-09-04); the SmoothRL rows are single draws of
10–18 episodes from one run. The comparison writes itself (docs/33).

**Their smoothness number is the shape ours should take.** RMS
acceleration and jerk per chunk is a fine metric; one rollout is not a
measurement. The certificate loop already issues every action it
executes; the same three derivatives per trial, aggregated with an
interval, is a small change.

**The residual-RL story lands on our teacher–student gap.** The
campaign 4 student scores 27/40 against the teacher's 33/40, and the
rows say the gap is tracking, not falling (near-threshold `err_ratio`
misses, docs/69 §2). That is precisely "the last millimeter": a bounded
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
| E0 | smoothness as a certificate column: RMS velocity, acceleration and jerk of the executed targets per trial, with an interval; the teacher and the campaign 3–4 students | an hour, this box | whether our students are rougher than the teacher, with a number instead of a video |
| E1 | the latency-budget certificate: emulate an inference budget `n ∈ {0, 1, 2, 4, 8}` ticks in the scheduler (the previous chunk keeps executing while the new one is "inferred"); certify the campaign 4 student at each `n` | 2 hours, this box or the pod | how much of a student's certificate survives real inference latency — the paper's motivation, measured with intervals; a new protocol field, hashed with the trials |
| E2 | SmoothRL-lite in sim: frozen ACT student + residual TD3 actor/critic on `[state, reference chunk]`, gradient truncation to `[n, 2n)`, BC anchor to the reference, the smoothness penalty, sparse reward = the certificate's criterion, rollouts under the timed loop in 48 worlds over the bridge; judged on the same 40 seeds as DAgger round 1 | 2–3 days to build; a pod-hour per run | whether value-gradient residual RL closes the tracking gap where DAgger's +4 did not, and whether truncation matters (we can ablate it, which they did not) |

E2's design, so the next session can start it: two processes over the
existing bridge, as in their Algorithm 1 - the rollout side (mjlab venv)
runs the timed loop and appends transitions `(s_t, ã[0,n), a[n,2n),
a_target, r, s_{t+2n}, ã'[0,n))` to a replay directory; the learner side
(train venv) trains the residual head and publishes weights the rollout
side reloads between episodes. Reward: the certificate's `survived ∧
tracked` at episode end, plus optionally the per-interval tracking error
as a dense shaping term reported separately. The ablation that the paper
lacks costs one flag: truncate or not.

## 5. Sources

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
