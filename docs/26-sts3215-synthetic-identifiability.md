# The synthetic STS3215 identifiability study

*Run 2026-08-25, hardware-free, per the strategy review's software-first
constraint. Machinery: `trainnr/trainnr/robot/sts_synth.py` riding
the same `identify()` wedge Paper 0 rehearsed. Full 32-cell matrix in
`data/sts3215-synthetic-identifiability.json`; two load-bearing cells
pinned by `trainnr/tests/test_sts_synth.py`.*

## The question

docs/e2e-research/27 §5 — which parameters can `mujoco.sysid` recover
through an STS3215's position loop, firmware dead zone (10 counts ≈
0.88°), 12-bit encoder quantization, and the registers a TTL bus can
actually deliver? Answered the way Paper 0 was rehearsed: a true model
generates data, the data is corrupted the way the servo corrupts it
(numbers from the third-party video bench test — vendor/reported-grade
evidence, fine for a model, never citable as our measurement), and the
fit must recover what it can.

One design axis is a protocol insight in itself: the link's mass is
FIXED in the fit — **a weighed link is a free torque anchor**. A
kitchen scale turns gravity into the scale reference the drivetrain
never had at 50 Hz.

## Findings (each one changes the bench protocol)

**1. Position + load is the protocol, and rate barely matters.**
With present-position and present-load fitted (velocity NOT fitted),
all four parameters — servo kp, damping, frictionloss, armature —
recover within ~6% at every rate tried, **including 25 Hz**:

| rate | kp err | damping err | frictionloss err | armature err |
|---|---|---|---|---|
| 200 Hz | +0.2% | −0.0% | +3.7% | −0.9% |
| 100 Hz | −0.1% | −1.2% | +5.8% | +0.4% |
| 50 Hz | +0.1% | +0.1% | +3.7% | +0.3% |
| 25 Hz | −0.6% | −1.3% | +4.9% | +0.2% |

The slowest plausible bus polling beats a fast bus with the wrong
registers by an order of magnitude. The TTL bus's rate ceiling is a
non-problem; its *register selection* is everything.

**2. The derived-velocity register is poison — never fit against it.**
present-speed is firmware-derived from the dead-zoned, quantized
position; fitting against it injects structured error the optimizer
absorbs into every parameter. pos+vel cells are the worst in every
rate class — at 50/25 Hz catastrophically (−53% to −88% errors), and
**all confidently reported "pinned"**. Adding a bad channel is worse
than having fewer channels.

**3. The dead zone masquerades as friction.** In position-only fits,
frictionloss comes back +17% to +130% — physically sensible, both
mechanisms eat small motions. The load channel mostly rescues it
(≤6%). Real-bench consequence: without a load register, frictionloss
estimates need an explicit dead-zone caveat or an explicit dead-zone
model.

**4. "Pinned" is not "true" under structured corruption.** Many biased
cells report pinned: the confidence intervals assume white residuals
and the servo's corruptions are structured. The honesty machinery
needs one more organ before the real bench: a residual-whiteness
diagnostic in `identify()` that demotes intervals to DISTRUSTED when
the residual is visibly structured. Queued.

**5. The clean cells recover truth exactly — after three convention
bugs.** All 12 corruption-free cells recover all parameters to machine
precision, which is the guarantee that findings 1–4 measure the servo
and not the harness. Getting there required killing three successive
sampling-convention bugs (end-of-hold rows, hand-rolled ZOH vs the
optimizer's interpolation, a one-sample control-grid shift) — **each
of which produced a biased fit with tight intervals around wrong
values**. That is R13's failure class, met three more times in one
module, each caught by the truth-recovery check. The real bench
inherits the fence: truth-recovery on synthetic data is now a
prerequisite gate for any new excitation harness.

## What this changes

- **Paper 1's bench protocol is now specced before any purchase**: poll
  present-position + present-load; log present-speed if convenient but
  never fit against it; weigh the link (and any payload) on a kitchen
  scale; 25 Hz suffices; excitation dwarfing the ~20-count blind band
  and inside the governor envelope (protocol rules already in
  docs/23-research-agenda.md).
- **The downmarket-boundary worry shrinks**: through every corruption
  the video test documented, parametric identification converges given
  the right two registers. The open risk moves to what the study could
  not model — backlash as a *mechanical* element (v1 folds it into the
  measurement path), the governor's regime, and real sensor noise
  statistics.
- The residual-whiteness diagnostic is the next `identify()` feature.
