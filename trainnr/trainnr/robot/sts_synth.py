"""The synthetic STS3215 identifiability study — Paper 1's rehearsal.

No arms are on the bench (deliberate, 2026-08-24: software-first), so
the question docs/e2e-research/27 §5 poses — *which parameters can
`mujoco.sysid` pin through a position-servo's firmware* — is answered
here the way Paper 0 was rehearsed: a TRUE model rolls out synthetic
measurements, the measurements are corrupted the way the real servo
corrupts them, and the fit must recover what it can and say NOT PINNED
to the rest.

The corruption numbers are the third-party video bench test of one
STS3215-12V (operator-supplied summary, banked with provenance in
docs/e2e-research/27 §5 — vendor/reported-grade evidence, fine for a
model, never citable as our measurement):

- encoder 4,096 counts/rev → 0.088°/count quantization
- firmware DEAD ZONE 10 counts (~0.88°): the *reported* position
  freezes until true motion exceeds it — invisible by firmware choice
- present-speed derived from reported position (the register is not an
  independent sensor — corrupting it independently would flatter it)
- present-load noisy and coarse (sigma + quantization guesses, labelled)

What the study varies (the purchase-independent design axes):
- telemetry rate: what the TTL bus can actually deliver per servo
- sensor set: position-only vs +velocity vs +load — Paper 0's lesson
  says the torque-side signal is where scale observability lives
- corruption: clean (upper bound) vs servo-real

One protocol insight is built into the truth model deliberately: the
link's mass is FIXED in the fit (a weighed link is a free torque
anchor — a kitchen scale turns gravity into the scale reference the
drivetrain never had).
"""

from __future__ import annotations

from dataclasses import dataclass

from trainnr.robot.identify import (
    ExcitationData,
    IdentificationResult,
    ParameterSpec,
    identify,
    staged_excitation,
)

# Truth values for an STS3215-class joint driving a weighed arm link.
# Plausible hobby-servo magnitudes; the study's claim is about
# OBSERVABILITY (which of these pin under which telemetry), never about
# these numbers themselves.
TRUE_KP = 6.0
TRUE_DAMPING = 0.08
TRUE_FRICTIONLOSS = 0.03
TRUE_ARMATURE = 0.002

# The video-sourced corruption constants (27 §5).
COUNTS_PER_REV = 4096
QUANTUM_RAD = 2.0 * 3.141592653589793 / COUNTS_PER_REV
DEAD_ZONE_RAD = 10 * QUANTUM_RAD
# Labelled guesses (no video number exists for these two):
LOAD_NOISE_NM = 0.02
LOAD_QUANTUM_NM = 0.01

PHYSICS_DT = 0.001
SENSOR_BLOCKS = {
    "pos": '<jointpos name="q" joint="hinge"/>',
    "vel": '<jointvel name="qd" joint="hinge"/>',
    "load": '<actuatorfrc name="tau" actuator="servo"/>',
}


def model_xml(sensors: tuple[str, ...]) -> str:
    """The joint under study; `sensors` selects the telemetry registers."""
    blocks = "\n    ".join(SENSOR_BLOCKS[name] for name in sensors)
    return f"""
<mujoco>
  <option timestep="{PHYSICS_DT}" gravity="0 0 -9.81"/>
  <worldbody>
    <body>
      <joint name="hinge" type="hinge" axis="0 1 0"
             damping="{TRUE_DAMPING}" frictionloss="{TRUE_FRICTIONLOSS}"
             armature="{TRUE_ARMATURE}"/>
      <!-- The weighed link: mass fixed in the fit = the torque anchor. -->
      <geom name="link" type="capsule" fromto="0 0 0 0 0 -0.10"
            size="0.012" mass="0.12"/>
    </body>
  </worldbody>
  <actuator>
    <position name="servo" joint="hinge" kp="{TRUE_KP}" ctrlrange="-1.5 1.5"/>
  </actuator>
  <sensor>
    {blocks}
  </sensor>
</mujoco>
"""


def _set_kp(spec, parameter) -> None:
    servo = spec.actuator("servo")
    servo.gainprm[0] = parameter.value[0]
    servo.biasprm[1] = -parameter.value[0]


def _set_damping(spec, parameter) -> None:
    spec.joint("hinge").damping[0] = parameter.value[0]


def _set_frictionloss(spec, parameter) -> None:
    # frictionloss and armature are scalars on MjsJoint, unlike damping
    spec.joint("hinge").frictionloss = parameter.value[0]


def _set_armature(spec, parameter) -> None:
    spec.joint("hinge").armature = parameter.value[0]


PARAMETERS = (
    ParameterSpec("servo_kp", TRUE_KP, 1.0, 20.0, _set_kp, initial_guess=3.0),
    ParameterSpec("damping", TRUE_DAMPING, 0.005, 0.5, _set_damping, initial_guess=0.2),
    ParameterSpec(
        "frictionloss",
        TRUE_FRICTIONLOSS,
        0.001,
        0.2,
        _set_frictionloss,
        initial_guess=0.05,
    ),
    ParameterSpec(
        "armature", TRUE_ARMATURE, 1e-4, 0.02, _set_armature, initial_guess=0.005
    ),
)


@dataclass(frozen=True)
class Condition:
    """One cell of the study matrix."""

    telemetry_hz: int
    sensors: tuple[str, ...]
    servo_real: bool  # apply quantization + dead zone + derived velocity

    @property
    def label(self) -> str:
        corruption = "servo-real" if self.servo_real else "clean"
        return f"{self.telemetry_hz}Hz {'+'.join(self.sensors)} {corruption}"


def synthesize(condition: Condition, seconds: float = 6.0) -> ExcitationData:
    """Roll the TRUE model, then report it the way the servo would."""
    import mujoco  # noqa: PLC0415 - sim extra
    import numpy as np  # noqa: PLC0415

    sample_times = np.arange(0.0, seconds, 1.0 / condition.telemetry_hz)
    # Amplitude dwarfs the ~20-count blind band (protocol rule 1,
    # docs/23-research-agenda.md) and stays inside ctrlrange — the
    # governor's territory is deliberately never entered (rule 2).
    commands = staged_excitation(
        sample_times, frequencies_hz=[0.3, 0.9, 2.1], peak_amplitude=0.8, stages=3
    )
    # The truth is generated through sysid's OWN rollout — evaluate its
    # residual at the true parameter vector and keep the predictions.
    # Two hand-rolled conventions (end-of-hold sampling, then ZOH
    # controls against sysid's interpolation) each produced a biased
    # fit with tight intervals around wrong values — the exact "hidden
    # convention poisons the fit silently" class Paper 0 named (R13).
    # Riding the same rollout for truth and fit removes the convention
    # axis by construction; the study then varies only what the servo
    # genuinely corrupts.
    from trainnr.robot.identify import _require_sysid  # noqa: PLC0415

    sysid = _require_sysid()
    spec = mujoco.MjSpec.from_string(model_xml(("pos", "vel", "load")))
    model = spec.compile()
    home = mujoco.MjData(model)
    initial_state = sysid.create_initial_state(model, home.qpos, home.qvel, home.act)
    control_ts = sysid.TimeSeries(sample_times, commands.reshape(-1, 1))
    # Ones, not zeros: the dummy measurements only exist so the
    # residual machinery runs; zeros make its per-sensor normalisation
    # divide by zero.
    sensor_ts = sysid.TimeSeries.from_names(
        sample_times, np.ones((len(sample_times), 3)), model
    )
    sequences = sysid.ModelSequences(
        "truth", spec, "excitation", initial_state, control_ts, sensor_ts
    )
    residual_fn = sysid.build_residual_fn(models_sequences=[sequences])
    true_params = sysid.ParameterDict()
    for entry in PARAMETERS:
        true_params.add(
            sysid.Parameter(
                entry.name,
                nominal=entry.nominal,
                min_value=entry.min_value,
                max_value=entry.max_value,
                modifier=entry.apply,
            )
        )
    _, preds, _ = residual_fn(
        true_params.as_vector(), true_params, return_pred_all=True
    )
    predicted = preds[0][0]  # one group, one rollout -> TimeSeries
    # GRID IDENTITY, not resampling: the fit must see the exact times
    # and commands the truth was generated on. Every softer coupling
    # tried here (end-of-hold rows, hand ZOH, values resampled onto the
    # prediction clock) shifted the control grid by one sample — a
    # 10 ms effective command delay the optimizer soaked up by biasing
    # kp LOW with tight intervals. The "hidden convention poisons
    # silently" class, three times in one module; grid identity ends it.
    body = np.asarray(predicted.data)
    truth_times = np.asarray(predicted.times)
    count = len(body) + 1
    if not np.allclose(truth_times, sample_times[1:count]):
        raise ValueError("prediction clock is not the sample grid")
    home.ctrl[0] = commands[0]
    mujoco.mj_forward(model, home)
    full = np.vstack([home.sensordata.copy(), body])
    times_out = sample_times[:count]
    commands_out = commands[:count]

    q_true = full[:, 0]
    if condition.servo_real:
        reported = np.empty_like(q_true)
        last = q_true[0]
        for index, value in enumerate(q_true):
            if abs(value - last) > DEAD_ZONE_RAD:
                last = value
            reported[index] = np.round(last / QUANTUM_RAD) * QUANTUM_RAD
        q_out = reported
        # present-speed is DERIVED from reported position by firmware;
        # an independently-clean velocity channel would flatter it.
        qd_out = np.gradient(reported, times_out)
        rng = np.random.default_rng(20260824)
        tau = full[:, 2] + rng.normal(0.0, LOAD_NOISE_NM, size=len(full))
        tau_out = np.round(tau / LOAD_QUANTUM_NM) * LOAD_QUANTUM_NM
    else:
        q_out, qd_out, tau_out = q_true, full[:, 1], full[:, 2]

    columns = {"pos": q_out, "vel": qd_out, "load": tau_out}
    measurements = np.column_stack([columns[name] for name in condition.sensors])
    return ExcitationData(
        times=times_out,
        controls=commands_out.reshape(-1, 1),
        measurements=measurements,
    )


def run_condition(condition: Condition, seconds: float = 6.0) -> IdentificationResult:
    data = synthesize(condition, seconds=seconds)
    return identify(model_xml(condition.sensors), data, PARAMETERS)


STUDY_MATRIX = tuple(
    Condition(rate, sensors, servo_real)
    for rate in (200, 100, 50, 25)
    for sensors in (("pos",), ("pos", "vel"), ("pos", "load"), ("pos", "vel", "load"))
    for servo_real in (False, True)
)
