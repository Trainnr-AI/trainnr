"""Stage ② as code: excite, fit, and report what is actually pinned.

This is the wedge module — the capability the research found nobody
ships: parametric identification with **confidence intervals and an
identifiability verdict**, wrapped thin over `mujoco.sysid` (Gauss-Newton
over batched CPU rollouts; the toolbox that has existed since MuJoCo
3.5.0 and been applied to no low-cost robot in public).

The deliverable is deliberately not "we identified your robot." It is
**"here is what is pinned, what is not, and by how much"** — a parameter
whose interval spans its allowed range was not identified by the data,
and the report says so instead of pretending
(docs/76, stage ② sub-step 4).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from trainnr.stats.intervals import DEFAULT_CONFIDENCE

if TYPE_CHECKING:  # pragma: no cover
    import numpy

# A parameter counts as PINNED when its confidence half-width is at most
# this fraction of its allowed range. Range-relative rather than
# estimate-relative, so a true value near zero cannot game the verdict.
DEFAULT_PINNED_FRACTION = 0.1


def _require_sysid() -> Any:
    try:
        # Lazy: sim-extra territory; must import cleanly without it so
        # this error can be raised helpfully.
        from mujoco import sysid  # noqa: PLC0415
    except ImportError as error:
        raise ImportError(
            "identification needs the 'sim' extra: uv sync --extra sim (mujoco[sysid])"
        ) from error
    return sysid


@dataclass(frozen=True)
class ParameterSpec:
    """One parameter to identify: bounds, and how it modifies the model.

    `apply(spec, parameter)` writes `parameter.value` into the MjSpec —
    the same modifier contract `mujoco.sysid.Parameter` uses. Keep it a
    named function in real use; a lambda cannot be audited.
    """

    name: str
    nominal: float
    min_value: float
    max_value: float
    apply: Callable[[Any, Any], None]
    initial_guess: float | None = None


@dataclass(frozen=True)
class ExcitationData:
    """One excitation run: what was commanded, what was measured, when.

    Paper 0 builds this from the rig's wire recordings; tests build it
    from synthetic rollouts. Either way the three arrays are row-aligned.
    """

    times: numpy.ndarray
    controls: numpy.ndarray
    measurements: numpy.ndarray

    def __post_init__(self) -> None:
        lengths = {
            len(self.times),
            len(self.controls),
            len(self.measurements),
        }
        if len(lengths) != 1:
            raise ValueError(f"times/controls/measurements lengths differ: {lengths}")


@dataclass(frozen=True)
class IdentifiedParameter:
    name: str
    estimate: float
    half_width: float
    allowed_range: float
    pinned: bool
    # The estimate sits on its declared search bound (2026-09-24): the
    # data pushed it there and could not go further. The bootstrap of
    # docs/e2e-research/72 §7 found the actuator bundles' check knew only
    # the observed rails and missed an exponent at its floor; a method
    # that declares its bounds says so here, per parameter.
    at_bound: bool = False

    @property
    def lower(self) -> float:
        return self.estimate - self.half_width

    @property
    def upper(self) -> float:
        return self.estimate + self.half_width


@dataclass(frozen=True)
class IdentificationResult:
    """The identifiability report: estimates with intervals and verdicts."""

    parameters: tuple[IdentifiedParameter, ...]
    confidence: float

    @property
    def pinned(self) -> tuple[IdentifiedParameter, ...]:
        return tuple(p for p in self.parameters if p.pinned)

    @property
    def unpinned(self) -> tuple[IdentifiedParameter, ...]:
        return tuple(p for p in self.parameters if not p.pinned)

    def summary(self) -> str:
        lines = [
            f"identified {len(self.pinned)}/{len(self.parameters)} "
            f"parameters at {self.confidence:.0%} confidence"
        ]
        for parameter in self.parameters:
            verdict = "pinned" if parameter.pinned else "NOT PINNED"
            if parameter.at_bound:
                verdict += ", AT BOUND"
            lines.append(
                f"  {parameter.name}: {parameter.estimate:.6g} "
                f"± {parameter.half_width:.3g} [{verdict}]"
            )
        return "\n".join(lines)


def staged_excitation(
    times: numpy.ndarray,
    frequencies_hz: Sequence[float],
    peak_amplitude: float,
    *,
    stages: int = 3,
) -> numpy.ndarray:
    """Sum-of-sines excitation with staged amplitude: conservative first.

    The staging is the on-site safety posture from stage ②: early
    portions of the signal stay well inside limits so a rough fit exists
    before the wide excitation runs. Returns the 1-D signal for `times`;
    tile per actuator at the call site.
    """
    import numpy as np  # noqa: PLC0415

    if stages < 1:
        raise ValueError(f"stages must be >= 1, got {stages}")
    if peak_amplitude <= 0:
        raise ValueError(f"peak amplitude must be positive, got {peak_amplitude}")
    times = np.asarray(times)
    # Distinct phases stop the components from peaking together.
    summed = np.sum(
        [
            np.sin(2.0 * np.pi * frequency * times + index)
            for index, frequency in enumerate(frequencies_hz)
        ],
        axis=0,
    )
    signal = np.asarray(summed / max(1.0, float(np.max(np.abs(summed)))))
    progress = np.arange(len(times)) / max(1, len(times))
    stage_index = np.minimum((progress * stages).astype(int), stages - 1)
    amplitude = peak_amplitude * (stage_index + 1) / stages
    return amplitude * signal


def identify(
    model_xml: str,
    data: ExcitationData,
    parameters: Sequence[ParameterSpec],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    pinned_fraction: float = DEFAULT_PINNED_FRACTION,
) -> IdentificationResult:
    """Fit `parameters` so the model reproduces `data.measurements` under
    `data.controls`, and report intervals plus pinned/unpinned verdicts.

    Measurements are the model's sensor outputs, row-aligned with times;
    the initial state is the model's home configuration.
    """
    sysid = _require_sysid()
    import mujoco  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    if not parameters:
        raise ValueError("no parameters to identify")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    spec = mujoco.MjSpec.from_string(model_xml)
    model = spec.compile()
    # `home` not `data`: the ExcitationData parameter is also `data`, and
    # shadowing it here cost a debugging round already.
    home = mujoco.MjData(model)
    initial_state = sysid.create_initial_state(model, home.qpos, home.qvel, home.act)

    parameter_dict = sysid.ParameterDict()
    for entry in parameters:
        parameter_dict.add(
            sysid.Parameter(
                entry.name,
                nominal=entry.nominal,
                min_value=entry.min_value,
                max_value=entry.max_value,
                modifier=entry.apply,
            )
        )
        if entry.initial_guess is not None:
            parameter_dict[entry.name].value[:] = entry.initial_guess

    times = np.asarray(data.times)
    control_ts = sysid.TimeSeries(times, np.asarray(data.controls))
    sensor_ts = sysid.TimeSeries.from_names(times, np.asarray(data.measurements), model)
    sequences = sysid.ModelSequences(
        "model", spec, "excitation", initial_state, control_ts, sensor_ts
    )
    residual_fn = sysid.build_residual_fn(models_sequences=[sequences])
    fitted, optimum = sysid.optimize(
        initial_params=parameter_dict,
        residual_fn=residual_fn,
        optimizer="mujoco",
        verbose=False,
    )
    residuals_star, _, _ = residual_fn(optimum.x, fitted, return_pred_all=True)
    _covariance, half_widths = sysid.calculate_intervals(
        residuals_star, optimum.jac, alpha=1.0 - confidence
    )

    identified = []
    for index, entry in enumerate(parameters):
        half_width = float(half_widths[index])
        allowed = entry.max_value - entry.min_value
        finite = np.isfinite(half_width)
        identified.append(
            IdentifiedParameter(
                name=entry.name,
                estimate=float(fitted[entry.name].value[0]),
                half_width=half_width if finite else float("inf"),
                allowed_range=allowed,
                pinned=bool(finite and half_width <= pinned_fraction * allowed),
            )
        )
    return IdentificationResult(parameters=tuple(identified), confidence=confidence)
