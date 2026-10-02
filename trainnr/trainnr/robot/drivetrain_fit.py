"""The drivetrain's ratio-form fit — Paper 0's producer, as a committed tool.

The 2026-08-25 product review found the repo's flagship artifacts
(robots/rig-drivetrain/fits/*.json) were produced by an uncommitted
scratchpad script: the instrument could not reproduce its own
deliverable. This module recreates that producer from the records' own
shape and the bundle README's data path, and stays — reproducing the
committed estimates is pinned by test_drivetrain_fit.

The ratio form (the 2026-08-24 rig session's central finding): at 50 Hz
the torque scale is structurally unobservable for this ~2 ms motor — a
free fit sends every parameter NOT PINNED. So damping is FIXED at
DAMPING_REF as the scale reference and gear/frictionloss are fitted
PER UNIT DAMPING; the record's anchor statement says so, and
scale_ref_damping ships with an unbounded half-width by construction.
"""

from __future__ import annotations

from pathlib import Path

from trainnr.bundles.hashing import stamp
from trainnr.bundles.profile import load_profile
from trainnr.collect.excitation import drivetrain_excitation
from trainnr.collect.wire import parse_recording
from trainnr.robot.fit_record import write_fit_record
from trainnr.robot.identify import (
    ExcitationData,
    IdentificationResult,
    ParameterSpec,
    identify,
)

# The scale reference. Everything named *_per_damp is relative to it.
DAMPING_REF = 1e-3

RATIO_ANCHOR = (
    "RATIO FIT: damping FIXED at 1e-3 as the scale reference, because the "
    "torque scale is structurally unobservable at 50 Hz for this ~2 ms motor "
    "(the armature anchor cannot bite; confirmed live 2026-08-24 when free "
    "bounds sent every parameter NOT PINNED). gear and frictionloss are "
    "per-unit-damping. Measurement sign flipped (+duty drives -ticks, "
    "DRIVETRAIN_SIGN). ticks/rev 4290 per profile.json, hand-count pending."
)

UNITS = {
    "scale_ref_damping": "N*m*s/rad — FIXED anchor, not estimated",
    "left_gear_per_damp": "actuator gear (N*m per duty%) per unit damping ratio",
    "left_fric_per_damp": "joint frictionloss (N*m) per unit damping ratio",
    "right_gear_per_damp": "actuator gear (N*m per duty%) per unit damping ratio",
    "right_fric_per_damp": "joint frictionloss (N*m) per unit damping ratio",
}


def _set_scale_ref(spec, parameter) -> None:
    spec.joint("left").damping[0] = parameter.value[0]
    spec.joint("right").damping[0] = parameter.value[0]


def _set_left_gear(spec, parameter) -> None:
    spec.actuator("left_motor").gear[0] = parameter.value[0]


def _set_right_gear(spec, parameter) -> None:
    spec.actuator("right_motor").gear[0] = parameter.value[0]


def _set_left_friction(spec, parameter) -> None:
    spec.joint("left").frictionloss = parameter.value[0]


def _set_right_friction(spec, parameter) -> None:
    spec.joint("right").frictionloss = parameter.value[0]


def ratio_parameters() -> tuple[ParameterSpec, ...]:
    """The ratio-form parameter set the committed fit records carry."""
    hair = 1e-6  # the anchor's near-zero freedom; keeps the Jacobian sane
    return (
        ParameterSpec(
            "scale_ref_damping",
            DAMPING_REF,
            DAMPING_REF - hair,
            DAMPING_REF + hair,
            _set_scale_ref,
        ),
        ParameterSpec("left_gear_per_damp", 1e-4, 1e-5, 1e-3, _set_left_gear),
        ParameterSpec("left_fric_per_damp", 1e-4, 0.0, 5e-3, _set_left_friction),
        ParameterSpec("right_gear_per_damp", 1e-4, 1e-5, 1e-3, _set_right_gear),
        ParameterSpec("right_fric_per_damp", 1e-4, 0.0, 5e-3, _set_right_friction),
    )


def drivetrain_data(bundle_dir: Path, wire_path: Path) -> ExcitationData:
    """Wire recording → identification data, with the drivetrain sign."""
    import numpy as np  # noqa: PLC0415 - sim extra territory

    recording = parse_recording(wire_path)
    profile = load_profile(bundle_dir)
    data = drivetrain_excitation(recording, profile)
    # +duty drives -ticks on this chassis (firmware DRIVETRAIN_SIGN);
    # the model's +gear expects +angle, so the measurement flips here
    # and the anchor statement says so.
    return ExcitationData(
        times=data.times,
        controls=data.controls,
        measurements=-np.asarray(data.measurements),
    )


def fit_drivetrain(
    bundle_dir: Path,
    wire_path: Path,
    *,
    write: bool = True,
) -> tuple[IdentificationResult, Path | None]:
    """The whole chain: record → excite → identify → fit record."""
    bundle_dir = Path(bundle_dir)
    wire_path = Path(wire_path)
    profile = load_profile(bundle_dir)
    model_xml = (bundle_dir / profile.model_file).read_text(encoding="utf-8")
    data = drivetrain_data(bundle_dir, wire_path)
    result = identify(model_xml, data, ratio_parameters())
    if not write:
        return result, None
    path = write_fit_record(
        bundle_dir,
        result,
        robot=bundle_dir.name,
        recording=stamp(wire_path.stem, wire_path),
        anchor=RATIO_ANCHOR,
        units=UNITS,
    )
    return result, path
