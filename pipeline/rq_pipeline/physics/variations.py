"""Variation appliers: how a drawn value reaches a MuJoCo model.

The env is engine-blind about WHAT a knob does; each knob is an
`Applier` registered by name (the part of the key after the last dot),
with three duties: `snapshot` the nominal values once, `restore` them
before every reset, `apply` the trial's value from the nominal. A new
knob is one decorated class here; a new engine registers its own set.
Unknown names are refused with the list of known ones — built from the
registry, never retyped.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from rq_pipeline.evaluate.variations import VariationKeys

XYZ = 3


class Applier(Protocol):
    def snapshot(self, model: Any) -> Any: ...
    def restore(self, model: Any, nominal: Any) -> None: ...
    def apply(self, model: Any, nominal: Any, host: str, value: Any) -> None: ...


APPLIERS: dict[str, Applier] = {}


def applier(name: str):
    """Register an `Applier` class under a knob name."""

    def decorate(cls: type) -> type:
        if name in APPLIERS:
            raise ValueError(f"applier {name!r} registered twice")
        APPLIERS[name] = cls()
        return cls

    return decorate


def known_keys() -> str:
    return ", ".join(sorted(APPLIERS))


def _expect_host(host: str, expected: str, name: str) -> None:
    if host != expected:
        raise ValueError(f"{name} lives on host {expected!r}, got {host!r}")


def _named(model: Any, kind: Any, name: str, what: str) -> int:
    import mujoco  # noqa: PLC0415 - sim extra

    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise ValueError(f"variation names {what} {name!r}, which the model lacks")
    return index


@applier(VariationKeys.DAMPING_SCALE)
class DampingScale:
    """Every joint's damping, scaled: `joints.damping_scale`."""

    def snapshot(self, model: Any) -> Any:
        return model.dof_damping.copy()

    def restore(self, model: Any, nominal: Any) -> None:
        model.dof_damping[:] = nominal

    def apply(self, model: Any, nominal: Any, host: str, value: Any) -> None:
        _expect_host(host, VariationKeys.JOINTS, VariationKeys.DAMPING_SCALE)
        model.dof_damping[:] = nominal * float(value)


@applier(VariationKeys.GAIN_SCALE)
class GainScale:
    """Every position servo's stiffness: `actuators.gain_scale`. BOTH kp
    terms — `gainprm[0]` on the command and `biasprm[1]` on the position
    — or the setpoint moves instead (`tasks.aloha2.scale_dynamics`)."""

    def snapshot(self, model: Any) -> Any:
        return model.actuator_gainprm[:, 0].copy(), model.actuator_biasprm[:, 1].copy()

    def restore(self, model: Any, nominal: Any) -> None:
        model.actuator_gainprm[:, 0], model.actuator_biasprm[:, 1] = nominal

    def apply(self, model: Any, nominal: Any, host: str, value: Any) -> None:
        _expect_host(host, VariationKeys.ACTUATORS, VariationKeys.GAIN_SCALE)
        gain, bias = nominal
        model.actuator_gainprm[:, 0] = gain * float(value)
        model.actuator_biasprm[:, 1] = bias * float(value)


@applier(VariationKeys.DIFFUSE_SCALE)
class DiffuseScale:
    """Every light's diffuse level: `lights.diffuse_scale`."""

    def snapshot(self, model: Any) -> Any:
        return model.light_diffuse.copy()

    def restore(self, model: Any, nominal: Any) -> None:
        model.light_diffuse[:] = nominal

    def apply(self, model: Any, nominal: Any, host: str, value: Any) -> None:
        _expect_host(host, VariationKeys.LIGHTS, VariationKeys.DIFFUSE_SCALE)
        model.light_diffuse[:] = nominal * float(value)


@applier(VariationKeys.MASS_SCALE)
class MassScale:
    """One body's mass, inertia scaled with it: `<body>.mass_scale`."""

    def snapshot(self, model: Any) -> Any:
        return model.body_mass.copy(), model.body_inertia.copy()

    def restore(self, model: Any, nominal: Any) -> None:
        model.body_mass[:], model.body_inertia[:] = nominal

    def apply(self, model: Any, nominal: Any, host: str, value: Any) -> None:
        import mujoco  # noqa: PLC0415 - sim extra

        body = _named(model, mujoco.mjtObj.mjOBJ_BODY, host, "body")
        mass, inertia = nominal
        model.body_mass[body] = mass[body] * float(value)
        model.body_inertia[body] = inertia[body] * float(value)


@applier(VariationKeys.OFFSET_M)
class CameraOffset:
    """One camera's position, nominal plus a metre offset: `<camera>.offset_m`."""

    def snapshot(self, model: Any) -> Any:
        return model.cam_pos.copy()

    def restore(self, model: Any, nominal: Any) -> None:
        model.cam_pos[:] = nominal

    def apply(self, model: Any, nominal: Any, host: str, value: Any) -> None:
        import mujoco  # noqa: PLC0415 - sim extra

        camera = _named(model, mujoco.mjtObj.mjOBJ_CAMERA, host, "camera")
        offset = np.asarray(value, dtype=float)
        if offset.shape != (XYZ,):
            raise ValueError(f"{host}.{VariationKeys.OFFSET_M} needs three components")
        model.cam_pos[camera] = nominal[camera] + offset


def snapshot_all(model: Any) -> dict[str, Any]:
    return {name: app.snapshot(model) for name, app in APPLIERS.items()}


def restore_all(model: Any, nominal: dict[str, Any]) -> None:
    for name, app in APPLIERS.items():
        app.restore(model, nominal[name])


def apply_key(model: Any, nominal: dict[str, Any], key: str, value: Any) -> None:
    host, name = VariationKeys.split(key)
    app = APPLIERS.get(name)
    if app is None:
        raise ValueError(f"unknown variation {key!r}; the engine knows {known_keys()}")
    app.apply(model, nominal[name], host, value)
