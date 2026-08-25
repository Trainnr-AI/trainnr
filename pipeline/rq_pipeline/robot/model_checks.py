"""Fail-loudly gates for imported models.

MuJoCo's USD importer has zero references to UsdPhysicsDriveAPI: a
well-formed generic USD robot imports with no actuators and no error —
a plausible file and a silently inert robot. Same shape for sensors and
for scene geometry. Never infer that an import worked because it did
not throw; count what came alive and refuse zero.
"""

from __future__ import annotations


class DeadModelError(ValueError):
    """An imported model is missing the parts that make it a robot."""


def assert_model_alive(  # noqa: PLR0913, PLR0917 - one census, one gate
    actuators: int,
    sensors: int,
    geoms: int,
    source: str,
    expect_sensors: bool = True,
    cameras: int | None = None,
) -> None:
    """Refuse a model whose import silently dropped its function.

    `expect_sensors=False` is for scene-only bundles, which legitimately
    carry no sensors but must still carry geometry. `cameras` is checked
    only when given: the VISION path passes its count, because a vision
    policy on a camera-less model scores every trial 0% with no error —
    the doctrine ("count what came alive and refuse zero") applied to
    the one census the review found unenforced (2026-08-26).
    """
    problems = []
    if actuators <= 0:
        problems.append("0 actuators (motors dropped by the importer?)")
    if expect_sensors and sensors <= 0:
        problems.append("0 sensors (the <sensor> block does not survive USD)")
    if geoms <= 0:
        problems.append("0 geoms (empty scene)")
    if cameras is not None and cameras <= 0:
        problems.append("0 cameras (a vision policy would stare at nothing)")
    if problems:
        raise DeadModelError(
            f"{source}: model imported but is not alive — " + "; ".join(problems)
        )
