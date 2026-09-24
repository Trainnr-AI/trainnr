"""The gate's course protocol: on a staged scene (`scenes.stage`) the
policy is commanded along the course the scene's author laid out —
forward at a drawn speed, steered to the next waypoint — instead of
holding a random twist for twenty seconds, which on a scene drives a
flat-ground policy into its hurdles, desks and walls (docs/78 §8.3:
0/4 on the course by that protocol, never falling).

What is stated, never hidden:
- the steering law is the one the policy trained under: mjlab's heading
  pursuit, yaw rate = clip(gain * wrap(heading to the waypoint - yaw),
  the manifest's yaw-rate range); the gain is the manifest's when the
  export recorded it, else this module's own, and the record says which;
- the speed is drawn per trial, seeded, in the upper part of the
  manifest's forward range (`COURSE_SPEED_FRACTION`), so a trial is
  never a standstill exam;
- the budget is the path's length at that speed times `COURSE_SLACK`;
- success is `survived and finished`: upright, every waypoint reached
  within `COURSE_REACH_M`, inside the budget. The tracking ratio is
  still measured and recorded; it no longer decides.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from rq_pipeline.deploy.manifest import Key, Manifest
from rq_pipeline.deploy.ticks import Ticks
from rq_pipeline.evaluate.tracking import TrackingOutcome

if TYPE_CHECKING:
    from rq_pipeline.deploy.mirror import GateMirror
    from rq_pipeline.deploy.runtimes import GateRuntime

COMMANDS_ALONG = "forward along the scene's course, steered to the next waypoint"
COURSE_SPEED_FRACTION = 0.5  # speeds drawn in [fraction * v_max, v_max]
COURSE_REACH_M = 0.3  # a waypoint is reached inside this planar radius
COURSE_SLACK = 2.0  # budget = path length / speed * slack
# mjlab's `heading_control_stiffness` for the Go2 velocity task, the law
# every Go2 policy here trained under; used when the manifest names none.
COURSE_STEER_GAIN = 0.5
STEER_LAW = (
    "yaw rate = clip(gain * wrap(heading to the waypoint - yaw), yaw-rate range); "
    "forward = speed * max(cos(that error), 0)"
)
GAIN_FROM_MANIFEST = "the manifest's heading_gain: what the policy trained under"
GAIN_FROM_PROTOCOL = "the protocol's own: the manifest names none"
COURSE_KEY = "course"
START_KEY = "start"


def course_criterion_text() -> str:
    """The rule as a record states it."""
    return "survived and finished the course within its budget"


def yaw_of(quat_wxyz: np.ndarray) -> float:
    """The heading of a w-x-y-z quaternion's x axis, radians."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Course:
    """The path: the stage's start, then the scene's waypoints, in the
    world frame; the steering and the arrival are planar, the picture
    draws the heights."""

    start: np.ndarray  # (3,)
    waypoints: np.ndarray  # (n, 3)
    reach_m: float = COURSE_REACH_M

    @classmethod
    def of_manifest(cls, manifest: Manifest) -> Course | None:
        """The course a staged manifest's scene block lays out; None for
        a plane (no course, or an empty one); a course without the
        stage's start is refused, a start is given, never guessed."""
        block = manifest.raw.get(Key.SCENE) or {}
        points = (block.get(COURSE_KEY) or {}).get("waypoints") or []
        if not points:
            return None
        start = block.get(START_KEY)
        if start is None:
            raise ValueError(
                f"{manifest.root}: the scene block lays out a course but no start"
            )
        return cls(
            start=np.asarray(start, dtype=np.float64)[:3],
            waypoints=np.asarray(points, dtype=np.float64)[:, :3],
        )

    @property
    def path(self) -> np.ndarray:
        """The start and every waypoint, (n + 1, 3)."""
        return np.vstack([self.start[None, :], self.waypoints])

    @property
    def length_m(self) -> float:
        return float(np.linalg.norm(np.diff(self.path[:, :2], axis=0), axis=1).sum())

    def budget_s(self, speed: float) -> float:
        return self.length_m / speed * COURSE_SLACK

    def reached_by(self, position: np.ndarray, reached: int) -> int:
        """How many waypoints are reached with the robot at `position`,
        having reached `reached` already: every next one within reach."""
        while reached < len(self.waypoints) and (
            np.linalg.norm(self.waypoints[reached, :2] - position[:2]) < self.reach_m
        ):
            reached += 1
        return reached

    def describe(self) -> dict[str, Any]:
        return {
            "waypoints": int(self.waypoints.shape[0]),
            "length_m": round(self.length_m, 3),
            "reach_m": self.reach_m,
            "slack": COURSE_SLACK,
            "speed_fraction": COURSE_SPEED_FRACTION,
        }


@dataclass(frozen=True)
class Steering:
    """Heading pursuit at a stated gain, clipped to the yaw-rate range."""

    gain: float
    yaw_rate: tuple[float, float]
    basis: str

    @classmethod
    def of_manifest(cls, manifest: Manifest, *, limit: float | None = None) -> Steering:
        commands = manifest.commands
        lo, hi = commands.ang_vel_z
        if limit is not None:  # a gamepad's sticks stop here too
            lo, hi = max(lo, -float(limit)), min(hi, float(limit))
        gain = commands.heading_gain
        return cls(
            gain=float(gain) if gain is not None else COURSE_STEER_GAIN,
            yaw_rate=(float(lo), float(hi)),
            basis=GAIN_FROM_MANIFEST if gain is not None else GAIN_FROM_PROTOCOL,
        )

    def command(self, speed: float, heading_error: float) -> np.ndarray:
        """The twist for one tick: forward at `speed` scaled by how much
        of it points at the waypoint (standing to turn when facing
        away), yaw rate by the pursuit law."""
        wz = min(max(self.gain * heading_error, self.yaw_rate[0]), self.yaw_rate[1])
        vx = speed * max(math.cos(heading_error), 0.0)
        return np.array([vx, 0.0, wz], dtype=np.float32)

    def describe(self) -> dict[str, Any]:
        return asdict(self) | {"law": STEER_LAW}


@dataclass(frozen=True)
class CourseTrial(TrackingOutcome):
    """One walk along the course, judged by arrival."""

    speed: float
    reached: int
    of: int
    seconds: float
    budget_s: float

    @property
    def finished(self) -> bool:
        return self.reached == self.of

    @property
    def success(self) -> bool:  # arrival decides here, not tracking
        return self.survived and self.finished

    def row(self) -> dict[str, Any]:
        return super().row() | {"finished": self.finished}


def draw_speeds(
    manifest: Manifest, trials: int, seed: int, *, limit: float | None = None
) -> np.ndarray:
    """Seeded forward speeds in the upper part of the manifest's forward
    range, capped by the runtime's envelope; a policy that cannot walk
    forward is refused by name."""
    _, hi = manifest.commands.lin_vel_x
    if limit is not None:
        hi = min(hi, float(limit))
    if hi <= 0:
        raise ValueError(
            f"{manifest.root}: the manifest's forward range tops out at {hi}: "
            "no speed walks the course"
        )
    from rq_pipeline.deploy.gate import trial_rng  # noqa: PLC0415 - gate imports this

    # one per trial from (seed, trial): the gate's own draw (`gate.DRAW_NOW`)
    return np.array(
        [
            trial_rng(seed, i).uniform(COURSE_SPEED_FRACTION * hi, hi)
            for i in range(trials)
        ]
    )


def run_course_trial(  # noqa: PLR0913 - the trial's own knobs, each named
    manifest: Manifest,
    runtime: GateRuntime,
    course: Course,
    steering: Steering,
    speed: float,
    *,
    mirror: GateMirror | None = None,
    index: int = 0,
    contacts: list[np.ndarray] | None = None,
) -> CourseTrial:
    """One walk along the course at `speed`, steered every tick from
    the runtime's pose; the budget from the course's length; with a
    `mirror`, every tick goes to the Studio; with a `contacts` list,
    every tick's contact points are appended to it."""
    dt = manifest.control.step_dt
    budget_s = course.budget_s(speed)
    runtime.reset()
    meter = Ticks(dt, mirror=mirror, contacts=contacts)
    reached = 0
    if mirror is not None:
        mirror.note(f"trial {index}: along the course at {speed:.2f} m/s")
    for _ in range(round(budget_s / dt)):
        position, quat, _joints = runtime.pose()
        reached = course.reached_by(position, reached)
        if reached == len(course.waypoints):
            break
        ahead = course.waypoints[reached, :2] - position[:2]
        error = wrap_to_pi(math.atan2(ahead[1], ahead[0]) - yaw_of(quat))
        if meter.tick(runtime, steering.command(speed, error)):
            break
    else:  # the budget ran out: the last tick may have arrived
        reached = course.reached_by(runtime.pose()[0], reached)
    return CourseTrial(
        **meter.outcome(),
        speed=float(speed),
        reached=reached,
        of=int(course.waypoints.shape[0]),
        seconds=meter.steps * dt,
        budget_s=budget_s,
    )
