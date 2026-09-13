"""The rollout -> dataset writer: an RL teacher presses demonstrations.

    cd rq_mjlab && LD_LIBRARY_PATH=/usr/lib/wsl/lib MUJOCO_GL=egl \\
        uv run python -m rq_mjlab.walk_press --latest --out ../runs/walk-demos \\
        [--episodes 12] [--worlds 9] [--seed 1000] [--frame-every 5] [--no-studio]

docs/66 §3 source 3 and D2: the trained walk policy (a state-based PPO
teacher with the certified stack under it) rolls out in the batched
env, and every world's first episode is JUDGED with the certificate's
own criterion (walk_verdict: survived AND tracked). Keepers become
`DemoLayout` episodes the press's loop writes - the actor's
observation and action per control tick, the qpos replayed through a
one-robot CPU mirror for chase-camera frames, an `EpisodeManifest`
stamped with the run identity, the checkpoint's content hash as the
expert, and the batched instrument. Discards are not thrown away: each
lands as a row in `failures.jsonl` with its command and error - the
failure mining the funnel->press loop (D6) will target.

The keep/discard LOOP is `press.press`, unchanged: the batched rollout
is a queue its serial `attempt_fn` drains, refilled with a fresh
seeded batch when empty, so datasheet, feed and accounting are the
ones every other press has. The batch writes an `ExportSpec` so
`demo_export.export_batch` turns it into a LeRobot dataset with no
Task object - the distillation input (a vision/proprio student).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

from rq_pipeline.bundles.hashing import fields_hash, stamp
from rq_pipeline.collect.demo_export import ExportSpec
from rq_pipeline.collect.press import DemoBatch, EpisodeManifest, PressResult, press
from rq_pipeline.collect.provenance import dagger_stamp
from rq_pipeline.evaluate.tracking import criterion_text

from rq_mjlab.walk_verdict import (
    WorldEpisode,
    instrument_for,
    rollout_episodes,
)

REPO = Path(__file__).resolve().parents[3]
CAMERA_KEY = "chase"
FAILURES_FILE = "failures.jsonl"
BUNDLE = "microduck"
INSTRUCTION = "walk at the commanded planar twist"
STATE_SEMANTICS = (
    "the actor's observation vector (rsl-rl): base angular velocity, "
    "projected gravity, joint positions, joint velocities, previous "
    "actions, commanded twist - the state-based teacher's own input"
)
ACTION_SEMANTICS = "joint position targets (the rsl-rl actor's output, ctrl order)"


def checkpoint_stamp(checkpoint: Path) -> str:
    """The expert's identity: the checkpoint file's content hash."""
    return stamp(checkpoint.stem, checkpoint)


def state_names(observation_manager: Any, group: str = "actor") -> list[str]:
    """One name per observation component, from the manager's own term
    list and dims - `joint_pos[3]`, `command[0]` - so the dataset says
    what each state column IS."""
    names = observation_manager.active_terms[group]
    dims = observation_manager.group_obs_term_dim[group]
    columns: list[str] = []
    for term, shape in zip(names, dims, strict=True):
        width = 1
        for side in shape:
            width *= int(side)
        columns.extend(f"{term}[{i}]" for i in range(width))
    return columns


def failure_row(batch_seed: int, world: int, episode: WorldEpisode) -> dict[str, Any]:
    """What a discard leaves behind - enough to press where it fails."""
    outcome = episode.outcome
    return {
        "batch_seed": batch_seed,
        "world": world,
        "command": episode.command,
        "steps": outcome.steps,
        "fell": outcome.fell,
        "mean_err_mps": round(outcome.mean_err, 4),
        "mean_cmd_mps": round(outcome.mean_cmd, 4),
        "err_ratio": round(outcome.err_ratio, 4),
        "tracked": outcome.tracked,
    }


class ChaseCamera:
    """Chase-camera frames for one world's episode, replayed through a
    one-robot CPU mirror (the batched engine's memory is invisible to
    any renderer; the recorder's camera leg is the same idea)."""

    def __init__(self, width: int, height: int, robot: str = BUNDLE) -> None:
        import mujoco  # noqa: PLC0415

        from rq_mjlab.walk_view import transport  # noqa: PLC0415
        from rq_mjlab.walks import walk_spec  # noqa: PLC0415

        self._mujoco = mujoco
        # The bare one-robot mirror: no sky, the model's own shadow map -
        # the picture the student's datasets were pressed with. It was
        # `walk_view.mirror_of` until that went with the many-worlds scene
        # (2026-09-09) and left this import pointing at nothing; mypy
        # against the real packages found it (2026-09-13).
        self.model = transport().walk_scene(robot, 1, max(width, height), dressed=False)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        # The walk's own chase framing (rq_mjlab.walks), not a duck's numbers.
        chase = walk_spec(robot).chase
        self.camera.distance = chase.distance
        self.camera.elevation = chase.elevation
        self.camera.azimuth = chase.azimuth

    def frame_at(self, qpos: Any) -> Any:
        """One chase frame for one world's qpos - the SAME picture the
        press writes and the student's verdict shows it (one camera for
        training and judging, or the certificate measures a camera
        mismatch instead of a policy)."""
        self.data.qpos[:] = qpos
        self._mujoco.mj_forward(self.model, self.data)
        self.camera.lookat[:] = self.data.xpos[1]  # the trunk, body 1
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()

    def frames(self, qpos: Any, every: int) -> list[tuple[int, Any]]:
        return [
            (tick, self.frame_at(qpos[tick])) for tick in range(0, len(qpos), every)
        ]

    def close(self) -> None:
        self.renderer.close()


def relabel(episode: WorldEpisode, teacher: Any) -> WorldEpisode:
    """DAgger's one move: the STUDENT drove (its observations, its
    states), the TEACHER labels — every captured observation gets the
    action the teacher would have taken there, and that pair is what
    the dataset learns from. `teacher` maps a (T, obs) array to a
    (T, nu) array (the rsl-rl actor on a batch of observations). The
    judgment, command and qpos are the student's episode, untouched:
    the referee gated the STUDENT's walk, not the teacher's labels."""
    import numpy as np  # noqa: PLC0415

    if episode.observations is None:
        raise ValueError("relabel needs a captured episode (observations)")
    labels = np.asarray(teacher(episode.observations), dtype=np.float32)
    if labels.shape != episode.actions.shape:
        raise ValueError(
            f"teacher labels {labels.shape} do not match the student's actions "
            f"{episode.actions.shape}"
        )
    return WorldEpisode(
        episode.outcome,
        episode.command,
        observations=episode.observations,
        actions=labels,
        qpos=episode.qpos,
    )


def teacher_labeler(policy: Any, device: str) -> Any:
    """The rsl-rl inference policy as a batch labeler: (T, obs) numpy
    in, (T, nu) numpy out, no gradients. The captured rows are the
    actor's observation group (`walk_verdict.ACTOR_OBS_GROUP`), and the
    actor indexes a TensorDict of groups — handing it the bare tensor
    is the IndexError that killed DAgger round 1's first launch
    (docs/07 2026-09-04)."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from tensordict import TensorDict  # noqa: PLC0415

    from rq_mjlab.walk_verdict import ACTOR_OBS_GROUP  # noqa: PLC0415

    def label(observations: Any) -> Any:
        with torch.no_grad():
            rows = torch.as_tensor(
                np.asarray(observations), dtype=torch.float32, device=device
            )
            obs = TensorDict({ACTOR_OBS_GROUP: rows}, batch_size=[rows.shape[0]])
            return policy(obs).detach().cpu().numpy()

    return label


def press_walk(  # noqa: PLR0913, PLR0915 - every knob of the press, named; one loop
    checkpoint: Path,
    out: Path,
    *,
    episodes: int,
    worlds: int,
    seed: int,
    frame_every: int,
    frame_size: tuple[int, int],
    device: str,
    feed: Any = None,
    say: Any = print,
    driver: Any = None,
    driver_name: str = "",
) -> DemoBatch:
    """Press `episodes` kept walk demonstrations from `checkpoint` under
    `out`; the batched rollouts run `worlds` at a time.

    `driver` is who ROLLS OUT; the checkpoint's teacher always LABELS.
    With no driver the teacher drives itself (the D2 press). With a
    driver — a `walk_verdict.StudentPolicy` over the bridge — this is
    DAgger: the student visits its own states, the teacher says what
    to do there, the referee keeps the episodes where the student's
    walk passed, and the dataset grows exactly where the student is
    weak (docs/66 §6; Ross et al. 2011)."""
    import numpy as np  # noqa: PLC0415

    from rq_mjlab.microduck_walk import microduck_walk_env_cfg  # noqa: PLC0415
    from rq_mjlab.walk_view import load_policy  # noqa: PLC0415

    _, identity = microduck_walk_env_cfg()
    env, policy = load_policy(checkpoint, worlds, device)
    teacher = teacher_labeler(policy, device)
    if driver is not None and hasattr(driver, "bind"):
        driver.bind(env)
    rolling = policy if driver is None else driver
    unwrapped = env.unwrapped
    source = f"microduck-walk@{fields_hash(identity)}"
    expert = checkpoint_stamp(checkpoint)
    if driver is not None:
        # The dataset's expert is still the teacher (its labels); the
        # driver rides on the stamp so a DAgger batch is never mistaken
        # for a teacher press.
        expert = dagger_stamp(expert, driver_name or "student")
    instrument = instrument_for(device)
    control_hz = round(1.0 / float(unwrapped.step_dt))
    basis = (
        f"{identity['dr_basis']}; per-world draws live inside the batched env "
        "and are not exported per episode (D2 v1)"
    )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ExportSpec(
        state_width=len(state_names(unwrapped.observation_manager)),
        state_names=state_names(unwrapped.observation_manager),
        cameras=[CAMERA_KEY],
        instruction=INSTRUCTION,
        bundle=BUNDLE,
        state_semantics=STATE_SEMANTICS,
        action_semantics=ACTION_SEMANTICS,
        action_names=[
            f"joint_pos_target[{i}]"
            for i in range(unwrapped.action_manager.total_action_dim)
        ],
        notes={
            "teacher": expert,
            "criterion": criterion_text(),
        },
    ).write_to(out)
    camera = ChaseCamera(*frame_size)
    queue: deque[tuple[int, int, WorldEpisode]] = deque()
    failures = out / FAILURES_FILE

    def refill(rng: np.random.Generator) -> None:
        batch_seed = int(rng.integers(2**31 - 1))
        unwrapped.seed(batch_seed)
        env.reset()
        rolled = rollout_episodes(env, rolling, worlds, capture=True)
        if driver is not None:
            rolled = [relabel(ep, teacher) for ep in rolled]
        queue.extend((batch_seed, world, ep) for world, ep in enumerate(rolled))
        passing = sum(e.outcome.success for e in rolled)
        say(f"batch seed {batch_seed}: {passing}/{worlds} pass the criterion")

    def attempt_fn(rng: np.random.Generator, *, frame_every: int) -> PressResult:
        if not queue:
            refill(rng)
        batch_seed, world, episode = queue.popleft()
        outcome = episode.outcome
        draws = {
            "batch_seed": batch_seed,
            "world": world,
            "command": episode.command,
            "err_ratio": round(outcome.err_ratio, 4),
            "survived": outcome.survived,
            "tracked": outcome.tracked,
        }
        if not outcome.success:
            with failures.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure_row(batch_seed, world, episode)) + "\n")
            reason = "fell" if outcome.fell else f"err_ratio {outcome.err_ratio:.2f}"
            return PressResult(False, {}, draws, [], basis, note=reason)
        return PressResult(
            True,
            {},
            draws,
            [],
            basis,
            episode.qpos,
            episode.observations,
            episode.actions,
            camera_frames={CAMERA_KEY: camera.frames(episode.qpos, frame_every)},
        )

    def manifest_fn(result: PressResult, attempt: int) -> EpisodeManifest:
        return EpisodeManifest(
            seed=seed,
            attempt=attempt,
            task=source,
            expert=expert,
            instrument=instrument,
            dynamics={},
            draws=result.draws,
            retries=[],
            control_hz=control_hz,
            frame_every_control_ticks=frame_every,
            dynamics_basis=basis,
            action_semantics=ACTION_SEMANTICS,
            verdict=f"success ({criterion_text()})",
        )

    try:
        return press(
            out,
            task=source,
            expert=expert,
            instrument=instrument,
            control_hz=control_hz,
            attempt_fn=attempt_fn,
            manifest_fn=manifest_fn,
            episodes=episodes,
            seed=seed,
            frame_every=frame_every,
            feed=feed,
            say=say,
        )
    finally:
        camera.close()
        if driver is not None and hasattr(driver, "close"):
            driver.close()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", nargs="?", type=Path, default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("../runs/walk-demos"))
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--worlds", type=int, default=9)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--frame-every",
        type=int,
        default=1,
        help="control ticks per dataset frame - the student's action cadence. "
        "For this gait it must be 1: the TEACHER, actions held 5 ticks, falls "
        "in 20-23 ticks; held 2, 38/40 survive (measured 2026-09-02)",
    )
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-studio", action="store_true")
    parser.add_argument(
        "--student",
        type=Path,
        default=None,
        help="DAgger: a LeRobot checkpoint (pretrained_model dir) DRIVES the "
        "rollouts over the bridge; the teacher labels every visited state",
    )
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--student-python",
        type=Path,
        default=REPO / "pipeline" / ".venv-train" / "bin" / "python",
    )
    args = parser.parse_args()

    from rq_mjlab.walk_view import latest_checkpoint  # noqa: PLC0415

    checkpoint = args.checkpoint if args.checkpoint else latest_checkpoint()

    import warp as wp  # noqa: PLC0415

    wp.init()
    device = args.device or ("cuda:0" if wp.is_cuda_available() else "cpu")

    feed = None
    if not args.no_studio:
        from rq_pipeline.collect.press_feed import StudioPressFeed  # noqa: PLC0415

        feed = StudioPressFeed.connect(f"walk-{args.out.name}")

    driver = None
    driver_name = ""
    if args.student is not None:
        from rq_pipeline.bundles.hashing import stamp as stamp_of  # noqa: PLC0415

        from rq_mjlab.walk_verdict import StudentPolicy  # noqa: PLC0415
        from rq_mjlab.walk_view import load_policy  # noqa: PLC0415

        # The student needs the env it will drive; press_walk builds its
        # own, so the driver is built lazily on that env below.
        driver_name = stamp_of(f"student-{args.student.parent.name}", args.student)

        class _LazyStudent:
            """Binds the StudentPolicy to press_walk's env on first call."""

            def __init__(self) -> None:
                self._inner: Any = None

            def bind(self, env: Any) -> None:
                self._inner = StudentPolicy(
                    env,
                    args.student,
                    python=args.student_python,
                    horizon=args.horizon,
                    stride=args.stride,
                    frame_size=(args.width, args.height),
                    device="cuda" if device.startswith("cuda") else "cpu",
                )

            def __call__(self, obs: Any) -> Any:
                return self._inner(obs)

            def close(self) -> None:
                if self._inner is not None:
                    self._inner.close()

        driver = _LazyStudent()
        del load_policy

    batch = press_walk(
        checkpoint,
        args.out,
        episodes=args.episodes,
        worlds=args.worlds,
        seed=args.seed,
        frame_every=args.frame_every,
        frame_size=(args.width, args.height),
        device=device,
        feed=feed,
        driver=driver,
        driver_name=driver_name,
    )
    print(f"[walk-press] kept {batch.kept}/{batch.attempts} attempts -> {batch.out}")
    if not batch.complete:
        sys.exit(1)


if __name__ == "__main__":
    main()
