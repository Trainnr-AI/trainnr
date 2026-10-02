"""The locomotion certificate — G4, in C1's shape.

    cd trainnr-mjlab && LD_LIBRARY_PATH=/usr/lib/wsl/lib uv run python -m \\
        trainnr_mjlab.walk_verdict ../runs/microduck-walk/<stamp>/model_7999.pt \\
        [--trials 40] [--device cuda:0|cpu] [--seed 1000]

One seeded episode per trial through the SAME certified env the
checkpoint trained under (full DR, pushes on — the robustness claim is
judged, not a sanitized demo). Per episode, two declared milestones:

- ``survived``: the episode ended by time_out, never by fell_over or
  nan_state.
- ``tracked``: the episode-mean planar velocity error, measured in the
  base frame against the commanded twist, closes at least half the
  gap that standing still would leave — err_ratio = mean‖v - v*‖ /
  max(mean‖v*‖, ERR_FLOOR) < 0.5. Dimensionless, so a gentle command
  is not an easier exam than a brisk one.

Success = survived AND tracked. Every row is an
`trainnr.evaluate.records.EpisodeRecord` — the run's identity as
the stamped source, the mjlab/mujoco/warp/device instrument string,
the drawn command and errors under `variations` — appended to
records.jsonl beside the checkpoint; the fold and the exact
Clopper-Pearson interval land in walk-verdict.json. Run once per
instrument (cuda:0 and cpu — warp's two device backends); the
plain-CPU-MuJoCo cross-check belongs to C4's deployment manifest,
where the policy leaves torch.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trainnr.evaluate.tracking import (
    ERR_FLOOR_MPS,
    ERR_RATIO_BOUND,
    TrackingOutcome,
    criterion_text,
)
from trainnr.mcp_jobs import (
    VIEWPORT_WALK,
    Tracker,
    default_jobs_root,
    jobs_dir_of,
    track,
)
from trainnr.viz import viewer_file

from trainnr_mjlab.envelope import (
    COMMAND_TERM,
    checkpoint_iteration,
    pin_command_envelope,
)
from trainnr_mjlab.walk_export import ACTOR_OBS_GROUP
from trainnr_mjlab.walk_view import (
    fit_of,
    require_same_identity,
    trained_with_cameras,
)
from trainnr_mjlab.walk_view import (
    trained_identity as read_trained_identity,
)
from trainnr_mjlab.walks import DEFAULT_ROBOT, ROBOTS, Identity, use_project, walk_spec

# The judgment's rule lives in `trainnr.evaluate.tracking`, shared
# with the deployment gate; these names stay for the callers here.
ERR_FLOOR = ERR_FLOOR_MPS

# The observation group the actor reads and the capture records: what a
# captured (T, obs) row IS, and the key a labeler must hand back to the
# actor — rsl-rl actors index a TensorDict of groups, never a bare tensor.


@dataclass(frozen=True)
class EpisodeOutcome(TrackingOutcome):
    """The tracking judgment (`trainnr.evaluate.tracking`, shared with
    the sim-to-sim gate) plus what the walk evaluation measures beside
    it: the RMS velocity, acceleration and jerk of the issued targets
    (docs/e2e-research/71 E0) - the smoothness a video shows, as numbers."""

    smoothness: Any = None


@dataclass(frozen=True)
class WorldEpisode:
    """One world's first episode: the judgment plus, when captured, the
    trajectory a dataset writer needs (D2) - the actor's observation
    and the action at every control tick, the qpos to replay for
    frames, and the commanded twist the episode began under."""

    outcome: EpisodeOutcome
    command: list[float]
    observations: Any = None  # (T, obs) float32
    actions: Any = None  # (T, nu) float32
    qpos: Any = None  # (T, nq) float32
    # what the world's own camera sensors saw, by name: (T, H, W, 3) uint8
    frames: dict[str, Any] = field(default_factory=dict)


def instrument_for(device: str) -> str:
    """The stamp every row carries: the batched stack and its device."""
    import mjlab  # noqa: PLC0415
    import mujoco  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    tag = "cuda" if device.startswith("cuda") else "cpu"
    return (
        f"mjlab-{getattr(mjlab, '__version__', '1.6.0')}"
        f"+mujoco-{mujoco.__version__}+warp-{wp.config.version}+{tag}"
    )


def _snap_cameras(
    unwrapped: Any, seen: dict[str, list[list[Any]]], worlds: Any
) -> None:
    """Each named camera sensor's picture, appended to the open worlds'."""
    for name, per_world in seen.items():
        rgb = unwrapped.scene.sensors[name].data.rgb
        if rgb is None:
            raise ValueError(f"camera sensor {name!r} renders no rgb")
        pictures = rgb.detach().cpu().numpy()
        for world in worlds:
            per_world[world].append(pictures[world])


def rollout_episodes(  # noqa: PLR0913, PLR0915 - one rollout loop: its judging and its capture
    env,
    policy,
    trials: int,
    *,
    capture: bool = False,
    max_ticks: int | None = None,
    capture_cameras: tuple[str, ...] = (),
    camera_every: int = 1,
) -> list[WorldEpisode]:
    """One completed episode per world, judged from the live managers:
    the commanded twist from the command manager, the base-frame
    velocity from the entity, the cause of death from the termination
    manager. Worlds that finish early keep stepping (the env auto-
    resets) but only each world's FIRST episode is recorded. With
    `capture`, the per-tick observation/action/qpos of that first
    episode ride along (the rollout->dataset writer's raw material),
    and with `capture_cameras` the named camera sensors' pictures too,
    one every `camera_every` ticks (a scene's head camera, docs/78 E3:
    the dataset keeps one frame in `frame_every`, so the rollout copies
    no more). With `max_ticks`, a world still
    open at that tick is closed as survived so far (the stills tool
    wants one frame, not a verdict)."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from trainnr_mjlab.actuator import as_torch  # noqa: PLC0415

    unwrapped = env.unwrapped
    device = unwrapped.device
    err_sum = torch.zeros(trials, device=device)
    cmd_sum = torch.zeros(trials, device=device)
    steps = torch.zeros(trials, dtype=torch.long, device=device)
    open_worlds = torch.ones(trials, dtype=torch.bool, device=device)
    fell = torch.zeros(trials, dtype=torch.bool, device=device)
    recorded_steps = torch.zeros(trials, dtype=torch.long, device=device)
    device_qpos = as_torch(unwrapped.sim.data.qpos) if capture else None
    trace: list[list[tuple[Any, Any, Any]]] = [[] for _ in range(trials)]
    seen: dict[str, list[list[Any]]] = {
        name: [[] for _ in range(trials)] for name in capture_cameras
    }
    tick = 0

    from trainnr.evaluate.smoothness import SmoothnessMeter  # noqa: PLC0415

    meter = SmoothnessMeter(trials, dt=float(unwrapped.step_dt))
    obs = env.get_observations()  # a TensorDict, not the (obs, extras) pair
    first_command = unwrapped.command_manager.get_command(COMMAND_TERM).clone()
    while open_worlds.any():
        with torch.inference_mode():
            actions = policy(obs)
        act = actions.detach().cpu().numpy()  # one copy per tick, reused below
        still_open = open_worlds.cpu().numpy()
        meter.observe(act, active=still_open)
        if capture and device_qpos is not None:
            actor = obs[ACTOR_OBS_GROUP].detach().cpu().numpy()
            pose = device_qpos.detach().cpu().numpy()
            open_ids = np.flatnonzero(still_open)
            for world in open_ids:
                trace[world].append((actor[world], act[world], pose[world]))
            if seen and tick % camera_every == 0:
                _snap_cameras(unwrapped, seen, open_ids)
        tick += 1
        obs, _, dones, _ = env.step(actions)
        command = unwrapped.command_manager.get_command(COMMAND_TERM)
        velocity = unwrapped.scene["robot"].data.root_link_lin_vel_b
        err = torch.linalg.norm(velocity[:, :2] - command[:, :2], dim=1)
        cmd = torch.linalg.norm(command[:, :2], dim=1)
        err_sum += err * open_worlds
        cmd_sum += cmd * open_worlds
        steps += open_worlds.long()
        closing = open_worlds & dones.bool()
        if max_ticks is not None:
            closing |= open_worlds & (steps >= max_ticks)
        if closing.any():
            fell |= closing & unwrapped.termination_manager.terminated
            recorded_steps = torch.where(closing, steps, recorded_steps)
            open_worlds &= ~closing

    episodes = []
    for i in range(trials):
        outcome = EpisodeOutcome(
            steps=int(recorded_steps[i]),
            fell=bool(fell[i]),
            mean_err=float(err_sum[i] / recorded_steps[i]),
            mean_cmd=float(cmd_sum[i] / recorded_steps[i]),
            smoothness=meter.result(i),
        )
        arrays: dict[str, Any] = {}
        if capture and trace[i]:
            observations, acts, poses = zip(*trace[i], strict=True)
            arrays = {
                "observations": np.stack(observations).astype(np.float32),
                "actions": np.stack(acts).astype(np.float32),
                "qpos": np.stack(poses).astype(np.float32),
            }
            arrays["frames"] = {
                name: np.stack(per_world[i]) for name, per_world in seen.items()
            }
        episodes.append(
            WorldEpisode(outcome, [float(v) for v in first_command[i]], **arrays)
        )
    return episodes


def rollout_outcomes(env, policy, trials: int) -> list[EpisodeOutcome]:
    """The certificate's view: judgments only (no trajectories)."""
    return [episode.outcome for episode in rollout_episodes(env, policy, trials)]


class VerdictFeed:
    """The certificate, watched (docs/66 §0): world 0's chase view while
    a student is judged, every trial's verdict as it lands, the fold at
    the end - into the Studio's one ingest address. Best-effort: no
    rerun-sdk means one loud line and an unwatched verdict."""

    ROOT = "verdict"

    def __init__(self, run_name: str, file: Path | None = None) -> None:
        import rerun as rr  # noqa: PLC0415 - viz extra
        from trainnr.viz import STUDIO_ADDRESS, open_stream  # noqa: PLC0415

        self._rr: Any = rr
        open_stream(f"trainnr-verdict-{run_name}", address=STUDIO_ADDRESS, file=file)
        self._tick = 0

    @classmethod
    def connect(cls, run_name: str, file: Path | None = None) -> VerdictFeed | None:
        try:
            return cls(run_name, file)
        except ImportError:
            print(
                "no rerun-sdk in this venv - the verdict runs UNWATCHED",
                file=sys.stderr,
            )
            return None

    def frame(self, image: Any) -> None:
        self._tick += 1
        self._rr.set_time("tick", sequence=self._tick)
        self._rr.log(f"{self.ROOT}/camera", self._rr.Image(image))

    def outcomes(self, outcomes: list[EpisodeOutcome]) -> None:
        rr = self._rr
        # The trial rows live on THEIR clock only: without the reset they
        # also carried the last tick and the viewer stacked all eight on
        # one x (the operator's screenshot, 2026-09-03).
        rr.reset_time()
        for trial, outcome in enumerate(outcomes):
            rr.set_time("trial", sequence=trial)
            rr.log(f"{self.ROOT}/err_ratio", rr.Scalars(outcome.err_ratio))
            rr.log(f"{self.ROOT}/survived", rr.Scalars(float(outcome.survived)))
            verdict = (
                "PASS" if outcome.success else ("fell" if outcome.fell else "drifted")
            )
            rr.log(
                f"{self.ROOT}/log",
                rr.TextLog(
                    f"trial {trial}: {verdict} (err_ratio {outcome.err_ratio:.2f})"
                ),
            )

    def done(self, line: str) -> None:
        self._rr.log(f"{self.ROOT}/log", self._rr.TextLog(line))


def blanked(images: Any) -> Any:
    """The camera, blanked: zeros of the frames' own shape and dtype.
    The control the campaign 3 plan asks first (docs/07 2026-09-03):
    a student judged WITHOUT its image, with the full state still in
    hand — if the score holds, the image was never used, and the next
    distill trains without the privileged state."""
    import numpy as np  # noqa: PLC0415

    return np.zeros_like(images)


CAMERA_SIGHTED = "chase"
CAMERA_BLANKED = "blanked"


STATE_GIVEN = "actor"
STATE_BLANKED = "blanked"


def verdict_suffix(  # noqa: PLR0913 - every control that earns its own file, named
    devicetag: str,
    *,
    student: bool,
    blank_camera: bool,
    blank_state: bool = False,
    latency: int = 0,
    delay: int = 0,
) -> str:
    """The certificate file's suffix — a control run must never land on
    the sighted run's file (`walk-verdict-student-cuda.json` is a
    campaign's headline number): `blank-` for the camera control,
    `blank-state-` for the state control, `blank-both-` for both,
    `latency-N-` for a run under an inference budget."""
    if not student:
        return teacher_suffix(devicetag, delay)
    control = {
        (False, False): "",
        (True, False): "blank-",
        (False, True): "blank-state-",
        (True, True): "blank-both-",
    }[(blank_camera, blank_state)]
    budget = f"latency-{latency}-" if latency else ""
    return f"student-{control}{budget}{devicetag}"


def teacher_suffix(devicetag: str, delay: int = 0) -> str:
    """The teacher's file, its own when judged under an action delay."""
    return f"delay-{delay}-{devicetag}" if delay else devicetag


class StudentPolicy:
    """A vision student (a LeRobot checkpoint) as the env's per-tick
    policy: the SAME chase camera the press wrote its frames with
    (walk_press.ChaseCamera, off a one-robot CPU mirror) renders every
    world's view from the live qpos, the actor observation rides along
    as the state, and the bridge answers from the train venv."""

    def __init__(  # noqa: PLR0913 - the student's six facts, named
        self,
        env,
        checkpoint: Path,
        *,
        python: Path,
        horizon: int,
        stride: int,
        frame_size: tuple[int, int],
        device: str,
        blank_camera: bool = False,
        blank_state: bool = False,
        latency: int = 0,
    ) -> None:
        from trainnr.envs.policy_bridge import BridgePolicy  # noqa: PLC0415

        from trainnr_mjlab.actuator import as_torch  # noqa: PLC0415
        from trainnr_mjlab.walk_press import CAMERA_KEY, ChaseCamera  # noqa: PLC0415

        self._env = env
        self._qpos = as_torch(env.unwrapped.sim.data.qpos)
        self._camera = ChaseCamera(*frame_size)
        self._bridge = BridgePolicy(
            python,
            checkpoint,
            camera=CAMERA_KEY,
            horizon=horizon,
            device=device,
            latency=latency,
        )
        self._first = True
        self.feed: VerdictFeed | None = None
        self._ticks = 0
        # The student's clock is the DATASET's: frames every `stride`
        # control ticks (walk_press --frame-every), so a chunk step is
        # `stride` ticks long. Asked every tick, the first student was
        # driven 5x too fast and fell in 18 ticks (2026-09-02).
        self._stride = stride
        self._held: Any = None
        self._blank = blank_camera
        # The complementary control (2026-09-04): the camera blanked
        # collapsed the student to 0/40, which proves the image is USED,
        # not that the state is unused — a blank frame is out of
        # distribution. Zeroing the state instead says whether the
        # student is purely visual.
        self._blank_state = blank_state

    def __call__(self, obs) -> Any:
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        if self._held is not None and self._ticks % self._stride:
            self._ticks += 1
            return self._held
        state = obs[ACTOR_OBS_GROUP].detach().cpu().numpy()
        if self._blank_state:
            state = blanked(state)
        qpos = self._qpos.detach().cpu().numpy()
        images = np.stack([self._camera.frame_at(row) for row in qpos])
        if self._blank:
            images = blanked(images)
        if self.feed is not None:
            self.feed.frame(images[0])
        resets = np.full(len(state), self._first, dtype=bool)
        self._first = False
        actions = self._bridge.act(state, images, resets)
        self._held = torch.as_tensor(actions, device=self._env.unwrapped.device)
        self._ticks += 1
        return self._held

    def close(self) -> None:
        self._bridge.close()
        self._camera.close()


# `--judge-in-fit declared`: the robot as the vendor declares it, no fit.
JUDGE_DECLARED = "declared"


def judged_fit(judge_in_fit: str | None, trained: dict[str, Any]) -> str | None:
    """The fit the judged world is built with: the policy's own (a normal
    certificate), another fit's stamp, or none for the declared robot."""
    if judge_in_fit is None:
        return fit_of(trained)
    return None if judge_in_fit == JUDGE_DECLARED else judge_in_fit


def cross_identity(
    identity: dict[str, Any], trained: dict[str, Any], judged: dict[str, Any]
) -> dict[str, Any]:
    """The record of a cross-evaluation: the trained fit and its basis under
    `fit` / `fit_basis` (absent when it trained on declared numbers), the
    judged world under `judged_in_fit` / `judged_fit_basis` - read from
    the environment as BUILT (`judged`), never as typed - so no key stands
    for both worlds."""
    record = {
        k: v for k, v in identity.items() if k not in (Identity.FIT, Identity.FIT_BASIS)
    }
    for key in (Identity.FIT, Identity.FIT_BASIS):
        if trained.get(key):
            record[key] = trained[key]
    record[Identity.JUDGED_IN_FIT] = judged.get(Identity.FIT) or JUDGE_DECLARED
    if judged.get(Identity.FIT_BASIS):
        record[Identity.JUDGED_FIT_BASIS] = judged[Identity.FIT_BASIS]
    return record


def require_another_world(trained: dict[str, Any], judged: dict[str, Any]) -> None:
    """A cross-evaluation in the world the policy trained in is its own
    certificate under another name: refused."""
    own = fit_of(trained) or JUDGE_DECLARED
    if (judged.get(Identity.FIT) or JUDGE_DECLARED) == own:
        raise SystemExit(
            f"--judge-in-fit names {own}, the world this policy trained in: "
            "that is its own certificate, judge it without --judge-in-fit"
        )


VERDICT_KIND = "evaluate-walk"  # the job table's kind, the door's own tool name


def verdict_viewer(checkpoint: Path) -> Path:
    """The saved stream of a judgment, in the run the checkpoint belongs to."""
    return viewer_file(checkpoint.parent, f"verdict-{checkpoint.stem}")


def verdict_job_name(args: argparse.Namespace) -> str:
    """What a judgment is called while it runs: the run, and the world
    when it is not the policy's own (a cross, a scaled law, a delay)."""
    words = [args.checkpoint.parent.name]
    if args.student is not None:
        words.append(f"student {args.student.parent.name}")
    if args.judge_in_fit is not None:
        words.append(f"in {args.judge_in_fit}")
    if args.judge_at_scale is not None:
        axis = "" if args.judge_param == "all" else f"{args.judge_param} "
        words.append(f"at {axis}x{args.judge_at_scale:g}")
    if args.delay:
        words.append(f"delay {args.delay}")
    return " ".join(words)


def cross_world_word(judge_in_fit: str) -> str:
    """The judged world as a file-name part: `declared`, or `fit-<hash>`."""
    return judge_in_fit.replace("@", "-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="a project root: its robots are searched first (the Go2 lives there)",
    )
    parser.add_argument(
        "--robot",
        choices=ROBOTS,
        default=DEFAULT_ROBOT,
        help="which walk (trainnr_mjlab.walks) the checkpoint belongs to",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=None,
        help="the captured scene the checkpoint trained on (a project's "
        "scenes/<name>): judged on it, the protocol naming it (docs/78 E2)",
    )
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--device", default=None, help="cuda:0 or cpu")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--student",
        type=Path,
        default=None,
        help="a LeRobot checkpoint (pretrained_model dir): judge the vision "
        "student distilled from this teacher's data instead of the teacher",
    )
    parser.add_argument(
        "--horizon", type=int, default=2, help="executed chunk steps before a replan"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=0,
        help="the TEACHER's action delay in control ticks (its delay margin, the "
        "control for the student's --latency; docs/e2e-research/71 E1)",
    )
    parser.add_argument(
        "--latency",
        type=int,
        default=0,
        help="the student's inference budget in control ticks (docs/e2e-research/71 "
        "E1): its chunk takes over that many ticks after it was asked; 0 = the "
        "synchronous loop",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="control ticks per chunk step: the press's --frame-every (1 for "
        "this gait - held 5 ticks even the teacher falls, measured 2026-09-02)",
    )
    parser.add_argument(
        "--student-python",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "trainnr"
        / ".venv-train"
        / "bin"
        / "python",
        help="the interpreter with LeRobot (the bridge's server side)",
    )
    parser.add_argument("--frame-width", type=int, default=320)
    parser.add_argument("--frame-height", type=int, default=240)
    parser.add_argument(
        "--judge-at-scale",
        type=float,
        default=None,
        help="the envelope probe: judge with every law parameter pinned at "
        "fit x SCALE (no draw); the certificate file carries the scale",
    )
    parser.add_argument(
        "--judge-span",
        type=float,
        default=None,
        help="judge under law DR drawn from a declared ±SPAN around the fit "
        "(the mismatch matrix's span columns); without it the non-at-fit "
        "certificate judges under LAW_DR_SPAN for every policy, whatever "
        "span it trained under — NOT each policy's own span (2026-09-04)",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="the actuator bundle the checkpoint trained on (microduck only); the "
        "identity check refuses a mismatch",
    )
    parser.add_argument(
        "--judge-param",
        default="all",
        help="with --judge-at-scale: pin only this mismatch axis (the walk's "
        "PIN_AXES, e.g. kt / R / friction on the microduck, kp / kd / armature "
        "on the Go1); 'all' moves every parameter together",
    )
    parser.add_argument(
        "--judge-in-fit",
        default=None,
        metavar="FIT|declared",
        help="a CROSS-evaluation: judge the policy in another robot world - a "
        "joints fit's (`fit@<stamp>`) or the declared constants (`declared`) - "
        "instead of the one it trained in. Recorded as such (its own suffix, "
        "`judged_at` says both worlds); never a certificate of the policy in "
        "its own world",
    )
    parser.add_argument(
        "--judge-at-fit",
        action="store_true",
        help="judge with NO law DR — the bundle's point fit exactly, the pinned "
        "truth the walk C1 study's arms share (pushes stay on)",
    )
    parser.add_argument(
        "--blank-state",
        action="store_true",
        help="judge the student with its state vector zeroed (image still "
        "given): the complement of --blank-camera",
    )
    parser.add_argument(
        "--blank-camera",
        action="store_true",
        help="judge the student with its image zeroed (state still given): "
        "the control that says whether the image is used at all",
    )
    parser.add_argument(
        "--no-studio", action="store_true", help="do not stream to the Studio"
    )
    return parser.parse_args()


def write_certificate(  # noqa: PLR0913 - every fact of one certificate, named
    out_dir: Path,
    suffix: str,
    outcomes: list[EpisodeOutcome],
    *,
    source: str,
    policy_name: str,
    instrument: str,
    protocol: dict[str, Any],
    identity: dict[str, Any],
    trials: int,
) -> None:
    """The fold and the exact interval, printed and written beside the rows."""
    from trainnr.stats.intervals import clopper_pearson  # noqa: PLC0415

    survived = sum(o.survived for o in outcomes)
    tracked = sum(o.tracked for o in outcomes)
    successes = sum(o.success for o in outcomes)
    low, high = clopper_pearson(successes, trials)
    row = {
        "source": source,
        "policy": policy_name,
        "instrument": instrument,
        "protocol": protocol,
        "identity": identity,
        "funnel": {"survived": survived, "tracked": tracked},
        "successes": successes,
        "trials": trials,
        "ci95": [round(low, 4), round(high, 4)],
        "median_err_ratio": round(
            sorted(o.err_ratio for o in outcomes)[len(outcomes) // 2], 4
        ),
    }
    verdict_path = out_dir / f"walk-verdict-{suffix}.json"
    if verdict_path.exists():
        # A certificate is never silently replaced: an 8-trial smoke run
        # once overwrote the flagship's 40-trial file (found 2026-09-04
        # by an audit reading the rows). The rows are appended and stay
        # the primary artifact; the previous file is kept beside the new.
        stamp_prev = json.loads(verdict_path.read_text())
        # Named by what it judged too: two checkpoints of one run judged
        # under the same suffix would otherwise share a backup name, and
        # the second rotation would overwrite the first (2026-09-11) - and
        # by how: the same checkpoint judged under another protocol or on
        # another instrument keeps its own backup (2026-09-22, E0).
        from trainnr.bundles.hashing import fields_hash  # noqa: PLC0415
        from trainnr.project.importer import PROTOCOL_HASH_CHARS  # noqa: PLC0415

        protocol_prev = stamp_prev.get("protocol") or {}
        previous = out_dir / (
            f"walk-verdict-{suffix}.{stamp_prev.get('policy', 'policy')}"
            f".seed{protocol_prev.get('seed')}"
            f".n{stamp_prev['trials']}"
            f".p{fields_hash(protocol_prev)[:PROTOCOL_HASH_CHARS]}.json"
        )
        if not previous.exists():
            verdict_path.rename(previous)
    verdict_path.write_text(json.dumps(row, indent=1))
    print(
        f"[verdict] survived {survived}/{trials}, tracked "
        f"{tracked}/{trials} -> success {successes}/{trials}, "
        f"CP95 [{low:.3f}, {high:.3f}]"
    )
    print(f"[verdict] rows in {out_dir}")


def main() -> None:
    """One judgment, in the project's job table while it runs: the Studio's
    Running now names it by the run and the world it is judged in."""
    args = parse_args()
    with track(
        VERDICT_KIND,
        jobs_dir=jobs_dir_of(args.project or default_jobs_root()),
        name=verdict_job_name(args),
        viewport=VIEWPORT_WALK,
        # no stream is written under --no-studio: point at none, not at an
        # older judgment's file
        viewer="" if args.no_studio else str(verdict_viewer(args.checkpoint)),
    ) as run:
        judge(args, run)


def judge(args: argparse.Namespace, run: Tracker) -> None:  # noqa: PLR0912, PLR0915 - the certificate's whole procedure, in order
    """The certificate's procedure; `run` is the job table's tracker."""
    import warp as wp  # noqa: PLC0415

    wp.init()
    device = args.device or ("cuda:0" if wp.is_cuda_available() else "cpu")

    from dataclasses import asdict  # noqa: PLC0415

    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: PLC0415
    from trainnr.bundles.hashing import fields_hash  # noqa: PLC0415
    from trainnr.evaluate.records import (  # noqa: PLC0415
        EpisodeRecord,
        append_records,
    )

    if args.judge_at_fit and args.judge_span is not None:
        raise SystemExit("--judge-at-fit and --judge-span exclude each other")
    use_project(args.project)
    spec = walk_spec(args.robot)
    default_span = spec.default_span
    judge_span = default_span if args.judge_span is None else args.judge_span
    if args.judge_param != "all" and args.judge_at_scale is None:
        raise SystemExit("--judge-param needs --judge-at-scale")
    trained_identity: Any = read_trained_identity(args.checkpoint)
    cfg, identity = spec.env_cfg(
        dr_span=None if args.judge_at_fit else judge_span,
        pin_scale=args.judge_at_scale,
        pin_axis=args.judge_param,
        bundle=args.bundle,
        # A policy trained with its head pinned is judged with it pinned:
        # the action space is part of what the run's identity records.
        head=str(trained_identity.get(Identity.HEAD, "free")),
        scene=args.scene,
        # the actor sees a camera exactly when it trained with one (the
        # scene walk's picture: 12,288 inputs a plain actor never had)
        cameras=trained_with_cameras(trained_identity),
        fit=judged_fit(args.judge_in_fit, trained_identity),
    )
    # The one gate (walk_view): robot, actuator and ground must match; the
    # DR basis may differ on purpose (a policy trained under one span is
    # judged at the fit), and the certificate records both. A cross-
    # evaluation moves the fit on purpose: the gate then holds every other
    # key and the record says which world it judged in.
    require_same_identity(
        {**trained_identity, Identity.FIT: identity.get(Identity.FIT)}
        if args.judge_in_fit is not None
        else trained_identity,
        identity,
    )
    # the judged world as built, before the trained identity is merged in
    judged = {k: identity.get(k) for k in (Identity.FIT, Identity.FIT_BASIS)}
    if args.judge_in_fit is not None:
        require_another_world(trained_identity, judged)
    cfg.scene.num_envs = args.trials
    cfg.seed = args.seed
    if trained_identity:
        identity = {
            **identity,
            "task": trained_identity.get("task"),
            "trained_dr_basis": trained_identity.get("dr_basis"),
            # the fit the policy trained under and whose robot it measured
            # (2026-09-25); absent for a run that trained on declared numbers
            **{
                k: trained_identity[k]
                for k in (Identity.FIT, Identity.FIT_BASIS)
                if trained_identity.get(k)
            },
            # The training run's seed (walk_train --seed; mjlab's default
            # 42 when the run predates the knob) — a replicate's name.
            "seed": trained_identity.get("seed", 42),
        }
    if args.judge_in_fit is not None:
        # a cross record names both worlds: `fit` stays the one the policy
        # trained under (none for declared), `judged_in_fit` the one it ran in
        identity = cross_identity(identity, trained_identity, judged)

    devicetag = "cuda" if device.startswith("cuda") else "cpu"
    instrument = instrument_for(device)
    # The environment the certificate cites: the declared task the run was
    # trained against (its content stamp, the project's environment card)
    # when the run recorded one; else the walk spec's identity hash.
    source = identity.get("task") or f"{spec.source_prefix}@{fields_hash(identity)}"
    agent = spec.agent(1)
    # The commands the checkpoint trained under, not the curriculum's
    # first stage a fresh env would restart at (trainnr_mjlab.envelope).
    envelope = pin_command_envelope(
        cfg, checkpoint_iteration(args.checkpoint.stem), agent.num_steps_per_env
    )
    protocol = {
        "trials": args.trials,
        "seed": args.seed,
        "criterion": criterion_text(),
        "err_ratio_bound": ERR_RATIO_BOUND,
        "err_floor_mps": ERR_FLOOR,
        "dr_basis": identity["dr_basis"],
        "commands": envelope["commands"],
        "command_basis": envelope["basis"],
        # The instrument that measured it: a judgment on another MuJoCo or
        # mujoco_warp is another evaluation, and both stand (E0, docs/78:
        # without this a 3.13 re-judge took the 3.11 certificate's name).
        "instrument": instrument,
    }
    if identity.get(Identity.SCENE):
        # judged on the captured scene it trained on: another protocol,
        # another certificate name (the hash below), never the plane's
        protocol[Identity.SCENE] = identity[Identity.SCENE]
        protocol[Identity.TERRAIN] = identity.get(
            Identity.TERRAIN, "the scene's heightfield"
        )
    print(f"[verdict] {source} on {instrument}, {args.trials} trials")
    print(f"[verdict] commands: {envelope['commands']} ({envelope['basis']})")

    run.stage(f"building {args.trials} worlds on {device}")
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg, device=device), clip_actions=agent.clip_actions
    )
    runner = MjlabOnPolicyRunner(env, asdict(agent), log_dir=None, device=device)
    runner.load(
        str(args.checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy: Any = runner.get_inference_policy(device=device)
    if args.delay:
        # The teacher's delay margin, the way the student's budget is
        # measured (docs/e2e-research/71 E1): its action from the
        # observation at t reaches the robot at t + delay.
        from trainnr.evaluate.scheduler import Delayed  # noqa: PLC0415

        policy = Delayed(policy, args.delay)
        protocol["delay"] = args.delay
    policy_name = args.checkpoint.stem
    if args.student is not None:
        # The student's identity: its checkpoint directory's content hash
        # (the same stamp a bundle gets), named by its run step.
        from trainnr.bundles.hashing import stamp as stamp_of  # noqa: PLC0415

        policy = StudentPolicy(
            env,
            args.student,
            python=args.student_python,
            horizon=args.horizon,
            stride=args.stride,
            frame_size=(args.frame_width, args.frame_height),
            device=devicetag,
            blank_camera=args.blank_camera,
            blank_state=args.blank_state,
            latency=args.latency,
        )
        policy_name = stamp_of(f"student-{args.student.parent.name}", args.student)
        protocol["student"] = {
            "teacher": args.checkpoint.stem,
            "horizon": args.horizon,
            "stride": args.stride,
            "latency": args.latency,
            # Every row says what the student SAW: the camera, or nothing.
            "camera": CAMERA_BLANKED if args.blank_camera else CAMERA_SIGHTED,
            "state": STATE_BLANKED if args.blank_state else STATE_GIVEN,
        }
    feed = (
        None
        if args.no_studio
        else VerdictFeed.connect(
            policy_name.split("@")[0],
            # The saved stream lands in the run the checkpoint belongs to.
            verdict_viewer(args.checkpoint),
        )
    )
    if isinstance(policy, StudentPolicy):
        policy.feed = feed
    run.stage(f"walking {args.trials} paired trials")
    outcomes = rollout_outcomes(env, policy, args.trials)
    if isinstance(policy, StudentPolicy):
        policy.close()
    env.close()
    if feed is not None:
        feed.outcomes(outcomes)

    records = [
        EpisodeRecord(
            source=source,
            policy=policy_name,
            trial=trial,
            success=outcome.success,
            steps=outcome.steps,
            instrument=instrument,
            protocol=protocol,
            seed=args.seed,
            events=(
                {"index": 0, "name": "survived", "passed": outcome.survived},
                {"index": 1, "name": "tracked", "passed": outcome.tracked},
            ),
            variations={
                "mean_err_mps": round(outcome.mean_err, 4),
                "mean_cmd_mps": round(outcome.mean_cmd, 4),
                "err_ratio": round(outcome.err_ratio, 4),
                **(outcome.smoothness.columns() if outcome.smoothness else {}),
            },
        )
        for trial, outcome in enumerate(outcomes)
    ]
    out_dir = args.checkpoint.parent / "verdict"
    out_dir.mkdir(exist_ok=True)
    suffix = verdict_suffix(
        devicetag,
        student=args.student is not None,
        blank_camera=args.blank_camera,
        blank_state=args.blank_state,
        latency=args.latency if args.student is not None else 0,
        delay=args.delay if args.student is None else 0,
    )
    if args.judge_at_fit:
        suffix = f"at-fit-{suffix}"
        protocol["judged_at"] = "the bundle's point fit (no law DR)"
    elif args.judge_span is not None:
        suffix = f"under-pm{args.judge_span:g}-{suffix}"
        protocol["judged_at"] = f"law DR: {identity['dr_basis']}"
    else:
        protocol["judged_at"] = f"law DR: {identity['dr_basis']}"
    if args.judge_at_scale is not None:
        axis = "" if args.judge_param == "all" else f"{args.judge_param}-"
        suffix = f"at-x{args.judge_at_scale:g}-{axis}{suffix}"
        moved = "every law parameter" if args.judge_param == "all" else args.judge_param
        protocol["judged_at"] = f"{moved} at fit x {args.judge_at_scale:g}"
    if args.judge_in_fit is not None:
        # last, so it wraps the rung it may be combined with (a cliff judged
        # in another world keeps both names)
        world = identity[Identity.JUDGED_IN_FIT]
        suffix = f"in-{cross_world_word(world)}-{suffix}"
        protocol["judged_at"] = (
            f"CROSS-evaluation: trained in "
            f"{fit_of(trained_identity) or JUDGE_DECLARED}, judged in {world}; "
            f"{protocol['judged_at']}"
        )
    append_records(out_dir / f"records-{suffix}.jsonl", records)

    run.stage(f"{sum(o.success for o in outcomes)}/{len(outcomes)} walked")
    write_certificate(
        out_dir,
        suffix,
        outcomes,
        source=source,
        policy_name=policy_name,
        instrument=instrument,
        protocol=protocol,
        identity=identity,
        trials=args.trials,
    )


if __name__ == "__main__":
    main()
