"""The trained walk policy in the Studio's simulator — the RL view.

    cd rq_mjlab && LD_LIBRARY_PATH=/usr/lib/wsl/lib MUJOCO_GL=egl \\
        uv run python -m rq_mjlab.walk_view --latest [--envs 9] [--shm=PATH]

The batched mjlab env steps N policy-driven worlds (GPU where there is
one, CPU here on the Mac at about a third of real time for nine). This
process is the PHYSICS side of the Studio's two-process stream
(docs/76 §10.2): it creates the state ring for a mirror model — one
copy of the walk robot per world on one ground plane, the same model
`tools/studio-render-stream.py` builds by recipe — spawns that script
as the render process on the Studio's own stdin and stdout, and each
control step publishes every world's qpos plus per-world reward and
done into the ring. The renderer draws at the display's rate, follows
a world when asked, and reports the worlds in its status.

Ctrl+drag shoves a POLICY-DRIVEN duck for real: the gesture resolves on
the renderer's copy (mjv_select / mjvPerturb), the wrench comes back
through the ring, and it lands in the batched sim's `xfrc_applied` for
that world every step — the policy must recover, live.

The env is built from the CHECKPOINT's identity (its actuator bundle and
its DR basis), and the identity gate then holds exactly as walk_play and
walk_verdict hold it: a checkpoint whose identity this env cannot
reproduce is refused by name.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OVERVIEW_HZ = 10.0  # the per-world markers into the Studio's viewer


def transport():
    """The viewport transport (StateRing, PhysicsPump, the mirror scene)
    from its one home — a hyphenated tool file, hence importlib."""
    tools = REPO / "tools"
    sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location(
        "studio_render_stream", tools / "studio-render-stream.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Slot:
    """What mjlab's velocity term reads off a viser handle: `.value`."""

    def __init__(self, value: float | bool) -> None:
        self.value = value


class Joystick:
    """The Studio's Commands tab on mjlab's own compute-time override:
    the velocity term's viser joystick (`create_gui` in mjlab's
    tasks/velocity/mdp/velocity_command.py) reads an enable handle,
    three slider handles and an env index at every `compute`, and
    writes the twist into the command the policy observes. The same
    three attributes, fed from the ring's mailbox instead of viser -
    nothing of the term's own logic is copied."""

    HANDLES = ("_joystick_enabled", "_joystick_sliders", "_joystick_get_env_idx")

    def __init__(self, term, axes: int) -> None:
        missing = [h for h in self.HANDLES if not hasattr(term, h)]
        if missing:
            raise SystemExit(
                f"mjlab's velocity term has no {missing}: the Commands tab drives "
                "its viewer joystick, which this mjlab does not have"
            )
        self.axes = axes
        self.world = 0
        self.enabled = _Slot(False)
        self.sliders = [_Slot(0.0) for _ in range(axes)]
        # mjlab's viewer seam: the three handles its `compute` reads.
        term._joystick_enabled = self.enabled
        term._joystick_sliders = self.sliders
        term._joystick_get_env_idx = lambda: self.world

    def on_command(self, name: str, arg_i: int, arg_f: float) -> None:
        """The mailbox's `twist`: arg_i = world * axes + axis, or -1 to
        hand the commands back to the task."""
        if name != "twist":
            return
        if arg_i < 0:
            self.enabled.value = False
            return
        self.world, axis = divmod(arg_i, self.axes)
        self.sliders[axis].value = arg_f
        if not self.enabled.value:
            print(
                f"[walk-view] w{self.world} commanded from the Studio", file=sys.stderr
            )
        self.enabled.value = True


def twist_switch(env, axes: int) -> tuple[object | None, str]:
    """The walk's velocity command term and its bounds as the render
    stream's `--twist-ranges` value (lo,hi per axis, the envelope's
    first `axes` keys). A walk without the term gets no Commands tab:
    (None, "")."""
    from rq_mjlab.envelope import (  # noqa: PLC0415
        COMMAND_TERM,
        RANGE_KEYS,
        command_ranges,
    )

    if COMMAND_TERM not in env.command_manager.active_terms:
        return None, ""
    ranges = command_ranges(env.cfg)
    bounds = ",".join(f"{lo},{hi}" for lo, hi in (ranges[k] for k in RANGE_KEYS[:axes]))
    return env.command_manager.get_term(COMMAND_TERM), bounds


def latest_checkpoint(project: Path | None = None) -> Path:
    """The newest model_*.pt: under the project's runs when a project is
    given (the Studio's walk scene rolls the project's latest policy),
    else under runs/microduck-walk or the flagship study's arms."""
    if project is not None:
        # A policy artifact first (a judged checkpoint, copied there by
        # the live loop), else any run's checkpoint; newest by time.
        root = Path(project)
        for pattern in ("policies/*/model_*.pt", "runs/*/model_*.pt"):
            checkpoints = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
            if checkpoints:
                return checkpoints[-1]
        raise SystemExit(
            f"no model_*.pt under {project}/policies or runs - train a walk first"
        )
    checkpoints = sorted(
        [
            *REPO.glob("runs/microduck-walk/*/model_*.pt"),
            *REPO.glob("docs/artifacts/walk-c1/*/train/model_*.pt"),
        ],
        key=lambda p: p.stat().st_mtime,
    )
    if not checkpoints:
        raise SystemExit(
            "no checkpoint under runs/microduck-walk or docs/artifacts/walk-c1 - "
            "train one (walk_train) or pull a run from the pod first"
        )
    return checkpoints[-1]


def project_walk_robot(project: Path) -> str:
    """The robot of the project's one declared walk (its task's family
    through the registry), the same rule the doors apply; refused by
    name when the project declares none or several."""
    import json  # noqa: PLC0415

    from rq_pipeline.tasks.walks import walk_robot  # noqa: PLC0415

    found = []
    for task_file in sorted(Path(project).glob("tasks/*/task.json")):
        try:
            robot = walk_robot(
                str(json.loads(task_file.read_text()).get("task_id", ""))
            )
        except KeyError:
            robot = None
        if robot:
            found.append((task_file.parent.name, robot))
    if len(found) != 1:
        raise SystemExit(
            f"{project}: {len(found)} declared walks {[n for n, _ in found]}; "
            "name the robot with --robot"
        )
    return found[0][1]


def dr_span_of(identity: dict) -> float:
    """The DR span the checkpoint was trained under, from its identity's
    basis string: none (a point fit), or a caller-declared ±span. An
    identified-set basis needs the bootstrap replicates and is refused."""
    basis = str(identity.get("dr_basis", ""))
    if basis.startswith("none"):
        return 0.0
    match = re.search(r"span ±([0-9.]+)", basis)
    if match:
        return float(match.group(1))
    raise SystemExit(
        f"this view cannot rebuild the env for basis {basis!r}; "
        "run a point or caller-declared-span checkpoint"
    )


def same_identity(trained: dict, env: dict) -> bool:
    """The gate: robot and actuator bundle byte-for-byte; the DR basis by
    what it MEANS (the same span, or both a point fit), since the basis
    string's wording changed on 2026-09-06 ("(bundle is point estimates)"
    became "around the bundle's point") and a checkpoint trained before
    that is the same physics. A wording difference is said on stderr."""
    for key in ("robot", "actuator"):
        if trained.get(key) != env.get(key):
            return False
    a, b = str(trained.get("dr_basis", "")), str(env.get("dr_basis", ""))
    if a != b:
        try:
            same = dr_span_of({"dr_basis": a}) == dr_span_of({"dr_basis": b})
        except SystemExit:
            return False
        if same:
            print(
                f"[walk-view] basis wording differs, same span: "
                f"trained {a!r}, env {b!r}",
                file=sys.stderr,
                flush=True,
            )
        return same
    return True


def body_maps(mirror, env_model) -> tuple[dict[int, int], dict[int, int]]:
    """Mirror body id -> (world, device body id): the shove's routing.
    Mirror bodies are named `wNN/<name>`; the device model carries the
    same robot once, possibly under the entity's own prefix."""
    world_of, device_of = {}, {}
    for body in range(1, mirror.nbody):
        name = mirror.body(body).name
        if "/" not in name:
            continue
        prefix, _, local = name.partition("/")
        for candidate in (local, f"robot/{local}"):
            try:
                device_of[body] = env_model.body(candidate).id
                world_of[body] = int(prefix.removeprefix("w"))
                break
            except KeyError:
                continue
    return world_of, device_of


def load_policy(checkpoint: Path, envs: int, device: str, robot: str = "microduck"):
    """The env built from the checkpoint's identity, and its inference
    policy, identity-gated (the same door walk_play and walk_verdict use)."""
    from dataclasses import asdict  # noqa: PLC0415

    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: PLC0415

    from rq_mjlab.walks import walk_spec  # noqa: PLC0415

    trained_file = checkpoint.parent / "identity.json"
    trained = json.loads(trained_file.read_text()) if trained_file.is_file() else {}
    spec = walk_spec(robot)
    # The walk in play mode at the nominal point, as walk_play rolls it:
    # a view, not a judgment, so the trained DR basis is not rebuilt
    # (the Go2's declared-constants basis has no span to parse, and this
    # gate refused it, 2026-09-12). Robot and actuator must still match.
    from rq_mjlab.walk_export import (  # noqa: PLC0415
        ACTOR_OBS_GROUP,
        trained_actor_width,
    )

    cfg, identity = spec.env_cfg(play=True, dr_span=None, pin_scale=None)
    for key in ("robot", "actuator"):
        if trained.get(key) not in (None, identity.get(key)):
            raise SystemExit(
                f"identity mismatch on {key}: this env is {identity}, trained {trained}"
            )
    cfg.scene.num_envs = envs
    agent = spec.agent(1)
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg, device=device), clip_actions=agent.clip_actions
    )
    # A checkpoint from the recipe before the actor became deployable
    # (48 terms against 47, 2026-09-11) is still a policy to watch: the
    # env is rebuilt with that actor, by name, never a shape traceback.
    trained_width = trained_actor_width(checkpoint, device)
    built_width = int(env.get_observations()[ACTOR_OBS_GROUP].shape[-1])
    if trained_width is not None and trained_width != built_width:
        env.close()
        try:
            cfg, _ = spec.env_cfg(
                play=True, dr_span=None, pin_scale=None, legacy_actor=True
            )
        except TypeError as error:
            raise SystemExit(
                f"{checkpoint.name}: actor observes {trained_width} terms, this "
                f"walk builds {built_width}, and {robot!r} has no earlier recipe"
            ) from error
        cfg.scene.num_envs = envs
        env = RslRlVecEnvWrapper(
            ManagerBasedRlEnv(cfg, device=device), clip_actions=agent.clip_actions
        )
        print(
            f"[walk-view] {checkpoint.name}: the earlier actor recipe "
            f"({trained_width} terms)",
            file=sys.stderr,
            flush=True,
        )
    runner = MjlabOnPolicyRunner(env, asdict(agent), log_dir=None, device=device)
    runner.load(
        str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    return env, runner.get_inference_policy(device=device)


class Overview:
    """The worlds in the Studio's viewer: one marker per world at its
    root, coloured by reward (green good, red fallen), and a reward
    series per world — the many-worlds overview the picture cannot be."""

    def __init__(self, worlds: int) -> None:
        import rerun as rr  # noqa: PLC0415
        import rerun.blueprint as rrb  # noqa: PLC0415

        self.rr = rr
        self.worlds = worlds
        from rq_pipeline.viz import leave_cleanly_on_term  # noqa: PLC0415

        rr.init("robotiq-walk-worlds", spawn=False)
        rr.connect_grpc()
        leave_cleanly_on_term(rr)
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial3DView(origin="worlds", name="worlds"),
                    rrb.TimeSeriesView(origin="worlds/reward", name="reward per world"),
                    column_shares=[2, 1],
                ),
                collapse_panels=True,
            )
        )
        for w in range(worlds):
            rr.log(f"worlds/reward/w{w}", rr.SeriesLines(names=[f"w{w}"]), static=True)
        self.last = 0.0

    def log(self, sim_time: float, roots, rewards, dones) -> None:
        now = time.monotonic()
        if now - self.last < 1.0 / OVERVIEW_HZ:
            return
        self.last = now
        rr = self.rr
        rr.set_time("sim", duration=sim_time)
        colors = [(220, 60, 60) if d else (70, 200, 110) for d in dones]
        rr.log(
            "worlds/markers",
            rr.Points3D(
                roots,
                radii=0.03,
                colors=colors,
                labels=[f"w{w}" for w in range(self.worlds)],
            ),
        )
        for w, r in enumerate(rewards):
            rr.log(f"worlds/reward/w{w}", rr.Scalars(float(r)))


# One process, one loop, read top to bottom: the physics side of the RL view.
def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", nargs="?", type=Path, default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--envs", type=int, default=9)
    parser.add_argument("--shm", default=None)
    parser.add_argument("--no-rerun", action="store_true")
    parser.add_argument("--robot", default=None, help="the walk's robot (bundle name)")
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="a project root: its robots first, its latest run, its declared walk",
    )
    args = parser.parse_args()
    from rq_mjlab.walks import DEFAULT_ROBOT, use_project  # noqa: PLC0415

    use_project(args.project)
    robot = args.robot or (
        project_walk_robot(args.project) if args.project else DEFAULT_ROBOT
    )
    checkpoint = args.checkpoint if args.checkpoint else latest_checkpoint(args.project)
    streamer = transport()

    import warp as wp  # noqa: PLC0415

    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    import mujoco  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from rq_mjlab.actuator import as_torch  # noqa: PLC0415

    # stderr: stdout is the render process's token channel.
    print(
        f"[walk-view] {robot} {checkpoint.name} on {device}, {args.envs} worlds",
        file=sys.stderr,
        flush=True,
    )
    env, policy = load_policy(checkpoint, args.envs, device, robot=robot)
    twist_term, twist_bounds = twist_switch(env.unwrapped, streamer.TWIST_AXES)
    mirror = streamer.walk_scene(robot, args.envs)
    mirror_data = mujoco.MjData(mirror)
    device_qpos = as_torch(env.unwrapped.sim.data.qpos)
    device_qvel = as_torch(env.unwrapped.sim.data.qvel)
    device_ctrl = as_torch(env.unwrapped.sim.data.ctrl)
    device_xfrc = as_torch(env.unwrapped.sim.data.xfrc_applied)
    nq = device_qpos.shape[1]
    if mirror.nq != args.envs * nq:
        raise SystemExit(f"mirror nq {mirror.nq} != {args.envs} x {nq}")
    world_of, device_of = body_maps(mirror, env.unwrapped.sim.mj_model)
    roots = [
        body
        for body in range(1, mirror.nbody)
        if mirror.body(body).name.endswith("/trunk_base")
        or mirror.body(body).name.split("/")[-1] == mirror.body(1).name.split("/")[-1]
    ][: args.envs]

    # The ring, then the render process on the Studio's own pipes.
    fd, ring_path = tempfile.mkstemp(prefix="studio-state-walk-", suffix=".ring")
    import os  # noqa: PLC0415

    os.close(fd)
    ring = streamer.StateRing(ring_path, mirror, create=True, nworld=args.envs)
    renderer = subprocess.Popen(
        [
            "uv",
            "run",
            "--extra",
            "sim",
            "--extra",
            "viz",
            "python",
            str(REPO / "tools" / "studio-render-stream.py"),
            streamer.WALK,
            f"--scene={streamer.WALK}:{robot}:{args.envs}",
            f"--ring={ring_path}",
            *([f"--shm={args.shm}"] if args.shm else []),
            *([f"--project={args.project}"] if args.project else []),
            f"--rig={robot}-rl",
            *([f"--twist-ranges={twist_bounds}"] if twist_bounds else []),
        ],
        cwd=str(REPO / "pipeline"),
    )
    overview = None if args.no_rerun else Overview(args.envs)
    pump = streamer.PhysicsPump(mirror, None, ring)
    if twist_term is not None:
        pump.on_command = Joystick(twist_term, streamer.TWIST_AXES).on_command

    step_seconds = float(env.unwrapped.step_dt)
    obs = env.get_observations()
    sim_time = 0.0
    rewards = torch.zeros(args.envs)
    dones = torch.zeros(args.envs, dtype=torch.bool)
    try:
        while renderer.poll() is None:
            if not ring.paused():
                with torch.inference_mode():
                    actions = policy(obs)
                obs, rewards, dones, _ = env.step(actions)
                sim_time += step_seconds
            mirror_data.qpos[:] = device_qpos.reshape(-1).cpu().numpy()
            mirror_data.qvel[:] = device_qvel.reshape(-1).cpu().numpy()
            if mirror.nu == device_ctrl.numel():
                mirror_data.ctrl[:] = device_ctrl.reshape(-1).cpu().numpy()
            mujoco.mj_forward(mirror, mirror_data)
            ring.world_stats[:, 0] = rewards.cpu().numpy()
            ring.world_stats[:, 1] = dones.cpu().numpy()
            mirror_data.time = sim_time
            device_xfrc[:] = 0.0
            try:
                pump.tick(mirror_data, step_seconds, hold_when_paused=False)
            except streamer.ResetScene:
                obs = (
                    env.reset()[0]
                    if isinstance(env.reset(), tuple)
                    else env.get_observations()
                )
                sim_time = 0.0
            except streamer.TakeOver:
                print(
                    "[walk-view] manual control is not available in the RL view: the "
                    "policy drives every world",
                    file=sys.stderr,
                    flush=True,
                )
            # The shove's round trip: the wrench the pump wrote on the mirror
            # goes to the batched sim's world.
            loaded = mirror_data.xfrc_applied.any(axis=1).nonzero()[0]
            for body in loaded:
                world = world_of.get(int(body))
                target = device_of.get(int(body))
                if world is not None and world < args.envs and target is not None:
                    device_xfrc[world, target] = torch.from_numpy(
                        mirror_data.xfrc_applied[body]
                    ).to(device_xfrc)
            if overview is not None:
                overview.log(
                    sim_time,
                    mirror_data.xpos[roots],
                    rewards.cpu().numpy(),
                    dones.cpu().numpy(),
                )
            if ring.paused():
                time.sleep(0.02)
    finally:
        renderer.terminate()
        Path(ring_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
