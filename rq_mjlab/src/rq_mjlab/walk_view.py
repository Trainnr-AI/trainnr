"""The trained walk policy in the Studio's viewport — the RL view.

    cd rq_mjlab && LD_LIBRARY_PATH=/usr/lib/wsl/lib MUJOCO_GL=egl \\
        uv run python -m rq_mjlab.walk_view --latest [--envs 9] [--shm=PATH]

The batched mjlab env steps N policy-driven worlds on the GPU; a CPU
MIRROR (one copy of the robot MJCF per world, on one ground plane)
takes every world's qpos each control step and rides the SAME
transport the scene previews use — the shared-memory frame ring, the
tagged stdin protocol, the orbit camera, contact-force arrows
(imported from tools/studio-render-stream.py, the transport's one
home). Ctrl+drag shoves a POLICY-DRIVEN duck for real: the gesture
resolves on the mirror (mjv_select / mjvPerturb), and the force it
produces is copied into the batched sim's `xfrc_applied` for that
world every step — the policy must recover live, in the panel.

Identity-gated like every checkpoint door: a run whose identity.json
differs from this env is refused.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def transport():
    """The viewport transport (FrameSink, OrbitCamera, Perturber, the
    stdin reader) from its one home — a hyphenated tool file, hence
    importlib rather than an import statement."""
    tools = REPO / "tools"
    sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location(
        "studio_render_stream", tools / "studio-render-stream.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def latest_checkpoint() -> Path:
    """The newest model_*.pt under runs/microduck-walk — what the
    Studio's walk button rolls without naming a file."""
    checkpoints = sorted(
        REPO.glob("runs/microduck-walk/*/model_*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not checkpoints:
        raise SystemExit(
            "no checkpoint under runs/microduck-walk - train one (walk_train) "
            "or pull a run from the pod first"
        )
    return checkpoints[-1]


def mirror_of(worlds: int, offscreen_side: int):
    """One CPU model holding `worlds` copies of the walk robot on one
    ground plane. Every frame sits at the origin: the batched env's
    world ORIGINS are already baked into each free joint's global qpos,
    so the copies land on their own env origins by the copy alone."""
    import mujoco  # noqa: PLC0415
    from rq_pipeline.tasks.scene import grid_of  # noqa: PLC0415

    xml = str(REPO / "robots" / "microduck" / "robot_walk.xml")
    scene, _ = grid_of(
        f"microduck-rl-{worlds}",
        (mujoco.MjSpec.from_file(xml) for _ in range(worlds)),
        pitch=0.0,
    )
    for geom in scene.geoms:
        if geom.name == "ground":
            geom.pos[2] = 0.0  # the display grids' table offset; ducks walk at z=0
    scene.visual.global_.offwidth = offscreen_side
    scene.visual.global_.offheight = offscreen_side
    return scene.compile()


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


def load_policy(checkpoint: Path, envs: int, device: str):
    """The env and its inference policy, identity-gated (the same door
    walk_play and walk_verdict use)."""
    from dataclasses import asdict  # noqa: PLC0415

    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: PLC0415

    from rq_mjlab.microduck_walk import microduck_walk_env_cfg  # noqa: PLC0415
    from rq_mjlab.walk_train import g3_agent  # noqa: PLC0415

    cfg, identity = microduck_walk_env_cfg()
    cfg.scene.num_envs = envs
    trained = checkpoint.parent / "identity.json"
    if trained.is_file() and json.loads(trained.read_text()) != identity:
        raise SystemExit(f"identity mismatch: this env is {identity}")
    agent = g3_agent(iterations=1)
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg, device=device), clip_actions=agent.clip_actions
    )
    runner = MjlabOnPolicyRunner(env, asdict(agent), log_dir=None, device=device)
    runner.load(
        str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    return env, runner.get_inference_policy(device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", nargs="?", type=Path, default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--envs", type=int, default=9)
    parser.add_argument("--shm", default=None)
    args = parser.parse_args()
    checkpoint = args.checkpoint if args.checkpoint else latest_checkpoint()
    streamer = transport()

    import warp as wp  # noqa: PLC0415

    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    import mujoco  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from rq_mjlab.actuator import as_torch  # noqa: PLC0415

    # stderr: stdout is the frame-token channel the controller reads.
    print(
        f"[walk-view] {checkpoint.name} on {device}, {args.envs} worlds",
        file=sys.stderr,
        flush=True,
    )
    env, policy = load_policy(checkpoint, args.envs, device)
    mirror = mirror_of(args.envs, streamer.MAX_RENDER_SIDE)
    mirror_data = mujoco.MjData(mirror)
    device_qpos = as_torch(env.unwrapped.sim.data.qpos)
    device_xfrc = as_torch(env.unwrapped.sim.data.xfrc_applied)
    nq = device_qpos.shape[1]
    if mirror.nq != args.envs * nq:
        raise SystemExit(f"mirror nq {mirror.nq} != {args.envs} x {nq}")
    world_of, device_of = body_maps(mirror, env.unwrapped.sim.mj_model)

    orbit = streamer.OrbitCamera(
        {"azimuth": 120.0, "elevation": -20.0, "distance": 3.0, "lookat": (0, 0, 0.1)}
    )
    perturber = streamer.Perturber(mirror)
    pump = streamer.RenderPump(
        mirror, orbit, None, perturber, streamer.FrameSink(args.shm)
    )
    threading.Thread(
        target=streamer._read_control_messages,
        # exit_on_eof: this leg only ever runs under the Studio; a
        # closed stdin means the controller died - never orphan the GPU.
        args=(orbit, perturber, pump._fresh.set, True),
        daemon=True,
    ).start()

    step_seconds = float(env.unwrapped.step_dt)
    obs = env.get_observations()
    while True:
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        mirror_data.qpos[:] = device_qpos.reshape(-1).cpu().numpy()
        # The shove's round trip: forward the mirror so the perturb has
        # fresh poses, let it write the mirror's xfrc, route the one
        # loaded row into the batched sim's world.
        mujoco.mj_forward(mirror, mirror_data)
        device_xfrc[:] = 0.0
        pump.tick(mirror_data, step_seconds)
        if perturber.pert.active:
            loaded = mirror_data.xfrc_applied.any(axis=1).nonzero()[0]
            for body in loaded:
                world = world_of.get(int(body))
                target = device_of.get(int(body))
                if world is not None and world < args.envs and target is not None:
                    device_xfrc[world, target] = torch.from_numpy(
                        mirror_data.xfrc_applied[body]
                    ).to(device_xfrc)


if __name__ == "__main__":
    main()
