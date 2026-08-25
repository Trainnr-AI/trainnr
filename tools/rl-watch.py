"""Watch reinforcement learning happen: thousands of worlds on the GPU,
sixteen of them on screen.

    cd pipeline && GALLIUM_DRIVER=d3d12 WGPU_BACKEND=vulkan MUJOCO_GL=egl \
        .venv-train/bin/python ../tools/rl-watch.py --timesteps 20000000

MuJoCo Playground's `AlohaHandOver` (two ALOHA arms, a box to pick up
with the right arm and hand to the left — Menagerie's MJX-patched
ALOHA model) trains with brax PPO on the GPU, `--envs` worlds stepping
in lockstep on MJX. This is the Isaac-Lab picture on the MuJoCo stack:
the policy ACTS in the simulator while it learns, and the rollouts are
the training data.

    MuJoCo window     a grid of `--show` worlds, the CURRENT policy
                      driving them, refreshed at every evaluation
    Rerun train/*     PPO metrics on the env_steps timeline: eval
                      episode reward, reward terms, losses
          worlds/<n>  each shown world's box height per step
          world/rig   the mesh-true mirror of the grid, 5 Hz
          stage       what is playing

The grid is a CPU MuJoCo model built from the env's own XML attached
`--show` times; the GPU rollouts' qpos are copied into it step by step,
so what you see is exactly what the batched simulator computed. The
mirror is deliberately slow (5 Hz): a Rerun mesh log costs ~46 ms for
400 meshes (measured), and the physics must not starve for it.

This is the exploratory rung (T6): Playground's task and reward, not
our harness's; its result is a watched learning curve, not a
certificate. The harness judges what comes out of it afterwards.
"""

import argparse
import contextlib
import functools
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import jax
import mujoco
import mujoco.viewer
import numpy as np
import rerun as rr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _rig3d import RigMirror  # noqa: E402

ENV_NAME = "AlohaHandOver"
COLLISION_GROUP = 3
HIDDEN_GROUP = 4
PITCH = 1.6
MIRROR_EVERY = 10  # control steps at 50 Hz -> 5 Hz
REALTIME_SLEEP = 0.02  # one control step


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--timesteps", type=int, default=20_000_000)
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--evals", type=int, default=20)
    parser.add_argument("--show", type=int, default=16, help="worlds on screen")
    parser.add_argument("--impl", default="jax", choices=("jax", "warp"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def materialise_assets(env) -> Path:
    """Playground keeps the env's XMLs and meshes as an in-memory dict
    keyed by basename; MjSpec resolves `<include>` and `meshdir` from
    disk. Write the dict out once — flat, plus an `assets/` copy so
    `meshdir="assets"` resolves — and build from the file."""
    root = Path(tempfile.mkdtemp(prefix="rl-watch-assets-"))
    (root / "assets").mkdir()
    for name, payload in env._model_assets.items():
        (root / name).write_bytes(payload)
        (root / "assets" / name).write_bytes(payload)
    return root / Path(env.xml_path).name


def grid_model(env, worlds: int):
    """The env's scene, attached `worlds` times on a grid, as a CPU model."""
    xml_path = materialise_assets(env)
    scene = mujoco.MjSpec()
    scene.modelname = f"{ENV_NAME}-grid-{worlds}"
    scene.visual.quality.shadowsize = 2048
    scene.worldbody.add_light(
        pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    )
    scene.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.1],
        pos=[0, 0, -0.75],
        rgba=[0.12, 0.12, 0.12, 1],
    )
    side = int(np.ceil(np.sqrt(worlds)))
    for n in range(worlds):
        child = mujoco.MjSpec.from_file(str(xml_path))
        for geom in child.geoms:
            if geom.name == "floor":
                geom.group = HIDDEN_GROUP
        row, col = divmod(n, side)
        frame = scene.worldbody.add_frame(
            pos=[(col - (side - 1) / 2) * PITCH, (row - (side - 1) / 2) * PITCH, 0]
        )
        scene.attach(child, prefix=f"w{n:02d}/", frame=frame)
    return scene.compile(), side


class Stage:
    """The grid on screen: copies batched rollouts into the CPU model."""

    def __init__(self, env, worlds: int):
        self.model, side = grid_model(env, worlds)
        self.data = mujoco.MjData(self.model)
        self.nq = env.mjx_model.nq
        self.worlds = worlds
        self.mirror = RigMirror(
            self.model, model_colors=True, skip_groups=(COLLISION_GROUP, HIDDEN_GROUP)
        )
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance = 2.2 * side
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -40
        self.viewer.cam.lookat[:] = [0, 0, 0]
        box = env.mj_model.body("box").id
        self.box_z = int(env.mj_model.jnt_qposadr[env.mj_model.body_jntadr[box]]) + 2

    def play(self, qpos_history: np.ndarray, env_steps: int, label: str):
        """qpos_history: (T, worlds, nq) from the GPU rollout."""
        rr.log("stage", rr.TextLog(f"{label}: playing {self.worlds} worlds"))
        for t in range(qpos_history.shape[0]):
            if not self.viewer.is_running():
                raise SystemExit("viewer closed")
            self.data.qpos[:] = qpos_history[t].reshape(-1)
            mujoco.mj_forward(self.model, self.data)
            self.viewer.sync()
            if t % MIRROR_EVERY == 0:
                rr.set_time("env_steps", sequence=env_steps)
                rr.set_time("sim_time", duration=t * 0.02)
                self.mirror.log(self.data, path="world/rig")
                for n in range(self.worlds):
                    rr.log(
                        f"worlds/{n:02d}/box_z",
                        rr.Scalars(float(qpos_history[t, n, self.box_z])),
                    )
            time.sleep(REALTIME_SLEEP)
        picked = int((qpos_history[-1, :, self.box_z] > 0.15).sum())  # noqa: PLR2004 - env's own threshold
        rr.log("stage", rr.TextLog(f"{label}: {picked}/{self.worlds} boxes held up"))
        print(
            f"[rl] {label}: {picked}/{self.worlds} boxes held up at the end", flush=True
        )


def rollout_fn(env, worlds: int, make_policy):
    """A jitted batched rollout of one full episode under the policy
    `make_policy(params)` — params are traced, so one compile serves
    every evaluation."""
    episode = env._config.episode_length

    def run(params, rng):
        policy = make_policy(params, deterministic=True)
        keys = jax.random.split(rng, worlds)
        state = jax.vmap(env.reset)(keys)

        def body(carry, _):
            state, key = carry
            key, sub = jax.random.split(key)
            action, _ = policy(state.obs, jax.random.split(sub, worlds))
            state = jax.vmap(env.step)(state, action)
            return (state, key), state.data.qpos

        (_, _), qpos = jax.lax.scan(body, (state, rng), None, length=episode)
        return qpos  # (T, worlds, nq)

    return jax.jit(run)


def main() -> None:
    args = parse_args()
    from brax.training.agents.ppo import networks as ppo_networks  # noqa: PLC0415
    from brax.training.agents.ppo import train as ppo  # noqa: PLC0415
    from mujoco_playground import registry, wrapper  # noqa: PLC0415
    from mujoco_playground.config import manipulation_params  # noqa: PLC0415

    env = registry.load(ENV_NAME, config_overrides={"impl": args.impl})
    cfg = manipulation_params.brax_ppo_config(ENV_NAME)
    cfg.num_timesteps = args.timesteps
    cfg.num_envs = args.envs
    cfg.num_evals = args.evals
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **cfg.network_factory
    )
    train_kwargs = dict(cfg)
    del train_kwargs["network_factory"]

    rr.init(f"robotiq-rl-watch-{ENV_NAME}", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    stage = Stage(env, args.show)
    rollouts = {}  # one jitted rollout per make_policy (brax passes the same one)
    started = time.time()
    rng = jax.random.PRNGKey(args.seed + 1)

    def progress(num_steps, metrics):
        rr.set_time("env_steps", sequence=num_steps)
        for key, value in metrics.items():
            with contextlib.suppress(TypeError, ValueError):
                rr.log(f"train/{key}", rr.Scalars(float(value)))
        reward = metrics.get("eval/episode_reward", float("nan"))
        print(
            f"[rl] {datetime.now():%H:%M:%S} steps {num_steps:>12,}  "
            f"eval reward {float(reward):8.2f}  ({time.time() - started:.0f} s)",
            flush=True,
        )

    def on_policy(current_step, make_policy, params):
        nonlocal rng
        rng, sub = jax.random.split(rng)
        if make_policy not in rollouts:
            rollouts[make_policy] = rollout_fn(env, args.show, make_policy)
        qpos = np.asarray(rollouts[make_policy](params, sub))
        stage.play(qpos, current_step, f"policy at {current_step:,} env steps")

    rr.log("stage", rr.TextLog(f"PPO on {ENV_NAME}: {args.envs} worlds on the GPU"))
    print(
        f"[rl] PPO on {ENV_NAME}, {args.envs} envs, {args.timesteps:,} steps",
        flush=True,
    )
    _make_inference_fn, _params, _ = ppo.train(
        environment=env,
        eval_env=registry.load(ENV_NAME, config_overrides={"impl": args.impl}),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        network_factory=network_factory,
        progress_fn=progress,
        policy_params_fn=on_policy,
        seed=args.seed,
        **train_kwargs,
    )
    rr.log("stage", rr.TextLog("training finished"))
    print("[rl] training finished - close the MuJoCo window to exit", flush=True)
    while stage.viewer.is_running():
        time.sleep(0.2)


if __name__ == "__main__":
    main()
