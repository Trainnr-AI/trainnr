"""Watch reinforcement learning happen: thousands of worlds on the GPU,
sixteen of them on screen.

    cd pipeline && ../tools/wsl-run.sh .venv-train/bin/python \
        ../tools/rl-watch.py --timesteps 20000000

`wsl.env`'s `LD_LIBRARY_PATH` is what lets Warp find the GPU under WSL
(measured: without it Warp reports "no CUDA-capable device" and falls
to the CPU, and the JAX->Warp FFI call fails). Warp is the only
backend fast enough for these mesh-contact environments: 36k
env-steps/s at 2,048 worlds versus 1.8k on MJX-JAX.

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
import atexit
import contextlib
import functools
import math
import multiprocessing
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import jax
import mujoco
import mujoco.viewer
import numpy as np
import rerun as rr

from _lab import bootstrap, frame_viewer, rr_session

bootstrap()
from rq_pipeline.tasks.scene import GeomGroup, RenderBudget, grid_of  # noqa: E402
from rq_pipeline.viz import RigMirror  # noqa: E402

ENV_NAME = "AlohaHandOver"
MIRROR_EVERY = 10  # control steps at 50 Hz -> 5 Hz
REALTIME_SLEEP = 0.02  # one control step
BOX_HELD_Z = 0.15  # the env's own "picked" threshold


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--timesteps", type=int, default=20_000_000)
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--evals", type=int, default=20)
    parser.add_argument("--show", type=int, default=16, help="worlds on screen")
    parser.add_argument("--impl", default="warp", choices=("jax", "warp"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def materialise_assets(env) -> Path:
    """Playground keeps the env's XMLs and meshes as an in-memory dict
    keyed by basename; MjSpec resolves `<include>` and `meshdir` from
    disk. Write the dict out once — flat, plus an `assets/` copy so
    `meshdir="assets"` resolves — and build from the file."""
    root = Path(tempfile.mkdtemp(prefix="rl-watch-assets-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    (root / "assets").mkdir()
    for name, payload in env._model_assets.items():
        (root / name).write_bytes(payload)
        (root / "assets" / name).write_bytes(payload)
    return root / Path(env.xml_path).name


def grid_model_from_xml(xml_path, worlds: int):
    """The env's scene, attached `worlds` times on a grid, as a CPU model."""
    scene, prefixes = grid_of(
        f"{ENV_NAME}-grid-{worlds}",
        (mujoco.MjSpec.from_file(str(xml_path)) for _ in range(worlds)),
    )
    scene.visual.quality.shadowsize = RenderBudget.SHADOWSIZE
    side = math.isqrt(max(worlds - 1, 0)) + 1
    del prefixes  # the mirror reads geoms straight off the compiled model
    return scene.compile(), side


def stage_process(spec, queue, closed):
    """The grid on screen, in its OWN PROCESS: loops the latest policy's
    rollout continuously; training sends a new one at every evaluation.

    A process, not a thread, on purpose: MJX-Warp steps the simulator
    through a Python callback on every training step, so the training
    thread holds the GIL almost continuously and a viewer thread starves
    — measured as "the arms moved while the GPU was compiling, then
    never again". Here the viewer owns an interpreter of its own; the
    only thing crossing is a (T, worlds, nq) array per evaluation.
    """
    import queue as queue_module  # noqa: PLC0415

    xml_path, worlds, box_z, recording_id = spec
    model, side = grid_model_from_xml(xml_path, worlds)
    data = mujoco.MjData(model)
    mirror = RigMirror(
        model, model_colors=True, skip_groups=(GeomGroup.COLLISION, GeomGroup.HIDDEN)
    )
    # mode="connect": the viewer the training process spawned exists.
    rr_session(
        f"robotiq-rl-watch-{ENV_NAME}", mode="connect", recording_id=recording_id
    )
    history, env_steps = None, 0
    viewer = mujoco.viewer.launch_passive(model, data)
    frame_viewer(viewer, 2.2 * side)
    try:
        while viewer.is_running():
            try:
                while True:  # drain to the newest rollout
                    history, env_steps = queue.get_nowait()
            except queue_module.Empty:
                pass
            if history is None:
                time.sleep(0.1)
                continue
            for t in range(history.shape[0]):
                if not viewer.is_running():
                    break
                with viewer.lock():
                    data.qpos[:] = history[t].reshape(-1)
                    mujoco.mj_forward(model, data)
                viewer.sync()
                if t % MIRROR_EVERY == 0:
                    rr.set_time("env_steps", sequence=env_steps)
                    rr.set_time("sim_time", duration=t * 0.02)
                    mirror.log(data)
                    for n in range(worlds):
                        rr.log(
                            f"worlds/{n:02d}/box_z",
                            rr.Scalars(float(history[t, n, box_z])),
                        )
                time.sleep(REALTIME_SLEEP)
    finally:
        viewer.close()
        closed.set()


class Stage:
    """The training side of the grid: starts the viewer process, ships
    each evaluation's rollout to it, logs the verdict."""

    def __init__(self, env, worlds: int, recording_id: str):
        self.worlds = worlds
        box = env.mj_model.body("box").id
        self.box_z = int(env.mj_model.jnt_qposadr[env.mj_model.body_jntadr[box]]) + 2
        context = multiprocessing.get_context("spawn")  # never fork a CUDA process
        self.queue = context.Queue()
        self.closed = context.Event()
        spec = (str(materialise_assets(env)), worlds, self.box_z, recording_id)
        self.process = context.Process(
            target=stage_process, args=(spec, self.queue, self.closed), daemon=True
        )
        self.process.start()

    def publish(self, qpos_history: np.ndarray, env_steps: int, label: str) -> int:
        """Ship a new rollout; returns how many worlds ended holding the box."""
        picked = int((qpos_history[-1, :, self.box_z] > BOX_HELD_Z).sum())
        self.queue.put((np.ascontiguousarray(qpos_history), env_steps))
        rr.set_time("env_steps", sequence=env_steps)
        rr.log("eval/boxes_held", rr.Scalars(picked))
        rr.log("stage", rr.TextLog(f"{label}: {picked}/{self.worlds} boxes held up"))
        print(
            f"[rl] {label}: {picked}/{self.worlds} boxes held up at the end", flush=True
        )
        return picked


def rollout_fn(env, worlds: int, make_policy):
    """A jitted batched rollout of one full episode under the policy
    `make_policy(params)` — params are traced, so one compile serves
    every evaluation. STOCHASTIC actions: this is what the learner's
    own rollouts look like (its exploration), which is the honest thing
    to watch — an early deterministic mean of this env's +-0.015 rad
    deltas is a robot standing still."""
    episode = env._config.episode_length

    def run(params, rng):
        policy = make_policy(params, deterministic=False)
        keys = jax.random.split(rng, worlds)
        state = jax.vmap(env.reset)(keys)

        def body(carry, _):
            state, key = carry
            key, sub = jax.random.split(key)
            action, _ = policy(state.obs, sub)  # one key serves the whole batch
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

    recording_id = f"rl-watch-{ENV_NAME}-{datetime.now():%Y%m%d-%H%M%S}"
    rr_session(f"robotiq-rl-watch-{ENV_NAME}", mode="spawn", recording_id=recording_id)
    stage = Stage(env, args.show, recording_id)
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
        stage.publish(qpos, current_step, f"policy at {current_step:,} env steps")
        if stage.closed.is_set():
            raise SystemExit("viewer closed")

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
    rr.log(
        "stage",
        rr.TextLog("training finished - the grid keeps looping the final policy"),
    )
    print("[rl] training finished - close the MuJoCo window to exit", flush=True)
    stage.closed.wait()


if __name__ == "__main__":
    main()
