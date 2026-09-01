"""Roll a trained walk checkpoint in the viewers — the G3 judgment.

    cd rq_mjlab && MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 \\
        LD_LIBRARY_PATH=/usr/lib/wsl/lib uv run python -m rq_mjlab.walk_play \\
        ../runs/microduck-walk/20260901-163412/model_7999.pt [--envs 9]

Builds the SAME certified env the checkpoint trained under
(`microduck_walk_env_cfg` — stamped robot, certified actuator,
declared-basis DR), loads the checkpoint through rsl-rl's own runner
(`load_cfg={"actor": True}`: inference needs the actor, not the
optimizer), and drives mjlab's native MuJoCo window with the
inference policy. The RerunRecorder streams the watched duck into the
Studio at the same time — mesh mirror, MuJoCo-lit camera, reward
series — so the gait is judged in BOTH instruments (the operator's
rule). Prints the run's identity beside this env's so a checkpoint
can never be rolled on a rig it was not trained for.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path, help="model_*.pt from a walk run")
    parser.add_argument("--envs", type=int, default=9)
    parser.add_argument(
        "--no-recorder",
        action="store_true",
        help="MuJoCo window only (no Studio listening on :9876)",
    )
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"no checkpoint at {args.checkpoint}")

    import warp as wp  # noqa: PLC0415

    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: PLC0415
    from mjlab.viewer import NativeMujocoViewer  # noqa: PLC0415

    from rq_mjlab.microduck_walk import microduck_walk_env_cfg  # noqa: PLC0415
    from rq_mjlab.recorder import RerunRecorderCfg  # noqa: PLC0415
    from rq_mjlab.walk_train import g3_agent  # noqa: PLC0415

    cfg, identity = microduck_walk_env_cfg()
    cfg.scene.num_envs = args.envs
    if not args.no_recorder:
        cfg.recorders = {
            "rerun": RerunRecorderCfg(app_id="rq-walk-play", every=5, frame_every=25)
        }

    # The identity beside the run's own: a checkpoint rolled on a rig it
    # was not trained for is a wrong answer with a straight face.
    trained = args.checkpoint.parent / "identity.json"
    if trained.is_file():
        recorded = json.loads(trained.read_text())
        print(f"[play] run identity:  {recorded}")
        if recorded != identity:
            raise SystemExit(f"identity mismatch: this env is {identity}")
    print(f"[play] env identity:  {identity}")
    print(f"[play] {args.envs} envs on {device}; checkpoint {args.checkpoint.name}")

    agent = g3_agent(iterations=1)  # the net shapes; iterations unused at inference
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
    policy = runner.get_inference_policy(device=device)
    NativeMujocoViewer(env, policy).run()
    env.close()


if __name__ == "__main__":
    main()
