"""Roll a trained walk checkpoint in the viewers — the G3 judgment.

    cd trainnr-mjlab && MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 \\
        LD_LIBRARY_PATH=/usr/lib/wsl/lib uv run python -m trainnr_mjlab.walk_play \\
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
from dataclasses import asdict
from pathlib import Path
from typing import Any

from trainnr_mjlab.walk_view import (
    fit_of,
    require_same_identity,
    trained_identity,
    trained_with_cameras,
)
from trainnr_mjlab.walks import (
    DEFAULT_ROBOT,
    ROBOTS,
    TERRAIN_SCAN_GROUP,
    use_project,
    walk_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path, help="model_*.pt from a walk run")
    parser.add_argument("--envs", type=int, default=9)
    parser.add_argument("--robot", default=DEFAULT_ROBOT, choices=ROBOTS)
    parser.add_argument(
        "--viewer",
        default="native",
        choices=("native", "viser"),
        help="mjlab's MuJoCo window, or its browser viewer (a URL on the log)",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="a project root: its robots are searched first (the Go2 lives there)",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=None,
        help="the captured scene the checkpoint trained on (a project's "
        "scenes/<name>): played on it, as the verdict judges it (docs/78 E2)",
    )
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
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer  # noqa: PLC0415

    from trainnr_mjlab.recorder import RerunRecorderCfg  # noqa: PLC0415

    use_project(args.project)
    spec = walk_spec(args.robot)
    # The walk in play mode: the curriculum and the pushes off, episodes
    # open-ended, the same robot, actuator and gains it trained under.
    recorded = trained_identity(args.checkpoint)
    cfg, identity = spec.env_cfg(
        play=True,
        dr_span=None,
        pin_scale=None,
        fit=fit_of(recorded),
        scene=args.scene,
        # the actor sees a camera exactly when it trained with one
        cameras=trained_with_cameras(recorded),
    )
    cfg.scene.num_envs = args.envs
    if not args.no_recorder:
        cfg.recorders = {
            "rerun": RerunRecorderCfg(
                app_id="trainnr-walk-play", every=5, frame_every=25
            )
        }

    # The identity beside the run's own: a checkpoint rolled on a rig it
    # was not trained for is a wrong answer with a straight face.
    if recorded:
        print(f"[play] run identity:  {recorded}")
    require_same_identity(recorded, identity)  # the one gate (walk_view)
    print(f"[play] env identity:  {identity}")
    print(f"[play] {args.envs} envs on {device}; checkpoint {args.checkpoint.name}")

    agent = spec.agent(1)  # the net shapes; iterations unused at inference
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
    # mjlab's two viewers as they are, drawing a scene's ground too. The
    # native window drew at 0 FPS on the WSLg box's X11 path (2026-09-11,
    # again 2026-09-25); the browser one does not touch that path.
    viewer: type[Any] = (
        ViserPlayViewer if args.viewer == "viser" else NativeMujocoViewer
    )
    if args.scene is not None:
        viewer = showing_the_ground(viewer)
    viewer(env, policy).run()
    env.close()


def showing_the_ground(viewer: type) -> type:
    """`viewer` with the scene's ground group drawn: mjviser's group list
    through its private `_sync_visibilities` (mjlab 1.6.0 / mjviser as
    installed; no public setter exists, and its "G4" checkbox still reads
    off until toggled) or MuJoCo's public `opt.geomgroup` in the native
    window. A viewer with neither is refused by name, never a silent
    invisible ground."""

    class Shown(viewer):  # type: ignore[misc, valid-type]
        def setup(self) -> None:
            super().setup()
            scene = getattr(self, "_scene", None)
            native = getattr(self, "viewer", None)
            if scene is not None:  # the browser viewer
                scene.geom_groups_visible[TERRAIN_SCAN_GROUP] = True
                scene._sync_visibilities()
            elif native is not None:  # the MuJoCo window
                native.opt.geomgroup[TERRAIN_SCAN_GROUP] = 1
            else:
                raise RuntimeError(
                    f"{viewer.__name__} has neither `_scene` (mjviser) nor `viewer` "
                    "(MuJoCo's window): cannot show the scene's ground group"
                )

    Shown.__name__ = f"{viewer.__name__}ShowingTheGround"
    return Shown


if __name__ == "__main__":
    main()
