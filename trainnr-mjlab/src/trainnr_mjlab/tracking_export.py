"""A trained tracking policy exported for deployment (docs/86 phase 3,
docs/87 §8):

    uv run python -m trainnr_mjlab.tracking_export <run>/model_N.pt \\
        --project <project> --name <deployment> --reference <recording> \\
        [--certificate <evaluation>] [--policy-stamp <policy>] [--lag-ms 20]

Writes `<project>/deploy/<name>/`:

- `policy.onnx`, mjlab's tracking export as it is: inputs `obs` and
  `time_step`, outputs the action and the reference at that step, with
  mjlab's metadata (joint names, gains, default pose, action scale, the
  observation names, the anchor and the tracked bodies). It is the file
  BeyondMimic's motion-tracking controller runs on a real G1.
- `motion.npz`, the reference as the runtime reads it: joint positions
  and velocities in the policy's joint order, the tracked bodies' poses
  and velocities in the world, one row per control tick.
- `deploy.json`, the manifest (`trainnr-deploy/1`) with a `tracking`
  section in place of a walk's commands: the reference, the torso the
  observations are taken against, mjlab's tracking bounds, and the
  training conditions an evaluation judges under (start draw, pushes,
  observation noise, command lag).
- `scene.xml` and `assets/`, the trained G1 and the plane it trained on,
  with the robot's meshes from mjlab's asset zoo (the run's robot is not a
  project bundle).

The torch actor and the ONNX are checked to agree on random observations,
and the ONNX's reference outputs against the reference written beside it.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from trainnr.deploy.manifest import (
    MANIFEST_FILE,
    TIME_STEP_INPUT,
    Key,
)
from trainnr.project.kinds import IDENTITY_FILE
from trainnr.tasks.tracking import SUCCESS

from trainnr_mjlab.checkpoint_guard import require_tensor_only
from trainnr_mjlab.tracking_train import TASK_ID, quiet_line_search
from trainnr_mjlab.walk_export import (
    ACTOR_OBS_GROUP,
    DEPLOY_FOLDER,
    EXPORT_CHECK_SAMPLES,
    EXPORT_TOLERANCE,
    POLICY_FILE,
    PROBE_SEED,
    UNRECORDED,
    Export,
    _manifest,
    _scene_xml,
    manifest_identity,
)
from trainnr_mjlab.walks import DeployFacts, use_project

MOTION_FILE = "raw/motion.npz"  # in the reference recording
DEPLOY_MOTION = "motion.npz"  # beside the policy
ASSETS = "assets"
TIME_STEP = "time_step"
ACTIONS = "actions"
ACTUATOR = "mjlab's G1 actuators, as declared (not identified)"
REFERENCE_OUTPUTS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w")
TERM_ANCHOR_Z, TERM_ANCHOR_TILT, TERM_EFFECTORS = (
    "anchor_pos",
    "anchor_ori",
    "ee_body_pos",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True, help="the deployment's folder name")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--certificate", default=None)
    parser.add_argument("--policy-stamp", default=None)
    parser.add_argument("--lag-ms", type=float, default=0.0)
    args = parser.parse_args()
    out = export(
        args.checkpoint,
        project=args.project,
        name=args.name,
        reference=args.reference,
        certificate=args.certificate,
        policy_stamp=args.policy_stamp,
        lag_ms=args.lag_ms,
    )
    print(f"[export] {out['manifest'][Key.STAMP_OF]} -> {out['dir']}", flush=True)


def _noise_of(cfg: Any) -> dict[str, list[float]]:
    """Each actor term's training noise, uniform [n_min, n_max]."""
    out = {}
    for name, term in cfg.observations[ACTOR_OBS_GROUP].terms.items():
        noise = getattr(term, "noise", None)
        if noise is not None and hasattr(noise, "n_min"):
            out[name] = [float(noise.n_min), float(noise.n_max)]
    return out


def _ranges(ranges: dict[str, Any]) -> dict[str, list[float]]:
    return {k: [float(v[0]), float(v[1])] for k, v in ranges.items()}


def randomization_of(play: Any, model_geoms: list[str]) -> dict[str, Any]:
    """Training's start-up draws the evaluation keeps (play mode leaves
    them on): the encoder bias, a body's centre-of-mass offset, the feet's
    shared friction, with the geoms the pattern names resolved."""
    import re  # noqa: PLC0415

    events = play.events
    out: dict[str, Any] = {}
    if "encoder_bias" in events:
        out["encoder_bias"] = [
            float(v) for v in events["encoder_bias"].params["bias_range"]
        ]
    com = events.get("base_com")
    if com is not None:
        (body,) = com.params["asset_cfg"].body_names
        if com.params.get("operation", "add") != "add":
            raise SystemExit(
                "the centre-of-mass draw is not an offset; the gate adds one"
            )
        out["body_com"] = {
            "body": body,
            "ranges": {
                str(k): [float(v[0]), float(v[1])]
                for k, v in com.params["ranges"].items()
            },
        }
    friction = events.get("foot_friction")
    if friction is not None:
        pattern = re.compile(friction.params["asset_cfg"].geom_names)
        if friction.params.get("operation", "abs") != "abs" or not friction.params.get(
            "shared_random", False
        ):
            raise SystemExit(
                "the feet's friction draw is not one shared value; the gate draws one"
            )
        out["foot_friction"] = {
            "geoms": [g for g in model_geoms if pattern.match(g)],
            "range": [float(v) for v in friction.params["ranges"]],
        }
    return out


def tracking_block(  # noqa: PLR0913 - the section's sources, each named
    play: Any,
    training: Any,
    command: Any,
    soft_limit_factor: float,
    model_geoms: list[str],
    *,
    lag_ms: float,
) -> dict[str, Any]:
    """The manifest's `tracking` section, every number read from the built
    task (`play`) and training's own config (`training`)."""
    motion_cfg, trained = play.commands["motion"], training.commands["motion"]
    terms = play.terminations
    push = training.events.get("push_robot")
    return {
        "motion": DEPLOY_MOTION,
        "frames": int(command.motion.time_step_total),
        "anchor_body": str(motion_cfg.anchor_body_name),
        "body_names": list(motion_cfg.body_names),
        "terminations": {
            "anchor_z_m": float(terms[TERM_ANCHOR_Z].params["threshold"]),
            "anchor_tilt": float(terms[TERM_ANCHOR_TILT].params["threshold"]),
            "end_effector_z_m": float(terms[TERM_EFFECTORS].params["threshold"]),
            "end_effectors": list(terms[TERM_EFFECTORS].params["body_names"]),
        },
        "start": {
            "pose_range": _ranges(trained.pose_range),
            "velocity_range": _ranges(trained.velocity_range),
            "joint_position_range": [float(v) for v in motion_cfg.joint_position_range],
            "soft_limit_factor": float(soft_limit_factor),
        },
        "pushes": (
            {
                "interval_s": [float(v) for v in push.interval_range_s],
                "velocity_range": _ranges(push.params["velocity_range"]),
            }
            if push is not None
            else None
        ),
        "noise": _noise_of(training),
        "randomization": randomization_of(play, model_geoms),
        "lag_ms": float(lag_ms),
        "criterion": SUCCESS,
    }


def write_reference(command: Any, out: Path) -> None:
    """The reference the runtime reads, from the command term itself: its
    joints in the robot's (the policy's) order, its tracked bodies."""
    motion = command.motion
    np.savez(
        out,
        joint_pos=motion.joint_pos.cpu().numpy().astype(np.float32),
        joint_vel=motion.joint_vel.cpu().numpy().astype(np.float32),
        body_pos_w=motion.body_pos_w.cpu().numpy().astype(np.float32),
        body_quat_w=motion.body_quat_w.cpu().numpy().astype(np.float32),
        body_lin_vel_w=motion.body_lin_vel_w.cpu().numpy().astype(np.float32),
        body_ang_vel_w=motion.body_ang_vel_w.cpu().numpy().astype(np.float32),
    )


def g1_assets() -> Path:
    """The G1's meshes in mjlab's asset zoo."""
    import mjlab.asset_zoo.robots.unitree_g1 as g1  # noqa: PLC0415

    return Path(g1.__file__).parent / "xmls" / ASSETS


def _check_onnx(path: Path, actor: Any, obs_dim: int, reference: Path) -> float:
    """The largest difference between the ONNX and the torch actor over
    random observations; refused past the tolerance, as is a reference
    output that differs from the reference written beside the policy."""
    import onnxruntime as ort  # noqa: PLC0415
    import torch  # noqa: PLC0415

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(PROBE_SEED)
    probe = rng.standard_normal((EXPORT_CHECK_SAMPLES, obs_dim)).astype(np.float32)
    with torch.inference_mode():
        expected = actor(torch.from_numpy(probe)).numpy()
    with np.load(reference) as ref:
        frames = int(ref["joint_pos"].shape[0])
        worst = 0.0
        for i, row in enumerate(probe):
            step = np.array([[i * frames // EXPORT_CHECK_SAMPLES]], dtype=np.float32)
            out = session.run(None, {"obs": row[None, :], TIME_STEP: step})
            worst = max(worst, float(np.max(np.abs(out[0][0] - expected[i]))))
            for name, got in zip(REFERENCE_OUTPUTS, out[1:], strict=False):
                if not np.allclose(got[0], ref[name][int(step[0, 0])], atol=1e-5):
                    raise SystemExit(
                        f"the ONNX's {name} disagrees with {DEPLOY_MOTION}"
                    )
    if worst > EXPORT_TOLERANCE:
        raise SystemExit(
            f"the ONNX disagrees with the actor by {worst:.2e} (> {EXPORT_TOLERANCE})"
        )
    return worst


def export(  # noqa: PLR0913, PLR0915 - the export's own knobs and steps, each named
    checkpoint: Path,
    *,
    project: Path,
    name: str,
    reference: Path,
    certificate: str | None = None,
    policy_stamp: str | None = None,
    lag_ms: float = 0.0,
) -> dict[str, Any]:
    """Write the deployment; returns the manifest and the directory."""
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.rl import RslRlVecEnvWrapper  # noqa: PLC0415
    from mjlab.rl.exporter_utils import (  # noqa: PLC0415
        attach_metadata_to_onnx,
        get_base_metadata,
    )
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: PLC0415
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner  # noqa: PLC0415

    use_project(project)
    checkpoint = Path(checkpoint).resolve()
    run_dir = checkpoint.parent
    identity_file = run_dir / IDENTITY_FILE
    trained = (
        json.loads(identity_file.read_text(encoding="utf-8"))
        if identity_file.is_file()
        else {}
    )
    from mjlab.tasks.tracking.mdp.commands import (  # noqa: PLC0415
        MotionCommand,
        MotionCommandCfg,
    )

    cfg = load_env_cfg(TASK_ID, play=True)
    training = load_env_cfg(TASK_ID)
    motion_cfg = cfg.commands["motion"]
    assert isinstance(motion_cfg, MotionCommandCfg)
    motion_cfg.motion_file = str(Path(reference).resolve() / MOTION_FILE)
    cfg.scene.num_envs = 1
    cfg.sim = quiet_line_search(cfg.sim)
    device = "cpu"
    agent = load_rl_cfg(TASK_ID)
    env = ManagerBasedRlEnv(cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = MotionTrackingOnPolicyRunner(
        wrapped, asdict(agent), log_dir=None, device=device
    )
    require_tensor_only(checkpoint)  # before any unpickling
    runner.load(
        str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )

    out_dir = Path(project) / DEPLOY_FOLDER / name
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; a deployment is never overwritten")
    out_dir.mkdir(parents=True)
    try:
        command = env.command_manager.get_term("motion")
        assert isinstance(command, MotionCommand)
        write_reference(command, out_dir / DEPLOY_MOTION)
        runner.export_policy_to_onnx(str(out_dir), POLICY_FILE)
        policy_path = out_dir / POLICY_FILE
        obs_dim = int(wrapped.get_observations()[ACTOR_OBS_GROUP].shape[-1])
        actor = runner.alg.get_policy().as_onnx(verbose=False).cpu().eval()
        max_abs_diff = _check_onnx(policy_path, actor, obs_dim, out_dir / DEPLOY_MOTION)
        metadata = get_base_metadata(env, run_dir.name)
        metadata.update(
            {
                "anchor_body_name": command.cfg.anchor_body_name,
                "body_names": list(command.cfg.body_names),
                "trainnr_manifest": MANIFEST_FILE,
            }
        )
        attach_metadata_to_onnx(str(policy_path), metadata)
        robot = str(trained.get(Key.ROBOT) or "unitree_g1")
        built = {Key.ROBOT: robot, Key.ACTUATOR: ACTUATOR, Key.TASK: TASK_ID}
        manifest = _manifest(
            env,
            Export(
                obs_dim=obs_dim,
                checkpoint=checkpoint,
                run_dir=run_dir,
                identity=manifest_identity(built, trained),
                robot=robot,
                certificate=certificate,
                policy_stamp=policy_stamp,
                max_abs_diff=max_abs_diff,
                command_basis="",
                input_name="obs",
                output_name=ACTIONS,
                clip_actions=agent.clip_actions,
                deploy=DeployFacts(),
            ),
        )
        frames = int(command.motion.time_step_total)
        manifest[Key.STAMP_OF] = f"{robot} tracking policy {checkpoint.stem}"
        manifest[Key.TASK] = trained.get("tracking_task") or manifest[Key.TASK]
        manifest[Key.CONTROL]["episode_length_s"] = frames * float(env.step_dt)
        manifest[Key.ONNX][TIME_STEP_INPUT] = TIME_STEP
        manifest[Key.ONNX]["outputs"] = (
            f"{ACTIONS}, then the reference at the step: "
            "joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, "
            "body_ang_vel_w"
        )
        manifest[Key.SCENE]["meshes"] = (
            f"{ASSETS}/ beside the scene: the robot's meshes from mjlab's asset zoo"
        )
        manifest.pop(Key.COMMANDS, None)
        manifest.pop(Key.COMMAND_BASIS, None)
        manifest.pop(Key.TERMINATION, None)
        if not manifest.get(Key.DR_BASIS):
            manifest[Key.DR_BASIS] = UNRECORDED
        manifest[Key.TRACKING] = tracking_block(
            cfg,
            training,
            command,
            env.scene["robot"].cfg.articulation.soft_joint_pos_limit_factor,
            [
                env.sim.mj_model.geom(i).name.rsplit("/", 1)[-1]
                for i in range(env.sim.mj_model.ngeom)
            ],
            lag_ms=lag_ms,
        )
        (out_dir / "scene.xml").write_text(_scene_xml(env), encoding="utf-8")
        # their own dates are not the deployment's: copied as new files
        shutil.copytree(g1_assets(), out_dir / ASSETS, copy_function=shutil.copyfile)
        (out_dir / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    finally:
        env.close()
    return {"dir": str(out_dir), "manifest": manifest}


if __name__ == "__main__":
    main()
