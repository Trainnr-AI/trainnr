"""Export a walk policy for deployment (docs/76 A6, docs/77): the actor
as ONNX with its observation normalization folded in, and a manifest
that says everything a runtime needs to drive it without this package -
the joint order the policy acts in, the actuator order the simulator
takes control in, gains, the home pose, the action scale and offset,
the ordered observation list with each term's source and width, the
control rate - plus the scene the policy was trained in, compiled from
the project's bundle with the injected actuators and keyframe, so a
plain MuJoCo runtime steps the same model.

    uv run python -m rq_mjlab.walk_export <run>/model_7999.pt \\
        --project <root> --robot go2 --name go2-c1-final

Everything in the manifest is read out of the BUILT environment, never
retyped from a config: the reference stack's hand-maintained deploy.yaml
drifted from its training config, and that is the failure this file
exists to make impossible (docs/77 §1).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "trainnr-deploy/1"
MANIFEST_FILE = "deploy.json"
POLICY_FILE = "policy.onnx"
SCENE_FILE = "scene.xml"
# The project's folder for deployments (`rq_pipeline.project.locate.FOLDERS`).
DEPLOY_FOLDER = "deploy"
# Manifest numbers are float32 facts of the model; print them as such.
FLOAT_DIGITS = 6
ACTOR_OBS_GROUP = "actor"
# How closely the ONNX must follow the torch actor on random inputs.
EXPORT_TOLERANCE = 1e-4
EXPORT_CHECK_SAMPLES = 64
# The reference's SDK joint order for the Go2 (deploy.yaml, docs/77 §1):
# the policy's MJCF order FL, FR, RL, RR against the SDK's FR, FL, RR, RL.
SDK_JOINT_MAPS = {"go2": [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]}
SDK_JOINT_MAP_SOURCE = (
    "unitree_rl_mjlab deploy/robots/go2 deploy.yaml joint_ids_map (declared)"
)
FLOOR_XML = (
    '  <worldbody>\n    <geom name="floor" type="plane" size="0 0 0.05" '
    'condim="3" friction="0.6" priority="1"/>\n'
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--name", required=True, help="the deployment's folder name")
    parser.add_argument(
        "--certificate",
        default=None,
        help="the evaluation this policy carries, by version",
    )
    parser.add_argument(
        "--policy-stamp", default=None, help="the policy artifact's version"
    )
    args = parser.parse_args()
    out = export(
        args.checkpoint,
        project=args.project,
        robot=args.robot,
        name=args.name,
        certificate=args.certificate,
        policy_stamp=args.policy_stamp,
    )
    print(f"[export] {out['manifest']['stamp_of']} -> {out['dir']}", flush=True)


def export(  # noqa: PLR0913 - the export's own knobs, each named
    checkpoint: Path,
    *,
    project: Path,
    robot: str,
    name: str,
    certificate: str | None = None,
    policy_stamp: str | None = None,
) -> dict[str, Any]:
    """Write `<project>/deploy/<name>/{policy.onnx, deploy.json,
    scene.xml}`; returns the manifest and the directory."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: PLC0415
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx  # noqa: PLC0415

    from rq_mjlab.envelope import (  # noqa: PLC0415
        checkpoint_iteration,
        pin_command_envelope,
    )
    from rq_mjlab.walks import use_project, walk_spec  # noqa: PLC0415

    use_project(project)
    spec = walk_spec(robot)
    checkpoint = Path(checkpoint).resolve()
    run_dir = checkpoint.parent
    identity_file = run_dir / "identity.json"
    trained = json.loads(identity_file.read_text()) if identity_file.is_file() else {}
    cfg, identity = spec.env_cfg(dr_span=None, pin_scale=None, bundle=None)
    for key in ("robot", "actuator"):
        if trained.get(key) not in (None, identity.get(key)):
            raise SystemExit(
                f"identity mismatch on {key}: the run was trained on "
                f"{trained.get(key)}, this project's is {identity.get(key)}"
            )
    cfg.scene.num_envs = 1
    device = "cpu"
    agent = spec.agent(1)
    # The commands the checkpoint trained under (rq_mjlab.envelope), so
    # the manifest's ranges - what the gate draws from - are the
    # policy's envelope, not the curriculum's first stage.
    envelope = pin_command_envelope(
        cfg, checkpoint_iteration(checkpoint.stem), agent.num_steps_per_env
    )
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg, device=device), clip_actions=agent.clip_actions
    )
    runner = MjlabOnPolicyRunner(env, asdict(agent), log_dir=None, device=device)
    runner.load(
        str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )

    out_dir = Path(project) / DEPLOY_FOLDER / name
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; a deployment is never overwritten")
    out_dir.mkdir(parents=True)
    runner.export_policy_to_onnx(str(out_dir), POLICY_FILE)
    policy_path = out_dir / POLICY_FILE

    # The torch actor and the ONNX must agree on random inputs.
    unwrapped = env.unwrapped
    obs = env.get_observations()
    obs_dim = int(obs[ACTOR_OBS_GROUP].shape[-1])
    rng = np.random.default_rng(0)
    probe = rng.standard_normal((EXPORT_CHECK_SAMPLES, obs_dim)).astype(np.float32)
    with torch.inference_mode():
        onnx_module = runner.alg.get_policy().as_onnx(verbose=False).cpu().eval()
        expected = onnx_module(torch.from_numpy(probe)).numpy()
    got = _run_onnx(policy_path, probe)
    max_abs_diff = float(np.max(np.abs(expected - got)))
    if max_abs_diff > EXPORT_TOLERANCE:
        raise SystemExit(
            f"the ONNX disagrees with the actor by {max_abs_diff:.2e} "
            f"(> {EXPORT_TOLERANCE})"
        )
    inputs = list(getattr(onnx_module, "input_names", None) or ["obs"])
    outputs = list(getattr(onnx_module, "output_names", None) or ["actions"])
    manifest = _manifest(
        unwrapped,
        Export(
            obs_dim=obs_dim,
            checkpoint=checkpoint,
            run_dir=run_dir,
            identity={
                **identity,
                **{k: trained[k] for k in ("seed", "task") if k in trained},
            },
            robot=robot,
            certificate=certificate,
            policy_stamp=policy_stamp,
            max_abs_diff=max_abs_diff,
            command_basis=envelope["basis"],
            input_name=inputs[0],
            output_name=outputs[0],
        ),
    )
    (out_dir / SCENE_FILE).write_text(_scene_xml(unwrapped))
    attach_metadata_to_onnx(
        str(policy_path),
        {
            "trainnr_schema": MANIFEST_SCHEMA,
            "robot": manifest["robot"],
            "actuator": manifest["actuator"],
            "task": manifest.get("task") or "unrecorded",
            "run": manifest["run"],
            "joint_names": manifest["joints"]["policy_order"],
            "observation_names": [t["name"] for t in manifest["observations"]],
        },
    )
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=1) + "\n")
    env.close()
    return {"dir": str(out_dir), "manifest": manifest}


class Export:
    """What the manifest writer needs beyond the environment."""

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


def _run_onnx(path: Path, probe: Any) -> Any:
    """The exported graph takes one observation at a time (rsl_rl exports a
    fixed batch of one, as a runtime uses it), so the probe runs row by row."""
    import numpy as np  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return np.stack([session.run(None, {name: row[None, :]})[0][0] for row in probe])


def _manifest(env: Any, ex: Export) -> dict[str, Any]:
    """Every number read from the built environment."""
    import torch  # noqa: PLC0415
    from mjlab.envs.mdp.actions import JointPositionAction  # noqa: PLC0415
    from rq_pipeline.project.kinds import stamp_run  # noqa: PLC0415

    entity = env.scene["robot"]
    joint_action = env.action_manager.get_term("joint_pos")
    assert isinstance(joint_action, JointPositionAction)
    model = env.sim.mj_model
    joint_names = list(entity.joint_names)
    # The actuators as the simulator orders them, and which joint each drives.
    # The scene namespaces the entity's names (`robot/FL_hip_joint`); the
    # manifest keeps the robot's own, the way its MJCF and its SDK name them.
    ctrl_names = [_unprefixed(model.actuator(i).name) for i in range(model.nu)]
    joint_of_ctrl = [
        _unprefixed(model.joint(model.actuator_trnid[i, 0]).name)
        for i in range(model.nu)
    ]
    action_to_ctrl = [joint_of_ctrl.index(j) for j in joint_names]
    stiffness = [float(model.actuator_gainprm[c, 0]) for c in action_to_ctrl]
    damping = [float(-model.actuator_biasprm[c, 2]) for c in action_to_ctrl]
    effort = [float(model.actuator_forcerange[c, 1]) for c in action_to_ctrl]
    default_pos = [
        round(float(v), FLOAT_DIGITS)
        for v in entity.data.default_joint_pos[0].cpu().tolist()
    ]
    scale = joint_action._scale
    scale_list = (
        [float(v) for v in scale[0].cpu().tolist()]
        if isinstance(scale, torch.Tensor)
        else [float(scale)] * len(joint_names)
    )
    observations = []
    manager = env.observation_manager
    dims = manager.group_obs_term_dim[ACTOR_OBS_GROUP]
    for term, dim in zip(manager.active_terms[ACTOR_OBS_GROUP], dims, strict=True):
        cfg = manager.get_term_cfg(ACTOR_OBS_GROUP, term)
        observations.append(
            {
                "name": term,
                "width": int(dim[0]),
                "source": _source_of(cfg),
                "params": {
                    k: v
                    for k, v in (cfg.params or {}).items()
                    if isinstance(v, (str, int, float))
                },
                "scale": _jsonable(cfg.scale) if cfg.scale is not None else 1.0,
                "clip": list(cfg.clip) if cfg.clip is not None else None,
                "history_length": int(cfg.history_length),
            }
        )
    widths = sum(t["width"] for t in observations)
    if widths != ex.obs_dim:
        raise SystemExit(
            f"observation widths sum to {widths}, the actor takes {ex.obs_dim}"
        )
    twist = env.command_manager.get_term("twist")
    ranges = getattr(getattr(twist, "cfg", None), "ranges", None)
    command_ranges = (
        {k: list(v) for k, v in asdict(ranges).items() if v is not None}
        if ranges
        else {}
    )
    physics_dt = float(env.cfg.sim.mujoco.timestep)
    decimation = int(env.cfg.decimation)
    terrain = env.cfg.scene.terrain.terrain_type if env.cfg.scene.terrain else None
    return {
        "schema": MANIFEST_SCHEMA,
        "stamp_of": f"{ex.robot} policy {ex.checkpoint.stem}",
        "policy": ex.policy_stamp or "unrecorded",
        "checkpoint": ex.checkpoint.name,
        "run": stamp_run(ex.run_dir),
        "robot": ex.identity["robot"],
        "actuator": ex.identity["actuator"],
        "task": ex.identity.get("task"),
        "dr_basis": ex.identity.get("dr_basis"),
        "seed": ex.identity.get("seed"),
        "certificate": ex.certificate,
        "control": {
            "physics_timestep_s": physics_dt,
            "decimation": decimation,
            "control_hz": round(1.0 / (physics_dt * decimation)),
            "episode_length_s": float(env.cfg.episode_length_s),
        },
        "joints": {
            "policy_order": joint_names,
            "ctrl_order": ctrl_names,
            "action_to_ctrl": action_to_ctrl,
            "sdk_order_map": SDK_JOINT_MAPS.get(ex.robot),
            "sdk_order_source": SDK_JOINT_MAP_SOURCE
            if ex.robot in SDK_JOINT_MAPS
            else None,
            "stiffness": stiffness,
            "damping": damping,
            "effort_limit": effort,
            "default_pos": default_pos,
        },
        "action": {
            "kind": "joint position target = default_pos + scale * action",
            "scale": scale_list,
            "offset": default_pos,
            "clip": None,
        },
        "observations": observations,
        "onnx": {
            "file": POLICY_FILE,
            "input": ex.input_name,
            "output": ex.output_name,
            "input_width": ex.obs_dim,
            "output_width": len(joint_names),
            "normalization": "folded into the graph",
            "export_check_max_abs_diff": ex.max_abs_diff,
        },
        "scene": {
            "file": SCENE_FILE,
            "meshes": "the robot bundle's assets/, by name (the bundle is the "
            "version above)",
            "terrain": terrain,
        },
        "commands": {"twist": command_ranges},
        "command_basis": ex.command_basis,
        "termination": {"fell_over_deg": _fell_over_deg(env)},
    }


def _unprefixed(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _source_of(cfg: Any) -> str:
    name = getattr(cfg.func, "__name__", str(cfg.func))
    params = cfg.params or {}
    if name == "builtin_sensor":
        return f"sensor {params.get('sensor_name', '')}".strip()
    if name == "generated_commands":
        return f"command {params.get('command_name', '')}".strip()
    return name


def _fell_over_deg(env: Any) -> float | None:
    term = env.cfg.terminations.get("fell_over")
    if term is None:
        return None
    angle = (term.params or {}).get("limit_angle")
    return round(math.degrees(float(angle)), 3) if angle is not None else None


def _scene_xml(env: Any) -> str:
    """The trained model as MJCF, actuators and keyframe injected, a
    plane under it, the training timestep - what a plain MuJoCo runtime
    loads with the bundle's meshes."""
    from mjlab.entity.entity import Entity  # noqa: PLC0415

    # A fresh entity: the scene's own is attached to the scene spec and
    # cannot be serialized on its own.
    xml = Entity(env.cfg.scene.entities["robot"]).spec.to_xml()
    physics_dt = float(env.cfg.sim.mujoco.timestep)
    xml = xml.replace("  <worldbody>\n", FLOOR_XML, 1)
    if re.search(r"<option[^>]*timestep=", xml):
        xml = re.sub(r'timestep="[^"]*"', f'timestep="{physics_dt:g}"', xml, count=1)
    elif "<option" in xml:
        xml = xml.replace("<option", f'<option timestep="{physics_dt:g}"', 1)
    else:
        xml = xml.replace(
            "  <worldbody>", f'  <option timestep="{physics_dt:g}"/>\n  <worldbody>', 1
        )
    return xml


def _jsonable(v: Any) -> Any:
    try:
        import torch  # noqa: PLC0415

        if isinstance(v, torch.Tensor):
            return v.cpu().tolist()
    except ImportError:
        pass
    return v


if __name__ == "__main__":
    main()
