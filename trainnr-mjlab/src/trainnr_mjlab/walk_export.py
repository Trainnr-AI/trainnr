"""Export a walk policy for deployment (docs/76 A6, docs/77): the actor
as ONNX with its observation normalization folded in, and a manifest
that says everything a runtime needs to drive it without this package -
the joint order the policy acts in, the actuator order the simulator
takes control in, gains, the home pose, the action scale and offset,
the ordered observation list with each term's source and width, the
control rate - plus the scene the policy was trained in, compiled from
the project's bundle with the injected actuators and keyframe, so a
plain MuJoCo runtime steps the same model.

    uv run python -m trainnr_mjlab.walk_export <run>/model_7999.pt \\
        --project <root> --robot go2 --name go2-c1-final

Everything in the manifest is read out of the BUILT environment, never
retyped from a config: the reference stack's hand-maintained deploy.yaml
drifted from its training config, and that is the failure this file
exists to make impossible (docs/77 §1). What the environment cannot
know - a vendor SDK's joint order, the vendor's own stack - the walk
declares (`walks.DeployFacts`) with its source, and the manifest says so.
The manifest's schema and keys are the pipeline's
(`trainnr.deploy.manifest`); this file writes them from there.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray
from trainnr.deploy.manifest import (
    HEADING_GAIN,
    MANIFEST_FILE,
    MANIFEST_SCHEMA,
    SOURCE_GAIT_PHASE,
    Key,
)
from trainnr.project.kinds import IDENTITY_FILE, stamp_run
from trainnr.scenes.stage import FLOOR_GEOM

from trainnr_mjlab.envelope import COMMAND_TERM
from trainnr_mjlab.walk_view import fit_of, require_same_identity
from trainnr_mjlab.walks import ROBOT_ENTITY, DeployFacts

POLICY_FILE = "policy.onnx"
SCENE_FILE = "scene.xml"
# The project's folder for deployments (`trainnr.project.locate.FOLDERS`).
DEPLOY_FOLDER = "deploy"
# Manifest numbers are float32 facts of the model; print them as such.
FLOAT_DIGITS = 6
DEGREES_DIGITS = 3
ACTOR_OBS_GROUP = "actor"
ACTION_TERM = "joint_pos"
FALL_TERM = "fell_over"
# How closely the ONNX must follow the torch actor on random inputs.
EXPORT_TOLERANCE = 1e-4
EXPORT_CHECK_SAMPLES = 64
PROBE_SEED = 0
FLOOR_NAME = FLOOR_GEOM  # the plane the stage deletes, by the same name
UNRECORDED = "unrecorded"
# mjlab's term functions, as `_source_of` names their sources.
BUILTIN_SENSOR = "builtin_sensor"
GENERATED_COMMANDS = "generated_commands"
GAIT_PHASE_FUNC = "gait_phase"


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
    print(f"[export] {out['manifest'][Key.STAMP_OF]} -> {out['dir']}", flush=True)


@dataclass(frozen=True)
class Export:
    """What the manifest writer needs beyond the environment."""

    obs_dim: int
    checkpoint: Path
    run_dir: Path
    identity: dict[str, Any]
    robot: str
    certificate: str | None
    policy_stamp: str | None
    max_abs_diff: float
    command_basis: str
    input_name: str
    output_name: str
    clip_actions: float | None
    deploy: DeployFacts


@dataclass(frozen=True)
class Floor:
    """The plane the policy trained on, read from the built model: its
    contact parameters decide what the feet touch."""

    size: tuple[float, float, float]
    condim: int
    priority: int
    friction: tuple[float, float, float]


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
    import torch  # noqa: PLC0415
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: PLC0415
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx  # noqa: PLC0415

    from trainnr_mjlab.envelope import (  # noqa: PLC0415
        checkpoint_iteration,
        pin_command_envelope,
    )
    from trainnr_mjlab.walks import use_project, walk_spec  # noqa: PLC0415

    use_project(project)
    spec = walk_spec(robot)
    checkpoint = Path(checkpoint).resolve()
    run_dir = checkpoint.parent
    identity_file = run_dir / IDENTITY_FILE
    trained = (
        json.loads(identity_file.read_text(encoding="utf-8"))
        if identity_file.is_file()
        else {}
    )
    cfg, identity = spec.env_cfg(
        dr_span=None, pin_scale=None, bundle=None, fit=fit_of(trained)
    )
    require_same_identity(trained, identity)  # the one gate (walk_view)
    cfg.scene.num_envs = 1
    device = "cpu"
    agent = spec.agent(1)
    # The commands the checkpoint trained under (trainnr_mjlab.envelope), so
    # the manifest's ranges - what the gate draws from - are the
    # policy's envelope, not the curriculum's first stage.
    envelope = pin_command_envelope(
        cfg, checkpoint_iteration(checkpoint.stem), agent.num_steps_per_env
    )
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg, device=device), clip_actions=agent.clip_actions
    )
    runner = MjlabOnPolicyRunner(env, asdict(agent), log_dir=None, device=device)
    _require_same_actor_width(checkpoint, env, device)
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
    rng = np.random.default_rng(PROBE_SEED)
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
            identity=manifest_identity(identity, trained),
            robot=robot,
            certificate=certificate,
            policy_stamp=policy_stamp,
            max_abs_diff=max_abs_diff,
            command_basis=envelope["basis"],
            input_name=inputs[0],
            output_name=outputs[0],
            clip_actions=agent.clip_actions,
            deploy=spec.deploy,
        ),
    )
    (out_dir / SCENE_FILE).write_text(_scene_xml(unwrapped), encoding="utf-8")
    attach_metadata_to_onnx(
        str(policy_path),
        {
            "trainnr_schema": MANIFEST_SCHEMA,
            Key.ROBOT: manifest[Key.ROBOT],
            Key.ACTUATOR: manifest[Key.ACTUATOR],
            Key.TASK: manifest[Key.TASK],
            Key.RUN: manifest[Key.RUN],
            "joint_names": manifest[Key.JOINTS]["policy_order"],
            "observation_names": [t["name"] for t in manifest[Key.OBSERVATIONS]],
        },
    )
    (out_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
    )
    env.close()
    return {"dir": str(out_dir), "manifest": manifest}


# The first actor layer's weight in rsl_rl's checkpoint: its input width
# is the observation width the checkpoint was trained on.
ACTOR_FIRST_LAYER = "mlp.0.weight"


def trained_actor_width(checkpoint: Path, device: str) -> int | None:
    """The observation width the checkpoint's actor was trained on, from
    its first layer; None when the file is not rsl_rl's layout."""
    import torch  # noqa: PLC0415

    state = torch.load(str(checkpoint), map_location=device, weights_only=False)
    weight = (state.get("actor_state_dict") or {}).get(ACTOR_FIRST_LAYER)
    return None if weight is None else int(weight.shape[1])


def _require_same_actor_width(checkpoint: Path, env: Any, device: str) -> None:
    """Refuse by name a checkpoint whose actor observes a different width
    than the declared walk's actor builds now: an older recipe (the Go2
    actor became deployable on 2026-09-11 and lost the base linear
    velocity for a gait clock), never a torch shape traceback."""
    trained_width = trained_actor_width(checkpoint, device)
    if trained_width is None:
        return  # not rsl_rl's layout; the loader's own refusal names it
    built_width = int(env.get_observations()[ACTOR_OBS_GROUP].shape[-1])
    if trained_width != built_width:
        raise SystemExit(
            f"{checkpoint}: its actor observes {trained_width} terms, the "
            f"declared walk's actor observes {built_width} now; the checkpoint "
            "was trained by an earlier recipe and cannot be exported through "
            "this one - retrain on the current recipe"
        )


# What the manifest takes from the RUN, not from the export's own env: the
# export builds its env with no randomization (it only runs the actor), so
# the env's `dr_basis` said "none" on every Go2 deployment whose run
# trained under a declared ±0.1 span (the review of 2026-09-24). The
# trained identity's word is the policy's; a run that predates the record
# says so rather than lending the export env's.
TRAINED_KEYS = (Key.SEED, Key.TASK, Key.DR_BASIS, Key.FIT, Key.FIT_BASIS)


def manifest_identity(built: dict[str, Any], trained: dict[str, Any]) -> dict[str, Any]:
    """The identity a deployment's manifest carries: the export env's
    robot and actuator (gated equal to the run's by
    `require_same_identity`), and the run's own seed, task and
    randomization basis; `unrecorded` where the run did not write one."""
    out = dict(built)
    out.update({k: trained[k] for k in TRAINED_KEYS if k in trained})
    if not trained.get(Key.DR_BASIS):  # never the export env's own "none"
        out[Key.DR_BASIS] = UNRECORDED
    return out


def _run_onnx(path: Path, probe: NDArray[np.float32]) -> NDArray[np.float32]:
    """The exported graph takes one observation at a time (rsl_rl exports a
    fixed batch of one, as a runtime uses it), so the probe runs row by row."""
    import onnxruntime as ort  # noqa: PLC0415

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return np.stack([session.run(None, {name: row[None, :]})[0][0] for row in probe])


def _manifest(env: Any, ex: Export) -> dict[str, Any]:
    """Every number read from the built environment; the vendor facts
    from the walk's declaration."""
    import torch  # noqa: PLC0415
    from mjlab.envs.mdp.actions import JointPositionAction  # noqa: PLC0415

    entity = env.scene[ROBOT_ENTITY]
    joint_action = env.action_manager.get_term(ACTION_TERM)
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
    twist = env.command_manager.get_term(COMMAND_TERM)
    twist_cfg = getattr(twist, "cfg", None)
    ranges = getattr(twist_cfg, "ranges", None)
    command_ranges = (
        {k: list(v) for k, v in asdict(ranges).items() if v is not None}
        if ranges
        else {}
    )
    commands: dict[str, Any] = {"twist": command_ranges}
    gain = getattr(twist_cfg, "heading_control_stiffness", None)
    if getattr(twist_cfg, "heading_command", False) and gain is not None:
        # the heading-pursuit gain the policy steered by, what a course
        # gate steers by (trainnr.deploy.course)
        commands[HEADING_GAIN] = float(gain)
    physics_dt = float(env.cfg.sim.mujoco.timestep)
    decimation = int(env.cfg.decimation)
    terrain = env.cfg.scene.terrain.terrain_type if env.cfg.scene.terrain else None
    clip = ex.clip_actions
    deploy = ex.deploy
    manifest: dict[str, Any] = {
        Key.SCHEMA: MANIFEST_SCHEMA,
        Key.STAMP_OF: f"{ex.robot} policy {ex.checkpoint.stem}",
        Key.POLICY: ex.policy_stamp or UNRECORDED,
        Key.CHECKPOINT: ex.checkpoint.name,
        Key.RUN: stamp_run(ex.run_dir),
        Key.ROBOT: ex.identity[Key.ROBOT],
        Key.ACTUATOR: ex.identity[Key.ACTUATOR],
        Key.TASK: ex.identity.get(Key.TASK) or UNRECORDED,
        Key.DR_BASIS: ex.identity.get(Key.DR_BASIS) or UNRECORDED,
        Key.SEED: ex.identity.get(Key.SEED),
        Key.CERTIFICATE: ex.certificate,
        Key.CONTROL: {
            "physics_timestep_s": physics_dt,
            "decimation": decimation,
            "control_hz": round(1.0 / (physics_dt * decimation)),
            "episode_length_s": float(env.cfg.episode_length_s),
        },
        Key.JOINTS: {
            "policy_order": joint_names,
            "ctrl_order": ctrl_names,
            "action_to_ctrl": action_to_ctrl,
            "sdk_order_map": (
                list(deploy.sdk_joint_map) if deploy.sdk_joint_map else None
            ),
            "sdk_order_source": deploy.sdk_joint_map_source,
            "stiffness": stiffness,
            "damping": damping,
            "effort_limit": effort,
            "default_pos": default_pos,
        },
        Key.ACTION: {
            "kind": "joint position target = default_pos + scale * action",
            "scale": scale_list,
            "offset": default_pos,
            "clip": [-float(clip), float(clip)] if clip is not None else None,
        },
        Key.OBSERVATIONS: observations,
        Key.ONNX: {
            "file": POLICY_FILE,
            "input": ex.input_name,
            "output": ex.output_name,
            "input_width": ex.obs_dim,
            "output_width": len(joint_names),
            "normalization": "folded into the graph",
            "export_check_max_abs_diff": ex.max_abs_diff,
        },
        Key.SCENE: {
            "file": SCENE_FILE,
            "meshes": "the robot bundle's assets/, by name (the bundle is the "
            "version above)",
            "terrain": terrain,
            "floor": asdict(_floor_of(model)),
        },
        Key.COMMANDS: commands,
        Key.COMMAND_BASIS: ex.command_basis,
        Key.TERMINATION: {"fell_over_deg": _fell_over_deg(env)},
    }
    if deploy.unitree is not None:
        manifest[Key.UNITREE] = asdict(deploy.unitree)
    return manifest


def _unprefixed(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _source_of(cfg: Any) -> str:
    """The manifest's name for what a term computes, from mjlab's term
    function and its params."""
    name = getattr(cfg.func, "__name__", str(cfg.func))
    params = cfg.params or {}
    if name == BUILTIN_SENSOR:
        return f"sensor {params.get('sensor_name', '')}".strip()
    if name == GENERATED_COMMANDS:
        return f"command {params.get('command_name', '')}".strip()
    if name == GAIT_PHASE_FUNC:
        return SOURCE_GAIT_PHASE
    return name


def _fell_over_deg(env: Any) -> float | None:
    term = env.cfg.terminations.get(FALL_TERM)
    if term is None:
        return None
    angle = (term.params or {}).get("limit_angle")
    return (
        round(math.degrees(float(angle)), DEGREES_DIGITS) if angle is not None else None
    )


def _floor_of(model: mujoco.MjModel) -> Floor:
    """The plane the trained scene stands on, from the built model - its
    contact parameters, never retyped."""
    planes = [
        g
        for g in range(model.ngeom)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE
    ]
    if len(planes) != 1:
        raise SystemExit(
            f"the built scene has {len(planes)} plane geoms; the exporter writes "
            "one floor"
        )
    g = planes[0]
    sx, sy, sz = (float(v) for v in model.geom_size[g])
    slide, spin, roll = (float(v) for v in model.geom_friction[g])
    return Floor(
        size=(sx, sy, sz),
        condim=int(model.geom_condim[g]),
        priority=int(model.geom_priority[g]),
        friction=(slide, spin, roll),
    )


def _scene_xml(env: Any) -> str:
    """The trained model as MJCF, actuators and keyframe injected, the
    plane it trained on under it, the training timestep - what a plain
    MuJoCo runtime loads with the bundle's meshes."""
    from mjlab.entity.entity import Entity  # noqa: PLC0415

    # A fresh entity: the scene's own is attached to the scene spec and
    # cannot be serialized on its own.
    spec = Entity(env.cfg.scene.entities[ROBOT_ENTITY]).spec
    spec.option.timestep = float(env.cfg.sim.mujoco.timestep)
    floor = _floor_of(env.sim.mj_model)
    geom = spec.worldbody.add_geom()
    geom.name = FLOOR_NAME
    geom.type = mujoco.mjtGeom.mjGEOM_PLANE
    geom.size = list(floor.size)
    geom.condim = floor.condim
    geom.priority = floor.priority
    geom.friction = list(floor.friction)
    return str(spec.to_xml())


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
