"""Stage a deployment for Unitree's controller and simulator, in the
layout their binaries expect, without touching their tree.

Their controller finds its project directory from its own binary's
real path (`<proj>/build/go2_ctrl` -> `<proj>`), reads
`<proj>/config/config.yaml` (the state machine) and the newest
`<proj>/config/policy/velocity/<version>/` holding `params/deploy.yaml`
and `exported/policy.onnx`. So a deployment gets a `unitree/` folder
shaped like that, with a COPY of their binary (a symlink would resolve
back to their tree), their config.yaml as shipped, our manifest as
their yaml, our ONNX. Their simulator reads `<simulate>/config.yaml`
beside its own binary; that one file is written into the cached
checkout with the Go2 scene, a joystick and the loopback interface.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from rq_pipeline.deploy.manifest import Manifest
from rq_pipeline.deploy.unitree_yaml import write_unitree_deploy

STAGE_DIR = "unitree"
POLICY_VERSION = "v0"
SIM_CONFIG = """robot: "go2"
robot_scene: "src/assets/robots/unitree_go2/xmls/scene_go2.xml"
domain_id: 0
interface: "lo"
use_joystick: 1
joystick_type: "xbox"
joystick_device: "{device}"
joystick_bits: 16
print_scene_information: 0
enable_elastic_band: 0
"""


def stage(manifest: Manifest, reference: Path, *, robot: str = "go2") -> Path:
    """`<deployment>/unitree/` ready for `build/<robot>_ctrl --network=lo`."""
    reference = Path(reference)
    proj = manifest.root / STAGE_DIR
    build = proj / "build"
    build.mkdir(parents=True, exist_ok=True)
    binary = reference / "deploy" / "robots" / robot / "build" / f"{robot}_ctrl"
    if not binary.is_file():
        raise FileNotFoundError(f"{binary}: build their controller first (docs/77 §7)")
    shutil.copy2(binary, build / binary.name)
    config = proj / "config"
    config.mkdir(exist_ok=True)
    shutil.copy2(
        reference / "deploy" / "robots" / robot / "config" / "config.yaml",
        config / "config.yaml",
    )
    version = config / "policy" / "velocity" / POLICY_VERSION
    write_unitree_deploy(manifest, version / "params" / "deploy.yaml")
    (version / "exported").mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest.policy_path, version / "exported" / "policy.onnx")
    return proj


def write_sim_config(reference: Path, *, device: str = "/dev/input/js0") -> Path:
    """Their simulator's config, beside its binary in the cached checkout."""
    out = Path(reference) / "simulate" / "config.yaml"
    out.write_text(SIM_CONFIG.format(device=device))
    return out


def sim_binary(reference: Path) -> Path:
    path = Path(reference) / "simulate" / "build" / "unitree_mujoco"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: build their simulator first (docs/77 §7)")
    return path
