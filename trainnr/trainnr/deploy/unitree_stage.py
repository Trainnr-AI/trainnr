"""Stage a deployment for Unitree's controller and simulator, in the
layout their binaries expect, without touching their tree; and run the
two around the gate.

Their controller finds its project directory from its own binary's
real path (`<proj>/build/<robot>_ctrl` -> `<proj>`), reads
`<proj>/config/config.yaml` (the state machine) and the newest
`<proj>/config/policy/velocity/<version>/` holding `params/deploy.yaml`
and `exported/policy.onnx`. Their simulator does the same one level up:
`<sim>/build/unitree_mujoco` reads `<sim>/config.yaml` and resolves a
relative scene against `<sim>/..` (their `simulate/src/main.cc`, read
2026-09-12). So a deployment gets a `unitree/` folder with COPIES of
both binaries (a symlink would resolve back to their tree), their
controller config as shipped, our manifest as their yaml, our ONNX, and
a simulator config of our own naming the robot, the pad and loopback,
with the scene as an absolute path into their checkout. Which robot,
controller and scene: the manifest's `unitree` block, declared by the
walk that trained the policy - nothing here names a robot.

The checkout: `$TRAINNR_UNITREE_REFERENCE`, else the cache the
onboarding tools use. Linux only (their SDK, uinput): the registry
refuses it by name elsewhere.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import IO, Any

from trainnr.deploy.dds_runtime import DOMAIN_ID, NETWORK
from trainnr.deploy.gamepad import JOYSTICK_BITS, VirtualPad
from trainnr.deploy.manifest import Manifest, UnitreeFacts
from trainnr.deploy.runtimes import RUNTIMES, require_platform
from trainnr.deploy.unitree_yaml import UNITREE_DEPLOY_FILE, write_unitree_deploy

STAGE_DIR = "unitree"
POLICY_VERSION = "v0"
REFERENCE_ENV = "TRAINNR_UNITREE_REFERENCE"
REFERENCE_CACHE = Path("~/.cache/trainnr/unitree_rl_mjlab")
SIM_BINARY = "unitree_mujoco"
SIM_DIR = "simulate"
CTRL_CONFIG = "config.yaml"
JOYSTICK_TYPE = "xbox"
SIM_CONFIG = """robot: "{robot}"
robot_scene: "{scene}"
domain_id: {domain_id}
interface: "{network}"
use_joystick: {use_joystick}
joystick_type: "{joystick_type}"
joystick_device: "{device}"
joystick_bits: {joystick_bits}
print_scene_information: 0
enable_elastic_band: 0
"""
# Unitree's SDK installs CycloneDDS under /usr/local/lib, which the
# loader does not search until ldconfig runs; both binaries are told.
SDK_LIB_DIRS = ("/usr/local/lib",)
INPUT_DIR = Path("/dev/input")
JOYSTICK_GLOB = "js*"
JOYDEV_APPEAR_S = 0.5  # the joystick node appears after the pad is created
SIM_WARMUP_S = 4.0  # their simulator loads the scene and opens DDS
CTRL_WARMUP_S = 3.0  # their controller waits for the first LowState
SHUTDOWN_WAIT_S = 5.0  # a process that ignores TERM this long is killed
LOG_DIR = "logs"


def reference_dir(explicit: Path | None = None) -> Path:
    """The reference checkout: the argument, the environment, the cache."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    named = os.environ.get(REFERENCE_ENV)
    return (Path(named) if named else REFERENCE_CACHE).expanduser().resolve()


def facts_of(manifest: Manifest) -> UnitreeFacts:
    """The manifest's Unitree block, or a refusal naming the deployment."""
    facts = manifest.unitree
    if facts is None:
        raise ValueError(
            f"{manifest.root}: the manifest names no Unitree stack for this robot "
            "(the walk that trained it declares none); their simulator and "
            "controller cannot run it"
        )
    return facts


BUILD_DIR = "build"  # where their CMake puts a binary, in every folder of theirs


def robot_dir(reference: Path, facts: UnitreeFacts) -> Path:
    """Their checkout's folder for this robot: the controller's source,
    build and config."""
    return Path(reference) / "deploy" / "robots" / facts.robot


def controller_binary(reference: Path, facts: UnitreeFacts) -> Path:
    path = robot_dir(reference, facts) / BUILD_DIR / facts.controller
    if not path.is_file():
        raise FileNotFoundError(f"{path}: build their controller first (docs/77 §7)")
    return path


def sim_binary(reference: Path) -> Path:
    path = reference / SIM_DIR / BUILD_DIR / SIM_BINARY
    if not path.is_file():
        raise FileNotFoundError(f"{path}: build their simulator first (docs/77 §7)")
    return path


def stage(manifest: Manifest, reference: Path) -> Path:
    """`<deployment>/unitree/` ready for `build/<controller> --network=lo`."""
    reference = Path(reference)
    facts = facts_of(manifest)
    proj = manifest.root / STAGE_DIR
    build = proj / BUILD_DIR
    build.mkdir(parents=True, exist_ok=True)
    binary = controller_binary(reference, facts)
    shutil.copy2(binary, build / binary.name)
    config = proj / "config"
    config.mkdir(exist_ok=True)
    shutil.copy2(
        robot_dir(reference, facts) / "config" / CTRL_CONFIG, config / CTRL_CONFIG
    )
    version = config / "policy" / "velocity" / POLICY_VERSION
    write_unitree_deploy(manifest, version / "params" / UNITREE_DEPLOY_FILE)
    (version / "exported").mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest.policy_path, version / "exported" / manifest.policy_path.name)
    return proj


def stage_simulator(
    manifest: Manifest, reference: Path, *, device: Path | None
) -> Path:
    """`<deployment>/unitree/simulate/`: a copy of their simulator beside a
    config of ours, the scene named absolutely in their checkout. With no
    `device` their simulator reads no joystick (a stack with no pad)."""
    reference = Path(reference)
    facts = facts_of(manifest)
    sim = manifest.root / STAGE_DIR / SIM_DIR
    build = sim / BUILD_DIR
    build.mkdir(parents=True, exist_ok=True)
    binary = sim_binary(reference)
    shutil.copy2(binary, build / binary.name)
    (sim / CTRL_CONFIG).write_text(
        SIM_CONFIG.format(
            robot=facts.robot,
            scene=reference / facts.scene,
            domain_id=DOMAIN_ID,
            network=NETWORK,
            use_joystick=int(device is not None),
            joystick_type=JOYSTICK_TYPE,
            device=device or "",
            joystick_bits=JOYSTICK_BITS,
        ),
        encoding="utf-8",
    )
    return sim


def joystick_nodes() -> list[Path]:
    return sorted(INPUT_DIR.glob(JOYSTICK_GLOB)) if INPUT_DIR.is_dir() else []


class UnitreeStack:
    """Their simulator and controller as two subprocesses for the gate's
    duration: the pad first (their simulator opens the joystick at
    start), the simulator, then the staged controller on loopback; logs
    beside the deployment. A context: enter starts, exit ends both. The
    robot is their simulator: a stand-in, and every record says so."""

    stand_in = True
    state_from = (
        "Unitree's simulator over DDS in their fixed stand (a simulation stand-in)"
    )

    def __init__(
        self,
        manifest: Manifest,
        *,
        reference: Path | None = None,
        log: IO[str] = sys.stdout,
        with_pad: bool = True,
    ) -> None:
        require_platform(RUNTIMES["dds"])
        # Without a pad their controller stays in its first state (passive:
        # damping, still publishing rt/lowcmd) - what a telemetry capture of
        # the stand-in needs, and all it can have where /dev/uinput is
        # root's (the virtual pad needs it writable).
        self.with_pad = with_pad
        self.manifest = manifest
        self.reference = reference_dir(reference)
        self._log = log
        self.pad: VirtualPad | None = None
        self.sim: subprocess.Popen[bytes] | None = None
        self.ctrl: subprocess.Popen[bytes] | None = None
        self._files: list[IO[bytes]] = []

    def __enter__(self) -> UnitreeStack:
        # Whatever fails on the way up - a missing binary, an exited
        # process, a Ctrl-C in the warm-up sleeps - the pad and the
        # processes already started come down: `__exit__` never runs when
        # `__enter__` raises. The children are killed as a group by the
        # job runner's TERM; a bare `kill <pid>` of the gate alone leaves
        # them (docs/77 §7): signal the group.
        try:
            return self._start()
        except BaseException:
            self.close()
            raise

    def _start(self) -> UnitreeStack:
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = ":".join(
            p for p in (env.get("LD_LIBRARY_PATH", ""), *SDK_LIB_DIRS) if p
        )
        device: Path | None = None
        if self.with_pad:
            self.pad = VirtualPad()
            time.sleep(JOYDEV_APPEAR_S)
            nodes = joystick_nodes()
            if not nodes:
                self.close()
                raise RuntimeError(
                    f"no {INPUT_DIR / JOYSTICK_GLOB} after creating the virtual "
                    "pad: modprobe joydev"
                )
            device = nodes[-1]
            print(f"[gate] virtual pad at {device}", file=self._log, flush=True)
        sim = stage_simulator(self.manifest, self.reference, device=device)
        proj = stage(self.manifest, self.reference)
        logs = proj / LOG_DIR
        logs.mkdir(exist_ok=True)
        self.sim = self._spawn(
            [str(sim / BUILD_DIR / SIM_BINARY)],
            sim / BUILD_DIR,
            env,
            logs / "simulate.log",
        )
        time.sleep(SIM_WARMUP_S)
        controller = facts_of(self.manifest).controller
        self.ctrl = self._spawn(
            [str(proj / BUILD_DIR / controller), f"--network={NETWORK}"],
            proj / BUILD_DIR,
            env,
            logs / "controller.log",
        )
        time.sleep(CTRL_WARMUP_S)
        for name, proc in (("simulator", self.sim), ("controller", self.ctrl)):
            if proc.poll() is not None:
                self.close()
                raise RuntimeError(
                    f"their {name} exited with {proc.returncode}; see {logs}"
                )
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def runtime_options(self) -> dict[str, Any]:
        """What the runtime opened inside this stack must share with it:
        the one pad their simulator reads. Two pads were two joystick
        nodes, and the runtime moved the one nobody read (2026-09-12)."""
        return {"pad": self.pad} if self.pad is not None else {}

    def _spawn(
        self, argv: list[str], cwd: Path, env: dict[str, str], log: Path
    ) -> subprocess.Popen[bytes]:
        handle = log.open("ab")
        self._files.append(handle)
        return subprocess.Popen(
            argv, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT
        )

    def close(self) -> None:
        for proc in (self.ctrl, self.sim):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(SHUTDOWN_WAIT_S)
                except subprocess.TimeoutExpired:
                    proc.kill()
        for handle in self._files:
            handle.close()
        self._files = []
        if self.pad is not None:
            self.pad.close()
            self.pad = None
