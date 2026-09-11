"""Run the sim-to-sim gate on a project's deployment (docs/76 A6).

    cd pipeline && uv run --extra sim --extra deploy \\
        python ../tools/gate-deployment.py --project ../projects/go2-walk \\
        --name go2-c1-final --trials 20

Drives the exported ONNX policy through its manifest in plain MuJoCo,
judges every trial the certificate's way, writes `gate.json` beside the
manifest, reindexes the project. Exits 1 when the gate fails.
"""

import argparse
import json
import sys
from pathlib import Path

from _lab import bootstrap

bootstrap()

from rq_pipeline.deploy.dds_runtime import open_dds_runtime  # noqa: E402
from rq_pipeline.deploy.gate import (  # noqa: E402
    DEFAULT_TOLERANCE,
    DEFAULT_TRIALS,
    gate,
)
from rq_pipeline.deploy.manifest import Manifest, load_manifest  # noqa: E402
from rq_pipeline.deploy.runtime import assets_dir_of, open_runtime  # noqa: E402
from rq_pipeline.project import index_project, write_index  # noqa: E402
from rq_pipeline.project.kinds import CERTIFICATE_FILE  # noqa: E402
from rq_pipeline.project.locate import Project  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True, help="the deployment's folder name")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--runtime",
        default="mujoco",
        choices=("mujoco", "dds"),
        help="plain MuJoCo through our manifest, or Unitree's simulator and controller",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path.home() / ".cache" / "trainnr" / "unitree_rl_mjlab",
        help="the reference checkout with their simulator and controller built",
    )
    args = parser.parse_args()
    project = Project(args.project.resolve()).use()
    folder = project.folder("deploy") / args.name
    manifest = load_manifest(folder)
    try:
        assets_dir = assets_dir_of(manifest)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    certificate = _certificate(project, manifest.raw.get("certificate"))
    stack = unitree_stack(manifest, args.reference) if args.runtime == "dds" else None
    try:
        record = gate(
            folder,
            assets_dir=assets_dir,
            trials=args.trials,
            seed=args.seed,
            tolerance=args.tolerance,
            certificate=certificate,
            open=open_dds_runtime if stack else open_runtime,
        )
    finally:
        if stack:
            stack.close()
    verdict = record["verdict"]
    print(
        f"[gate] {record['successes']}/{record['trials']} ci95 {record['ci95']} "
        f"passed={verdict.get('passed')}",
        flush=True,
    )
    write_index(project, index_project(project))
    sys.exit(0 if verdict.get("passed") in (True, None) else 1)


class unitree_stack:  # noqa: N801 - a context, named for what it holds
    """Their simulator and controller as two subprocesses for the gate's
    duration: the pad first (their simulator opens the joystick at
    start), the simulator, then the staged controller on loopback; logs
    beside the deployment. `close()` ends both."""

    def __init__(self, manifest: Manifest, reference: Path) -> None:
        import os  # noqa: PLC0415
        import subprocess  # noqa: PLC0415
        import time  # noqa: PLC0415

        from rq_pipeline.deploy.gamepad import VirtualPad  # noqa: PLC0415
        from rq_pipeline.deploy.unitree_stage import (  # noqa: PLC0415
            sim_binary,
            stage,
            write_sim_config,
        )

        # Unitree's SDK installs CycloneDDS under /usr/local/lib, which the
        # loader does not search until ldconfig runs; say so for both.
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = ":".join(
            p for p in (env.get("LD_LIBRARY_PATH", ""), SDK_LIB) if p
        )
        self.pad = VirtualPad()
        time.sleep(0.5)  # the joystick node appears
        nodes = (
            sorted(Path("/dev/input").glob("js*"))
            if Path("/dev/input").is_dir()
            else []
        )
        if not nodes:
            self.pad.close()
            raise SystemExit(
                "no /dev/input/js* after creating the virtual pad: modprobe joydev"
            )
        print(f"[gate] virtual pad at {nodes[-1]}", flush=True)
        write_sim_config(reference)
        proj = stage(manifest, reference)
        logs = proj / "logs"
        logs.mkdir(exist_ok=True)
        self._sim_log = open(logs / "simulate.log", "ab")  # noqa: SIM115
        self._ctrl_log = open(logs / "controller.log", "ab")  # noqa: SIM115
        self.sim = subprocess.Popen(
            [str(sim_binary(reference))],
            cwd=reference / "simulate" / "build",
            env=env,
            stdout=self._sim_log,
            stderr=subprocess.STDOUT,
        )
        time.sleep(SIM_WARMUP_S)
        self.ctrl = subprocess.Popen(
            [str(proj / "build" / "go2_ctrl"), "--network=lo"],
            cwd=proj / "build",
            env=env,
            stdout=self._ctrl_log,
            stderr=subprocess.STDOUT,
        )
        time.sleep(CTRL_WARMUP_S)
        for name, proc in (("simulator", self.sim), ("controller", self.ctrl)):
            if proc.poll() is not None:
                self.close()
                raise SystemExit(
                    f"their {name} exited with {proc.returncode}; see {logs}"
                )

    def close(self) -> None:
        for proc in (getattr(self, "ctrl", None), getattr(self, "sim", None)):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(5)
                except Exception:  # a stuck one is killed
                    proc.kill()
        for log in (getattr(self, "_sim_log", None), getattr(self, "_ctrl_log", None)):
            if log is not None:
                log.close()
        pad = getattr(self, "pad", None)
        if pad is not None:
            pad.close()


SDK_LIB = "/usr/local/lib"  # where `make install` of unitree_sdk2 puts libddsc
SIM_WARMUP_S = 4.0  # their simulator loads the scene and opens DDS
CTRL_WARMUP_S = 3.0  # their controller waits for the first LowState


def _certificate(project: Project, stamp: str | None) -> dict | None:
    """The cited evaluation's record, found by its version's hash."""
    if not stamp or "@" not in stamp:
        return None
    wanted = stamp.split("@", 1)[1]
    for folder in project.folder("certificates").iterdir():
        path = folder / CERTIFICATE_FILE
        if not path.is_file():
            continue
        from rq_pipeline.project.kinds import Kind, stamp_kind  # noqa: PLC0415

        try:
            if stamp_kind(Kind.CERTIFICATE, folder).split("@", 1)[1] == wanted:
                return json.loads(path.read_text())
        except Exception:  # a folder that is not a certificate any more
            continue
    return None


if __name__ == "__main__":
    main()
