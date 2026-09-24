#!/usr/bin/env python3
"""Show a legged-joints fit in both viewers.

Rerun: per joint, the measured torque beside the rigid model's and the
fitted model's torque on the recording's clock, the residual left, and
the bootstrap replicates of every term as a bar chart — streamed to the
Studio and saved as `.viewer/fit-legged.rrd` inside the recording.
MuJoCo: the bundle's model under the fitted terms, replaying the
recorded joint trajectory, rendered to stills.

    uv run --extra sim --extra viz python tools/show-legged-fit.py \\
        <bundle-dir> <recording-dir> [--stills DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pipeline"))

STILL_WIDTH, STILL_HEIGHT = 960, 600
STILL_TIMES = (0.15, 0.5, 0.85)  # fractions of the recording
CAMERA = {"distance": 1.6, "azimuth": 135.0, "elevation": -18.0}


def apply_terms(model, balance) -> None:
    """Write the fitted armature, damping and frictionloss into the model."""
    import mujoco  # noqa: PLC0415

    for parameter in balance.result.parameters:
        joint, term = parameter.name.split(".")
        dof = model.jnt_dofadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        ]
        getattr(model, f"dof_{term}")[dof] = parameter.estimate


def stream(fit, recording_dir: Path, app_id: str) -> Path:
    """The fit into the Studio when one listens, and into the recording's
    `.viewer/` file (the headless door, docs/76 §10.5)."""
    import numpy as np  # noqa: PLC0415
    from rq_pipeline.viz import open_stream  # noqa: PLC0415

    out = recording_dir / ".viewer" / "fit-legged.rrd"
    out.parent.mkdir(parents=True, exist_ok=True)
    rr = open_stream(app_id, file=out)
    balance, samples = fit.balance, fit.samples
    rr.log("fit/anchor", rr.TextDocument(fit.anchor), static=True)
    for k, joint in enumerate(samples.joints):
        base = f"torque/{joint}"
        for i, t in enumerate(samples.times):
            rr.set_time("time", duration=float(t))
            rr.log(f"{base}/measured", rr.Scalars(float(samples.torque[i, k])))
            rr.log(f"{base}/rigid", rr.Scalars(float(balance.rigid_torque[i, k])))
            rr.log(f"{base}/modelled", rr.Scalars(float(balance.modelled_torque[i, k])))
            rr.log(
                f"residual/{joint}",
                rr.Scalars(float(samples.torque[i, k] - balance.modelled_torque[i, k])),
            )
        jf = balance.joints[k]
        for term, picks in jf.replicates.items():
            counts, edges = np.histogram(picks, bins=20)
            rr.log(
                f"bootstrap/{joint}/{term}",
                rr.BarChart(counts.astype(np.int64)),
                static=True,
            )
            rr.log(
                f"bootstrap/{joint}/{term}/edges",
                rr.TextDocument(f"{edges[0]:.5g} … {edges[-1]:.5g}"),
                static=True,
            )
    lines = [
        f"{jf.joint}: {jf.samples_used} samples, explained {jf.explained:.1%}, "
        f"rms {jf.rms_before:.3f} → {jf.rms_after:.3f} N·m"
        for jf in balance.joints
    ]
    rr.log("fit/summary", rr.TextDocument("\n".join(lines)), static=True)
    rr.log("fit/verdict", rr.TextDocument(balance.result.summary()), static=True)
    return out


def stills(model_file: Path, fit, out_dir: Path) -> list[Path]:
    import mujoco  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    model = mujoco.MjModel.from_xml_path(str(model_file))
    apply_terms(model, fit.balance)
    # The offscreen framebuffer defaults to 640 wide; the stills are wider.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, STILL_WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, STILL_HEIGHT)
    data = mujoco.MjData(model)
    samples = fit.samples
    qpos_adr = [
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
        for j in samples.joints
    ]
    renderer = mujoco.Renderer(model, STILL_HEIGHT, STILL_WIDTH)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = (
        CAMERA["distance"],
        CAMERA["azimuth"],
        CAMERA["elevation"],
    )
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    n = len(samples.times)
    for fraction in STILL_TIMES:
        i = int(fraction * (n - 1))
        data.qpos[:] = mujoco.MjData(model).qpos
        if samples.base_pose is not None:
            data.qpos[:7] = samples.base_pose[i]
            data.qpos[2] = 0.4
        data.qpos[qpos_adr] = samples.position[i]
        mujoco.mj_forward(model, data)
        camera.lookat[:] = (
            data.qpos[:3] if samples.base_pose is not None else (0, 0, 0.3)
        )
        renderer.update_scene(data, camera, option)
        elapsed = samples.times[i] - samples.times[0]
        path = out_dir / f"fit-t{elapsed:05.1f}s.png"
        Image.fromarray(renderer.render()).save(path)
        written.append(path)
    renderer.close()
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("bundle", type=Path)
    parser.add_argument("recording", type=Path)
    parser.add_argument(
        "--stills", type=Path, default=None, help="where the MuJoCo stills go"
    )
    args = parser.parse_args()
    from rq_pipeline.bundles.bundle import model_file_of  # noqa: PLC0415
    from rq_pipeline.robot.legged_fit import LeggedJoints  # noqa: PLC0415

    fit = LeggedJoints().fit_balance(args.bundle, args.recording)
    print(fit.balance.result.summary())
    saved = stream(fit, args.recording, f"rq-fit-{args.recording.name}")
    print(f"viewer recording: {saved}")
    if args.stills is not None:
        model_file = model_file_of(args.bundle)
        assert model_file is not None
        for path in stills(model_file, fit, args.stills):
            print(f"still: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
