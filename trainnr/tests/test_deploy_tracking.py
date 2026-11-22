"""A tracking policy in the plain-MuJoCo runtime (docs/87 §8): the
manifest's tracking section, the reference terms against hand-computed
values, the clock one frame ahead (mjlab's reset advance), each tracking
bound, the start-up draws restored per trial, and the gate judging the
clip under the cited evaluation's conditions. A toy robot stands in for
the G1; the parity with mjlab's own terms was measured on the G1 (the
record g1-court-dance-gated-2026-11-23)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from tests._extras import needs_sim

SCENE = """<mujoco><option timestep="0.005"/><worldbody>
<geom name="floor" type="plane" size="5 5 0.1"/>
<body name="pelvis" pos="0 0 1"><freejoint/><geom type="box" size=".1 .1 .1" mass="5"/>
 <body name="torso_link" pos="0 0 .2"><joint name="waist" axis="0 0 1" range="-1 1"/>
  <geom type="box" size=".1 .1 .1" mass="3"/>
  <body name="hand" pos=".3 0 0">
   <joint name="arm" pos="-.3 0 0" axis="0 1 0" range="-2 2"/>
   <geom type="sphere" size=".05" mass=".5"/></body></body></body>
</worldbody><actuator>
<position joint="waist" kp="50"/><position joint="arm" kp="20"/></actuator>
</mujoco>"""
FRAMES = 6
BODIES = ["pelvis", "torso_link", "hand"]


def _deployment(root: Path, **tracking: Any) -> Path:
    import numpy as np  # noqa: PLC0415

    folder = root / "dance"
    folder.mkdir()
    (folder / "scene.xml").write_text(SCENE)
    (folder / "policy.onnx").write_bytes(b"not used: the tests act themselves")
    t = np.arange(FRAMES, dtype=np.float32)
    pos = np.zeros((FRAMES, 3, 3), dtype=np.float32)
    pos[:, 0] = [0, 0, 1]
    pos[:, 1] = [0, 0, 1.2]
    pos[:, 2] = [0.3, 0, 1.2]
    pos[:, 1, 0] = 0.01 * t  # the reference torso moves forward a centimetre a frame
    quat = np.zeros((FRAMES, 3, 4), dtype=np.float32)
    quat[..., 0] = 1.0
    np.savez(
        folder / "motion.npz",
        joint_pos=np.stack([0.1 * t, -0.1 * t], axis=1),
        joint_vel=np.stack([t, -t], axis=1),
        body_pos_w=pos,
        body_quat_w=quat,
        body_lin_vel_w=np.zeros_like(pos),
        body_ang_vel_w=np.zeros_like(pos),
    )
    terms = [
        ("command", "command motion", 4, {}),
        ("motion_anchor_pos_b", "motion_anchor_pos_b", 3, {}),
        ("motion_anchor_ori_b", "motion_anchor_ori_b", 6, {}),
        ("joint_pos", "joint_pos_rel", 2, {"biased": True}),
        ("actions", "last_action", 2, {}),
    ]
    block = {
        "motion": "motion.npz",
        "frames": FRAMES,
        "anchor_body": "torso_link",
        "body_names": BODIES,
        "terminations": {
            "anchor_z_m": 0.25,
            "anchor_tilt": 0.8,
            "end_effector_z_m": 0.25,
            "end_effectors": ["hand"],
        },
        "start": {"soft_limit_factor": 0.9},
        "pushes": None,
        "noise": {},
        "lag_ms": 0.0,
        "criterion": "the clip played to its end",
    } | tracking
    manifest = {
        "schema": "trainnr-deploy/1",
        "robot": "toy",
        "run": "dance@0",
        "control": {
            "physics_timestep_s": 0.005,
            "decimation": 4,
            "control_hz": 50,
            "episode_length_s": FRAMES * 0.02,
        },
        "joints": {
            "policy_order": ["waist", "arm"],
            "action_to_ctrl": [0, 1],
            "default_pos": [0.0, 0.0],
        },
        "action": {"scale": [0.5, 0.5]},
        "observations": [
            {
                "name": n,
                "width": w,
                "source": s,
                "params": p,
                "scale": 1.0,
                "clip": None,
                "history_length": 1,
            }
            for n, s, w, p in terms
        ],
        "onnx": {"file": "policy.onnx", "input_width": sum(w for _, _, w, _ in terms)},
        "scene": {"file": "scene.xml"},
        "tracking": block,
    }
    (folder / "deploy.json").write_text(json.dumps(manifest))
    return folder


def _runtime(folder: Path) -> Any:
    import mujoco  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from trainnr.deploy.manifest import load_manifest  # noqa: PLC0415
    from trainnr.deploy.runtime import Runtime  # noqa: PLC0415

    manifest = load_manifest(folder)
    model = mujoco.MjModel.from_xml_string(SCENE)
    runtime = Runtime(
        manifest=manifest, model=model, data=mujoco.MjData(model), session=None
    )
    runtime.act = lambda obs: np.zeros(2, dtype=np.float32)  # type: ignore[method-assign]
    runtime.reset()
    return runtime


@needs_sim
class TheManifest(unittest.TestCase):
    def test_a_tracking_section_stands_in_for_commands(self) -> None:
        from trainnr.deploy.manifest import load_manifest  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            folder = _deployment(Path(tmp))
            tracking = load_manifest(folder).tracking
            assert tracking is not None
            self.assertEqual(tracking.end_effectors, ("hand",))
            raw = json.loads((folder / "deploy.json").read_text())
            (folder / "motion.npz").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "motion.npz"):
                load_manifest(folder)
            raw.pop("tracking")
            (folder / "deploy.json").write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "'commands'"):
                load_manifest(folder)


@needs_sim
class TheReferenceTerms(unittest.TestCase):
    def test_the_clock_leads_by_one_and_the_anchor_is_in_the_torsos_frame(self) -> None:
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        from trainnr.deploy.tracking import start_at_clip  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            rt = _runtime(_deployment(Path(tmp)))
            start_at_clip(rt, np.random.default_rng(0), drawn=False)
            obs = rt.observe()
            # tick 0 reads frame 1: joints 0.1, -0.1 and velocities 1, -1
            np.testing.assert_allclose(obs[:4], [0.1, -0.1, 1.0, -1.0], atol=1e-6)
            # the reference torso a centimetre ahead of the robot's (frame 1)
            np.testing.assert_allclose(obs[4:7], [0.01, 0.0, 0.0], atol=1e-6)
            np.testing.assert_allclose(obs[7:13], [1, 0, 0, 1, 0, 0], atol=1e-6)
            rt.data.qpos[rt.joint_qpos[0]] = (
                np.pi / 2
            )  # the torso turned a quarter left
            mujoco.mj_forward(rt.model, rt.data)
            obs = rt.observe()
            np.testing.assert_allclose(obs[4:7], [0.0, -0.01, 0.0], atol=1e-6)
            # Rz(-90°): the reference is a quarter turn right of the robot's torso
            np.testing.assert_allclose(obs[7:13], [0, 1, -1, 0, 0, 0], atol=1e-6)

    def test_a_biased_joint_term_reads_the_encoder_bias(self) -> None:
        import numpy as np  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            rt = _runtime(_deployment(Path(tmp)))
            rt.encoder_bias = np.array([0.01, -0.02])
            np.testing.assert_allclose(rt.observe()[13:15], [0.01, -0.02], atol=1e-9)


@needs_sim
class TheBounds(unittest.TestCase):
    def test_each_bound_is_named_when_broken(self) -> None:
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        from trainnr.deploy.tracking import (  # noqa: PLC0415
            ANCHOR_TILT,
            ANCHOR_Z,
            END_EFFECTOR_Z,
            broken_bound,
            start_at_clip,
        )

        with tempfile.TemporaryDirectory() as tmp:
            rt = _runtime(_deployment(Path(tmp)))
            start_at_clip(rt, np.random.default_rng(0), drawn=False)
            self.assertIsNone(broken_bound(rt, 1))
            q = rt.base_qpos
            rt.data.qpos[q + 2] -= 0.3  # the whole robot 30 cm low
            mujoco.mj_forward(rt.model, rt.data)
            self.assertEqual(broken_bound(rt, 1), ANCHOR_Z)
            start_at_clip(rt, np.random.default_rng(0), drawn=False)
            rt.data.qpos[q + 3 : q + 7] = [
                np.cos(0.75),
                np.sin(0.75),
                0,
                0,
            ]  # rolled 86°
            mujoco.mj_forward(rt.model, rt.data)
            self.assertEqual(broken_bound(rt, 1), ANCHOR_TILT)
            start_at_clip(rt, np.random.default_rng(0), drawn=False)
            rt.data.qpos[rt.joint_qpos[1]] = 1.5  # the arm swung down
            mujoco.mj_forward(rt.model, rt.data)
            self.assertEqual(broken_bound(rt, 1), END_EFFECTOR_Z)


@needs_sim
class TheDraws(unittest.TestCase):
    def test_each_trial_draws_on_the_models_own_values(self) -> None:
        import numpy as np  # noqa: PLC0415

        from trainnr.deploy.tracking import randomize  # noqa: PLC0415

        draws = {
            "encoder_bias": [-0.01, 0.01],
            "body_com": {"body": "torso_link", "ranges": {"0": [0.1, 0.2]}},
            "foot_friction": {"geoms": ["floor"], "range": [0.3, 0.4]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            rt = _runtime(_deployment(Path(tmp)))
            body = rt.model.body("torso_link").id
            before = float(rt.model.body_ipos[body, 0])
            (com_lo, com_hi), (mu_lo, mu_hi) = (
                draws["body_com"]["ranges"]["0"],
                draws["foot_friction"]["range"],
            )
            for seed in (1, 2):
                randomize(rt, draws, np.random.default_rng(seed))
                shift = float(rt.model.body_ipos[body, 0]) - before
                self.assertTrue(com_lo <= shift <= com_hi, shift)  # never stacked
                friction = float(rt.model.geom_friction[0, 0])
                self.assertTrue(mu_lo <= friction <= mu_hi)
                bias = draws["encoder_bias"][1]
                self.assertTrue(np.all(np.abs(rt.encoder_bias) <= bias))


@needs_sim
class TheGate(unittest.TestCase):
    def test_the_clip_is_judged_under_the_cited_evaluations_conditions(self) -> None:
        from trainnr.deploy.gate import gate  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            folder = _deployment(Path(tmp))
            certificate = {
                "successes": 3,
                "trials": 4,
                "ci95": [0.19, 0.99],
                "protocol": {
                    "pushes": False,
                    "lag_ms": 40.0,
                    "observation_noise": False,
                },
            }
            record = gate(
                folder,
                assets_dir=None,
                trials=3,
                certificate=certificate,
                open=lambda manifest, assets_dir=None: _runtime(manifest.root),
            )
            conditions = record["protocol"]["conditions"]
            self.assertEqual(conditions["lag_ms"], 40.0)
            self.assertEqual(conditions["from"], "the cited evaluation's protocol")
            self.assertNotIn("err_ratio_bound", record["protocol"])
            for row in record["records"]:
                self.assertEqual(row["frames"], FRAMES)
                self.assertLessEqual(row["lag_ticks"], 2)  # 40 ms at 50 Hz
            self.assertIn("passed", record["verdict"])
            self.assertTrue((folder / "gate.json").is_file())


if __name__ == "__main__":
    unittest.main()
