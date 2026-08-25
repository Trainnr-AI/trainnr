"""solve_arm_ik's own pins — the grip-centre mode earned them twice over."""

import importlib.util
import unittest

MUJOCO_PRESENT = importlib.util.find_spec("mujoco") is not None

# A planar 3-hinge arm with two "pad" geoms on the last link — enough
# to pin both target modes without a robot bundle.
ARM = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body>
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      <body pos="0.2 0 0">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
        <body pos="0.2 0 0">
          <joint name="j3" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.015"/>
          <site name="tip" pos="0.1 0 0"/>
          <geom name="pad_a" type="sphere" size="0.01" pos="0.1 0.03 0"/>
          <geom name="pad_b" type="sphere" size="0.01" pos="0.1 -0.03 0"/>
        </body>
      </body>
    </body>
    <body name="ball" pos="0 0.5 0">
      <freejoint name="ball_free"/>
      <geom type="sphere" size="0.02" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

JOINTS = ("j1", "j2", "j3")


@unittest.skipUnless(MUJOCO_PRESENT, "sim extra not installed (uv sync --extra sim)")
class SolveArmIk(unittest.TestCase):
    def _model_data(self):
        import mujoco  # noqa: PLC0415

        model = mujoco.MjModel.from_xml_string(ARM)
        return model, mujoco.MjData(model)

    def test_site_mode_reaches_a_target(self) -> None:
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.robot.arm_ik import solve_arm_ik  # noqa: PLC0415

        model, data = self._model_data()
        target = [0.3, 0.0, 0.2]
        self.assertTrue(
            solve_arm_ik(
                model,
                data,
                site="tip",
                joints=JOINTS,
                target_pos=target,
                # y-only hinges cannot satisfy the down term; this test
                # pins the POSITION contract.
                down_weight=0.0,
            )
        )
        import mujoco  # noqa: PLC0415

        mujoco.mj_forward(model, data)
        tip = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")]
        self.assertLess(float(np.linalg.norm(tip - np.array(target))), 0.006)

    def test_grip_geoms_mode_positions_the_pad_midpoint(self) -> None:
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.robot.arm_ik import solve_arm_ik  # noqa: PLC0415

        model, data = self._model_data()
        target = [0.25, 0.0, 0.25]
        self.assertTrue(
            solve_arm_ik(
                model,
                data,
                site="tip",
                joints=JOINTS,
                grip_geoms=("pad_a", "pad_b"),
                target_pos=target,
                down_weight=0.0,
            )
        )
        mujoco.mj_forward(model, data)
        pads = [
            data.geom_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)]
            for name in ("pad_a", "pad_b")
        ]
        midpoint = 0.5 * (pads[0] + pads[1])
        self.assertLess(float(np.linalg.norm(midpoint - np.array(target))), 0.006)

    def test_unreachable_target_returns_false_not_a_pretty_lie(self) -> None:
        from rq_pipeline.robot.arm_ik import solve_arm_ik  # noqa: PLC0415

        model, data = self._model_data()
        # Total reach is 0.5 m; a metre away is honest failure territory.
        self.assertFalse(
            solve_arm_ik(
                model,
                data,
                site="tip",
                joints=JOINTS,
                target_pos=[1.0, 0.0, 0.0],
                down_weight=0.0,
            )
        )

    def test_refusals_name_the_offender(self) -> None:
        from rq_pipeline.robot.arm_ik import solve_arm_ik  # noqa: PLC0415

        model, data = self._model_data()
        with self.assertRaises(ValueError) as caught:
            solve_arm_ik(
                model, data, site="tip", joints=("j1", "ghost"), target_pos=[0, 0, 0]
            )
        self.assertIn("ghost", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            solve_arm_ik(
                model, data, site="nowhere", joints=JOINTS, target_pos=[0, 0, 0]
            )
        self.assertIn("nowhere", str(caught.exception))
        # The multi-dof guard: a free joint must be refused, not driven.
        with self.assertRaises(ValueError) as caught:
            solve_arm_ik(
                model,
                data,
                site="tip",
                joints=("ball_free",),
                target_pos=[0, 0, 0],
            )
        self.assertIn("single-dof", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
