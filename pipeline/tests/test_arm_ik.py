"""solve_arm_ik's own pins — the grip-centre mode earned them twice over."""

import unittest

from tests._extras import needs_sim

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

# The same arm with a wrist ROLL before the pads: the closing plane (the
# site's y, the pad-to-pad line) can then turn about the link, which is
# the nullspace the kitting expert's grasp fell into (docs/07 2026-08-27).
WRIST_ARM = ARM.replace(
    '<joint name="j3" type="hinge" axis="0 1 0"/>',
    '<joint name="j3" type="hinge" axis="0 1 0"/>'
    '<joint name="roll" type="hinge" axis="1 0 0"/>',
)
WRIST_JOINTS = (*JOINTS, "roll")
CLOSING_TOL_DEG = 2.0
BENT_POSE = (0.3, -0.6, 0.3)  # j1..j3, radians: the arm reaching out and down
ROLL_START_DEG = 40.0
DESCEND_M = 0.10  # the kitting descend's length


@needs_sim
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

    def test_closing_axis_pins_the_pad_line(self) -> None:
        """With `closing_axis` the pad-to-pad line (the site's y) is
        pulled onto the wanted axis, either sign; without it the roll
        is nullspace and stays where it started. The reach is the
        kitting descend's 10 cm: the orientation terms are a bias and
        get only the iterations the position needs (a 2 cm nudge leaves
        the line 23° out — which is why the choreography's correction
        rounds carry the closing flag too)."""
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.robot.arm_ik import solve_arm_ik  # noqa: PLC0415

        model = mujoco.MjModel.from_xml_string(WRIST_ARM)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
        roll = model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "roll")
        ]

        def posed() -> object:
            data = mujoco.MjData(model)
            data.qpos[:3] = BENT_POSE
            data.qpos[roll] = np.radians(ROLL_START_DEG)
            mujoco.mj_forward(model, data)
            return data

        def pad_line_angle_to_y(data: object) -> float:
            mujoco.mj_forward(model, data)
            site_y = data.site_xmat[site].reshape(3, 3)[:, 1]
            return float(np.degrees(np.arccos(abs(site_y[1]))))

        data = posed()
        target = data.site_xpos[site].copy()
        target[2] -= DESCEND_M
        self.assertAlmostEqual(pad_line_angle_to_y(data), ROLL_START_DEG, places=3)
        self.assertTrue(
            solve_arm_ik(
                model,
                data,
                site="tip",
                joints=WRIST_JOINTS,
                target_pos=target,
                down_weight=0.0,
            )
        )
        self.assertGreater(pad_line_angle_to_y(data), ROLL_START_DEG - 1.0)  # nullspace
        data = posed()
        self.assertTrue(
            solve_arm_ik(
                model,
                data,
                site="tip",
                joints=WRIST_JOINTS,
                target_pos=target,
                down_weight=0.0,
                closing_axis=(0.0, -1.0, 0.0),  # either sign: the same line
                closing_weight=0.5,
            )
        )
        self.assertLess(pad_line_angle_to_y(data), CLOSING_TOL_DEG)

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
