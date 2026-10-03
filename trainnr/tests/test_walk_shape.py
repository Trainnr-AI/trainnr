"""The shape a walk family reads off its robot by name, censused before
the learnability smoke, and the census every onboarding reply carries
(stranger test 2026-10-03: a Menagerie Go2 died in mjlab's regex match)."""

from __future__ import annotations

import unittest

from tests._extras import needs_sim

# The Menagerie Go2's shape, in miniature: an unnamed foot geom, a body `base`.
MENAGERIE_LIKE = """
<mujoco>
  <worldbody>
    <body name="base">
      <geom name="base1_collision" type="box" size="0.1 0.1 0.1"/>
      <body name="FR_calf">
        <geom type="sphere" size="0.02"/>
        <geom name="FR" type="sphere" size="0.02" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# Unitree's reference, in miniature: every named part the trainer wants.
REFERENCE_LIKE = """
<mujoco>
  <worldbody>
    <body name="base_link">
      <site name="imu"/>
      <geom name="base1_collision" type="box" size="0.1 0.1 0.1"/>
      {legs}
    </body>
  </worldbody>
</mujoco>
""".format(
    legs="".join(
        f'<body name="{leg}_calf"><geom name="{leg}_foot_collision" type="sphere" '
        f'size="0.02"/><site name="{leg}"/></body>'
        for leg in ("FR", "FL", "RR", "RL")
    )
)


@needs_sim
class TheWalkShape(unittest.TestCase):
    def test_the_menagerie_shape_is_named_part_by_part(self) -> None:
        import mujoco  # noqa: PLC0415

        from trainnr.tasks.walks import (  # noqa: PLC0415
            walk_shape_missing,
            walk_shape_sentence,
        )

        model = mujoco.MjModel.from_xml_string(MENAGERIE_LIKE)
        missing = walk_shape_missing("go2", model)
        self.assertEqual(
            missing,
            [
                "geom FR_foot_collision",
                "geom FL_foot_collision",
                "geom RR_foot_collision",
                "geom RL_foot_collision",
                "site FR",
                "site FL",
                "site RR",
                "site RL",
                "site imu",
                "body base_link",
            ],
        )
        sentence = walk_shape_sentence("go2", missing)
        self.assertIn("unitree_rl_mjlab", sentence)
        self.assertIn("Menagerie", sentence)
        self.assertTrue(sentence.startswith("the robot bundle is not in the shape"))

    def test_the_reference_shape_is_complete_and_other_walks_are_not_judged(
        self,
    ) -> None:
        import mujoco  # noqa: PLC0415

        from trainnr.tasks.walks import walk_shape_missing  # noqa: PLC0415

        model = mujoco.MjModel.from_xml_string(REFERENCE_LIKE)
        self.assertEqual(walk_shape_missing("go2", model), [])
        menagerie = mujoco.MjModel.from_xml_string(MENAGERIE_LIKE)
        self.assertEqual(walk_shape_missing("microduck", menagerie), [])


@needs_sim
class TheShapeCensus(unittest.TestCase):
    def test_named_collision_geoms_sites_and_the_trunk(self) -> None:
        import mujoco  # noqa: PLC0415

        from trainnr.robot.onboarding import shape_census  # noqa: PLC0415

        census = shape_census(mujoco.MjModel.from_xml_string(REFERENCE_LIKE))
        self.assertEqual(census["trunk"], "base_link")
        self.assertEqual(census["sites"], ["imu", "FR", "FL", "RR", "RL"])
        self.assertEqual(
            census["collision_geoms"],
            [
                "base1_collision",
                "FR_foot_collision",
                "FL_foot_collision",
                "RR_foot_collision",
                "RL_foot_collision",
            ],
        )
        # Menagerie-like: the unnamed foot geom is not listed, nor the
        # named one that does not collide; the trunk is `base`.
        census = shape_census(mujoco.MjModel.from_xml_string(MENAGERIE_LIKE))
        self.assertEqual(
            census,
            {"collision_geoms": ["base1_collision"], "sites": [], "trunk": "base"},
        )
