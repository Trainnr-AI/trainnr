"""The microduck bundle: their walk model, byte-true, census-pinned —
the flagship's G1 (docs/e2e-research/63)."""

from __future__ import annotations

import unittest
from pathlib import Path

from tests._extras import needs_mjx, needs_sim

BUNDLE = Path(__file__).resolve().parents[2] / "robots" / "microduck"
XML = BUNDLE / "robot_walk.xml"

ACTUATED = 14  # their walk model: 5 per leg, 2 neck, 2 head
JOINTS = ACTUATED + 1  # + the trunk freejoint
MESHES = 38
TOTAL_MASS_KG = 0.737  # measured at wrap (2026-09-01); the duck is 737 g


@needs_sim
class MicroduckBundle(unittest.TestCase):
    def _model(self):
        import mujoco  # noqa: PLC0415

        from trainnr.bundles.fetch import unavailable  # noqa: PLC0415

        reason = unavailable(BUNDLE)  # the meshes are fetched on first use
        if reason:
            self.skipTest(reason)
        return mujoco.MjModel.from_xml_path(str(XML))

    def test_the_stamp_is_pinned_fetched_or_not(self) -> None:
        """Every record citing the microduck names this stamp; a fresh
        clone without the meshes reports it too (FETCH.json records it)."""
        from trainnr.bundles.hashing import stamp  # noqa: PLC0415

        self.assertEqual(stamp("microduck", BUNDLE), "microduck@ad90736153cc")

    def test_census_matches_their_walk_model(self) -> None:
        model = self._model()
        self.assertEqual(model.njnt, JOINTS)
        self.assertEqual(model.nu, ACTUATED)
        self.assertAlmostEqual(
            float(model.body_subtreemass[1]), TOTAL_MASS_KG, places=3
        )

    def test_every_referenced_mesh_is_in_the_fetch_manifest(self) -> None:
        """The meshes are not carried (their licence: Creative Commons
        BY-SA-NC); each is fetched from Pollen Robotics by blob id."""
        import re  # noqa: PLC0415

        from trainnr.bundles.fetch import fetch_manifest  # noqa: PLC0415

        referenced = set(re.findall(r'file="([^"]+)"', XML.read_text()))
        self.assertEqual(len(referenced), MESHES)
        manifest = fetch_manifest(BUNDLE)
        self.assertIsNotNone(manifest)
        listed = {rel.removeprefix("assets/") for rel in manifest["files"]}
        self.assertEqual(listed, referenced)

    def test_the_baked_in_fit_signature_is_theirs(self) -> None:
        # The flagship's opening exhibit (63 §1, README), measured at
        # wrap: the default class says kp=50, but every actuator
        # overrides to kp=0.55, and every servo dof carries
        # armature=0.0018 - BAM's fitted XL330 armature baked in as bare
        # numbers with no provenance. Pin the physics so an upstream
        # refresh cannot slip in as "just a mesh update".
        model = self._model()
        for i in range(model.nu):
            self.assertAlmostEqual(float(model.actuator_gainprm[i][0]), 0.55)
        joint_ids = set(model.actuator_trnid[:, 0])
        for j in joint_ids:
            self.assertAlmostEqual(
                float(model.dof_armature[model.jnt_dofadr[j]]), 0.0018
            )

    def test_stamp_is_stable(self) -> None:
        from trainnr.bundles.hashing import stamp  # noqa: PLC0415

        first = stamp("microduck", BUNDLE)
        self.assertEqual(first, stamp("microduck", BUNDLE))
        self.assertIn("@", first)


@needs_mjx
class MicroduckOnTheBatchedEngine(unittest.TestCase):
    def test_loads_and_steps_on_mjx(self) -> None:
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        from trainnr.physics.mjx_backend import MJXWarpBackend  # noqa: PLC0415

        spec = mujoco.MjSpec.from_file(str(XML))
        backend = MJXWarpBackend(impl="jax", naconmax=512, njmax=1024)
        backend.load_spec(spec)
        home = backend.default_initial_state()
        controls = np.zeros((2, 25, backend.model.nu))
        states = backend.rollout(np.stack([home, home]), controls)
        self.assertEqual(states.shape[0], 2)
        self.assertTrue(np.isfinite(states).all())


if __name__ == "__main__":
    unittest.main()
