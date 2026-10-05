"""What the importer changed (robot/import_audit):
a bundle's compiled model against the description it came from.

The URDF fixture is written here: a root link with mass, a link with a
rotated inertial origin, a limited revolute, a continuous mimic, a
velocity limit and a transmission — every conversion MuJoCo's loader
makes, each either explained by the reader at this mujoco build or
refused by name. The MJCF fixtures are the library's own bundles, which
must audit clean against themselves. The USD reader is pinned in
`test_usd_import`.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests._extras import needs_sim
from trainnr.bundles.bundle import (
    AUDIT_KEY,
    BUNDLE_SCHEMA,
    read_audit,
    read_bundle_record,
)
from trainnr.robot import import_audit as audit

ROBOTS = Path(__file__).resolve().parents[2] / "robots"
LIBRARY_MJCF = {
    "so101-nominal": "so101.xml",
    "microduck": "robot_walk.xml",
    "aloha2-nominal": None,  # the record names it
}
URDF = """<robot name="fixture">
  <link name="base">
    <inertial><mass value="2"/>
      <inertia ixx="0.1" iyy="0.2" izz="0.3" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual>
  </link>
  <link name="upper">
    <inertial><origin xyz="0 0 0.1" rpy="0 1.5707963 0"/><mass value="1"/>
      <inertia ixx="0.01" iyy="0.02" izz="0.03" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><box size="0.05 0.05 0.2"/></geometry></visual>
  </link>
  <link name="lower">
    <inertial><mass value="0.5"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><box size="0.05 0.05 0.1"/></geometry></visual>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="upper"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="5" velocity="2"/>
    <dynamics damping="0.5" friction="0.2"/>
  </joint>
  <joint name="elbow" type="continuous">
    <parent link="upper"/><child link="lower"/><origin xyz="0 0 0.2"/>
    <axis xyz="0 1 0"/>
    <mimic joint="shoulder" multiplier="-1" offset="0"/>
  </joint>
  <transmission name="t"/>
</robot>
"""
MIMIC_FIXED_IN = "3.13.0"
INERTIA_FIXED_IN = "3.14.0"


def write_urdf(folder: Path, text: str = URDF) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "fixture.urdf"
    path.write_text(text, encoding="utf-8")
    return path


class Snapshots(unittest.TestCase):
    def test_a_rotated_inertial_is_a_full_tensor_in_the_link_frame(self) -> None:
        from trainnr.robot.urdf_import import parse_urdf  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            snapshot, root, advisories = parse_urdf(write_urdf(Path(tmp)))
        self.assertEqual(root, "base")
        upper = snapshot.bodies["upper"]
        self.assertEqual(upper.mass, 1.0)
        self.assertEqual(upper.com, (0.0, 0.0, 0.1))
        # rpy (0, 90°, 0) swaps x and z: diag(0.01, 0.02, 0.03) → diag(0.03, 0.02, 0.01)
        tensor = np.array(upper.inertia).reshape(3, 3)
        np.testing.assert_allclose(np.diag(tensor), [0.03, 0.02, 0.01], atol=1e-6)
        self.assertEqual(list(snapshot.joints), ["shoulder", "elbow"])
        shoulder = snapshot.joints["shoulder"]
        self.assertEqual((shoulder.kind, shoulder.limited), (audit.HINGE, True))
        self.assertEqual(shoulder.range, (-1.0, 1.0))
        self.assertEqual(shoulder.force_range, (-5.0, 5.0))
        self.assertEqual((shoulder.damping, shoulder.frictionloss), (0.5, 0.2))
        self.assertFalse(snapshot.joints["elbow"].limited)
        self.assertEqual(snapshot.counts[audit.MIMICS], 1)
        self.assertTrue(any("velocity" in a for a in advisories))
        self.assertTrue(any("transmission" in a for a in advisories))

    def test_quaternion_and_rpy_rotations_agree_on_a_quarter_turn(self) -> None:
        about_y = audit.rpy_to_matrix(0.0, math.pi / 2, 0.0)
        quat = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)
        np.testing.assert_allclose(audit.quat_to_matrix(quat), about_y, atol=1e-12)

    def test_version_tuples_order_builds(self) -> None:
        self.assertLess(audit.version_tuple("3.11.0"), audit.version_tuple("3.13.0"))
        self.assertEqual(audit.version_tuple("3.14.0-dev+abc"), (3, 14, 0))


class Comparison(unittest.TestCase):
    def body(
        self, mass: float, inertia: tuple[float, ...] | None = None
    ) -> audit.BodyFacts:
        return audit.BodyFacts(mass, (0.0, 0.0, 0.0), inertia)

    def test_equal_snapshots_have_no_changes(self) -> None:
        a = audit.Snapshot(
            {"b": self.body(1.0)},
            {"j": audit.JointFacts(audit.HINGE)},
            {audit.MIMICS: 0},
        )
        self.assertEqual(audit.compare(a, a), [])

    def test_every_kind_of_change_is_named(self) -> None:
        src = audit.Snapshot(
            {"b": self.body(1.0, (1.0,) * 9), "gone": self.body(2.0)},
            {
                "j": audit.JointFacts(
                    audit.HINGE, (0, 0, 1), True, (-1, 1), 0.1, 0.5, 0.2, 0.0, (-5, 5)
                ),
                "k": audit.JointFacts(audit.SLIDE),
            },
            {audit.MIMICS: 1},
            {"angle": "degree"},
        )
        dst = audit.Snapshot(
            {"b": self.body(1.1, (2.0,) * 9), "new": self.body(3.0)},
            {
                "k": audit.JointFacts(audit.SLIDE),
                "j": audit.JointFacts(
                    audit.HINGE, (0, 1, 0), True, (-2, 1), 0.0, 0.5, 0.2, 0.0, None
                ),
            },
            {audit.MIMICS: 0},
            {"angle": "radian"},
        )
        kinds = {(c.kind, c.element) for c in audit.compare(src, dst)}
        self.assertEqual(
            kinds,
            {
                (audit.MASS, "b"),
                (audit.INERTIA, "b"),
                (audit.BODY_MISSING, "gone"),
                (audit.BODY_ADDED, "new"),
                (audit.JOINT_AXIS, "j"),
                (audit.JOINT_RANGE, "j"),
                (audit.ARMATURE, "j"),
                (audit.FORCE_RANGE, "j"),
                (audit.JOINT_ORDER, audit.ANY),
                (audit.COUNT, audit.MIMICS),
                (audit.UNIT, "angle"),
            },
        )

    def test_names_map_a_source_name_to_the_bundle_s(self) -> None:
        src = audit.Snapshot({"/robot/base": self.body(1.0)}, {})
        dst = audit.Snapshot({"base": self.body(1.0)}, {})
        self.assertEqual(audit.compare(src, dst, {"/robot/base": "base"}), [])
        self.assertEqual(len(audit.compare(src, dst)), 2)

    def test_explanations_cover_a_kind_for_one_element_or_all(self) -> None:
        changes = [
            audit.Change(audit.MASS, "a", 1, 2),
            audit.Change(audit.MASS, "b", 1, 2),
            audit.Change(audit.COUNT, audit.MIMICS, 1, 0),
        ]
        explained = audit.explain(
            changes,
            [
                audit.Explanation(audit.MASS, "a", "a is special"),
                audit.Explanation(audit.COUNT, audit.ANY, "counts differ"),
            ],
        )
        self.assertEqual(
            [c.explanation for c in explained], ["a is special", None, "counts differ"]
        )
        report = audit.Audit("t", "s", "3.11.0", tuple(explained))
        self.assertEqual(report.summary(), "2 explained, 1 UNEXPLAINED")
        self.assertIn("UNEXPLAINED", report.unexplained[0].line())
        with self.assertRaises(audit.ImportAuditError) as ctx:
            audit.require_explained(report)
        self.assertIn("mass of b", str(ctx.exception))
        self.assertIn("accept_changes", str(ctx.exception))
        again = audit.Audit.from_record(json.loads(json.dumps(report.to_record())))
        self.assertEqual(again.changes, report.changes)
        self.assertEqual(audit.Audit("t", "s", "3", ()).summary(), "nothing")


class Readers(unittest.TestCase):
    def test_the_registry_dispatches_by_suffix(self) -> None:
        names = set(audit.readers())
        self.assertEqual(names, {"mjcf", "urdf", "usd"})
        self.assertEqual(audit.reader_for(Path("a.urdf")).name, "urdf")
        self.assertEqual(audit.reader_for(Path("a.USDA")).name, "usd")
        with self.assertRaises(ValueError) as ctx:
            audit.reader_for(Path("a.sdf"))
        self.assertIn("urdf", str(ctx.exception))
        with self.assertRaises(ValueError):
            audit.source_reader("mjcf", (".zzz",))(lambda p, o: None)  # type: ignore[arg-type,return-value]


@needs_sim
class TheUrdfDoor(unittest.TestCase):
    def test_the_loader_s_losses_at_this_build_are_the_explanations(self) -> None:
        """Measured on mujoco 3.11.0 and 3.13.0: the root link's 2 kg
        becomes the world (dropped), the mimic is dropped before 3.13.0
        and kept from it, the rotated inertial is lost before 3.14.0.
        Every one explained at the build that has it; nothing else."""
        import mujoco  # noqa: PLC0415

        from trainnr.robot.onboarding import onboard  # noqa: PLC0415
        from trainnr.robot.urdf_import import LOADER_LOSSES  # noqa: PLC0415

        version = audit.version_tuple(mujoco.__version__)
        with tempfile.TemporaryDirectory() as tmp:
            source = write_urdf(Path(tmp) / "src")
            out = onboard(source, "fixture", Path(tmp) / "robots" / "fixture")
            bundle = Path(tmp) / "robots" / "fixture"
            record = {**read_bundle_record(bundle), AUDIT_KEY: read_audit(bundle)}
            self.assertNotIn(AUDIT_KEY, read_bundle_record(bundle))  # its own file
        self.assertEqual(record["schema"], BUNDLE_SCHEMA)
        self.assertEqual(record["provenance"]["format"], "urdf")
        report = audit.Audit.from_record(record[AUDIT_KEY])
        self.assertEqual(report.unexplained, ())
        kinds = {(c.kind, c.element): c for c in report.changes}
        self.assertIn((audit.BODY_MISSING, "base"), kinds)
        self.assertIn(
            "world body", kinds[(audit.BODY_MISSING, "base")].explanation or ""
        )
        mimic_lost = version < audit.version_tuple(MIMIC_FIXED_IN)
        self.assertEqual((audit.COUNT, audit.MIMICS) in kinds, mimic_lost)
        inertia_lost = version < audit.version_tuple(INERTIA_FIXED_IN)
        self.assertEqual((audit.INERTIA, "upper") in kinds, inertia_lost)
        self.assertEqual(len(LOADER_LOSSES), 2)
        self.assertTrue(any("velocity" in a for a in report.advisories))
        self.assertEqual(out[AUDIT_KEY], report.summary())
        self.assertTrue(out["stamp"].startswith("fixture@"))

    def test_an_unexplained_change_is_refused_and_the_bundle_removed(self) -> None:
        """The same fixture with the reader's explanation for the root
        link withheld: the door must refuse by name and leave nothing."""
        from unittest import mock  # noqa: PLC0415

        from trainnr.robot import urdf_import  # noqa: PLC0415
        from trainnr.robot.onboarding import onboard  # noqa: PLC0415

        silent = urdf_import.LoaderLoss(audit.MASS, audit.ANY, None, "x", "not it")
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(urdf_import, "ROOT_LINK_LOSS", silent),
        ):
            source = write_urdf(Path(tmp) / "src")
            destination = Path(tmp) / "robots" / "fixture"
            with self.assertRaises(audit.ImportAuditError) as ctx:
                onboard(source, "fixture", destination)
            self.assertIn("body missing of base", str(ctx.exception))
            self.assertFalse(destination.exists())
            out = onboard(source, "fixture", destination, {"accept_changes": True})
            self.assertIn("UNEXPLAINED", out[AUDIT_KEY])
            self.assertTrue(read_audit(destination)["accepted"])
            self.assertEqual(len(out["audit_unexplained"]), 1)

    def test_the_door_refuses_an_option_a_urdf_does_not_take(self) -> None:
        from trainnr.robot.onboarding import onboard, sources  # noqa: PLC0415

        self.assertIn("urdf", sources())
        with tempfile.TemporaryDirectory() as tmp:
            source = write_urdf(Path(tmp))
            with self.assertRaises(ValueError) as ctx:
                onboard(source, "f", Path(tmp) / "out", {"root": "free"})
            self.assertIn("root", str(ctx.exception))


@needs_sim
class TheLibraryAuditsClean(unittest.TestCase):
    """An MJCF bundle is a copy of its source: the audit of each library
    bundle against its own model file finds nothing."""

    def test_every_mjcf_bundle_against_itself(self) -> None:
        from trainnr.bundles.bundle import model_file_of  # noqa: PLC0415

        for name in LIBRARY_MJCF:
            model_file = model_file_of(ROBOTS / name)
            self.assertIsNotNone(model_file, name)
            assert model_file is not None
            report = audit.audit_bundle(model_file, model_file)
            self.assertEqual(report.changes, (), name)
            self.assertEqual(report.summary(), "nothing")

    def test_the_mjcf_door_records_a_clean_audit(self) -> None:
        from trainnr.robot.onboarding import onboard  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            src.mkdir()
            (src / "one.xml").write_text(
                '<mujoco><worldbody><body name="b"><joint name="j"/>'
                '<geom size="0.1"/></body></worldbody></mujoco>'
            )
            out = onboard(src / "one.xml", "one", Path(tmp) / "robots" / "one")
            record = {AUDIT_KEY: read_audit(Path(tmp) / "robots" / "one")}
        self.assertEqual(out[AUDIT_KEY], "nothing")
        self.assertEqual(record[AUDIT_KEY]["changes"], [])
        self.assertEqual(record[AUDIT_KEY]["format"], "mjcf")


@needs_sim
class TheDrawerAndTheCard(unittest.TestCase):
    def test_the_robot_summary_and_section_carry_the_audit(self) -> None:
        from trainnr.project.details import (  # noqa: PLC0415
            AUDIT_TITLE,
            _importer_audit,
        )
        from trainnr.project.index import _summary_robot  # noqa: PLC0415
        from trainnr.robot.onboarding import onboard  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            source = write_urdf(Path(tmp) / "src")
            destination = Path(tmp) / "robots" / "fixture"
            onboard(source, "fixture", destination)
            summary = _summary_robot(destination)
            section = _importer_audit(destination)
        self.assertIn("explained", summary["audit"])
        self.assertEqual(section["title"], AUDIT_TITLE)
        self.assertEqual(section["kind"], "table")
        self.assertTrue(any(row[0] == audit.BODY_MISSING for row in section["rows"]))
        unaudited = _importer_audit(ROBOTS / "so101-nominal")
        self.assertEqual(unaudited["kind"], "kv")
        self.assertIn("not audited", unaudited["rows"][0][1])
        self.assertNotIn("importer changed", _summary_robot(ROBOTS / "so101-nominal"))
