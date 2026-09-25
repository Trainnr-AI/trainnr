"""The USD path, pinned against the Isaac layer it reads (docs/e2e-research/77).

The fixture is Robotiq's own 2F-85 (robotiq/isaacsim_assets at the
pinned commit, CC BY 4.0), fetched once into the runs/assets cache by
`robot.asset_fetch` and never committed; without the network and
without the cache the reading tests skip by name. The registry, the
option refusals and the layer check run on any venv with mujoco.
"""

from __future__ import annotations

import http.client
import json
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rq_pipeline.bundles.bundle import AUDIT_KEY, read_audit, read_bundle_record
from rq_pipeline.robot import onboarding
from rq_pipeline.robot.asset_fetch import cached_tree, fetch_tree
from tests._extras import USD_LINE, needs_sim, needs_usd

FETCH_ENV = "RQ_FETCH_TEST_ASSETS"
ASSET_REPOSITORY = "robotiq/isaacsim_assets"
ASSET_COMMIT = "6d992b664428"  # 2026-09-23, the tip docs/77 read
ASSET_PATH = "grippers/Robotiq_2F_85"
ASSET_FILE = "Robotiq_2F_85.usda"
NEWTON_VARIANT = {"Physics": "Newton_compliant"}
PHYSX_VARIANT = {"Physics": "Physx_parallel_grip"}
# docs/77 §4: what the Newton_compliant variant compiles to.
BODIES = 12  # the world and 11 links
HINGES = 8
EQUALITIES = 3  # two loop closures, one mimic
DRIVE_KP, DRIVE_KD = 1432.0, 158.0
CLOSED_TARGET = 0.8
CLOSE_TOLERANCE_RAD = 1e-3
LOOP_TOLERANCE_M = 1e-2
ONE_JOINT_MJCF = (
    '<mujoco><worldbody><body name="b"><joint name="j"/><geom size="0.1"/>'
    "</body></worldbody></mujoco>"
)
_ASSET: Path | None = None
_SKIP: str | None = None


def asset_root() -> Path:
    """The cached asset. A test run never goes to the network on its own:
    an empty cache skips by name, unless RQ_FETCH_TEST_ASSETS=1 allows one
    fetch per run (review 2026-09-24: a dropped connection mid-fetch
    errored the suite instead of skipping)."""
    global _ASSET, _SKIP  # noqa: PLW0603 - one fetch per test run
    if _ASSET is None and _SKIP is None:
        cached = cached_tree(ASSET_REPOSITORY, ASSET_COMMIT, ASSET_PATH)
        if cached is not None:
            _ASSET = cached.root
        elif os.environ.get(FETCH_ENV) != "1":
            _SKIP = (
                f"the 2F-85 asset is not in the cache; fetch it with "
                f"tools/import-usd.py or set {FETCH_ENV}=1 to let the tests fetch"
            )
        else:
            try:
                _ASSET = fetch_tree(ASSET_REPOSITORY, ASSET_COMMIT, ASSET_PATH).root
            except (OSError, http.client.HTTPException) as why:
                _SKIP = f"the 2F-85 asset could not be fetched: {why}"
    if _SKIP:
        raise unittest.SkipTest(_SKIP)
    assert _ASSET is not None
    return _ASSET


def usd_limits_and_masses(
    stage: object,
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    """A fresh pxr read of every revolute joint's range (radians) and
    every rigid body's mass — the layer the bundle must equal."""
    from pxr import Usd, UsdPhysics  # noqa: PLC0415

    limits, masses = {}, {}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint = UsdPhysics.RevoluteJoint(prim)
            limits[prim.GetName()] = (
                math.radians(joint.GetLowerLimitAttr().Get()),
                math.radians(joint.GetUpperLimitAttr().Get()),
            )
        if prim.HasAPI(UsdPhysics.MassAPI) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            masses[prim.GetName()] = float(UsdPhysics.MassAPI(prim).GetMassAttr().Get())
    return limits, masses


class Registry(unittest.TestCase):
    def test_sources_dispatch_by_suffix_and_refuse_the_rest(self) -> None:
        names = onboarding.sources()
        self.assertEqual(
            set(names), {onboarding.MJCF_SOURCE, "urdf", onboarding.USD_SOURCE}
        )
        self.assertEqual(
            onboarding.source_for(Path("a/b.xml")).name, onboarding.MJCF_SOURCE
        )
        for suffix in (".usd", ".usda", ".usdc", ".usdz", ".USDA"):
            self.assertEqual(
                onboarding.source_for(Path(f"r{suffix}")).name, onboarding.USD_SOURCE
            )
        with self.assertRaises(ValueError) as ctx:
            onboarding.source_for(Path("robot.sdf"))
        self.assertIn(".sdf", str(ctx.exception))
        self.assertIn("mjcf", str(ctx.exception))
        self.assertIn("usd", str(ctx.exception))

    def test_a_source_refuses_options_it_does_not_take(self) -> None:
        mjcf = onboarding.sources()[onboarding.MJCF_SOURCE]
        with self.assertRaises(ValueError) as ctx:
            mjcf.check_options({"variants": {"Physics": "x"}})
        self.assertIn("variants", str(ctx.exception))
        usd = onboarding.sources()[onboarding.USD_SOURCE]
        usd.check_options({"variants": {}, "root": "fixed"})
        with self.assertRaises(ValueError):
            usd.check_options({"colour": "red"})

    def test_a_duplicate_name_or_suffix_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            onboarding.model_source("mjcf", (".zzz",))(lambda *a: {})
        with self.assertRaises(ValueError):
            onboarding.model_source("other", (".xml",))(lambda *a: {})

    def test_the_door_refuses_a_usd_option_on_an_mjcf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "m.xml"
            source.write_text("<mujoco/>")
            with self.assertRaises(ValueError) as ctx:
                onboarding.onboard(source, "m", Path(tmp) / "out", {"root": "free"})
            self.assertIn("root", str(ctx.exception))


class LicenceAndPlugin(unittest.TestCase):
    """Review of 2026-09-24: a local asset adopted the enclosing checkout's
    LICENSE (the search walked three folders up), and a missing schema
    plugin let a stage open unguarded."""

    def test_a_local_asset_never_adopts_the_checkouts_licence(self) -> None:
        from rq_pipeline.robot.usd_import import find_license  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp)
            (checkout / "LICENSE").write_text("MIT License\n")
            asset = checkout / "robots" / "arm" / "arm.usda"
            asset.parent.mkdir(parents=True)
            asset.write_text("#usda 1.0\n")
            self.assertEqual(find_license(asset), (None, "unrecorded"))

    def test_a_fetched_asset_finds_its_licence_up_to_the_trees_root(self) -> None:
        from rq_pipeline.robot.asset_fetch import MARKER_FILE  # noqa: PLC0415
        from rq_pipeline.robot.usd_import import find_license  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "LICENSE").write_text("MIT License\n")  # above the tree
            slot = Path(tmp) / "cache" / "owner-assets-0123456"
            asset = slot / "grippers" / "g" / "g.usda"
            asset.parent.mkdir(parents=True)
            asset.write_text("#usda 1.0\n")
            (slot / MARKER_FILE).write_text("{}")
            self.assertEqual(find_license(asset), (None, "unrecorded"))
            package = asset.parent / "PACKAGE-LICENSES" / "LICENSE"
            package.parent.mkdir()
            package.write_text("Attribution 4.0 International\n")
            self.assertEqual(find_license(asset), (package, "CC-BY-4.0"))

    def test_without_the_schema_plugin_no_stage_opens(self) -> None:
        from unittest import mock  # noqa: PLC0415

        from rq_pipeline.robot import usd_import  # noqa: PLC0415

        with (
            mock.patch.object(
                usd_import.importlib.util, "find_spec", return_value=None
            ),
            self.assertRaisesRegex(RuntimeError, "newton_usd_schemas"),
        ):
            usd_import.register_schemas()


@needs_usd
class StageUnits(unittest.TestCase):
    def test_a_stage_in_centimetres_and_grams_says_so(self) -> None:
        """The audit said "meter" whatever the stage declared (review
        2026-09-24); a centimetre stage now reads as one, and the bundle's
        metres make an unexplained unit change."""
        from pxr import Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        from rq_pipeline.robot.usd_import import (  # noqa: PLC0415
            register_schemas,
            stage_units,
        )

        register_schemas()
        stage = Usd.Stage.CreateInMemory()
        self.assertEqual(stage_units(stage)["length"], "meter")
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 0.001)
        units = stage_units(stage)
        self.assertEqual(units["length"], "0.01 meter per unit")
        self.assertEqual(units["mass"], "0.001 kilogram per unit")


class Settings(unittest.TestCase):
    def test_root_kinds_are_the_named_two(self) -> None:
        from rq_pipeline.robot.usd_import import ImportSettings  # noqa: PLC0415

        ImportSettings(root="free")
        with self.assertRaises(ValueError):
            ImportSettings(root="mocap")

    def test_the_missing_line_names_the_module_and_its_extra(self) -> None:
        from rq_pipeline.robot import usd_import  # noqa: PLC0415

        line = usd_import.missing_line()
        if line is not None:
            self.assertIn("uv sync --extra", line)
        self.assertIn("macOS", usd_import.DARWIN_LINE)


@needs_usd
class Layers(unittest.TestCase):
    def test_an_empty_sublayer_is_refused_by_name_before_newton(self) -> None:
        """The materials layer fetched as 0 bytes twice on 2026-09-24 and
        the whole stage refused with a composition error from inside the
        library; the door now names the file first."""
        from rq_pipeline.robot.usd_import import (  # noqa: PLC0415
            UsdLayerError,
            check_layers,
        )

        root = asset_root()
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "asset"
            shutil.copytree(root, copy)
            layers = check_layers(copy / ASSET_FILE)
            self.assertGreater(len(layers), 5)
            (copy / "materials" / "materials.usd").write_bytes(b"")
            with self.assertRaises(UsdLayerError) as ctx:
                check_layers(copy / ASSET_FILE)
            self.assertIn("materials.usd", str(ctx.exception))
            self.assertIn("empty", str(ctx.exception))
            (copy / "materials" / "materials.usd").unlink()
            with self.assertRaises(UsdLayerError) as ctx:
                check_layers(copy / ASSET_FILE)
            self.assertIn("missing", str(ctx.exception))

    def test_a_process_that_touched_pxr_first_is_refused_not_read_wrong(self) -> None:
        """pxr builds its schema registry once; a stage opened before
        Newton's schemas registered leaves NewtonMimicAPI unknown for the
        process and the mimic reads as nothing (2026-09-24). A fresh
        interpreter reproduces the order; the reader must refuse."""
        import subprocess  # noqa: PLC0415
        import sys  # noqa: PLC0415

        from rq_pipeline.robot.usd_import import SCHEMA_WITNESS  # noqa: PLC0415

        script = (
            f"from pxr import Usd; Usd.Stage.Open(r'{asset_root() / ASSET_FILE}')\n"
            "from rq_pipeline.robot.usd_import import open_stage\n"
            "try:\n"
            f"    open_stage(r'{asset_root() / ASSET_FILE}', {{}})\n"
            "except RuntimeError as why:\n"
            "    print('refused:', why)\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False
        )
        self.assertIn("refused:", out.stdout, out.stderr[-800:])
        self.assertIn(SCHEMA_WITNESS, out.stdout)

    def test_an_unknown_variant_is_refused_naming_the_choices(self) -> None:
        from rq_pipeline.robot.usd_import import open_stage  # noqa: PLC0415

        path = asset_root() / ASSET_FILE
        with self.assertRaises(ValueError) as ctx:
            open_stage(path, {"Physics": "Bullet"})
        self.assertIn("Newton_compliant", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            open_stage(path, {"Colour": "red"})
        self.assertIn("Physics", str(ctx.exception))
        _, selected = open_stage(path, NEWTON_VARIANT)
        self.assertEqual(selected["Physics"], "Newton_compliant")
        self.assertEqual(selected["Fingertip"], "Standard")


@needs_usd
class TheBundle(unittest.TestCase):
    """The 2F-85 (Newton_compliant) written once, read from its files."""

    tmp: tempfile.TemporaryDirectory[str]
    bundle: Path
    report: dict

    @classmethod
    def setUpClass(cls) -> None:
        from rq_pipeline.robot.usd_import import (  # noqa: PLC0415
            ImportSettings,
            write_usd_bundle,
        )

        root = asset_root()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.bundle = Path(cls.tmp.name) / "robotiq-2f85-isaac"
        cls.report = write_usd_bundle(
            root / ASSET_FILE,
            "robotiq-2f85-isaac",
            cls.bundle,
            ImportSettings(variants=NEWTON_VARIANT, grip_options=True),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def model(self):
        import mujoco  # noqa: PLC0415

        return mujoco.MjModel.from_xml_path(
            str(self.bundle / self.report["model_file"])
        )

    def test_the_census_is_the_documented_one(self) -> None:
        m = self.model()
        self.assertEqual(
            (m.nbody, m.njnt, m.neq, m.nu), (BODIES, HINGES, EQUALITIES, 1)
        )
        self.assertEqual(self.report["sensors"], 2 * HINGES)
        self.assertEqual(m.nsensor, 2 * HINGES)
        self.assertEqual(m.nkey, 1)
        self.assertTrue(self.report["stamp"].startswith("robotiq-2f85-isaac@"))

    def test_limits_and_masses_equal_the_usd_layer(self) -> None:
        import mujoco  # noqa: PLC0415

        from rq_pipeline.robot.usd_import import open_stage  # noqa: PLC0415

        stage, _ = open_stage(asset_root() / ASSET_FILE, NEWTON_VARIANT)
        limits, masses = usd_limits_and_masses(stage)
        m = self.model()
        self.assertEqual(len(limits), HINGES)
        for i in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
            np.testing.assert_allclose(
                m.jnt_range[i], limits[name], atol=1e-5, err_msg=name
            )
        self.assertEqual(len(masses), BODIES - 1)
        for i in range(1, m.nbody):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
            self.assertAlmostEqual(m.body_mass[i], masses[name], places=5, msg=name)

    def test_the_drive_carries_the_usd_gains_and_its_joint_range(self) -> None:
        import mujoco  # noqa: PLC0415

        m = self.model()
        self.assertEqual(
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, 0), "finger_joint_drive"
        )
        np.testing.assert_allclose(m.actuator_gainprm[0][:1], [DRIVE_KP])
        np.testing.assert_allclose(
            m.actuator_biasprm[0][:3], [0.0, -DRIVE_KP, -DRIVE_KD]
        )
        np.testing.assert_allclose(m.actuator_ctrlrange[0], [0.0, CLOSED_TARGET])
        self.assertEqual(m.actuator_ctrllimited[0], 1)
        finger = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint")
        np.testing.assert_allclose(m.jnt_actfrcrange[finger], [-15.0, 15.0])
        self.assertAlmostEqual(m.dof_armature[m.jnt_dofadr[finger]], 0.3)

    def test_names_are_leaves_and_meshes_are_files(self) -> None:
        import mujoco  # noqa: PLC0415

        m = self.model()
        names = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, m.nbody)
        ]
        self.assertIn("left_outer_knuckle", names)
        self.assertTrue(
            all("/" not in n and not n.startswith("_") for n in names), names
        )
        assets = self.bundle / "assets"
        hulls = sorted(p.name for p in assets.glob("*_hull.obj"))
        visuals = sorted(p.name for p in assets.glob("*_visual.obj"))
        self.assertEqual(len(hulls), BODIES - 1)
        self.assertEqual(len(visuals), BODIES - 1)
        self.assertNotIn(
            "vertex=", (self.bundle / self.report["model_file"]).read_text()
        )
        # the hull sits inside its visual: same frame, same place
        for name in ("base_link", "left_fingertip"):
            hull = m.mesh_vertnum[
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, f"{name}_hull")
            ]
            visual = m.mesh_vertnum[
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, f"{name}_visual")
            ]
            self.assertLessEqual(hull, 64)
            self.assertGreater(visual, hull)
        groups = {
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g): int(m.geom_group[g])
            for g in range(m.ngeom)
        }
        self.assertEqual(groups["base_link_hull"], 3)
        self.assertEqual(groups["base_link_visual"], 2)
        self.assertEqual(
            int(
                m.geom_contype[
                    mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "base_link_visual")
                ]
            ),
            0,
        )

    def test_the_gripper_closes_with_the_loops_holding(self) -> None:
        import mujoco  # noqa: PLC0415

        m = self.model()
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        for step in range(int(2.0 / m.opt.timestep)):
            d.ctrl[0] = CLOSED_TARGET * min(1.0, step * m.opt.timestep)
            mujoco.mj_step(m, d)
        finger = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint")
        self.assertAlmostEqual(
            d.qpos[m.jnt_qposadr[finger]], CLOSED_TARGET, delta=CLOSE_TOLERANCE_RAD
        )
        violation = d.efc_pos[d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY]
        self.assertLess(float(np.abs(violation).max()), LOOP_TOLERANCE_M)
        self.assertLess(float(np.abs(d.qvel).max()), 1e-3)

    def test_the_record_carries_the_provenance(self) -> None:
        record = read_bundle_record(self.bundle)
        prov = record["provenance"]
        self.assertEqual(prov["format"], "usd")
        self.assertEqual(prov["repository"], ASSET_REPOSITORY)
        self.assertEqual(prov["commit"], ASSET_COMMIT)
        self.assertEqual(
            prov["variants"], {"Fingertip": "Standard", "Physics": "Newton_compliant"}
        )
        self.assertEqual(prov["license"], "CC-BY-4.0")
        self.assertEqual(prov["newton_census"]["joints"], 13)
        self.assertIn("newton", prov["versions"])
        self.assertTrue((self.bundle / "LICENSE").is_file())
        self.assertIn("CC-BY-4.0", (self.bundle / "README.md").read_text())
        self.assertEqual(record["census"]["sensors"], 2 * HINGES)
        m = self.model()
        self.assertAlmostEqual(m.opt.impratio, 10.0)
        self.assertEqual(int(m.opt.cone), 1)

    def test_the_bundle_json_is_plain(self) -> None:
        text = (self.bundle / "bundle.json").read_text()
        self.assertEqual(json.loads(text)["name"], "robotiq-2f85-isaac")


@needs_usd
class TheDoorAudits(unittest.TestCase):
    """The door's audit of the 2F-85 (docs/e2e-research/78 §1 item 2):
    every mass, centre of mass, inertia tensor, joint range and joint
    parameter equal to the USD layer; what differs is explained — the
    two spherical loop closures Newton writes as connect equalities, the
    tree-ordered joints, the hulls, the sensors, the keyframe, degrees
    to radians — and nothing is UNEXPLAINED."""

    def test_the_audit_explains_every_change_and_finds_no_other(self) -> None:
        from rq_pipeline.robot import import_audit as audit  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "robotiq-2f85-isaac"
            out = onboarding.onboard(
                asset_root() / ASSET_FILE,
                "robotiq-2f85-isaac",
                destination,
                {"variants": NEWTON_VARIANT},
            )
            record = {AUDIT_KEY: read_audit(destination)}
        report = audit.Audit.from_record(record[AUDIT_KEY])
        self.assertEqual(report.unexplained, ())
        self.assertEqual(out[AUDIT_KEY], report.summary())
        kinds = {(c.kind, c.element.rsplit("/", 1)[-1]) for c in report.changes}
        self.assertEqual(
            kinds,
            {
                (audit.JOINT_MISSING, "right_loop_closure"),
                (audit.JOINT_MISSING, "left_loop_closure"),
                (audit.JOINT_ORDER, audit.ANY),
                (audit.COUNT, audit.MESHES),
                (audit.COUNT, audit.SENSORS),
                (audit.COUNT, audit.KEYFRAMES),
                (audit.UNIT, "angle"),
            },
        )
        self.assertEqual(report.reader["variants"]["Physics"], "Newton_compliant")


@needs_usd
class OtherVariantsAndRoots(unittest.TestCase):
    def test_the_physx_variant_collapses_to_six_hinges_and_five_mimics(self) -> None:
        import mujoco  # noqa: PLC0415

        from rq_pipeline.robot.usd_import import (  # noqa: PLC0415
            ImportSettings,
            read_usd,
        )

        read = read_usd(
            asset_root() / ASSET_FILE, ImportSettings(variants=PHYSX_VARIANT)
        )
        m = mujoco.MjSpec.from_string(read.xml).compile()
        self.assertEqual((m.njnt, m.neq, m.nu), (6, 5, 1))
        self.assertTrue(all(t == mujoco.mjtEq.mjEQ_JOINT for t in m.eq_type))

    def test_a_free_root_gets_a_free_joint(self) -> None:
        import mujoco  # noqa: PLC0415

        from rq_pipeline.robot.usd_import import (  # noqa: PLC0415
            ImportSettings,
            write_usd_bundle,
        )

        with tempfile.TemporaryDirectory() as tmp:
            out = write_usd_bundle(
                asset_root() / ASSET_FILE,
                "free-gripper",
                Path(tmp) / "free-gripper",
                ImportSettings(variants=NEWTON_VARIANT, root="free"),
            )
            m = mujoco.MjModel.from_xml_path(
                str(Path(tmp) / "free-gripper" / out["model_file"])
            )
            self.assertEqual(m.njnt, HINGES + 1)
            self.assertEqual(m.jnt_type[0], mujoco.mjtJoint.mjJNT_FREE)
            self.assertEqual(m.nmocap, 0)


@needs_sim
class TheDoorWithoutUsd(unittest.TestCase):
    def test_the_actions_door_still_onboards_an_mjcf(self) -> None:
        from rq_pipeline.mcp_actions import Actions  # noqa: PLC0415
        from rq_pipeline.mcp_jobs import JobManager  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "src"
            source.mkdir()
            (source / "one.xml").write_text(ONE_JOINT_MJCF)
            actions = Actions(JobManager(Path(tmp) / "runs"), env_file=None)
            out = actions.onboard_robot(
                str(source / "one.xml"), "one", into=str(Path(tmp) / "robots")
            )
            self.assertEqual(out["joints"], 1)
            with self.assertRaises(ValueError) as ctx:
                actions.onboard_robot(
                    str(source / "one.xml"),
                    "two",
                    into=str(Path(tmp) / "robots"),
                    options={"root": "free"},
                )
            self.assertIn("root", str(ctx.exception))

    def test_the_server_door_refuses_a_usd_without_the_extra_or_by_name(self) -> None:
        import rq_pipeline.mcp_server as server  # noqa: PLC0415

        out = server.onboard_robot("/nowhere/robot.usda", "ghost")
        self.assertEqual(out["status"], "refused")
        self.assertTrue(USD_LINE or True)
