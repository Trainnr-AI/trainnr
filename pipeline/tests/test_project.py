"""A project: located by name, created never over an existing one, its
artifacts detected by marker and refused when ambiguous, its loop map
proved by artifacts, its index rebuildable from the files alone."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rq_pipeline.bundles.hashing import is_stamp, stamp
from rq_pipeline.project import (
    PROJECT_ENV,
    Kind,
    Project,
    create_project,
    current_project,
    detect,
    index_project,
    stamp_kind,
    write_index,
    write_task_reference,
)
from rq_pipeline.project.index import STATES, UNRECORDED
from rq_pipeline.project.kinds import UnknownKindError
from rq_pipeline.project.locate import FOLDERS


def make_project(tmp: Path, name: str = "go2-warehouse") -> Project:
    return create_project(tmp / name, name, "a test project")


def make_batch(root: Path, episodes: int = 2) -> Path:
    root.mkdir(parents=True)
    (root / "datasheet.md").write_text("# Datasheet\n")
    for k in range(episodes):
        ep = root / f"episode_{k:04d}"
        ep.mkdir()
        (ep / "manifest.json").write_text(
            json.dumps(
                {
                    "seed": k,
                    "task": "kitting@aaaaaaaaaaaa",
                    "expert": "kitting-expert@bbbbbbbbbbbb",
                    "instrument": "mujoco-3.11.0+x86_64",
                }
            )
        )
    return root


def make_robot(root: Path, with_fits: bool = False) -> Path:
    root.mkdir(parents=True)
    (root / "robot.xml").write_text("<mujoco/>")
    (root / "README.md").write_text("a robot\n")
    if with_fits:
        (root / "fits").mkdir()
        (root / "fits" / "a.json").write_text("{}")
    return root


class Locating(unittest.TestCase):
    def test_create_then_find_by_the_variable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = make_project(Path(tmp))
            self.assertEqual(project.name, "go2-warehouse")
            for folder in FOLDERS:
                self.assertTrue((project.root / folder).is_dir(), folder)
            with mock.patch.dict(os.environ, {PROJECT_ENV: str(project.root)}):
                self.assertEqual(current_project().root, project.root)

    def test_a_missing_project_names_the_variable(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, {PROJECT_ENV: tmp}),
            self.assertRaisesRegex(FileNotFoundError, PROJECT_ENV),
        ):
            current_project()

    def test_never_overwrites_an_existing_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = make_project(Path(tmp))
            with self.assertRaises(FileExistsError):
                create_project(project.root, "other")
            self.assertEqual(project.name, "go2-warehouse")

    def test_bad_names_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for bad in ("", "a/b", ".hidden"):
                with self.assertRaises(ValueError):
                    create_project(Path(tmp) / "x", bad)

    def test_a_manifest_with_the_wrong_schema_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = make_project(Path(tmp))
            raw = json.loads(project.manifest_path.read_text())
            raw["schema"] = "something-else/9"
            project.manifest_path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                project.manifest()


class Kinds(unittest.TestCase):
    def test_each_marker_detects_its_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIs(detect(make_batch(root / "b")), Kind.BATCH)
            self.assertIs(detect(make_robot(root / "r")), Kind.ROBOT)
            ds = root / "d"
            ds.mkdir()
            (ds / "provenance.json").write_text("{}")
            self.assertIs(detect(ds), Kind.DATASET)
            run = root / "w"
            run.mkdir()
            (run / "run.json").write_text("{}")
            self.assertIs(detect(run), Kind.RUN)
            rl = root / "rl"
            rl.mkdir()
            (rl / "identity.json").write_text("{}")
            (rl / "model_7999.pt").write_bytes(b"\x00")
            self.assertIs(detect(rl), Kind.RUN)
            cert = root / "c"
            cert.mkdir()
            (cert / "certificate.json").write_text("{}")
            self.assertIs(detect(cert), Kind.CERTIFICATE)
            pol = root / "p"
            pol.mkdir()
            (pol / "model.safetensors").write_bytes(b"\x00")
            self.assertIs(detect(pol), Kind.POLICY)

    def test_no_marker_and_two_markers_are_both_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "e"
            empty.mkdir()
            (empty / "notes.txt").write_text("hi")
            with self.assertRaises(UnknownKindError):
                detect(empty)
            two = Path(tmp) / "t"
            two.mkdir()
            (two / "datasheet.md").write_text("")
            (two / "certificate.json").write_text("{}")
            with self.assertRaisesRegex(UnknownKindError, "one kind"):
                detect(two)

    def test_stamp_kind_checks_the_kind_and_strips_an_old_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            batch = make_batch(Path(tmp) / "press-1@000000000000")
            identity = stamp_kind(Kind.BATCH, batch)
            self.assertTrue(is_stamp(identity))
            self.assertTrue(identity.startswith("press-1@"))
            self.assertEqual(identity, stamp("press-1", batch))
            with self.assertRaisesRegex(UnknownKindError, "is a batch, not a robot"):
                stamp_kind(Kind.ROBOT, batch)


class Indexing(unittest.TestCase):
    def test_an_empty_project_has_every_state_missing_and_a_first_move(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index = index_project(make_project(Path(tmp)))
            self.assertEqual(index.artifacts, [])
            self.assertEqual([s.present for s in index.states], [False] * len(STATES))
            self.assertIn("record", index.next_move or "")

    def test_artifacts_are_stamped_cited_and_prove_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = make_project(Path(tmp))
            make_robot(project.robots / "go2", with_fits=True)
            make_batch(project.folder("batches") / "press-1", episodes=3)
            index = index_project(project)
            kinds = {a.kind for a in index.artifacts}
            self.assertEqual(kinds, {"robot", "batch"})
            batch = index.by_kind(Kind.BATCH)[0]
            self.assertEqual(batch.cites["task"], "kitting@aaaaaaaaaaaa")
            self.assertEqual(batch.summary["episodes"], 3)
            present = {s.name for s in index.states if s.present}
            # The robot carries fits, so identification is proved by it.
            self.assertEqual(
                present, {"asset onboarded", "system identified", "data generated"}
            )
            self.assertIn("task", index.next_move or "")

    def test_unrecorded_stamps_are_said_not_invented(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = make_project(Path(tmp))
            ds = project.folder("datasets") / "legacy"
            ds.mkdir()
            (ds / "provenance.json").write_text(json.dumps({"bundle": "aloha2"}))
            index = index_project(project)
            self.assertEqual(index.by_kind(Kind.DATASET)[0].cites["bundle"], UNRECORDED)

    def test_an_ambiguous_directory_is_listed_as_refused_not_indexed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = make_project(Path(tmp))
            odd = project.folder("runs") / "odd"
            odd.mkdir()
            (odd / "notes.txt").write_text("")
            index = index_project(project)
            self.assertEqual(index.artifacts, [])
            self.assertEqual(index.refused[0]["path"], "runs/odd")

    def test_the_index_is_written_atomically_and_ignored_by_git(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = make_project(Path(tmp))
            make_batch(project.folder("batches") / "press-1")
            out = write_index(project)
            self.assertEqual(out, project.index_path)
            raw = json.loads(out.read_text())
            self.assertEqual(raw["project"], "go2-warehouse")
            self.assertEqual(len(raw["artifacts"]), 1)
            self.assertEqual((out.parent / ".gitignore").read_text(), "*\n")
            self.assertFalse(list(out.parent.glob("*.tmp")))
            # Rebuildable: delete and re-index gives the same artifacts.
            out.unlink()
            again = json.loads(write_index(project).read_text())
            self.assertEqual(again["artifacts"], raw["artifacts"])

    def test_an_rl_run_nested_under_train_is_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = make_project(Path(tmp))
            arm = project.folder("runs") / "identified#1" / "train"
            arm.mkdir(parents=True)
            (arm / "identity.json").write_text(
                json.dumps(
                    {
                        "robot": "microduck@14b51b63c52e",
                        "dr_basis": "identified",
                        "seed": 42,
                    }
                )
            )
            (arm / "model_7999.pt").write_bytes(b"\x00")
            index = index_project(project)
            run = index.by_kind(Kind.RUN)[0]
            self.assertEqual(run.cites["robot"], "microduck@14b51b63c52e")
            self.assertEqual(run.summary["randomization"], "identified")
            self.assertEqual(run.path, "runs/identified#1/train")
            # The ARM names the run, not the folder the marker sat in.
            self.assertTrue(run.stamp.startswith("identified#1@"))

    def test_a_task_reference_proves_the_task_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = make_project(Path(tmp))
            out = write_task_reference(
                project, "robotiq/kitting", "kitting@aaaaaaaaaaaa"
            )
            self.assertEqual(out.parent.name, "robotiq--kitting")
            index = index_project(project)
            task = index.by_kind(Kind.TASK)[0]
            self.assertEqual(task.summary["stamp"], "kitting@aaaaaaaaaaaa")
            self.assertEqual(task.summary["kind"], "registered")
            self.assertIn(
                "environment defined", {s.name for s in index.states if s.present}
            )
            # Unstamped is said, never invented; ids are namespaced.
            write_task_reference(project, "robotiq/reach", None)
            self.assertEqual(
                json.loads(
                    (
                        project.folder("tasks") / "robotiq--reach" / "task.json"
                    ).read_text()
                )["stamp"],
                "unstamped",
            )
            with self.assertRaises(ValueError):
                write_task_reference(project, "kitting", "kitting@aaaaaaaaaaaa")


class TimeAndLineage(unittest.TestCase):
    def test_every_artifact_has_times_and_the_lineage_reads_both_ways(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = make_project(Path(tmp))
            arm = project.root / "runs" / "a" / "train"
            arm.mkdir(parents=True)
            (arm / "identity.json").write_text(
                json.dumps({"robot": "microduck@14b51b63c52e", "seed": 1})
            )
            (arm / "model_1.pt").write_bytes(b"\x00")
            index = index_project(project)
            run = index.by_kind(Kind.RUN)[0]
            self.assertIsNotNone(run.created)
            self.assertIsNotNone(run.updated)
            assert run.created is not None and run.updated is not None
            self.assertLessEqual(run.created, run.updated)
            self.assertTrue(run.updated.endswith("+00:00"))
            # A policy citing the run: the run lists it back.
            policy = project.root / "policies" / "p"
            policy.mkdir(parents=True)
            (policy / "model_1.pt").write_bytes(b"\x00")
            (policy / "policy.json").write_text(
                json.dumps({"schema": "trainnr-policy/1", "run": run.stamp})
            )
            index = index_project(project)
            run = index.by_kind(Kind.RUN)[0]
            pol = index.by_kind(Kind.POLICY)[0]
            self.assertEqual(run.cited_by, [pol.stamp])
            self.assertEqual(pol.cites["run"], run.stamp)

    def test_a_cite_by_hash_under_another_name_links_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = make_project(Path(tmp))
            task = project.root / "tasks" / "tray-far"
            task.mkdir(parents=True)
            (task / "task.json").write_text(
                json.dumps(
                    {
                        "task_id": "robotiq/kitting",
                        "stamp": "kitting@abc123abc123",
                        "kind": "declared",
                        "name": "tray-far",
                    }
                )
            )
            cert = project.root / "certificates" / "c"
            cert.mkdir(parents=True)
            (cert / "certificate.json").write_text(
                json.dumps(
                    {"successes": 1, "trials": 2, "task": "kitting@abc123abc123"}
                )
            )
            index = index_project(project)
            declared = index.by_kind(Kind.TASK)[0]
            self.assertEqual(declared.stamp, "tray-far@abc123abc123")
            self.assertEqual(
                declared.cited_by, [index.by_kind(Kind.CERTIFICATE)[0].stamp]
            )

    def test_a_finding_is_dated_by_its_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = make_project(Path(tmp))
            findings = project.root / "findings"
            findings.mkdir(parents=True, exist_ok=True)
            (findings / "x.json").write_text(
                json.dumps({"id": "x", "date": "2026-09-04", "claim": "c"})
            )
            finding = index_project(project).by_kind(Kind.FINDING)[0]
            self.assertEqual(finding.created, "2026-09-04T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
