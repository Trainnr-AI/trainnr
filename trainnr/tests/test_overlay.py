"""A task declared by conversation: a spec overlay over a family, built
for real, named by its content, written into the project; the readers
that rebuild it honour the overlay; refusals name what exists."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._extras import needs_sim
from trainnr.project import Kind, create_project, index_project
from trainnr.project.task_ref import declare_task
from trainnr.tasks.experts import expert_for
from trainnr.tasks.overlay import (
    build_from_reference,
    build_variant,
    families,
    spec_fields,
)


class Families(unittest.TestCase):
    def test_families_are_the_builders_with_a_spec(self) -> None:
        ids = set(families())
        self.assertIn("trainnr/kitting", ids)
        self.assertIn("trainnr/lift-study", ids)
        self.assertNotIn("trainnr/reach", ids)

    def test_fields_carry_types_and_defaults(self) -> None:
        fields = spec_fields("kitting")
        self.assertEqual(fields["trials"]["default"], 4)
        self.assertEqual(fields["tray_center"]["default"], [0.0, -0.02])
        self.assertIn("part_spawn", fields)

    def test_refusals_name_the_families_and_the_fields(self) -> None:
        with self.assertRaises(ValueError) as caught:
            spec_fields("trainnr/reach")
        self.assertIn("trainnr/kitting", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            build_variant("kitting", {"tray_centre": [0, 0]})
        self.assertIn("tray_center", str(caught.exception))
        with self.assertRaises(KeyError):
            build_variant("acme/pour", {})

    def test_a_wrong_typed_overlay_is_refused_by_name(self) -> None:
        with self.assertRaises(ValueError) as caught:
            build_variant("kitting", {"tray_center": "far"})
        self.assertIn("tray_center", str(caught.exception))
        self.assertIn("tuple of 2", str(caught.exception))
        with self.assertRaises(ValueError):
            build_variant("kitting", {"trials": "four"})

    def test_an_expert_exists_for_kitting_only(self) -> None:
        self.assertTrue(callable(expert_for("kitting")))
        with self.assertRaises(ValueError) as caught:
            expert_for("trainnr/reach")
        self.assertIn("trainnr/kitting", str(caught.exception))


@needs_sim
class Declaring(unittest.TestCase):
    def test_a_variant_is_built_stamped_and_written_with_its_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            base, _ = build_variant("kitting", {})
            out = declare_task(
                project, "kitting", "tray-far", {"tray_center": [0.0, 0.9]}
            )
            self.assertEqual(out["task_id"], "trainnr/kitting")
            self.assertNotEqual(out["stamp"], base.stamp)
            self.assertTrue(out["stamp"].startswith("kitting@"))
            ref = json.loads((project.root / "tasks/tray-far/task.json").read_text())
            self.assertEqual(ref["kind"], "declared")
            self.assertEqual(ref["name"], "tray-far")
            self.assertEqual(ref["spec"]["tray_center"], [0.0, 0.9])
            # The readers rebuild the variant, not the family default.
            rebuilt = build_from_reference(ref)
            self.assertEqual(rebuilt.stamp, out["stamp"])
            self.assertEqual(rebuilt.task_spec.tray_center, (0.0, 0.9))
            index = index_project(project)
            task = index.by_kind(Kind.TASK)[0]
            # The project's name for it, the spec's own version.
            self.assertEqual(task.stamp, "tray-far@" + out["stamp"].split("@")[1])
            self.assertEqual(task.summary["acceptance"], "unreviewed")
            with self.assertRaises(FileExistsError):
                declare_task(project, "kitting", "tray-far", {})
            with self.assertRaises(ValueError):
                declare_task(project, "kitting", "a/b", {})

    def test_the_same_numbers_get_the_same_name(self) -> None:
        a, _ = build_variant("kitting", {"in_slot_xy_m": 0.04})
        b, _ = build_variant("trainnr/kitting", {"in_slot_xy_m": 0.04})
        self.assertEqual(a.stamp, b.stamp)
