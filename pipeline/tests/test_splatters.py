"""The splat trainers as a registry: found by name, refused with this
machine's install line, `auto` taking the best one here; both leave the
same file so the chain after them is one path."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from rq_pipeline.scenes import splatters
from rq_pipeline.scenes.splatters import (
    AUTO,
    AUTO_ORDER,
    SPLAT_EXPORT,
    SPLATTERS,
    MissingToolError,
    choose_splatter,
)


class TheRegistry(unittest.TestCase):
    def test_both_trainers_are_registered_with_their_licences(self) -> None:
        self.assertEqual(set(SPLATTERS), {"brush", "gsplat"})
        self.assertEqual(tuple(AUTO_ORDER), ("gsplat", "brush"))
        for spec in SPLATTERS.values():
            self.assertTrue(spec.license)
            self.assertTrue(spec.title)
            self.assertEqual(spec.folder, spec.name)

    def test_both_command_lines_leave_the_same_file(self) -> None:
        for spec in SPLATTERS.values():
            argv = [
                str(a)
                for a in spec.argv(
                    Path("/t"), Path("/d"), Path("/o"), steps=7, narrate=True
                )
            ]
            self.assertIn(SPLAT_EXPORT, argv, spec.name)
            self.assertIn("7", argv)
        brush = [
            str(a)
            for a in SPLATTERS["brush"].argv(
                Path("/b"), Path("/d"), Path("/o"), steps=7, narrate=False
            )
        ]
        self.assertNotIn("--rerun-enabled", brush)
        gsplat = [
            str(a)
            for a in SPLATTERS["gsplat"].argv(
                Path("/py"), Path("/d"), Path("/o"), steps=7, narrate=True
            )
        ]
        self.assertEqual(gsplat[:3], ["/py", "-m", splatters.TRAINER_MODULE])
        self.assertIn("--rerun", gsplat)

    def test_an_unknown_name_is_refused_by_name(self) -> None:
        with self.assertRaises(MissingToolError) as caught:
            choose_splatter("nerf")
        self.assertIn("nerf", str(caught.exception))
        self.assertIn("brush", str(caught.exception))

    def test_auto_takes_gsplat_when_it_answers_and_brush_otherwise(self) -> None:
        with (
            mock.patch.dict(splatters.LOCATORS, {"gsplat": lambda _b: Path("/py")}),
            mock.patch.dict(splatters.LOCATORS, {"brush": lambda _b: Path("/b")}),
            mock.patch("platform.system", return_value="Linux"),
        ):
            spec, tool = choose_splatter(AUTO)
        self.assertEqual((spec.name, tool), ("gsplat", Path("/py")))
        with (
            mock.patch.dict(splatters.LOCATORS, {"gsplat": lambda _b: None}),
            mock.patch.dict(splatters.LOCATORS, {"brush": lambda _b: Path("/b")}),
            mock.patch("platform.system", return_value="Linux"),
        ):
            spec, tool = choose_splatter(AUTO)
        self.assertEqual((spec.name, tool), ("brush", Path("/b")))

    def test_a_mac_never_gets_gsplat_and_the_refusal_says_so(self) -> None:
        with (
            mock.patch("platform.system", return_value="Darwin"),
            self.assertRaises(MissingToolError) as caught,
        ):
            choose_splatter("gsplat")
        self.assertIn("Darwin", str(caught.exception))
        self.assertIn("Brush", str(caught.exception))

    def test_nothing_here_names_every_install_line(self) -> None:
        with (
            mock.patch.dict(splatters.LOCATORS, {"gsplat": lambda _b: None}),
            mock.patch.dict(splatters.LOCATORS, {"brush": lambda _b: None}),
            mock.patch("platform.system", return_value="Linux"),
            self.assertRaises(MissingToolError) as caught,
        ):
            choose_splatter(AUTO)
        why = str(caught.exception)
        self.assertIn("install-gsplat.py", why)
        self.assertIn("install-brush.py", why)
        self.assertIn("brush_app", why)


if __name__ == "__main__":
    unittest.main()
