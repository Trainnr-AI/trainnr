"""The engine registry: engines by name, plugins by entry point, and the
rule that a missing extra fails at construction, never at listing."""

import unittest

from rq_pipeline.physics.registry import (
    CPU_ENGINE,
    GPU_ENGINE,
    engine,
    engines,
    resolve,
)
from tests._extras import MJX, needs_sim


class EngineRegistry(unittest.TestCase):
    def test_the_built_ins_are_listed_with_a_line_each(self) -> None:
        known = engines()
        self.assertIn(CPU_ENGINE, known)
        self.assertIn(GPU_ENGINE, known)
        for entry in known.values():
            self.assertTrue(entry.doc, entry.name)

    @needs_sim
    def test_the_cpu_engine_resolves_to_an_engine(self) -> None:
        from rq_pipeline.evaluate.harness import Engine  # noqa: PLC0415

        self.assertIsInstance(resolve(CPU_ENGINE).build(), Engine)

    def test_an_unknown_engine_is_refused_with_the_known_ones_named(self) -> None:
        with self.assertRaises(KeyError) as caught:
            resolve("juggling")
        self.assertIn(CPU_ENGINE, str(caught.exception))

    @unittest.skipIf(MJX, "the mjx extra is installed here")
    def test_a_missing_extra_fails_at_construction_not_at_listing(self) -> None:
        entry = resolve(GPU_ENGINE)  # listing is free
        with self.assertRaises(ImportError) as caught:
            entry.build()
        self.assertIn("mjx", str(caught.exception))

    def test_a_plugin_that_fails_to_import_is_named(self) -> None:
        from rq_pipeline.plugins import load_entries  # noqa: PLC0415

        class Broken:
            name, value = "acme-sim", "acme.sim:register"

            def load(self):
                raise ModuleNotFoundError("No module named 'acme'")

        with self.assertRaises(ImportError) as caught:
            load_entries("rq_pipeline.engines", [Broken()])
        self.assertIn("rq_pipeline.engines:acme-sim", str(caught.exception))
        self.assertIn("acme.sim:register", str(caught.exception))

    def test_a_duplicate_name_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            engine(CPU_ENGINE)(object)
        self.assertIs(engines()[CPU_ENGINE].build, resolve(CPU_ENGINE).build)


if __name__ == "__main__":
    unittest.main()
