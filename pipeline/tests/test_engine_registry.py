"""The engine registry: engines by name, plugins by entry point, and the
rule that a missing extra fails at construction, never at listing."""

import importlib.util
import unittest

from rq_pipeline.physics.registry import (
    CPU_ENGINE,
    GPU_ENGINE,
    engine,
    engines,
    resolve,
)

MUJOCO_PRESENT = importlib.util.find_spec("mujoco") is not None
MJX_PRESENT = MUJOCO_PRESENT and importlib.util.find_spec("mujoco.mjx") is not None


class EngineRegistry(unittest.TestCase):
    def test_the_built_ins_are_listed_with_a_line_each(self) -> None:
        known = engines()
        self.assertIn(CPU_ENGINE, known)
        self.assertIn(GPU_ENGINE, known)
        for entry in known.values():
            self.assertTrue(entry.doc, entry.name)

    @unittest.skipUnless(MUJOCO_PRESENT, "sim extra not installed")
    def test_the_cpu_engine_resolves_to_an_engine(self) -> None:
        from rq_pipeline.evaluate.harness import Engine  # noqa: PLC0415

        self.assertIsInstance(resolve(CPU_ENGINE).build(), Engine)

    def test_an_unknown_engine_is_refused_with_the_known_ones_named(self) -> None:
        with self.assertRaises(KeyError) as caught:
            resolve("juggling")
        self.assertIn(CPU_ENGINE, str(caught.exception))

    @unittest.skipIf(MJX_PRESENT, "the mjx extra is installed here")
    def test_a_missing_extra_fails_at_construction_not_at_listing(self) -> None:
        entry = resolve(GPU_ENGINE)  # listing is free
        with self.assertRaises(ImportError) as caught:
            entry.build()
        self.assertIn("mjx", str(caught.exception))

    def test_a_duplicate_name_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            engine(CPU_ENGINE)(object)
        self.assertIs(engines()[CPU_ENGINE].build, resolve(CPU_ENGINE).build)


if __name__ == "__main__":
    unittest.main()
