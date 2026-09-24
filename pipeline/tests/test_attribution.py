"""Attribution (docs/77 §9): the knobs are one table with one applier
each; a ladder stops at the cliff; the ranking answers "what breaks it
first"; a gate that did not pass, cites nothing or walks a course is
refused by name; the wrappers delay, noise and push exactly as said; the
appliers turn a real MuJoCo model's fields; the door spawns the tool."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import rq_pipeline.mcp_server as server
from rq_pipeline.deploy import attribution as attr
from rq_pipeline.deploy.attribution import (
    APPLIERS,
    ATTRIBUTION_FILE,
    ATTRIBUTION_SCHEMA,
    KNOBS,
    NOMINAL,
    SURVIVED,
    Delayed,
    Knob,
    Noisy,
    Pushed,
    Rung,
    attribute,
    cliff_of,
    knob,
    knob_names,
    rank,
    read_attribution,
    require_passing_gate,
    sensitivity_line,
)
from rq_pipeline.deploy.gate import DRAW_KEY, DRAW_NOW
from rq_pipeline.deploy.manifest import GATE_SCHEMA, load_manifest
from rq_pipeline.project import PROJECT_ENV, create_project
from rq_pipeline.project.index import _summary_deploy
from tests._extras import needs_sim
from tests.test_deploy import _manifest
from tests.test_mcp_actions import PIPELINE_DIR, TOOLS_DIR, harness

CERT = {"successes": 38, "trials": 40, "ci95": [0.8308, 0.9939]}


def _passing_gate(
    folder: Path, *, passed: bool | None = True, trials: int = 20, seed: int = 1000
) -> None:
    (folder / "gate.json").write_text(
        json.dumps(
            {
                "schema": GATE_SCHEMA,
                "runtime": "mujoco",
                "successes": trials,
                "trials": trials,
                "ci95": [0.8316, 1.0],
                "protocol": {"trials": trials, "seed": seed, DRAW_KEY: DRAW_NOW},
                "verdict": {"passed": passed},
                "judged": "2026-09-24T00:00:00+00:00",
            }
        )
    )


@dataclass
class FakeRuntime:
    """A runtime that tracks until its severity passes a threshold: the
    knob's applier sets `severity`, the trial falls when it is too high."""

    manifest: Any
    threshold: float = 1.0
    severity: float = 0.0
    command: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    command_limit: float | None = None
    instrument: str = "fake"
    closed: int = 0

    def reset(self) -> None:
        pass

    def observe(self) -> np.ndarray:
        return np.zeros(1, np.float32)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(1, np.float32)

    def apply(self, action: np.ndarray) -> None:
        pass

    def base_velocity_b(self) -> np.ndarray:
        return np.array([self.command[0], self.command[1], 0.0])

    def fell_over(self) -> bool:
        return self.severity >= self.threshold

    def contact_points(self) -> np.ndarray | None:
        return None

    def pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(1)

    def close(self) -> None:
        self.closed += 1


def _set_severity(runtime: FakeRuntime, level: float, _seed: int) -> FakeRuntime:
    runtime.severity = level
    return runtime


def _leave(runtime: FakeRuntime, _level: float, _seed: int) -> FakeRuntime:
    return runtime


SOFT = Knob("soft", "u", (0.5, 0.9, 2.0, 3.0), "falls past 1")
HARD = Knob("hard", "u", (0.1, 0.2), "never falls")
FAKE_APPLIERS = {"soft": _set_severity, "hard": _leave, NOMINAL.name: _leave}


class TheKnobTable(unittest.TestCase):
    def test_every_knob_has_an_applier_and_a_monotone_ladder(self) -> None:
        for k in KNOBS:
            self.assertIn(k.name, APPLIERS, k.name)
            self.assertGreater(len(k.ladder), 1, k.name)
            steps = np.diff(np.asarray(k.ladder, dtype=float))
            self.assertTrue(
                np.all(steps > 0) or np.all(steps < 0),
                f"{k.name}'s ladder is not monotone: {k.ladder}",
            )
        self.assertEqual(knob("latency").ladder[0], 1)  # the budget finding's tick
        self.assertIn(NOMINAL.name, APPLIERS)
        self.assertNotIn(NOMINAL.name, knob_names())
        with self.assertRaises(ValueError):
            knob("gravity_constant")

    def test_labels_carry_the_unit(self) -> None:
        self.assertEqual(knob("friction").label(0.6), "friction 0.6 x")
        self.assertEqual(knob("latency").label(2), "latency 2 ticks")


def _rung(level: float, k: int, n: int = 20) -> Rung:
    from rq_pipeline.stats.intervals import clopper_pearson  # noqa: PLC0415

    lo, hi = clopper_pearson(k, n)
    return Rung(level, k, n, (round(lo, 4), round(hi, 4)), 0.1)


class TheCliffAndTheRanking(unittest.TestCase):
    def test_the_cliff_is_the_first_rung_under_the_certificates_bound(self) -> None:
        rungs = [_rung(1, 20), _rung(2, 19), _rung(3, 10)]
        self.assertEqual(cliff_of(rungs, 0.8308).level, 2)  # 19/20 -> lo 0.7513
        self.assertIsNone(cliff_of([_rung(1, 20)], 0.8308))

    def test_knobs_are_ranked_by_the_rung_they_fall_at(self) -> None:
        climbed = {
            "friction": [_rung(0.8, 20), _rung(0.6, 20), _rung(0.4, 12)],
            "latency": [_rung(1, 15)],
            "payload": [_rung(1, 20), _rung(2, 20)],  # survived its (short) ladder
            "kp": [_rung(0.8, 20), _rung(0.6, 20), _rung(0.4, 18)],
        }
        order = [name for name, _ in rank(climbed, 0.8308)]
        # latency fell at rung 1; friction and kp at rung 3, friction lower
        self.assertEqual(order, ["latency", "friction", "kp", "payload"])
        line = sensitivity_line(rank(climbed, 0.8308))
        self.assertEqual(line, "most sensitive to latency 1 ticks, then friction 0.4 x")
        self.assertEqual(
            sensitivity_line(rank({"payload": climbed["payload"]}, 0.83)), SURVIVED
        )


class TheRefusals(unittest.TestCase):
    def test_every_case_that_has_no_cliff_to_find_is_named(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = _manifest(Path(tmp))
            with self.assertRaises(ValueError) as no_gate:
                require_passing_gate(folder, "mujoco", CERT)
            self.assertIn("no mujoco gate record", str(no_gate.exception))
            _passing_gate(folder, passed=None)
            with self.assertRaises(ValueError) as reported:
                require_passing_gate(folder, "mujoco", CERT)
            self.assertIn("reported, not judged", str(reported.exception))
            _passing_gate(folder, passed=False)
            with self.assertRaises(ValueError) as failed:
                require_passing_gate(folder, "mujoco", CERT)
            self.assertIn("failed (20/20)", str(failed.exception))
            _passing_gate(folder)
            with self.assertRaises(ValueError) as none_cited:
                require_passing_gate(folder, "mujoco", None)
            self.assertIn("cites no evaluation", str(none_cited.exception))
            with self.assertRaises(ValueError) as other:
                require_passing_gate(folder, "dds", CERT)
            self.assertIn("another process", str(other.exception))
            _loaded, base, cited = require_passing_gate(folder, "mujoco", CERT)
            self.assertEqual(base["successes"], 20)
            self.assertIs(cited, CERT)
            (Path(tmp) / "staged").mkdir()
            staged = _manifest(
                Path(tmp) / "staged", scene={"file": "scene.xml", "scene": "garden@1"}
            )
            _passing_gate(staged)
            with self.assertRaises(ValueError) as course:
                require_passing_gate(staged, "mujoco", CERT)
            self.assertIn("another protocol", str(course.exception))


class TheSweep(unittest.TestCase):
    def test_ladders_stop_at_the_cliff_and_the_record_says_everything(self) -> None:
        opened: list[FakeRuntime] = []

        def fake_open(manifest: Any, *, assets_dir: Any) -> FakeRuntime:
            runtime = FakeRuntime(manifest)
            opened.append(runtime)
            return runtime

        with tempfile.TemporaryDirectory() as tmp:
            folder = _manifest(Path(tmp))
            _passing_gate(folder, trials=4, seed=7)
            # four trials cannot clear a lower bound of 0.83 (4/4 gives
            # 0.398): a certificate judged looser lets a small draw climb.
            # Trials and seed are the gate's own, read from its record.
            loose = {"successes": 8, "trials": 10, "ci95": [0.3, 0.99]}
            record = attribute(
                folder,
                assets_dir=None,
                certificate=loose,
                knobs=(SOFT, HARD),
                open=fake_open,
                appliers=FAKE_APPLIERS,
            )
            self.assertEqual(record["schema"], ATTRIBUTION_SCHEMA)
            self.assertEqual(record["baseline"]["successes"], 4)
            soft, hard = record["knobs"]
            # 0.5 and 0.9 track (4/4, lo 0.398); 2.0 falls (0/4): the cliff,
            # and the 3.0 rung is never run
            self.assertEqual([r["level"] for r in soft["rungs"]], [0.5, 0.9, 2.0])
            self.assertEqual(soft["cliff"], {"level": 2.0, "rung": 3})
            self.assertIsNone(hard["cliff"])
            self.assertEqual(len(hard["rungs"]), 2)
            self.assertEqual(record["protocol"]["workers"], 0)
            self.assertEqual(
                (record["protocol"]["trials"], record["protocol"]["seed"]), (4, 7)
            )
            self.assertEqual(record["certificate"]["lower"], 0.3)
            self.assertTrue((folder / ATTRIBUTION_FILE).is_file())
            self.assertEqual(read_attribution(folder), record)
            # every opened runtime was closed (a runtime may hold a bus)
            self.assertTrue(all(r.closed == 1 for r in opened))

    def test_with_enough_trials_the_cliff_is_where_the_policy_falls(self) -> None:
        def fake_open(manifest: Any, *, assets_dir: Any) -> FakeRuntime:
            return FakeRuntime(manifest)

        with tempfile.TemporaryDirectory() as tmp:
            folder = _manifest(Path(tmp))
            _passing_gate(folder, seed=7)
            record = attribute(
                folder,
                assets_dir=None,
                certificate=CERT,
                trials=20,
                seed=7,
                knobs=(SOFT, HARD),
                open=fake_open,
                appliers=FAKE_APPLIERS,
            )
            soft, hard = record["knobs"]
            self.assertEqual([r["level"] for r in soft["rungs"]], [0.5, 0.9, 2.0])
            self.assertEqual(soft["cliff"], {"level": 2.0, "rung": 3})
            self.assertEqual([r["successes"] for r in soft["rungs"]], [20, 20, 0])
            self.assertIsNone(hard["cliff"])
            self.assertEqual([r["name"] for r in record["ranking"]], ["soft", "hard"])
            self.assertEqual(record["sensitivity"], "most sensitive to soft 2 u")
            self.assertTrue(all(r["within_tolerance"] for r in soft["rungs"][:2]))
            self.assertFalse(soft["rungs"][2]["within_tolerance"])
            summary = _summary_deploy(folder)
            self.assertEqual(summary["sensitivity"], "most sensitive to soft 2 u")

    def test_a_baseline_under_the_bound_is_refused(self) -> None:
        def fake_open(manifest: Any, *, assets_dir: Any) -> FakeRuntime:
            return FakeRuntime(manifest, threshold=-1.0)  # falls untouched

        with tempfile.TemporaryDirectory() as tmp:
            folder = _manifest(Path(tmp))
            _passing_gate(folder)
            with self.assertRaises(ValueError) as why:
                attribute(
                    folder,
                    assets_dir=None,
                    certificate=CERT,
                    trials=20,
                    knobs=(SOFT,),
                    open=fake_open,
                    appliers=FAKE_APPLIERS,
                )
            self.assertIn("does not reproduce here", str(why.exception))
            self.assertFalse((folder / ATTRIBUTION_FILE).exists())

    def test_another_protocol_or_no_interval_is_refused_by_name(self) -> None:
        """The review of 2026-09-24: an attribution at a count or seed the
        gate never ran ran other trials than the gate it cited, and a
        certificate without `ci95` read a lower bound of 0 (no cliff ever)."""

        def fake_open(manifest: Any, *, assets_dir: Any) -> FakeRuntime:
            return FakeRuntime(manifest)

        with tempfile.TemporaryDirectory() as tmp:
            folder = _manifest(Path(tmp))
            _passing_gate(folder, trials=20, seed=1000)
            for asked, word in (
                ({"trials": 8}, "trials 20"),
                ({"seed": 5}, "seed 1000"),
            ):
                with self.assertRaisesRegex(ValueError, word):
                    attribute(
                        folder,
                        assets_dir=None,
                        certificate=CERT,
                        knobs=(SOFT,),
                        open=fake_open,
                        appliers=FAKE_APPLIERS,
                        **asked,
                    )
            with self.assertRaisesRegex(ValueError, "no ci95"):
                attribute(
                    folder,
                    assets_dir=None,
                    certificate={"successes": 38, "trials": 40},
                    knobs=(SOFT,),
                    open=fake_open,
                    appliers=FAKE_APPLIERS,
                )

    def test_a_record_of_another_schema_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ATTRIBUTION_FILE).write_text(
                '{"schema": "trainnr-attribution/0"}'
            )
            with self.assertRaises(ValueError):
                read_attribution(Path(tmp))
            self.assertIsNone(read_attribution(Path(tmp) / "nowhere"))


class TheWrappers(unittest.TestCase):
    def _inner(self) -> SimpleNamespace:
        applied: list[np.ndarray] = []
        inner = SimpleNamespace(
            applied=applied,
            last_action=np.zeros(2, np.float32),
            command=np.zeros(3, np.float32),
            ticks=0,
            reset=lambda: None,
        )

        def apply(action: np.ndarray) -> None:
            applied.append(np.array(action))
            inner.last_action = np.array(action)
            inner.ticks += 1

        inner.apply = apply
        return inner

    def test_delayed_applies_older_actions_and_shows_the_policy_its_own(self) -> None:
        inner = self._inner()
        late = Delayed(inner, 2)
        for i in (1, 2, 3):
            late.apply(np.array([i, i], np.float32))
            self.assertEqual(inner.last_action.tolist(), [i, i])
        self.assertEqual([a.tolist() for a in inner.applied], [[0, 0], [0, 0], [1, 1]])
        late.reset()
        self.assertEqual(len(late.queue), 0)
        late.command = np.ones(3, np.float32)
        self.assertEqual(inner.command.tolist(), [1, 1, 1])

    def test_noisy_adds_noise_to_the_named_term_only(self) -> None:
        manifest = SimpleNamespace(
            observations=[
                SimpleNamespace(name="base_ang_vel", width=3),
                SimpleNamespace(name="joint_pos", width=2),
                SimpleNamespace(name="actions", width=2),
            ]
        )
        inner = SimpleNamespace(
            manifest=manifest,
            command=np.zeros(3, np.float32),
            observe=lambda: np.zeros(7, np.float32),
        )
        noisy = Noisy(inner, "joint_pos", 0.1, np.random.default_rng(1))
        self.assertEqual((noisy.lo, noisy.hi), (3, 5))
        obs = noisy.observe()
        self.assertEqual(obs[:3].tolist(), [0, 0, 0])
        self.assertEqual(obs[5:].tolist(), [0, 0])
        self.assertTrue(np.any(obs[3:5] != 0))
        with self.assertRaises(ValueError):
            Noisy(inner, "no_such_term", 0.1, np.random.default_rng(1))

    def test_pushed_shoves_the_base_every_period(self) -> None:
        inner = self._inner()
        inner.data = SimpleNamespace(qvel=np.zeros(6))
        inner.base_qvel = 0  # the free joint's first dof
        pushed = Pushed(inner, 1.0, np.random.default_rng(3), period_ticks=3)
        for _ in range(7):
            pushed.apply(np.zeros(2, np.float32))
        # shoved at ticks 3 and 6 (never at 0): two unit velocities, summed
        self.assertEqual(len(inner.applied), 7)
        self.assertGreater(float(np.linalg.norm(inner.data.qvel[:2])), 0.0)
        self.assertLessEqual(float(np.linalg.norm(inner.data.qvel[:2])), 2.0 + 1e-9)


TINY_XML = """
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.01" friction="1 0.005 0.0001"/>
    <body name="base" pos="0 0 0.3">
      <joint name="root" type="free"/>
      <geom type="box" size="0.1 0.1 0.05" mass="2" friction="0.6 0.005 0.0001"/>
      <body name="leg" pos="0 0 -0.05">
        <joint name="j" axis="0 1 0" range="-1 1"/>
        <geom type="capsule" size="0.02 0.1" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <general name="j" joint="j" biastype="affine" gainprm="20" biasprm="0 -20 -1"/>
  </actuator>
  <keyframe><key name="home" qpos="0 0 0.3 1 0 0 0 0"/></keyframe>
</mujoco>
"""


@needs_sim
class TheAppliersOnAModel(unittest.TestCase):
    def _runtime(self) -> Any:
        import mujoco  # noqa: PLC0415

        from rq_pipeline.deploy.runtime import Runtime  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            folder = _manifest(
                Path(tmp),
                joints={
                    "policy_order": ["j"],
                    "action_to_ctrl": [0],
                    "default_pos": [0.0],
                },
            )
            (folder / "scene.xml").write_text(TINY_XML)
            manifest = load_manifest(folder)
        model = mujoco.MjModel.from_xml_string(TINY_XML)
        session = SimpleNamespace(
            get_inputs=lambda: [SimpleNamespace(name="obs")],
            run=lambda _o, _i: [np.zeros((1, 1), np.float32)],
        )
        return Runtime(
            manifest=manifest, model=model, data=mujoco.MjData(model), session=session
        )

    def test_each_knob_turns_the_field_it_names(self) -> None:
        rt = self._runtime()
        friction_before = rt.model.geom_friction[:, 0].copy()
        APPLIERS["friction"](rt, 0.5, 0)
        np.testing.assert_allclose(rt.model.geom_friction[:, 0], friction_before * 0.5)

        rt = self._runtime()
        base = rt.model.body("base").id
        mass, inertia = (
            float(rt.model.body_mass[base]),
            rt.model.body_inertia[base].copy(),
        )
        APPLIERS["payload"](rt, 2.0, 0)
        self.assertAlmostEqual(float(rt.model.body_mass[base]), mass + 2.0)
        np.testing.assert_allclose(
            rt.model.body_inertia[base], inertia * (mass + 2) / mass
        )

        rt = self._runtime()
        APPLIERS["kp"](rt, 0.5, 0)
        self.assertAlmostEqual(float(rt.model.actuator_gainprm[0, 0]), 10.0)
        self.assertAlmostEqual(float(rt.model.actuator_biasprm[0, 1]), -10.0)
        self.assertAlmostEqual(float(rt.model.actuator_biasprm[0, 2]), -1.0)
        APPLIERS["kd"](rt, 4.0, 0)
        self.assertAlmostEqual(float(rt.model.actuator_biasprm[0, 2]), -4.0)

        rt = self._runtime()
        APPLIERS["tilt"](rt, 10.0, 0)
        g = np.asarray(rt.model.opt.gravity)
        self.assertAlmostEqual(float(np.linalg.norm(g)), 9.81, places=6)
        self.assertAlmostEqual(np.degrees(np.arctan2(g[0], -g[2])), 10.0, places=6)

        rt = self._runtime()
        self.assertIsInstance(APPLIERS["latency"](rt, 2, 0), Delayed)
        self.assertIsInstance(APPLIERS["joint_pos_noise"](rt, 0.01, 0), Noisy)
        pushed = APPLIERS["push"](rt, 1.0, 0)
        self.assertIsInstance(pushed, Pushed)
        self.assertEqual(pushed.period_ticks, round(attr.PUSH_PERIOD_S / rt.step_dt))
        self.assertIs(APPLIERS[NOMINAL.name](rt, 1.0, 0), rt)


class TheDoor(unittest.TestCase):
    def test_the_action_spawns_the_tool_verbatim(self) -> None:
        with harness() as (actions, spawner):
            actions.attribute_deployment("final", project="/p", trials=8, seed=5)
            (argv, cwd), *_ = spawner.calls
            tool = argv[argv.index(str(TOOLS_DIR / "attribute-deployment.py")) :]
            for flag, value in (
                ("--project", "/p"),
                ("--name", "final"),
                ("--runtime", "mujoco"),
                ("--trials", "8"),
                ("--seed", "5"),
            ):
                self.assertEqual(tool[tool.index(flag) + 1], value)
            self.assertEqual(cwd, PIPELINE_DIR)
            with self.assertRaises(ValueError):
                actions.attribute_deployment("a/b", project="/p")

    def test_the_server_door_refuses_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            os.environ[PROJECT_ENV] = str(project.root)
            try:
                out = server.attribute_deployment("ghost")
                self.assertEqual(out["status"], "refused")
                self.assertIn("ghost", out["reason"])
                out = server.attribute_deployment("ghost", runtime="webots")
                self.assertEqual(out["status"], "refused")
                self.assertIn("webots", out["reason"])
            finally:
                os.environ.pop(PROJECT_ENV, None)


if __name__ == "__main__":
    unittest.main()
