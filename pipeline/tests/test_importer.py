"""Importing what exists into a project: an rq_mjlab experiment becomes a
run, a policy and evaluations that cite each other by version; a ledger
record becomes a finding; the index, previews and details know all three."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import rq_pipeline.mcp_server as server
from rq_pipeline.envs.rsl_rl_log import parse_rsl_rl_log
from rq_pipeline.project import (
    PROJECT_ENV,
    Kind,
    create_project,
    index_project,
    write_index,
)
from rq_pipeline.project.details import details_path
from rq_pipeline.project.importer import (
    _training_log,
    import_experiment,
    import_finding,
    ledger_findings,
)

IDENTITY = {
    "robot": "microduck@14b51b63c52e",
    "actuator": "xl330-m6@e57c25635c89",
    "dr_basis": "none: the bundle's point fit exactly (no law DR)",
    "seed": 42,
}
SUCCESSES, TRIALS = 3, 4  # the synthetic arm's outcome
VERDICT = {
    "source": "microduck-walk@29f89d7b28ed",
    "policy": "model_7999",
    "instrument": "mjlab-1.6.0+mujoco-3.11.0+warp-1.17.0+cuda",
    "protocol": {"trials": 4, "seed": 1000, "judged_at": "the bundle's point fit"},
    "funnel": {"survived": 4, "tracked": 3},
    "successes": 3,
    "trials": 4,
    "ci95": [0.194, 0.994],
}


def _arm(root: Path, name: str) -> Path:
    train = root / name / "train"
    (train / "verdict").mkdir(parents=True)
    (train / "identity.json").write_text(json.dumps(IDENTITY))
    (train / "model_7999.pt").write_bytes(b"\x80\x02not a real checkpoint")
    (train / "verdict" / "walk-verdict-at-fit-cuda.json").write_text(
        json.dumps(VERDICT)
    )
    lines = [
        json.dumps(
            {
                "trial": t,
                "success": t < SUCCESSES,
                "steps": 1000,
                "policy": "model_7999",
                "seed": t,
            }
        )
        for t in range(TRIALS)
    ]
    (train / "verdict" / "records-at-fit-cuda.jsonl").write_text(
        "\n".join(lines) + "\n"
    )
    return root / name


class Experiment(unittest.TestCase):
    def test_an_experiment_becomes_run_policy_and_evaluation_citing_each_other(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            arm = _arm(Path(tmp) / "study", "point#2")
            out = import_experiment(project, arm)
            self.assertEqual(out["name"], "point#2")
            self.assertTrue(out["policy"].startswith("point#2@"))
            self.assertEqual(len(out["evaluations"]), 1)
            index = index_project(project)
            policy = index.by_kind(Kind.POLICY)[0]
            self.assertEqual(policy.cites["run"], out["run"])
            self.assertEqual(policy.cites["robot"], IDENTITY["robot"])
            self.assertEqual(policy.summary["iterations"], 8000)
            self.assertEqual(policy.summary["randomization"], "none")
            cert = index.by_kind(Kind.CERTIFICATE)[0]
            self.assertEqual(cert.cites["policy"], out["policy"])
            self.assertEqual(cert.summary["success"], "3 / 4")
            self.assertEqual(cert.summary["95% CI"], "[0.19, 0.99]")
            states = {s.name: s.present for s in index.states}
            self.assertTrue(states["policy trained"] and states["policy evaluated"])
            with self.assertRaises(FileExistsError):
                import_experiment(project, arm)
            with self.assertRaisesRegex(FileNotFoundError, "identity.json"):
                import_experiment(project, Path(tmp) / "study")
            # Previews and details for the new kinds.
            write_index(project, index)
            previews = project.root / ".index" / "previews"
            self.assertTrue(
                (previews / f"{cert.stamp}.png").is_file(),
                "the evaluation draws itself",
            )
            detail = json.loads(details_path(project, policy.stamp).read_text())
            titles = [s["title"] for s in detail["sections"]]
            self.assertEqual(titles, ["Policy", "Evaluations of this policy"])
            evals = detail["sections"][1]["rows"]
            self.assertEqual(evals[0][1], "3 / 4")


class Finding(unittest.TestCase):
    def test_a_ledger_record_becomes_a_finding_with_its_outcome_table(self) -> None:
        listed = ledger_findings("walk-c1")
        self.assertTrue(listed, "the ledger holds walk-c1 records")
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            out = import_finding(project, listed[0]["id"])
            self.assertTrue(out["finding"].startswith(listed[0]["id"]))
            with self.assertRaises(FileExistsError):
                import_finding(project, listed[0]["id"])
            with self.assertRaisesRegex(FileNotFoundError, "no finding"):
                import_finding(project, "nothing-here")
            index = index_project(project)
            finding = index.by_kind(Kind.FINDING)[0]
            self.assertIn("claim", finding.summary)
            write_index(project, index)
            detail = json.loads(details_path(project, finding.stamp).read_text())
            titles = [s["title"] for s in detail["sections"]]
            self.assertEqual(titles[:2], ["Claim", "Record"])
            self.assertIn("Outcome by condition", titles)
            self.assertTrue(
                (
                    project.root / ".index" / "previews" / f"{finding.stamp}.png"
                ).is_file()
            )


class RslRlLogTest(unittest.TestCase):
    LOG = (
        "[g3] 4096 envs on cuda:0, 8000 iterations\n"
        "####\n Learning iteration 0/8000 \n"
        "   Steps per second: 58756 \n   Mean reward: -1.61\n"
        "   Mean episode length: 23.88\n"
        "   Iteration time: 1.67s\n   Time elapsed: 0:00:01\n"
        "####\n Learning iteration 1/8000 \n"
        "   Steps per second: 60000 \n   Mean reward: 130.37\n"
        "   Mean episode length: 999.0\n"
        "   Time elapsed: 3:43:02\n"
    )

    def test_parses_facts_and_curve(self) -> None:
        record = parse_rsl_rl_log(self.LOG)
        assert record is not None
        self.assertEqual(
            (record.envs, record.device, record.iterations), (4096, "cuda:0", 8000)
        )
        self.assertEqual(record.iterations_logged, 2)
        self.assertEqual(record.wall_seconds, 3 * 3600 + 43 * 60 + 2)
        self.assertEqual(record.final["reward"], 130.37)
        self.assertEqual(record.best_reward, 130.37)
        self.assertEqual(record.columns[:3], ["iteration", "reward", "episode_length"])
        self.assertEqual(len(record.curve), 2)

    def test_total_steps_is_a_curve_column(self) -> None:
        record = parse_rsl_rl_log(
            " Learning iteration 0/2\n   Total steps: 98304 \n   Mean reward: 1.0\n"
        )
        assert record is not None
        self.assertIn("total_steps", record.columns)
        self.assertEqual(record.curve[0][record.columns.index("total_steps")], 98304.0)

    def test_no_iteration_is_none(self) -> None:
        self.assertIsNone(parse_rsl_rl_log("Warp initialized\n"))


class StudyLogTest(unittest.TestCase):
    def test_a_run_finds_its_segment_in_a_study_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp)
            (study / "rep-a.log").write_text(
                "[g3] log_dir: /w/point#2/train\n Learning iteration 0/8\n"
                "[g3] log_dir: /w/point#3/train\n Learning iteration 0/8\n"
                " Learning iteration 1/8\n"
            )
            (study / "rep-d.log").write_text(
                "[g3] log_dir: /w/point#3/train\n Learning iteration 0/8\n"
            )
            (study / "point#3" / "train").mkdir(parents=True)
            seg = _training_log(
                study / "point#3", study / "point#3" / "train", "point#3"
            )
            assert seg is not None
            self.assertEqual(seg.count("Learning iteration"), 2)
            self.assertNotIn("point#2", seg)
            self.assertIsNone(
                _training_log(study / "point#3", study / "point#3" / "train", "wide")
            )


class ImportDoorsTest(unittest.TestCase):
    """The doors answer a refusal record, by name, rather than raising."""

    def test_doors_refuse_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ[PROJECT_ENV] = str(Path(tmp) / "p")
            try:
                server.create_project_dir(str(Path(tmp) / "p"), "p")
                out = server.import_experiment(str(Path(tmp) / "nowhere"))
                self.assertEqual(out["status"], "refused")
                self.assertIn("identity.json", out["reason"])
                out = server.import_finding("no-such-finding")
                self.assertEqual(out["status"], "refused")
                self.assertIn("no-such-finding", out["reason"])
            finally:
                os.environ.pop(PROJECT_ENV, None)


class ImportEdgesTest(unittest.TestCase):
    def test_train_dir_form_finds_the_study_log_and_copies_its_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp) / "study"
            arm = _arm(study, "point#2")
            (study / "rep.log").write_text(
                "[g3] log_dir: /w/point#2/train\n Learning iteration 0/8\n"
                "   Mean reward: 1.0\n"
            )
            project = create_project(Path(tmp) / "p", "p")
            out = import_experiment(project, arm / "train")
            self.assertEqual(out["name"], "point#2")
            run = project.root / "runs" / "point#2"
            self.assertTrue((run / "training.json").is_file())
            cert = next((project.root / "certificates").iterdir())
            self.assertTrue((cert / "records-at-fit-cuda.jsonl").is_file())

    def test_own_log_beats_the_study_log_and_last_checkpoint_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp) / "study"
            arm = _arm(study, "a")
            (arm / "train" / "train.log").write_text(
                "[g3] log_dir: /w/a/train\n Learning iteration 0/8\n"
                "   Mean reward: 7.0\n"
            )
            (study / "rep.log").write_text(
                "[g3] log_dir: /w/a/train\n Learning iteration 0/8\n"
                "   Mean reward: 1.0\n"
            )
            (arm / "train" / "model_3999.pt").write_bytes(b"\x01")
            project = create_project(Path(tmp) / "p", "p")
            import_experiment(project, arm)
            record = json.loads((project.root / "runs/a/training.json").read_text())
            self.assertEqual(record["final"]["reward"], 7.0)
            policy = json.loads((project.root / "policies/a/policy.json").read_text())
            self.assertEqual(policy["checkpoint"], "model_7999.pt")

    def test_a_refused_import_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            arm = _arm(Path(tmp) / "study", "b")
            (arm / "train" / "model_7999.pt").unlink()
            project = create_project(Path(tmp) / "p", "p")
            with self.assertRaises(FileNotFoundError):
                import_experiment(project, arm)
            self.assertFalse((project.root / "runs" / "b").exists())
            self.assertFalse((project.root / "policies" / "b").exists())

    def test_a_missing_cell_is_null_not_nan(self) -> None:
        record = parse_rsl_rl_log(
            "[g3] log_dir: x\n Learning iteration 0/2\n   Mean reward: 1.0\n"
            " Learning iteration 1/2\n   Mean episode length: 5\n"
            "   Time elapsed: 1 day, 2:00:00\n"
        )
        assert record is not None
        text = json.dumps(record.to_json())
        self.assertNotIn("NaN", text)
        self.assertIsNone(record.curve[1][record.columns.index("reward")])
        self.assertEqual(record.wall_seconds, 86400 + 7200)


if __name__ == "__main__":
    unittest.main()
