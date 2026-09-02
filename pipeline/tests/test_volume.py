"""The volume door's accounting, pure: prefixes, sizes, the delete
plan, and the credentials reader that reads ONLY what it is told."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rq_pipeline.cloud.volume import (
    CREDENTIAL_NAMES,
    Entry,
    Usage,
    human,
    load_credentials,
    prefix_of,
    read_env_file,
    total,
    under,
    usage_by_prefix,
)

ENTRIES = [
    Entry("robotiq/runs/campaign-1/demos/episode_0000/frames/000000.jpg", 30_000),
    Entry("robotiq/runs/campaign-1/dataset/data/chunk-000/file.parquet", 800_000),
    Entry("robotiq/runs/paired-study/checkpoints/model.safetensors", 600_000),
    Entry("robotiq/pipeline/.venv-train/lib/torch.so", 5_000_000),
    Entry("robotiq/campaign2.log", 700),
]


class TheAccounting(unittest.TestCase):
    def test_prefixes_by_depth(self) -> None:
        self.assertEqual(prefix_of("robotiq/runs/x/y", 1), "robotiq/")
        self.assertEqual(prefix_of("robotiq/runs/x/y", 2), "robotiq/runs/")
        self.assertEqual(prefix_of("robotiq/campaign2.log", 2), "robotiq/campaign2.log")

    def test_usage_groups_largest_first(self) -> None:
        by_top = usage_by_prefix(ENTRIES, 3)
        self.assertEqual(next(iter(by_top)), "robotiq/pipeline/.venv-train/")
        self.assertEqual(by_top["robotiq/runs/campaign-1/"], Usage(830_000, 2))
        self.assertEqual(total(ENTRIES), Usage(6_430_700, 5))

    def test_a_delete_takes_the_prefix_and_nothing_wider(self) -> None:
        doomed = under(ENTRIES, "robotiq/runs/campaign-1/demos/")
        self.assertEqual([e.key for e in doomed], [ENTRIES[0].key])
        self.assertEqual(under(ENTRIES, "robotiq/runs/campaign-1"), ENTRIES[:2])

    def test_human_sizes(self) -> None:
        self.assertEqual(human(512), "512 B")
        self.assertEqual(human(1536), "1.5 KB")
        self.assertEqual(human(35 * 1024**3), "35.0 GB")


class TheCredentialsReader(unittest.TestCase):
    def test_reads_only_the_named_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            env.write_text(
                "# comment\nRUNPOD_API_KEY=never-me\nexport AWS_ACCESS_KEY_ID='ak'\n"
                'AWS_SECRET_ACCESS_KEY="sk"\nMALFORMED\n'
            )
            self.assertEqual(
                read_env_file(env, CREDENTIAL_NAMES),
                {"AWS_ACCESS_KEY_ID": "ak", "AWS_SECRET_ACCESS_KEY": "sk"},
            )
            target: dict[str, str] = {"AWS_ACCESS_KEY_ID": "already"}
            load_credentials(env, target)
            self.assertEqual(target["AWS_ACCESS_KEY_ID"], "already")  # never overrides
            self.assertEqual(target["AWS_SECRET_ACCESS_KEY"], "sk")
            self.assertNotIn("RUNPOD_API_KEY", target)


if __name__ == "__main__":
    unittest.main()
