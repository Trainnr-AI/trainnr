"""The ssh/rsync command lines for a rented machine: the archive flags
that avoid chown, the filter order rsync needs, and what never leaves."""

import unittest
from pathlib import Path

from rq_pipeline.cloud.provider import SshEndpoint
from rq_pipeline.cloud.transfer import (
    Rsync,
    follow_filters,
    pull_filters,
    push_filters,
    remote_path,
    rsync_argv,
    ssh_argv,
)

DOOR = SshEndpoint(host="203.0.113.7", port=12345, username="root", command="")
KEY = Path("/keys/id_ed25519")


class TheCommandLines(unittest.TestCase):
    def test_ssh_names_the_port_and_the_key_only_when_given(self) -> None:
        with_key = ssh_argv(DOOR, KEY)
        self.assertEqual(with_key[:3], ["ssh", "-i", str(KEY)])
        self.assertEqual(with_key[-1], "root@203.0.113.7")
        self.assertIn("12345", with_key)
        self.assertNotIn("-i", ssh_argv(DOOR))

    def test_rsync_avoids_chown_and_puts_filters_before_the_paths(self) -> None:
        argv = rsync_argv(DOOR, KEY, "/a/", "root@h:/b/", filters=["--exclude=x"])
        self.assertEqual(argv[:3], ["rsync", "-rlptDz", "--info=progress2"])
        self.assertNotIn("-a", argv)
        self.assertEqual(argv[-3:], ["--exclude=x", "/a/", "root@h:/b/"])
        self.assertIn("-p 12345", argv[argv.index("-e") + 1])
        quiet = rsync_argv(DOOR, None, "/a/", "/b/", filters=[], progress=False)
        self.assertNotIn(Rsync.PROGRESS, quiet)

    def test_remote_path(self) -> None:
        self.assertEqual(
            remote_path(DOOR, "/workspace/x"), "root@203.0.113.7:/workspace/x"
        )


class TheFilters(unittest.TestCase):
    def test_push_never_ships_the_credentials(self) -> None:
        self.assertIn("--exclude=.env", push_filters())
        self.assertIn("--exclude=pipeline/runs", push_filters())

    def test_pull_takes_a_directory_and_its_contents_then_excludes_the_rest(
        self,
    ) -> None:
        self.assertEqual(
            pull_filters("t5"),
            ["--include=t5-*/", "--include=t5-*/**", "--exclude=*"],
        )

    def test_follow_leaves_the_weights_behind(self) -> None:
        filters = follow_filters("t5")
        self.assertEqual(filters[0], "--exclude=model.safetensors")
        self.assertEqual(filters[-1], "--exclude=*")  # the pull rule, last
        self.assertLess(
            filters.index("--exclude=training_state"), filters.index("--include=t5-*/")
        )


if __name__ == "__main__":
    unittest.main()
