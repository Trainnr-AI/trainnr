"""The downloaded Studio: the release asset for this platform, verified by
its checksum, unpacked into the user's cache, and found by the launcher.
Served here from local files, so the tests need no network."""

import hashlib
import io
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from trainnr import studio_install as si


def _release(
    where: Path, tag: str, triple: str, *, corrupt: bool = False
) -> dict[str, str]:
    """A fake release: the archive with a binary inside, and its checksum."""
    name = si.asset_name(tag, triple)
    archive = where / name
    payload = b"#!/bin/sh\necho studio\n"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(f"trainnr-studio-{tag}/{si.EXE}")
        info.size = len(payload)
        info.mode = 0o755
        tar.addfile(info, io.BytesIO(payload))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if corrupt:
        digest = "0" * 64
    (where / (name + ".sha256")).write_text(f"{digest}  {name}\n")
    return {n: (where / n).as_uri() for n in (name, name + ".sha256")}


@unittest.skipIf(sys.platform.startswith("win"), "the fixture writes a POSIX archive")
class DownloadedStudio(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.env = mock.patch.dict(
            os.environ,
            {"XDG_CACHE_HOME": str(self.tmp / "cache"), si.RELEASE_ENV: "v9.9.9"},
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.triple = "x86_64-unknown-linux-gnu"
        self.target = mock.patch.object(si, "target", return_value=self.triple)
        self.target.start()
        self.addCleanup(self.target.stop)

    def test_install_verifies_unpacks_and_is_found_after(self) -> None:
        urls = _release(self.tmp, "v9.9.9", self.triple)
        self.assertIsNone(si.installed_binary())
        with mock.patch.object(si, "_asset_urls", return_value=urls) as fetched:
            path = si.install(say=lambda _line: None)
            self.assertEqual(path, si.installed_binary())
            self.assertTrue(os.access(path, os.X_OK))
            # a second call downloads nothing
            self.assertEqual(si.install(say=lambda _line: None), path)
            self.assertEqual(fetched.call_count, 1)

    def test_a_checksum_mismatch_installs_nothing(self) -> None:
        urls = _release(self.tmp, "v9.9.9", self.triple, corrupt=True)
        with (
            mock.patch.object(si, "_asset_urls", return_value=urls),
            self.assertRaisesRegex(si.StudioInstallError, "checksum mismatch"),
        ):
            si.install(say=lambda _line: None)
        self.assertIsNone(si.installed_binary())

    def test_an_unbuilt_platform_says_how_to_build(self) -> None:
        with (
            mock.patch.object(si, "target", return_value=None),
            self.assertRaisesRegex(si.StudioInstallError, "cargo build --release"),
        ):
            si.install(say=lambda _line: None)


@unittest.skipIf(sys.platform.startswith("win"), "the fixture writes a POSIX archive")
class ThreeInstallsAtOnce(unittest.TestCase):
    """The plugin's session hook and launch_studio can install the same
    version at the same moment: each must end with the binary in place
    and none may fail or remove the others' work (review, 2026-10-03)."""

    def test_concurrent_installs_all_succeed(self) -> None:
        import json  # noqa: PLC0415
        import subprocess  # noqa: PLC0415

        tmp = Path(tempfile.mkdtemp())
        urls = _release(tmp, "v9.9.9", "x86_64-unknown-linux-gnu")
        script = (
            "import json, sys\n"
            "from unittest import mock\n"
            "from trainnr import studio_install as si\n"
            f"urls = json.loads({json.dumps(json.dumps(urls))})\n"
            "triple = 'x86_64-unknown-linux-gnu'\n"
            "with mock.patch.object(si, 'target', return_value=triple), "
            "mock.patch.object(si, '_asset_urls', return_value=urls):\n"
            "    print(si.install(say=lambda _l: None))\n"
        )
        env = {
            **os.environ,
            "XDG_CACHE_HOME": str(tmp / "cache"),
            si.RELEASE_ENV: "v9.9.9",
        }
        runs = [
            subprocess.Popen(
                [sys.executable, "-c", script],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(3)
        ]
        outputs = [run.communicate(timeout=120) for run in runs]
        for run, (out, err) in zip(runs, outputs, strict=True):
            self.assertEqual(run.returncode, 0, err)
            self.assertTrue(Path(out.strip()).is_file(), out)
        self.assertEqual(len({out.strip() for out, _ in outputs}), 1)


class Names(unittest.TestCase):
    def test_assets_are_named_by_tag_and_target(self) -> None:
        self.assertEqual(
            si.asset_name("v0.1.0", "aarch64-apple-darwin"),
            "trainnr-studio-v0.1.0-aarch64-apple-darwin.tar.gz",
        )
        self.assertTrue(
            si.asset_name("v0.1.0", "x86_64-pc-windows-msvc").endswith(".zip")
        )

    def test_the_tag_follows_the_package_version_unless_named(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(si.RELEASE_ENV, None)
            self.assertEqual(si.release_tag(), f"v{si.__version__}")


if __name__ == "__main__":
    unittest.main()
