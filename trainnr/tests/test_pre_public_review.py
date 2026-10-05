"""The smaller fixes of the second pre-public security review (2026-10-05):
each guard refuses the input the review reproduced."""

from __future__ import annotations

import struct
import tempfile
import unittest
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from unittest import mock

from trainnr.deploy.viewport_source import deployment_folder
from trainnr.robot import asset_fetch
from trainnr.robots import public_logs


class AnEmptyRawAnswer(unittest.TestCase):
    """With no raw bytes to check, the media itself must be the listed blob;
    it was returned unchecked."""

    def test_unchecked_media_is_refused(self) -> None:
        good = b"the real file\n"
        answers = {asset_fetch.RAW_FILE: b"", asset_fetch.MEDIA_FILE: b"tampered\n"}

        def get(url: str, timeout: float) -> bytes:
            for template, body in answers.items():
                if url.startswith(template.split("{", 1)[0]):
                    return body
            raise AssertionError(url)

        with (
            mock.patch.object(asset_fetch, "_get", get),
            self.assertRaises(asset_fetch.AssetFetchError),
        ):
            asset_fetch.fetch_file(
                "o/r", "0" * 40, "a.stl", blob=asset_fetch.git_blob_id(good)
            )
        answers[asset_fetch.MEDIA_FILE] = good
        with mock.patch.object(asset_fetch, "_get", get):
            self.assertEqual(
                asset_fetch.fetch_file(
                    "o/r", "0" * 40, "a.stl", blob=asset_fetch.git_blob_id(good)
                ),
                good,
            )


class AZipMemberPastItsSize(unittest.TestCase):
    def test_a_member_never_inflates_past_its_recorded_bytes(self) -> None:
        bomb = zlib.compressobj(9, zlib.DEFLATED, public_logs.RAW_DEFLATE)
        packed = bomb.compress(b"\0" * 10_000_000) + bomb.flush()
        name = b"log.bin"
        header = (
            public_logs.ZIP_LOCAL_SIGNATURE
            + b"\0" * 22
            + struct.pack("<HH", len(name), 0)
        )
        piece = public_logs.Piece(
            path="log.bin",
            bytes=100,
            sha256="0" * 64,
            span=public_logs.ZipSpan(
                offset=0, compressed=len(packed), method=public_logs.ZIP_DEFLATED
            ),
        )
        entry = public_logs.PublicLog(
            name="bomb",
            robot="go2",
            url="https://example.invalid/x.zip",
            fetch=public_logs.FETCH_ZIP_MEMBERS,
            pieces=(piece,),
            member="log.bin",
            adapter="none",
            source="test",
            recorded="2026-10-05",
            licence="test",
        )

        def ranged(url: str, start: int, length: int, opener: object) -> bytes:
            return (
                header
                if start == 0 and length == public_logs.ZIP_LOCAL_HEADER
                else packed
            )

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(public_logs, "_range", ranged),
            self.assertRaisesRegex(ValueError, "inflates past"),
        ):
            public_logs._fetch_zip_members(entry, Path(tmp), opener=None)


class NamesThatBecomePaths(unittest.TestCase):
    def test_a_deployment_is_one_plain_word(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("../x", "a/b", ".hidden", "a\\b", ""):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    deployment_folder(name, Path(tmp))

    def test_an_actuator_and_its_tier_are_plain_words(self) -> None:
        from trainnr.mcp_server import describe_actuator  # noqa: PLC0415

        for actuator, tier in (("../../x", "m6"), ("xl330", "../m6")):
            with (
                self.subTest(actuator=actuator, tier=tier),
                self.assertRaises(ValueError),
            ):
                describe_actuator(actuator, tier)


class TheStudioInstaller(unittest.TestCase):
    def test_a_redirect_away_from_https_is_refused(self) -> None:
        from trainnr.studio_install import _NoTokenAcrossHosts  # noqa: PLC0415

        handler = _NoTokenAcrossHosts()
        request = urllib.request.Request("https://github.com/x")
        with self.assertRaises(urllib.error.HTTPError):
            handler.redirect_request(
                request, None, 302, "Found", {}, "http://evil.example/x"
            )


if __name__ == "__main__":
    unittest.main()
