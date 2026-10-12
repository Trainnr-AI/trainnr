#!/usr/bin/env python3
"""Notarize a signed macOS file with Apple, and staple the ticket when the
file can carry one (a disk image, an app); fail with Apple's own log when
it is refused. The release workflow runs it on the macOS runner, after
signing, with the maintainer's App Store Connect team key (docs/85 §7):

    python3 tools/notarize.py <file> --key <AuthKey.p8> --key-id <id> --issuer <uuid>

A bare executable is submitted inside a zip (Apple takes zip, dmg and pkg)
and is not stapled: a Mach-O file has nowhere to keep a ticket, so
Gatekeeper asks Apple for it online. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

#: What notarytool reports for a submission Apple accepted.
ACCEPTED = "Accepted"
#: Files Apple notarizes as they are; anything else goes in a zip.
SUBMITTABLE = (".dmg", ".zip", ".pkg")
#: Files that can hold a stapled ticket.
STAPLEABLE = (".dmg", ".app", ".pkg")


def outcome(reply: str) -> tuple[str, str]:
    """notarytool's JSON reply: (status, submission id)."""
    data = json.loads(reply)
    return str(data.get("status", "")), str(data.get("id", ""))


def notarize(path: Path, credentials: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        submitted = path
        if path.suffix not in SUBMITTABLE:
            submitted = Path(tmp) / f"{path.name}.zip"
            subprocess.run(
                ["ditto", "-c", "-k", "--keepParent", str(path), str(submitted)],
                check=True,
            )
        reply = subprocess.run(
            [
                "xcrun",
                "notarytool",
                "submit",
                str(submitted),
                "--wait",
                "--output-format",
                "json",
                *credentials,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    try:
        status, submission = outcome(reply.stdout)
    except json.JSONDecodeError:
        sys.exit(f"notarytool did not answer in JSON:\n{reply.stdout}\n{reply.stderr}")
    print(f"{path.name}: {status} ({submission})")
    if status != ACCEPTED:
        if submission:
            subprocess.run(
                ["xcrun", "notarytool", "log", submission, *credentials], check=False
            )
        sys.exit(f"{path.name} was not notarized: {status}")
    if path.suffix in STAPLEABLE:
        subprocess.run(["xcrun", "stapler", "staple", str(path)], check=True)
        subprocess.run(["xcrun", "stapler", "validate", str(path)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("file", type=Path)
    parser.add_argument("--key", required=True, help="the team key's .p8 file")
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--issuer", required=True)
    args = parser.parse_args()
    notarize(
        args.file,
        ["--key", str(args.key), "--key-id", args.key_id, "--issuer", args.issuer],
    )


if __name__ == "__main__":
    main()
