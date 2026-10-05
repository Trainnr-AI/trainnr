#!/usr/bin/env python3
"""The Studio's THIRD_PARTY_LICENSES.md, completed and checked.

    python3 tools/third-party-notices.py append THIRD_PARTY_LICENSES.md
    python3 tools/third-party-notices.py check THIRD_PARTY_LICENSES.md

cargo-about (crates/trainnr-studio/about.toml, about.hbs) writes each
crate's licence text. Two things a binary distribution also owes are not
licence texts, so `append` adds them, read from the locked dependency
graph (`cargo metadata --locked`, normal dependencies of the Studio):

- the NOTICE file of every Apache-2.0 crate that ships one (Apache-2.0
  section 4(d): Apache Arrow, DataFusion, object_store, ...), each
  distinct text once with the crates it covers;
- the fonts compiled into the binary (`include_bytes!` of a .ttf or .otf
  in a dependency's source), each with the copyright and licence lines
  from the font's own name table: the SIL Open Font Licence asks that the
  copyright notice travel with the font.

`check` refuses a file that lacks an attribution a past review found
missing, or that carries HTML escapes (a template printing `{{text}}`
instead of `{{{text}}}` turns every quote in a licence into `&quot;`).
The release workflow (studio-release.yml) runs both after cargo-about.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CRATE = REPO / "crates" / "trainnr-studio"
NOTICE_NAMES = ("NOTICE", "NOTICE.txt", "NOTICE.md")
FONT = re.compile(r'include_bytes!\(\s*"([^"]+\.(?:ttf|otf))"\s*\)')

# Text the finished file must contain, and where it comes from.
REQUIRED: dict[str, str] = {
    "Apache Arrow": "the NOTICE of the arrow, parquet and object_store crates",
    "Meta Platforms": "zstd's BSD-3-Clause licence (zstd-sys, about.toml)",
    "Yann Collet": "liblz4's BSD-2-Clause licence (lz4-sys, about.toml)",
    "Bitstream": "the Hack font's Bitstream Vera licence (epaint_default_fonts)",
    "John Slegers": "emoji-icon-font's MIT licence (epaint_default_fonts)",
}
ESCAPES = re.compile(r"&(?:quot|amp|lt|gt|#x27|#x60|#x3D);")


def metadata() -> dict:
    run = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--locked"],
        cwd=CRATE,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(run.stdout)


def shipped(meta: dict) -> list[dict]:
    """The packages linked into the Studio: the root's normal dependencies,
    transitively (build and dev dependencies left out, as about.toml
    does), every target platform included."""
    packages = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    root = meta["resolve"]["root"]
    seen, stack = set(), [root]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        for dep in nodes[node]["deps"]:
            if any(kind["kind"] is None for kind in dep["dep_kinds"]):
                stack.append(dep["pkg"])
    seen.discard(root)
    return sorted((packages[i] for i in seen), key=lambda p: (p["name"], p["version"]))


def notices(packages: list[dict]) -> list[tuple[str, list[str]]]:
    """(NOTICE text, the crates that ship it) for each distinct text."""
    texts: dict[str, list[str]] = {}
    for package in packages:
        if "Apache-2.0" not in (package.get("license") or ""):
            continue
        root = Path(package["manifest_path"]).parent
        for name in NOTICE_NAMES:
            path = root / name
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="replace").strip()
                texts.setdefault(text, []).append(
                    f"{package['name']} {package['version']}"
                )
                break
    return sorted(texts.items(), key=lambda item: item[1][0])


def font_names(data: bytes) -> dict[int, str]:
    """The font's name table: copyright (0), family (1), licence (13)."""
    count = struct.unpack(">H", data[4:6])[0]
    for i in range(count):
        tag, _, offset, length = struct.unpack(
            ">4sIII", data[12 + 16 * i : 28 + 16 * i]
        )
        if tag != b"name":
            continue
        table = data[offset : offset + length]
        _, records, strings = struct.unpack(">HHH", table[:6])
        names: dict[int, str] = {}
        for j in range(records):
            platform, _, _, name_id, size, at = struct.unpack(
                ">HHHHHH", table[6 + 12 * j : 18 + 12 * j]
            )
            raw = table[strings + at : strings + at + size]
            text = raw.decode(
                "utf-16-be" if platform in (0, 3) else "latin-1", "replace"
            )
            # The first line of a long field (a licence summary), whitespace
            # collapsed; a copyright with two holders keeps both.
            first = next((line for line in text.splitlines() if line.strip()), "")
            names.setdefault(name_id, " ".join(first.split()))
        return names
    return {}


def fonts(packages: list[dict]) -> list[tuple[str, str, dict[int, str]]]:
    """(crate, font file, its names) for every font a crate compiles in."""
    found = []
    for package in packages:
        root = Path(package["manifest_path"]).parent
        for source in sorted((root / "src").rglob("*.rs")):
            text = source.read_text(encoding="utf-8", errors="replace")
            for relative in FONT.findall(text):
                path = (source.parent / relative).resolve()
                if path.is_file():
                    found.append(
                        (
                            f"{package['name']} {package['version']}",
                            path.name,
                            font_names(path.read_bytes()),
                        )
                    )
    return found


def append(target: Path) -> int:
    packages = shipped(metadata())
    parts = [
        "",
        "# NOTICE files of Apache-2.0 components",
        "",
        "Apache-2.0 section 4(d): the NOTICE files these crates ship, each text once.",
        "",
    ]
    found = notices(packages)
    for text, crates in found:
        parts += ["Used by: " + ", ".join(crates), "", "```", text, "```", ""]
    parts += [
        "# Fonts embedded in the Studio",
        "",
        "Each font compiled into the binary, with the copyright and licence it",
        "carries in its own name table; the licence texts are above.",
        "",
    ]
    embedded = fonts(packages)
    none = "none in the font; see the licences above"
    for crate, name, names in embedded:
        parts += [
            f"- {names.get(1, name)}, `{name}` ({crate})",
            f"  - Copyright: {names.get(0) or none}",
            f"  - Licence: {names.get(13) or 'see the licence texts above'}",
        ]
    with target.open("a", encoding="utf-8") as out:
        out.write("\n".join(parts) + "\n")
    print(
        f"{target}: appended {len(found)} NOTICE text(s) from "
        f"{sum(len(c) for _, c in found)} crates and {len(embedded)} font(s)"
    )
    return 0


def check(target: Path) -> int:
    text = target.read_text(encoding="utf-8")
    problems = [
        f"no {needle!r}: {source}"
        for needle, source in REQUIRED.items()
        if needle not in text
    ]
    escapes = sorted(set(ESCAPES.findall(text)))
    if escapes:
        problems.append(
            f"HTML escapes {escapes}: the template must print licence texts "
            "with triple braces"
        )
    for line in problems:
        print(f"{target}: {line}")
    if not problems:
        print(f"{target}: every required attribution present, no HTML escapes")
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("action", choices=("append", "check"))
    parser.add_argument("file", type=Path)
    args = parser.parse_args()
    return append(args.file) if args.action == "append" else check(args.file)


if __name__ == "__main__":
    sys.exit(main())
