"""The files an MJCF model reaches: itself, its includes, and the meshes,
textures, height fields and skins they name, resolved the way MuJoCo
resolves them (the compiler's `meshdir`, `texturedir` and `assetdir`,
relative to the model file's folder). Onboarding copies this closure, not
the folder the model sits in (#120): a folder of earlier variants and
analysis output is not the robot, and a stamp over it would move with
every unrelated file.

Standard library only: this reads the XML, it does not compile it. The
onboarding compiles the copy afterwards, so a reference this reader
misses refuses the onboarding by name instead of leaving a broken bundle.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

# Elements whose `file` attribute names an asset, and the compiler
# directory each resolves against (MuJoCo's XML reference: meshdir is for
# meshes, skins and height fields; texturedir for textures; assetdir sets
# both when they are not given).
MESH_LIKE = ("mesh", "skin", "hfield")
TEXTURE = "texture"
TEXTURE_FILES = (
    "file",
    "fileright",
    "fileleft",
    "fileup",
    "filedown",
    "filefront",
    "fileback",
)
# Files beside the model kept with it although it does not name them:
# the licence and the notes that say where it came from.
KEPT_BESIDE = ("LICENSE", "LICENCE", "COPYING", "NOTICE", "README")
# trainnr's own files about the robot, kept too: its limits over several
# joints and, for a servo bench, what the servo is (docs/88).
TRAINNR_BESIDE = ("limits.json", "bench.json")


def _xml_files(model_file: Path) -> list[Path]:
    """The model file and every file it includes, recursively. MuJoCo
    resolves an include against the top-level model's folder."""
    base = model_file.parent
    seen: list[Path] = []
    todo = [model_file]
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.append(path)
        for include in ET.parse(path).getroot().iter("include"):
            name = include.get("file")
            if name:
                todo.append((base / name).resolve())
    return seen


def model_closure(model_file: Path) -> list[Path]:
    """Every file the model reaches, the model first, resolved."""
    model_file = model_file.resolve()
    base = model_file.parent
    xmls = _xml_files(model_file)
    roots = [ET.parse(path).getroot() for path in xmls]
    asset_dir = mesh_dir = texture_dir = ""
    for root in roots:
        for compiler in root.iter("compiler"):
            asset_dir = compiler.get("assetdir", asset_dir)
            mesh_dir = compiler.get("meshdir", mesh_dir)
            texture_dir = compiler.get("texturedir", texture_dir)
    mesh_dir = mesh_dir or asset_dir
    texture_dir = texture_dir or asset_dir
    found: list[Path] = list(xmls)
    for root in roots:
        for tag in MESH_LIKE:
            for element in root.iter(tag):
                name = element.get("file")
                if name:
                    found.append((base / mesh_dir / name).resolve())
        for element in root.iter(TEXTURE):
            for attribute in TEXTURE_FILES:
                name = element.get(attribute)
                if name:
                    found.append((base / texture_dir / name).resolve())
    unique: list[Path] = []
    for path in found:
        if path not in unique:
            unique.append(path)
    return unique


def kept_beside(model_file: Path) -> list[Path]:
    """The licence and readme files in the model's folder, and trainnr's own
    files about the robot (its limits, a bench's servo), kept with it."""
    return sorted(
        path
        for path in model_file.resolve().parent.iterdir()
        if path.is_file()
        and (path.name.upper().startswith(KEPT_BESIDE) or path.name in TRAINNR_BESIDE)
    )
