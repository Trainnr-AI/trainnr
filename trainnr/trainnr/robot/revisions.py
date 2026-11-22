"""A robot's revisions (#119): onboarding a robot again under its own name
with different content keeps the bundle it replaces, out of the lists, and
the new one says what it replaced and what changed in model terms.

The current revision is `robots/<name>/`, the one every tool uses. An
earlier one moves to `robots/.revisions/<name>/<hash>/` whole, records
included, so a run or an evaluation that cites its stamp can still be
traced to the files it was built on; the index marks such an artifact as
built on a superseded robot. Nothing is ever overwritten: the same content
onboarded again is the same stamp, and the call says so.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from trainnr.bundles.bundle import model_file_for_use
from trainnr.bundles.hashing import REVISION_FILE, STAMP_SEPARATOR, stamp

REVISIONS_DIR = ".revisions"
# How many changed values a revision's record lists; the count is exact.
CHANGES_SHOWN = 60
# Below this, a value did not change (float32 model values read back).
SAME = 1e-6
HINGE = 3  # mjtJoint.mjJNT_HINGE


def revisions_root(robot_dir: Path) -> Path:
    """Where `robot_dir`'s earlier revisions are kept."""
    return robot_dir.parent / REVISIONS_DIR / robot_dir.name


def revise(name: str, current: Path, incoming: Path) -> dict[str, Any]:
    """Make `incoming` (a bundle just written beside `current`) the robot's
    current revision. The same content leaves everything as it was and says
    so; different content moves `current` to the revisions and records, in
    the new bundle, the stamp it replaced and what changed."""
    old, new = stamp(name, current), stamp(name, incoming)
    if old.split(STAMP_SEPARATOR)[1] == new.split(STAMP_SEPARATOR)[1]:
        shutil.rmtree(incoming)
        return {
            "revision": f"unchanged: {name} already holds this content",
            "stamp": old,
        }
    changes = model_changes(current, incoming)
    kept = revisions_root(current) / old.split(STAMP_SEPARATOR)[1]
    kept.parent.mkdir(parents=True, exist_ok=True)
    current.rename(kept)
    incoming.rename(current)
    record = {
        "replaces": old,
        "changes": changes[:CHANGES_SHOWN],
        "changed": len(changes),
    }
    (current / REVISION_FILE).write_text(json.dumps(record, indent=1) + "\n")
    return {"replaces": old, "changes": record["changes"], "changed": len(changes)}


def revisions(robot_dir: Path) -> list[dict[str, Any]]:
    """The robot's earlier revisions, newest first: each one's stamp and
    the record it carried of the one before it."""
    root = revisions_root(robot_dir)
    if not root.is_dir():
        return []
    found = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        record = _read(folder / REVISION_FILE)
        found.append(
            {
                "stamp": stamp(robot_dir.name, folder),
                "kept": folder.stat().st_mtime,
                "replaces": record.get("replaces"),
            }
        )
    return sorted(found, key=lambda r: -r["kept"])


def superseded(robots: Path) -> dict[str, str]:
    """Every earlier revision's stamp in a robots folder, mapped to the
    robot's current stamp."""
    out: dict[str, str] = {}
    root = robots / REVISIONS_DIR
    if not root.is_dir():
        return out
    for named in root.iterdir():
        current = robots / named.name
        if not current.is_dir():
            continue
        now = stamp(named.name, current)
        for folder in named.iterdir():
            if folder.is_dir():
                out[stamp(named.name, folder)] = now
    return out


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def model_changes(old_dir: Path, new_dir: Path) -> list[str]:
    """What changed between two revisions of a robot, in model terms: the
    joints (ranges, hinge ranges in degrees too, damping, armature,
    friction loss), the bodies' masses, the actuators (gain, control and
    force ranges), the keyframes, and what was added or removed."""
    import mujoco  # noqa: PLC0415 - sim extra

    files = [model_file_for_use(old_dir), model_file_for_use(new_dir)]
    if None in files:
        return ["a revision with no MJCF to compare"]
    a, b = (mujoco.MjModel.from_xml_path(str(f)) for f in files)
    out: list[str] = []
    out += _compare(a, b, mujoco.mjtObj.mjOBJ_JOINT, a.njnt, b.njnt, "joint", _joint)
    out += _compare(a, b, mujoco.mjtObj.mjOBJ_BODY, a.nbody, b.nbody, "body", _body)
    out += _compare(
        a, b, mujoco.mjtObj.mjOBJ_ACTUATOR, a.nu, b.nu, "actuator", _actuator
    )
    out += _compare(a, b, mujoco.mjtObj.mjOBJ_KEY, a.nkey, b.nkey, "keyframe", _key)
    return out


def _names(m: Any, kind: Any, count: int) -> dict[str, int]:
    import mujoco  # noqa: PLC0415

    return {mujoco.mj_id2name(m, kind, i) or f"#{i}": i for i in range(count)}


def _compare(  # noqa: PLR0913, PLR0917 - two models, the kind, its counts, its facts
    a: Any,
    b: Any,
    kind: Any,
    na: int,
    nb: int,
    word: str,
    facts: Any,
) -> list[str]:
    old, new = _names(a, kind, na), _names(b, kind, nb)
    out = [f"{word} {n} added" for n in new if n not in old]
    out += [f"{word} {n} removed" for n in old if n not in new]
    for n in old:
        if n in new:
            fa, fb = facts(a, old[n]), facts(b, new[n])
            for key, va in fa.items():
                vb = fb[key]
                if np.shape(va) != np.shape(vb) or not np.allclose(va, vb, atol=SAME):
                    out.append(
                        f"{word} {n} {key}: {_shown(va, key)} → {_shown(vb, key)}"
                    )
    return out


def _shown(value: Any, key: str) -> str:
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    text = (
        "[" + ", ".join(f"{v:.4g}" for v in arr) + "]"
        if arr.size > 1
        else f"{arr[0]:.4g}"
    )
    if key == "range (rad)":
        text += " (" + ", ".join(f"{np.degrees(v):.1f}°" for v in arr) + ")"
    return text


def _joint(m: Any, j: int) -> dict[str, Any]:
    dof = m.jnt_dofadr[j]
    rng = "range (rad)" if m.jnt_type[j] == HINGE else "range"
    return {
        rng: m.jnt_range[j] if m.jnt_limited[j] else np.array([0.0, 0.0]),
        "damping": m.dof_damping[dof],
        "armature": m.dof_armature[dof],
        "friction loss": m.dof_frictionloss[dof],
    }


def _body(m: Any, b: int) -> dict[str, Any]:
    return {"mass (kg)": m.body_mass[b]}


def _actuator(m: Any, a: int) -> dict[str, Any]:
    return {
        "gain": m.actuator_gainprm[a][0],
        "control range": m.actuator_ctrlrange[a],
        "force range": m.actuator_forcerange[a],
    }


def _key(m: Any, k: int) -> dict[str, Any]:
    return {"qpos": m.key_qpos[k]}
