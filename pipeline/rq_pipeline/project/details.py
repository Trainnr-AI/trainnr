"""Detail views: what each artifact IS, in the field's terms, as data.

The index carries a summary; a platform's detail page shows the thing.
This module reads each artifact and writes `<project>/.index/details/
<stamp>.json`: a list of titled sections, each a table (columns and rows)
or a key/value list, that the Studio renders with no domain knowledge of
its own. One writer per kind; the Studio has one renderer.

Vocabulary (decision 2026-09-09): the field's words and nothing new —
Isaac Sim and Isaac Lab for assets and environments, MuJoCo for the
model, RL and sysid for training and identification, LeRobot for
datasets, MLOps for versions. Our own words appear nowhere a user reads:
a stamp is a **version**, a batch is a **dataset**, the referee is the
**success criterion**, an expert is the **scripted policy**, a certificate
is an **evaluation**, a fit is **system identification**.

Like previews, details are a cache keyed by version: an unchanged artifact
is not re-read, and deleting `.index/` loses nothing.
"""

from __future__ import annotations

import contextlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from rq_pipeline.project.index import Artifact, ProjectIndex
from rq_pipeline.project.kinds import ACCEPTANCE_FILE, TASK_FILE
from rq_pipeline.project.locate import INDEX_DIR, Project

DETAILS_DIR = "details"
# /4 (2026-09-10): a walk's gate and episode; /3 (2026-09-09): fit records
# read as written.
SCHEMA = "trainnr-detail/5"
MAX_ROWS = 400  # a table longer than this is truncated, and says so
MAX_MARKDOWN = 6000  # a datasheet is a page, not a book
SMALL = 1e-3  # below this, print in scientific notation

JOINT_TYPES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
ACTUATOR_TRN = {
    0: "joint",
    1: "jointinparent",
    2: "slidercrank",
    3: "tendon",
    4: "site",
    5: "body",
}
INTEGRATORS = {0: "Euler", 1: "RK4", 2: "implicit", 3: "implicitfast"}
INTERVAL_ENDS = 2  # a confidence interval is two numbers
# The protocol fields with a field-word label; the rest are shown as recorded.
PROTOCOL_KEYS = frozenset(
    {"trials", "seed", "criterion", "err_floor_mps", "dr_basis", "judged_at"}
)


def details_path(project: Project, stamp: str) -> Path:
    return project.root / INDEX_DIR / DETAILS_DIR / f"{stamp}.json"


def _stale(out: Path, artifact: Artifact) -> bool:
    """The detail predates the artifact's newest file. The index keeps
    `updated` to the second, so a detail written within that second of
    the change counts as stale too — one extra write, never a stale view."""
    if not artifact.updated:
        return False
    try:
        updated = datetime.fromisoformat(artifact.updated).timestamp()
    except ValueError:
        return False
    return out.stat().st_mtime < updated + UPDATED_RESOLUTION_S


UPDATED_RESOLUTION_S = 1.0


def _schema_of(path: Path) -> str:
    try:
        return str(json.loads(path.read_text()).get("schema", ""))
    except (OSError, ValueError):
        return ""


def write_details(project: Project, index: ProjectIndex) -> dict[str, str]:
    """Write a detail file for every artifact whose kind has a writer and
    that has none yet — or one written by an older schema, or one older
    than the artifact's last change (a review written beside a task, a
    record appended to a run) — since a detail is a view and the writer
    is its only source; return `{stamp: relative path}` for those present."""
    written: dict[str, str] = {}
    for artifact in index.artifacts:
        out = details_path(project, artifact.stamp)
        if not out.is_file() or _schema_of(out) != SCHEMA or _stale(out, artifact):
            writer = _WRITERS.get(artifact.kind)
            if writer is None:
                continue
            try:
                sections = writer(project.root / artifact.path, artifact)
            except Exception as why:  # a detail is a view; the index is not
                sections = [
                    _kv(
                        "Detail unavailable",
                        [("reason", str(why)), ("path", artifact.path)],
                    )
                ]
            out.parent.mkdir(parents=True, exist_ok=True)
            staging = out.with_suffix(".json.tmp")
            staging.write_text(
                json.dumps(
                    {"schema": SCHEMA, "version": artifact.stamp, "sections": sections},
                    indent=1,
                    default=_jsonable,
                )
            )
            staging.replace(out)
        written[artifact.stamp] = str(out.relative_to(project.root))
    return written


# -- section builders ------------------------------------------------------------


# Words that older artifacts carry inside their data (an episode's verdict
# string, a datasheet written before 2026-09-09) and that the field says
# differently. The data is not rewritten; what a reader sees is (docs/76 §2).
FIELD_WORDS: tuple[tuple[str, str], ...] = (
    ("task referee", "success criterion"),
    ("referee", "success criterion"),
    ("expert retries", "scripted-policy retries"),
    ("expert", "scripted policy"),
    ("episodes kept", "successful episodes"),
    ("keep rate", "success rate"),
    ("last keep", "last success"),
    ("## Stamps", "## Versions"),
    ("instrument", "simulator build"),
    ("Dynamics draws", "Domain randomization draws"),
    ("Basis:", "Range:"),
)


def field_words(text: str) -> str:
    """The field's word for each house word inside a displayed string."""
    for ours, theirs in FIELD_WORDS:
        text = text.replace(ours, theirs)
    return text


def _markdown(title: str, text: str, note: str | None = None) -> dict[str, Any]:
    """A section rendered as markdown (a datasheet, a README)."""
    return {
        "title": title,
        "kind": "markdown",
        "columns": [],
        "rows": [],
        "text": text,
        "note": note,
    }


def _kv(
    title: str, rows: list[tuple[str, Any]], note: str | None = None
) -> dict[str, Any]:
    return {
        "title": title,
        "kind": "kv",
        "rows": [[k, v] for k, v in rows],
        "note": note,
    }


def _table(
    title: str, columns: list[str], rows: list[list[Any]], note: str | None = None
) -> dict[str, Any]:
    truncated = len(rows) > MAX_ROWS
    return {
        "title": title,
        "kind": "table",
        "columns": columns,
        "rows": rows[:MAX_ROWS],
        "note": (f"{len(rows)} rows; showing {MAX_ROWS}" if truncated else note),
    }


def _f(x: Any, digits: int = 4) -> Any:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return x
    if v == 0:
        return 0
    if v.is_integer() and isinstance(x, (int, float)) and not isinstance(x, bool):
        return int(v)  # 8000, not 8000.0
    return round(v, digits) if abs(v) >= SMALL else float(f"{v:.3g}")


def _rng(lo: Any, hi: Any) -> str:
    return f"[{_f(lo)}, {_f(hi)}]"


# -- robot (Asset) ---------------------------------------------------------------


def _robot(root: Path, artifact: Artifact) -> list[dict[str, Any]]:
    import mujoco  # noqa: PLC0415

    from rq_pipeline.project.previews import _robot_model_file  # noqa: PLC0415

    model_file = _robot_model_file(root)
    if model_file is None:
        raise ValueError("no MJCF at the asset's root")
    m = mujoco.MjModel.from_xml_path(str(model_file))
    name = lambda kind, i: mujoco.mj_id2name(m, kind, i) or f"#{i}"  # noqa: E731
    obj_joint = mujoco.mjtObj.mjOBJ_JOINT
    obj_actuator = mujoco.mjtObj.mjOBJ_ACTUATOR
    obj_body = mujoco.mjtObj.mjOBJ_BODY
    obj_sensor = mujoco.mjtObj.mjOBJ_SENSOR
    overview = _kv(
        "Asset",
        [
            ("version", artifact.stamp),
            ("model file", model_file.name),
            ("bodies", m.nbody - 1),
            ("joints", m.njnt),
            ("degrees of freedom", m.nv),
            ("actuators", m.nu),
            ("sensors", m.nsensor),
            ("geoms", m.ngeom),
            ("meshes", m.nmesh),
            ("keyframes", m.nkey),
            ("total mass (kg)", _f(float(np.sum(m.body_mass[1:])), 4)),
            ("timestep (s)", _f(m.opt.timestep, 5)),
            (
                "integrator",
                INTEGRATORS.get(int(m.opt.integrator), str(m.opt.integrator)),
            ),
            ("gravity (m/s²)", [_f(g, 3) for g in m.opt.gravity]),
        ],
    )
    joints = _table(
        "Joints",
        ["joint", "type", "range", "damping", "armature", "friction loss", "stiffness"],
        [
            [
                name(obj_joint, j),
                JOINT_TYPES.get(int(m.jnt_type[j]), str(m.jnt_type[j])),
                _rng(*m.jnt_range[j]) if m.jnt_limited[j] else "unlimited",
                _f(m.dof_damping[m.jnt_dofadr[j]]),
                _f(m.dof_armature[m.jnt_dofadr[j]]),
                _f(m.dof_frictionloss[m.jnt_dofadr[j]]),
                _f(m.jnt_stiffness[j]),
            ]
            for j in range(m.njnt)
        ],
        note="MuJoCo joint parameters; damping, armature and friction loss are what "
        "system identification measures.",
    )
    actuators = _table(
        "Actuators",
        [
            "actuator",
            "transmission",
            "target",
            "gear",
            "control range",
            "kp",
            "force range",
        ],
        [
            [
                name(obj_actuator, a),
                ACTUATOR_TRN.get(
                    int(m.actuator_trntype[a]), str(m.actuator_trntype[a])
                ),
                name(obj_joint, int(m.actuator_trnid[a][0]))
                if int(m.actuator_trntype[a]) in (0, 1)
                else str(m.actuator_trnid[a][0]),
                _f(m.actuator_gear[a][0]),
                _rng(*m.actuator_ctrlrange[a])
                if m.actuator_ctrllimited[a]
                else "unlimited",
                _f(m.actuator_gainprm[a][0]),
                _rng(*m.actuator_forcerange[a])
                if m.actuator_forcelimited[a]
                else "unlimited",
            ]
            for a in range(m.nu)
        ],
    )
    bodies = _table(
        "Bodies",
        ["body", "parent", "mass (kg)", "inertia diag (kg·m²)", "geoms"],
        [
            [
                name(obj_body, b),
                name(obj_body, int(m.body_parentid[b])),
                _f(m.body_mass[b]),
                [_f(v, 6) for v in m.body_inertia[b]],
                int(m.body_geomnum[b]),
            ]
            for b in range(1, m.nbody)
        ],
    )
    sensors = _table(
        "Sensors",
        ["sensor", "type", "dimension", "attached to"],
        [
            [
                name(obj_sensor, s),
                mujoco.mjtSensor(int(m.sensor_type[s]))
                .name.removeprefix("mjSENS_")
                .lower(),
                int(m.sensor_dim[s]),
                mujoco.mj_id2name(m, int(m.sensor_objtype[s]), int(m.sensor_objid[s]))
                or "",
            ]
            for s in range(m.nsensor)
        ],
    )
    collision = _kv(
        "Collision",
        [
            ("geoms with collision", int(np.sum(m.geom_contype != 0))),
            ("visual-only geoms", int(np.sum(m.geom_contype == 0))),
            ("mesh geoms", int(np.sum(m.geom_type == mujoco.mjtGeom.mjGEOM_MESH))),
        ],
    )
    files = sorted(p.name for p in root.iterdir() if not p.name.startswith("."))
    sections = [
        overview,
        joints,
        actuators,
        bodies,
        sensors,
        collision,
        _kv("Files", [(f, "") for f in files]),
    ]
    fits = root / "fits"
    if fits.is_dir():
        sections.insert(1, _fit_table(fits))
    return sections


def _fit_table(fits: Path) -> dict[str, Any]:
    """Every fit record as the record writer wrote it (`robot/fit_record`):
    a row per parameter with its estimate, confidence interval, identified
    or not, and unit; then the cross-run spread verdict per parameter when
    two or more records exist. (Until 2026-09-09 this guessed a dict shape
    the records never had and showed only the SPREAD line.)"""
    from rq_pipeline.robot.fit_record import (  # noqa: PLC0415
        load_fit_records,
        spread_verdicts,
    )

    rows: list[list[Any]] = []
    try:
        records = load_fit_records(fits.parent)
    except Exception as why:  # a malformed record is a row, not a crash
        return _table(
            "System identification",
            [
                "record",
                "parameter",
                "estimate",
                "confidence interval",
                "status",
                "unit",
            ],
            [["unreadable", str(why), "", "", "", ""]],
        )
    for record in records:
        units = record.units or {}
        for parameter in record.parameters:
            interval = (
                "unbounded"
                if parameter.half_width == float("inf")
                else _rng(parameter.lower, parameter.upper)
            )
            rows.append(
                [
                    record.recording,
                    parameter.name,
                    _f(parameter.estimate, 6),
                    interval,
                    "identified" if parameter.pinned else "unidentified",
                    units.get(parameter.name, ""),
                ]
            )
    if len(records) >= 2:  # noqa: PLR2004 - a spread needs two fits to disagree
        for name, verdict in spread_verdicts(records).items():
            rows.append(
                [
                    "SPREAD",
                    name,
                    _rng(verdict.lowest, verdict.highest),
                    ""
                    if verdict.mean_half_width is None
                    else f"mean half-width {_f(verdict.mean_half_width, 4)}",
                    "trust the spread" if verdict.exceeds else "runs agree",
                    "",
                ]
            )
    return _table(
        "System identification",
        ["record", "parameter", "estimate", "confidence interval", "status", "unit"],
        rows,
        note="From telemetry recorded on the real robot. 'identified': the confidence "
        "half-width is within 10 % of the parameter's allowed range. SPREAD rows: "
        "the estimate's span across records against the mean interval; 'trust the "
        "spread' means the runs disagree by more than their intervals claim.",
    )


# -- task (Environment) ------------------------------------------------------------


def _acceptance(a: dict[str, Any]) -> dict[str, Any]:
    """The critic's verdict as facts: accepted or rejected with the
    reasons, the counts, and the expert's funnel."""
    funnel = a.get("funnel") or {}

    def count(key: str) -> str:
        k, n = a.get(key), a.get("trials")
        return f"{k} / {n}" if k is not None and n is not None else "unrecorded"

    rows: list[tuple[str, Any]] = [
        ("verdict", "accepted" if a.get("accepted") else "rejected")
    ]
    gate = a.get("gate")
    if gate:
        rows.append(("gate", gate))
    else:
        rows += [
            ("scripted policy successes", count("expert_successes")),
            ("floor policy successes", count("floor_successes")),
        ]
    rows += [
        ("judged", a.get("judged", "unrecorded")),
        ("simulator build", a.get("instrument", "unrecorded")),
    ]
    for key, value in (a.get("identity") or {}).items():
        rows.append((f"identity · {key}", value))
    milestones = a.get("milestones") or []
    trials = a.get("trials")
    for name, counts in funnel.items():
        stages = [
            f"{milestones[i]} {c}/{trials}" if i < len(milestones) else f"{c}/{trials}"
            for i, c in enumerate(counts)
        ]
        rows.append((f"funnel · {name}", " · ".join(stages)))
    for i, reason in enumerate(a.get("reasons") or [], start=1):
        rows.append((f"reason {i}", reason))
    for i, refusal in enumerate(a.get("refusals") or [], start=1):
        rows.append((f"refusal {i}", refusal))
    return _kv(
        "Acceptance",
        rows,
        note=(
            "A walk is accepted when its environment builds from the project's "
            "robot and PPO iterations run: learnability, not a scripted expert."
            if gate
            else "The scripted policy must succeed on every paired trial and the "
            "floor policy (holding home) on none."
        ),
    )


def _task(root: Path, artifact: Artifact) -> list[dict[str, Any]]:
    ref = json.loads((root / TASK_FILE).read_text())
    task_id = ref.get("task_id", "")
    sections = [
        _kv(
            "Environment",
            [
                ("task", task_id),
                ("version", ref.get("stamp", "unversioned")),
                ("declared as", ref.get("kind", "")),
            ],
        )
    ]
    acceptance = (
        json.loads((root / ACCEPTANCE_FILE).read_text())
        if (root / ACCEPTANCE_FILE).is_file()
        else None
    )
    if acceptance is not None:
        sections.append(_acceptance(acceptance))
    else:
        sections.append(
            _kv(
                "Acceptance",
                [("verdict", "unreviewed")],
                note="run accept_task to review it",
            )
        )
    try:
        from dataclasses import asdict  # noqa: PLC0415

        from rq_pipeline.tasks.overlay import build_from_reference  # noqa: PLC0415

        task = build_from_reference(ref)
    except Exception as why:
        sections.append(_kv("Scene", [("not built", str(why))]))
        return sections
    proto = getattr(task, "protocol", None)
    control_hz = getattr(task, "control_hz", None)
    steps = getattr(proto, "steps", None)
    episode_s = (steps / control_hz) if (steps and control_hz) else None
    walk_spec = getattr(task, "task_spec", None) if proto is None else None
    if walk_spec is not None:  # a walk: the episode is the spec's
        episode_s = getattr(walk_spec, "episode_s", None)
        steps = int(episode_s * control_hz) if (episode_s and control_hz) else None
    sections.append(
        _kv(
            "Episode",
            [
                ("robot asset", Path(str(getattr(task, "bundle_dir", ""))).name),
                ("control rate (Hz)", control_hz),
                ("episode length (steps)", steps),
                ("episode length (s)", _f(episode_s, 2) if episode_s else ""),
                (
                    "paired trials per evaluation",
                    getattr(proto, "trials", None)
                    if proto is not None
                    else getattr(walk_spec, "trials", None),
                ),
                ("instruction", getattr(task, "instruction", "")),
                *(
                    []
                    if walk_spec is not None
                    else [
                        ("cameras", list(getattr(task, "cameras", []) or [])),
                        ("state dimension", getattr(task, "state_width", None)),
                    ]
                ),
            ],
        )
    )
    spec = getattr(task, "task_spec", None)
    if spec is not None:
        fields = asdict(spec)
        spawn = fields.pop("part_spawn", None)
        sections.append(
            _kv("Task specification", [(k, _jsonable(v)) for k, v in fields.items()])
        )
        if spawn:
            sections.append(
                _table(
                    "Spawn ranges (object initial positions)",
                    ["object", "x range (m)", "y range (m)"],
                    [[k, _rng(*v[0]), _rng(*v[1])] for k, v in spawn.items()],
                    note="Domain randomization of the initial state; paired trials "
                    "draw the same starts for every policy.",
                )
            )
    if proto is not None:
        milestones = list(getattr(proto, "milestones", []) or [])
        sections.append(
            _kv(
                "Success criterion and terminations",
                [
                    ("success", _describe(getattr(proto, "success", None))),
                    ("milestones (funnel stages)", milestones),
                    (
                        "placements checked",
                        list(getattr(proto, "placements", []) or []),
                    ),
                    ("observables", list(getattr(proto, "observables", []) or [])),
                    (
                        "executed horizon (actions per policy call)",
                        getattr(proto, "executed_horizon", None),
                    ),
                ],
            )
        )
    return sections


def _describe(obj: Any) -> str:
    if obj is None:
        return ""
    doc = (getattr(obj, "__doc__", None) or "").strip().split("\n")[0]
    return doc or getattr(obj, "__name__", None) or type(obj).__name__


# -- recording ---------------------------------------------------------------------


def _recording(root: Path, artifact: Artifact) -> list[dict[str, Any]]:
    from rq_pipeline.robots.recording import Recording  # noqa: PLC0415

    rec = Recording.read(root)
    rows = []
    for name, ch in rec.channels.items():
        v = ch.values.reshape(len(ch.times), -1)
        finite = v[np.isfinite(v)]
        rows.append(
            [
                name,
                ch.unit,
                len(ch.times),
                ch.width,
                _f(ch.rate_hz, 2) if ch.rate_hz else "",
                _f(finite.min()) if finite.size else "",
                _f(finite.max()) if finite.size else "",
                _f(finite.mean()) if finite.size else "",
                _f(finite.std()) if finite.size else "",
                ", ".join(ch.components) if ch.components else "",
            ]
        )
    return [
        _kv(
            "Recording",
            [
                ("version", artifact.stamp),
                ("source", rec.source),
                ("read by", rec.adapter),
                ("collected via", rec.collection),
                ("duration (s)", _f(rec.duration_s, 2)),
                ("channels", len(rec.channels)),
            ],
        ),
        _table(
            "Channels",
            [
                "channel",
                "unit",
                "samples",
                "dims",
                "rate (Hz)",
                "min",
                "max",
                "mean",
                "std",
                "components",
            ],
            rows,
        ),
        _kv(
            "What the robot reported",
            [(k, _jsonable(v)) for k, v in rec.census.items()],
        ),
        _kv("Notes", [(f"{i + 1}", n) for i, n in enumerate(rec.notes)])
        if rec.notes
        else _kv("Notes", []),
    ]


# -- batch (a generated dataset) and dataset (an exported one) -------------------------


def _batch(root: Path, artifact: Artifact) -> list[dict[str, Any]]:
    from rq_pipeline.collect.datasheet import summarize  # noqa: PLC0415

    s = summarize(root)
    d = _dataclass_dict(s)
    episodes = sorted(root.glob("episode_*"))
    rows = []
    for ep in episodes[:MAX_ROWS]:
        man = ep / "manifest.json"
        raw = json.loads(man.read_text()) if man.is_file() else {}
        frames = len(list(ep.glob("frames/*.jpg")) or list(ep.glob("frames/*/*.jpg")))
        dyn = raw.get("dynamics") or {
            k: raw[k] for k in ("damping_scale", "gain_scale") if k in raw
        }
        rows.append(
            [
                ep.name,
                raw.get("seed"),
                raw.get("attempt", 1),
                frames,
                field_words(str(raw.get("verdict", ""))),
                _jsonable(dyn),
                raw.get("dynamics_basis")
                or ("hand-set ±" + str(raw["dr_span"]) if "dr_span" in raw else ""),
            ]
        )
    return [
        _kv(
            "Dataset (generated in simulation)",
            [
                ("version", artifact.stamp),
                ("episodes (successful only)", d.get("episodes")),
                ("success rate (upper bound)", _f(d.get("keep_rate_bound"), 3)),
                ("scripted policy", d.get("expert") or d.get("expert_stamp", "")),
                ("environment", d.get("task") or d.get("task_stamp", "")),
                (
                    "simulator build",
                    d.get("instrument") or d.get("instrument_stamp", ""),
                ),
                ("domain randomization", d.get("basis") or d.get("dynamics_basis", "")),
            ],
        ),
        _table(
            "Episodes",
            [
                "episode",
                "seed",
                "attempt",
                "frames",
                "outcome",
                "dynamics draw",
                "randomization range",
            ],
            rows,
            note="'hand-set' ranges were chosen by the author; 'identified' ranges "
            "come from system identification.",
        ),
        _markdown(
            "Datasheet",
            field_words((root / "datasheet.md").read_text()[:MAX_MARKDOWN])
            if (root / "datasheet.md").is_file()
            else "",
            note="The datasheet as written at generation time; house words in an "
            "older file are shown in the field's terms.",
        ),
    ]


def _dataset(root: Path, artifact: Artifact) -> list[dict[str, Any]]:
    info = (
        json.loads((root / "meta" / "info.json").read_text())
        if (root / "meta" / "info.json").is_file()
        else {}
    )
    prov = (
        json.loads((root / "provenance.json").read_text())
        if (root / "provenance.json").is_file()
        else {}
    )
    features = info.get("features", {})
    feat_rows = [
        [
            k,
            v.get("dtype"),
            _jsonable(v.get("shape")),
            ", ".join(v.get("names") or []) if isinstance(v.get("names"), list) else "",
        ]
        for k, v in features.items()
    ]
    return [
        _kv(
            "Dataset (LeRobot)",
            [
                ("version", artifact.stamp),
                ("format", info.get("codebase_version", "")),
                ("episodes", info.get("total_episodes")),
                ("frames", info.get("total_frames")),
                ("fps", info.get("fps")),
                ("robot type", info.get("robot_type", "")),
                ("robot asset", prov.get("bundle", "unrecorded")),
                ("scripted policy", prov.get("expert", "")),
                (
                    "source dataset",
                    prov.get("source_stamp") or prov.get("source", "unrecorded"),
                ),
            ],
        ),
        _table(
            "Features (observation and action spaces)",
            ["feature", "dtype", "shape", "names"],
            feat_rows,
        ),
        _kv(
            "Provenance",
            [(k, _jsonable(v)) for k, v in prov.items() if k != "manifests"],
            note=f"{len(prov.get('manifests', []))} per-episode manifests carried"
            if prov.get("manifests")
            else None,
        ),
    ]


# -- run (Experiment) and evaluation ----------------------------------------------


def _run(root: Path, artifact: Artifact) -> list[dict[str, Any]]:
    run = (
        json.loads((root / "run.json").read_text())
        if (root / "run.json").is_file()
        else None
    )
    if run is not None:
        from rq_pipeline.envs.lerobot_train_log import parse_train_line  # noqa: PLC0415

        log = root / "chain.log"
        last = None
        n = 0
        if log.is_file():
            for line in log.read_text(errors="replace").splitlines():
                p = parse_train_line(line)
                if p is not None:
                    last, n = p, n + 1
        ckpts = (
            sorted(
                (root.parent / f"{run.get('name', '')}-act" / "checkpoints").glob("*")
            )
            if run.get("name")
            else []
        )
        return [
            _kv(
                "Experiment (imitation learning)",
                [
                    ("version", artifact.stamp),
                    ("policy", run.get("policy")),
                    ("training steps", run.get("steps")),
                    ("batch size", run.get("batch_size")),
                    ("learning rate", run.get("learning_rate")),
                    ("device", run.get("device")),
                    ("dataset", run.get("dataset_repo_id")),
                    (
                        "robot asset",
                        (run.get("provenance") or {}).get("bundle", "unrecorded"),
                    ),
                    (
                        "scripted policy",
                        (run.get("provenance") or {}).get("expert", "unrecorded"),
                    ),
                    ("started", run.get("started")),
                    ("checkpoint every", run.get("checkpoint_every")),
                    ("checkpoints on disk", len(ckpts)),
                    ("metric lines logged", n),
                    ("last step", getattr(last, "step", None)),
                    (
                        "last metrics",
                        _jsonable(dict(getattr(last, "metrics", {}) or {})),
                    ),
                ],
            ),
            _kv("Command", [("argv", run.get("command", ""))]),
        ]
    ident = (
        json.loads((root / "identity.json").read_text())
        if (root / "identity.json").is_file()
        else {}
    )
    training = (
        json.loads((root / "training.json").read_text())
        if (root / "training.json").is_file()
        else {}
    )
    final = training.get("final") or {}
    wall = training.get("wall_seconds")
    sections = [
        _kv(
            "Experiment (reinforcement learning)",
            [
                ("version", artifact.stamp),
                ("trainer", training.get("trainer", "unrecorded")),
                ("iterations", training.get("iterations_logged") or "unrecorded"),
                ("parallel environments", training.get("envs", "unrecorded")),
                ("device", training.get("device", "unrecorded")),
                ("wall time", f"{wall / 3600:.1f} h" if wall else "unrecorded"),
                ("final mean reward", final.get("reward", "unrecorded")),
                ("best mean reward", training.get("best_reward", "unrecorded")),
                ("final episode length", final.get("episode_length", "unrecorded")),
                ("steps per second", final.get("steps_per_second", "unrecorded")),
                ("robot asset", ident.get("robot", "unrecorded")),
                ("actuator model", ident.get("actuator", "unrecorded")),
                ("domain randomization", ident.get("dr_basis", "")),
                ("seed", ident.get("seed")),
            ],
        ),
    ]
    columns = training.get("columns") or []
    if columns:
        sections.append(
            _table(
                "Training curve",
                [c.replace("_", " ") for c in columns],
                [[_f(v) for v in row] for row in training.get("curve", [])],
                note="sampled from the console log; the full log is train.log",
            )
        )
    return sections


def _certificate(root: Path, artifact: Artifact) -> list[dict[str, Any]]:
    c = json.loads((root / "certificate.json").read_text())
    k, n = c.get("successes"), c.get("trials")
    ci = c.get("ci95") or c.get("ci") or []
    rows = []
    records = sorted(root.glob("records*.jsonl"))
    for rec in records:
        for line in rec.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(
                [
                    r.get("trial"),
                    r.get("seed"),
                    r.get("policy"),
                    "success" if r.get("success") else "failure",
                    r.get("steps"),
                    ", ".join(
                        e.get("name", "")
                        for e in r.get("events", [])
                        if isinstance(e, dict)
                    ),
                ]
            )
    funnel = c.get("funnel") or {}
    protocol = c.get("protocol") if isinstance(c.get("protocol"), dict) else {}
    return [
        _kv(
            "Evaluation",
            [
                ("version", artifact.stamp),
                ("success rate", f"{k} / {n}" if k is not None and n else ""),
                (
                    "95% confidence interval (exact)",
                    _rng(*ci) if len(ci) == INTERVAL_ENDS else "",
                ),
                ("policy", c.get("policy", "")),
                ("robot asset", c.get("robot", c.get("identity", {}).get("robot", ""))),
                ("environment", c.get("task", c.get("source", ""))),
                ("simulator build", c.get("instrument", "")),
                *[
                    (f"{stage} (of {n})" if n else f"{stage} (trials)", count)
                    for stage, count in funnel.items()
                ],
            ],
        ),
        _kv(
            "Protocol",
            [
                ("trials", protocol.get("trials")),
                ("seed", protocol.get("seed")),
                ("success criterion", protocol.get("criterion")),
                ("error floor (m/s)", protocol.get("err_floor_mps")),
                ("domain randomization", protocol.get("dr_basis")),
                ("judged at", protocol.get("judged_at")),
                *[
                    (k2, _jsonable(v))
                    for k2, v in protocol.items()
                    if k2 not in PROTOCOL_KEYS
                ],
            ],
        ),
        _table(
            "Trials", ["trial", "seed", "policy", "outcome", "steps", "events"], rows
        ),
    ]


# -- deployment (A6) -------------------------------------------------------------------


def _deploy(root: Path, artifact: Artifact) -> list[dict[str, Any]]:
    """A deployment: what the manifest says a runtime needs, the joints
    and observations as tables, and the sim-to-sim gate's verdict."""
    from rq_pipeline.deploy.manifest import (  # noqa: PLC0415
        GATE_INSTRUMENTS,
        MANIFEST_FILE,
        gate_word,
        read_gates,
    )

    m = json.loads((root / MANIFEST_FILE).read_text())
    control = m.get("control") or {}
    joints = m.get("joints") or {}
    onnx = m.get("onnx") or {}
    sections = [
        _kv(
            "Deployment",
            [
                ("version", artifact.stamp),
                ("policy", m.get("policy", "unrecorded")),
                ("checkpoint", m.get("checkpoint", "unrecorded")),
                ("experiment", m.get("run", "unrecorded")),
                ("robot asset", m.get("robot", "unrecorded")),
                ("actuator model", m.get("actuator", "unrecorded")),
                ("environment", m.get("task") or "unrecorded"),
                ("domain randomization", m.get("dr_basis") or "unrecorded"),
                ("evaluation cited", m.get("certificate") or "none"),
                ("control rate (Hz)", control.get("control_hz")),
                ("physics timestep (s)", control.get("physics_timestep_s")),
                ("decimation", control.get("decimation")),
                ("episode length (s)", control.get("episode_length_s")),
                (
                    "policy file",
                    f"{onnx.get('file')} ({onnx.get('input_width')} in, "
                    f"{onnx.get('output_width')} out)",
                ),
                ("normalization", onnx.get("normalization", "unrecorded")),
                (
                    "export check (max |onnx - torch|)",
                    onnx.get("export_check_max_abs_diff"),
                ),
                ("action", (m.get("action") or {}).get("kind", "unrecorded")),
                (
                    "scene",
                    f"{(m.get('scene') or {}).get('file')} · terrain "
                    f"{(m.get('scene') or {}).get('terrain')}",
                ),
                (
                    "fell-over limit (deg)",
                    (m.get("termination") or {}).get("fell_over_deg"),
                ),
            ],
        )
    ]
    gates = read_gates(root)
    for runtime, g in gates.items():
        instrument = GATE_INSTRUMENTS[runtime]
        verdict = g.get("verdict") or {}
        cert = g.get("certificate") or {}
        rows: list[tuple[str, Any]] = [
            ("verdict", gate_word(g)),
            ("successes", f"{g.get('successes')} / {g.get('trials')}"),
            (
                "95% confidence interval (exact)",
                _rng(*g["ci95"]) if g.get("ci95") else "",
            ),
            ("rule", verdict.get("rule", "")),
            ("tolerance", verdict.get("tolerance", "n/a")),
            (
                "certificate",
                f"{cert.get('successes')} / {cert.get('trials')}"
                if cert
                else "none cited",
            ),
            ("runtime", (g.get("protocol") or {}).get("runtime", "")),
            (
                "simulator build",
                (g.get("protocol") or {}).get("instrument", "unrecorded"),
            ),
            ("judged", g.get("judged", "unrecorded")),
        ]
        sections.append(
            _kv(
                f"Sim-to-sim gate: {instrument}",
                rows,
                note="The exported policy driven through its manifest alone, judged "
                "the certificate's way: survived and tracked the held command.",
            )
        )
        sections.append(
            _table(
                f"Gate trials: {instrument}",
                ["command (vx, vy, wz)", "steps", "fell", "error ratio", "success"],
                [
                    [
                        ", ".join(f"{c:.2f}" for c in r.get("command", [])),
                        r.get("steps"),
                        "yes" if r.get("fell") else "no",
                        _f(r.get("err_ratio")),
                        "success" if r.get("success") else "failure",
                    ]
                    for r in g.get("records", [])
                ],
            )
        )
    if not gates:
        sections.append(
            _kv(
                "Sim-to-sim gate",
                [("verdict", "not run")],
                note="run gate_deployment to judge it",
            )
        )
    names = joints.get("policy_order") or []
    sdk = joints.get("sdk_order_map") or [None] * len(names)
    sections.append(
        _table(
            "Joints (policy order)",
            [
                "joint",
                "kp",
                "kd",
                "effort limit",
                "default (rad)",
                "action scale",
                "ctrl index",
                "SDK index",
            ],
            [
                [
                    n,
                    _f(joints.get("stiffness", [None] * len(names))[i]),
                    _f(joints.get("damping", [None] * len(names))[i]),
                    _f(joints.get("effort_limit", [None] * len(names))[i]),
                    _f(joints.get("default_pos", [None] * len(names))[i]),
                    _f((m.get("action") or {}).get("scale", [None] * len(names))[i]),
                    (joints.get("action_to_ctrl") or [None] * len(names))[i],
                    sdk[i] if i < len(sdk) else None,
                ]
                for i, n in enumerate(names)
            ],
            note=joints.get("sdk_order_source") or None,
        )
    )
    sections.append(
        _table(
            "Observations (in order)",
            ["term", "width", "source", "scale", "clip", "history"],
            [
                [
                    t.get("name"),
                    t.get("width"),
                    t.get("source"),
                    _jsonable(t.get("scale")),
                    t.get("clip"),
                    t.get("history_length"),
                ]
                for t in m.get("observations", [])
            ],
            note="The concatenation order is the policy's input order.",
        )
    )
    return sections


# -- helpers ------------------------------------------------------------------------


def outcome_of(raw: Any) -> Any:
    """A finding's outcome: a dict, or the dict a record wrote as a JSON
    string; anything else as it came (the writers stay honest about it)."""
    if isinstance(raw, str):
        with contextlib.suppress(ValueError):
            return json.loads(raw)
    return raw


def _dataclass_dict(obj: Any) -> dict[str, Any]:
    from dataclasses import asdict, is_dataclass  # noqa: PLC0415

    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return dict(obj) if isinstance(obj, dict) else {}


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _policy(root: Path, artifact: Artifact) -> list[dict[str, Any]]:
    manifest = (
        json.loads((root / "policy.json").read_text())
        if (root / "policy.json").is_file()
        else {}
    )
    identity = (
        json.loads((root / "identity.json").read_text())
        if (root / "identity.json").is_file()
        else {}
    )
    weights = sorted(
        p.name
        for p in (root.iterdir() if root.is_dir() else [root])
        if p.suffix in (".pt", ".safetensors")
    )
    facts = _kv(
        "Policy",
        [
            ("version", artifact.stamp),
            (
                "checkpoint",
                manifest.get("checkpoint") or (weights[0] if weights else ""),
            ),
            ("format", manifest.get("format", "")),
            ("training iterations", manifest.get("iterations")),
            ("experiment", manifest.get("run", "")),
            ("robot asset", manifest.get("robot") or identity.get("robot", "")),
            (
                "actuator model",
                manifest.get("actuator") or identity.get("actuator", ""),
            ),
            (
                "domain randomization",
                manifest.get("dr_basis") or identity.get("dr_basis", ""),
            ),
            ("seed", manifest.get("seed") if manifest else identity.get("seed")),
        ],
    )
    # Every evaluation in the project that judged this policy.
    rows = []
    for cert in sorted(
        (root.parent.parent / "certificates").glob("*/certificate.json")
    ):
        c = json.loads(cert.read_text())
        if c.get("policy") != artifact.stamp:
            continue
        ci = c.get("ci95") or c.get("ci") or []
        rows.append(
            [
                cert.parent.name,
                f"{c.get('successes')} / {c.get('trials')}",
                _rng(*ci) if len(ci) == INTERVAL_ENDS else "",
                (c.get("protocol") or {}).get("judged_at", ""),
                c.get("instrument", ""),
            ]
        )
    return [
        facts,
        _table(
            "Evaluations of this policy",
            ["evaluation", "success", "95% interval", "judged at", "simulator build"],
            rows,
            note="Success is the environment's own criterion; the interval is exact "
            "(Clopper-Pearson) for the trials run.",
        ),
    ]


def _finding(root: Path, artifact: Artifact) -> list[dict[str, Any]]:
    raw = json.loads(root.read_text())
    outcome = raw.get("outcome")
    outcome = outcome_of(outcome)
    sections: list[dict[str, Any]] = [
        _markdown("Claim", f"**{raw.get('claim', '')}**"),
        _kv(
            "Record",
            [
                ("id", raw.get("id", "")),
                ("date", raw.get("date", "")),
                ("repository commit", raw.get("repo_commit", "")),
                ("simulator build", raw.get("instrument", "")),
                ("protocol", raw.get("protocol", "")),
                (
                    "command",
                    " ".join(raw.get("argv", []))
                    if isinstance(raw.get("argv"), list)
                    else raw.get("argv", ""),
                ),
            ],
        ),
    ]
    arms = (outcome or {}).get("arms") if isinstance(outcome, dict) else None
    if isinstance(arms, dict) and all(isinstance(a, dict) for a in arms.values()):
        keys: list[str] = []
        for arm in arms.values():
            for key in arm:
                if key not in keys and isinstance(arm[key], (int, float, str, list)):
                    keys.append(key)
        keys = [k for k in keys if k not in ("per_run", "instrument")][:8]
        sections.append(
            _table(
                "Outcome by condition",
                ["condition", *keys],
                [
                    [name, *[_jsonable(arm.get(k)) for k in keys]]
                    for name, arm in arms.items()
                ],
            )
        )
    elif isinstance(outcome, dict):
        sections.append(_kv("Outcome", [(k, _jsonable(v)) for k, v in outcome.items()]))
    elif outcome is not None:
        sections.append(_markdown("Outcome", str(outcome)))
    for title, key in (
        ("Inputs", "inputs"),
        ("Artifacts", "artifacts"),
        ("Sources", "sources"),
    ):
        block = raw.get(key)
        if isinstance(block, dict) and block:
            sections.append(_kv(title, [(k, _jsonable(v)) for k, v in block.items()]))
    caveats = raw.get("caveats")
    if isinstance(caveats, list) and caveats:
        sections.append(_markdown("Caveats", "\n".join(f"- {c}" for c in caveats)))
    return sections


_WRITERS = {
    "deploy": _deploy,
    "robot": _robot,
    "task": _task,
    "recording": _recording,
    "batch": _batch,
    "dataset": _dataset,
    "run": _run,
    "certificate": _certificate,
    "policy": _policy,
    "finding": _finding,
}
