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

import json
from pathlib import Path
from typing import Any

import numpy as np

from rq_pipeline.project.index import Artifact, ProjectIndex
from rq_pipeline.project.locate import INDEX_DIR, Project

DETAILS_DIR = "details"
SCHEMA = "trainnr-detail/3"  # /3 (2026-09-09): fit records read as written
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


def details_path(project: Project, stamp: str) -> Path:
    return project.root / INDEX_DIR / DETAILS_DIR / f"{stamp}.json"


def _schema_of(path: Path) -> str:
    try:
        return str(json.loads(path.read_text()).get("schema", ""))
    except (OSError, ValueError):
        return ""


def write_details(project: Project, index: ProjectIndex) -> dict[str, str]:
    """Write a detail file for every artifact whose kind has a writer and
    that has none yet — or one written by an older schema, since a detail
    is a view and the writer is its only source; return `{stamp: relative
    path}` for those present."""
    written: dict[str, str] = {}
    for artifact in index.artifacts:
        out = details_path(project, artifact.stamp)
        if not out.is_file() or _schema_of(out) != SCHEMA:
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


def _task(root: Path, artifact: Artifact) -> list[dict[str, Any]]:
    ref = json.loads((root / "task.json").read_text())
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
    try:
        from dataclasses import asdict  # noqa: PLC0415

        from rq_pipeline.tasks.registry import resolve  # noqa: PLC0415

        entry = resolve(task_id)
        task = entry.build()
    except Exception as why:
        sections.append(_kv("Scene", [("not built", str(why))]))
        return sections
    proto = getattr(task, "protocol", None)
    control_hz = getattr(task, "control_hz", None)
    steps = getattr(proto, "steps", None)
    episode_s = (steps / control_hz) if (steps and control_hz) else None
    sections.append(
        _kv(
            "Episode",
            [
                ("robot asset", Path(str(getattr(task, "bundle_dir", ""))).name),
                ("control rate (Hz)", control_hz),
                ("episode length (steps)", steps),
                ("episode length (s)", _f(episode_s, 2) if episode_s else ""),
                ("paired trials per evaluation", getattr(proto, "trials", None)),
                ("instruction", getattr(task, "instruction", "")),
                ("cameras", list(getattr(task, "cameras", []) or [])),
                ("state dimension", getattr(task, "state_width", None)),
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
    accept = root / "acceptance.json"
    if accept.is_file():
        a = json.loads(accept.read_text())
        sections.append(
            _kv(
                "Acceptance (scripted policy must succeed; hold-still must fail)",
                [(k, _jsonable(v)) for k, v in a.items()],
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
    weights = sorted(root.glob("*.pt"))
    return [
        _kv(
            "Experiment (reinforcement learning)",
            [
                ("version", artifact.stamp),
                ("robot asset", ident.get("robot", "unrecorded")),
                ("actuator model", ident.get("actuator", "unrecorded")),
                ("domain randomization", ident.get("dr_basis", "")),
                ("seed", ident.get("seed")),
                ("checkpoints", [w.name for w in weights]),
            ],
        ),
        _kv("Identity", [(k, _jsonable(v)) for k, v in ident.items()]),
    ]


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
                ("protocol version", c.get("protocol", "")),
            ],
        ),
        _table(
            "Funnel (trials reaching each stage)",
            ["stage", "trials"],
            [[k2, v] for k2, v in funnel.items()],
        )
        if funnel
        else _kv("Funnel", []),
        _table(
            "Trials", ["trial", "seed", "policy", "outcome", "steps", "events"], rows
        ),
    ]


# -- helpers ------------------------------------------------------------------------


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


_WRITERS = {
    "robot": _robot,
    "task": _task,
    "recording": _recording,
    "batch": _batch,
    "dataset": _dataset,
    "run": _run,
    "certificate": _certificate,
}
