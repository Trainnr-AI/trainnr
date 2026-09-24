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
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rq_pipeline.collect.datasheet import DATASHEET_FILE
from rq_pipeline.collect.provenance import PROVENANCE_FILE
from rq_pipeline.deploy.assay import read_assay
from rq_pipeline.deploy.attribution import SURVIVED, read_attribution
from rq_pipeline.deploy.preflight import card_line, read_preflight
from rq_pipeline.envs.lerobot_train_log import (
    CHAIN_LOG_FILE,
    RUN_MANIFEST_FILE,
    RunLayout,
    parse_train_line,
)
from rq_pipeline.envs.rsl_rl_log import (
    COL_EPISODE_LENGTH,
    COL_REWARD,
    COL_STEPS_PER_SECOND,
    TRAINING_FILE,
)
from rq_pipeline.evaluate.commands import TWIST_LABEL
from rq_pipeline.project.files import read_json, read_text, write_json
from rq_pipeline.project.index import (
    UNRECORDED,
    Artifact,
    ProjectIndex,
    interval_of,
    ratio_of,
)
from rq_pipeline.project.kinds import (
    ACCEPTANCE_FILE,
    ACCEPTED,
    CERTIFICATE_FILE,
    FITS_DIR,
    IDENTITY_FILE,
    POLICY_FILE,
    REJECTED,
    TASK_FILE,
    UNREVIEWED,
)
from rq_pipeline.project.locate import INDEX_DIR, Project
from rq_pipeline.tasks.overlay import jsonable as _plain_jsonable

if TYPE_CHECKING:
    from rq_pipeline.tasks.task import Task

DETAILS_DIR = "details"
# /6 (2026-09-22): a gate's trials in its protocol's shape (a course gate's
# arrival); /5 (2026-09-12): one gate section per runtime; /4 (2026-09-10):
# a walk's gate and episode; /3 (2026-09-09): fit records read as written.
SCHEMA = "trainnr-detail/6"
UNAVAILABLE_KEY = "unavailable"  # the writer failed; the Studio ignores the key
MAX_ROWS = 400  # a table longer than this is truncated, and says so
MAX_MARKDOWN = 6000  # a datasheet is a page, not a book
SMALL = 1e-3  # below this, print in scientific notation
EPISODE_MANIFEST = "manifest.json"  # a generated episode's own record
LEROBOT_INFO = Path("meta") / "info.json"  # LeRobot's dataset description
CHECKPOINTS_DIR = "checkpoints"  # LeRobot's trainer keeps them here
Section = dict[str, Any]
# A writer reads one artifact and returns its sections.
Writer = Callable[[Project, Path, Artifact], list[Section]]

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
    return str(read_json(path, missing_ok=True).get("schema", ""))


def _unavailable(path: Path) -> bool:
    """A detail the writer could not build last time: written so the
    Studio has something to show, never counted as fresh."""
    return bool(read_json(path, missing_ok=True).get(UNAVAILABLE_KEY))


def write_details(project: Project, index: ProjectIndex) -> dict[str, str]:
    """Write a detail file for every artifact whose kind has a writer and
    that has none yet — or one written by an older schema, or one older
    than the artifact's last change (a review written beside a task, a
    record appended to a run) — since a detail is a view and the writer
    is its only source; return `{stamp: relative path}` for those present."""
    written: dict[str, str] = {}
    for artifact in index.artifacts:
        out = details_path(project, artifact.stamp)
        if (
            not out.is_file()
            or _schema_of(out) != SCHEMA
            or _unavailable(out)
            or _stale(out, artifact)
        ):
            writer = _WRITERS.get(artifact.kind)
            if writer is None:
                continue
            body: dict[str, Any] = {"schema": SCHEMA, "version": artifact.stamp}
            try:
                body["sections"] = writer(
                    project, project.root / artifact.path, artifact
                )
            except Exception as why:  # a detail is a view; the index is not
                body["sections"] = [
                    _kv(
                        "Detail unavailable",
                        [("reason", str(why)), ("path", artifact.path)],
                    )
                ]
                body[UNAVAILABLE_KEY] = True  # retried on the next index
            write_json(out, body, default=jsonable)
        written[artifact.stamp] = out.relative_to(project.root).as_posix()
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


def _interval(record: dict[str, Any]) -> str:
    """The record's exact interval as `[lo, hi]`, or `unrecorded`."""
    interval = interval_of(record)
    return _rng(*interval) if interval is not None else UNRECORDED


# -- robot (Asset) ---------------------------------------------------------------


NOT_AUDITED = (
    "not audited: onboarded before the importer audit existed (2026-09-24); "
    "`tools/audit-bundle.py <bundle> <source>` audits it against its source"
)
AUDIT_NOTE = (
    "The bundle's compiled model against the description it came from "
    "(robot/import_audit): a change the format's reader explains is a documented "
    "conversion or a loader loss at that mujoco build; UNEXPLAINED means the "
    "operator accepted it at onboarding."
)
AUDIT_TITLE = "What the importer changed"


def _importer_audit(root: Path) -> Section:
    """What the importer changed, as the bundle record carries it: the
    facts as key/values when nothing changed, the changes as a table."""
    from rq_pipeline.bundles.bundle import (  # noqa: PLC0415
        AUDIT_KEY,
        read_bundle_record,
    )
    from rq_pipeline.robot.import_audit import Audit  # noqa: PLC0415

    record = read_bundle_record(root).get(AUDIT_KEY)
    if not record:
        return _kv(AUDIT_TITLE, [("audit", NOT_AUDITED)])
    audit = Audit.from_record(record)
    head: list[tuple[str, Any]] = [
        ("format", audit.format),
        ("source", audit.source),
        ("mujoco", audit.mujoco),
        ("changed", audit.summary()),
    ]
    if record.get("accepted"):
        head.append(("accepted", "yes: onboarded despite unexplained changes"))
    if not audit.changes and not audit.advisories:
        return _kv(AUDIT_TITLE, head, AUDIT_NOTE)
    rows = [
        [
            c.kind,
            c.element,
            _fact(c.source),
            _fact(c.bundle),
            c.explanation or "UNEXPLAINED",
        ]
        for c in audit.changes
    ]
    rows += [["advisory", "", "", "", a] for a in audit.advisories]
    note = "; ".join(f"{k} {v}" for k, v in head) + ". " + AUDIT_NOTE
    return _table(
        AUDIT_TITLE, ["what", "element", "source", "bundle", "why"], rows, note
    )


def _fact(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_f(v) for v in value]
    return _f(value) if isinstance(value, (int, float)) else value


def _robot(project: Project, root: Path, artifact: Artifact) -> list[Section]:
    import mujoco  # noqa: PLC0415

    from rq_pipeline.project.previews import _robot_model_file  # noqa: PLC0415

    model_file = _robot_model_file(root)
    if model_file is None:
        raise ValueError("no MJCF at the asset's root")
    m = mujoco.MjModel.from_xml_path(str(model_file))
    name = lambda kind, i: mujoco.mj_id2name(m, kind, i) or f"#{i}"  # noqa: E731
    audit = _importer_audit(root)
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
        audit,
        joints,
        actuators,
        bodies,
        sensors,
        collision,
        _kv("Files", [(f, "") for f in files]),
    ]
    fits = root / FITS_DIR
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
        MIN_FITS_FOR_SPREAD,
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
    if len(records) >= MIN_FITS_FOR_SPREAD:
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

    rows: list[tuple[str, Any]] = [
        ("verdict", ACCEPTED if a.get("accepted") else REJECTED)
    ]
    gate = a.get("gate")
    if gate:
        rows.append(("gate", gate))
    else:
        rows += [
            ("scripted policy successes", ratio_of(a, "expert_successes")),
            ("floor policy successes", ratio_of(a, "floor_successes")),
        ]
    rows += [
        ("judged", a.get("judged", UNRECORDED)),
        ("simulator build", a.get("instrument", UNRECORDED)),
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


def _task(project: Project, root: Path, artifact: Artifact) -> list[Section]:
    ref = read_json(root / TASK_FILE)
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
    if (root / ACCEPTANCE_FILE).is_file():
        sections.append(_acceptance(read_json(root / ACCEPTANCE_FILE)))
    else:
        sections.append(
            _kv(
                "Acceptance",
                [("verdict", UNREVIEWED)],
                note="run accept_task to review it",
            )
        )
    try:
        from rq_pipeline.tasks.overlay import build_from_reference  # noqa: PLC0415

        task = build_from_reference(ref)
    except Exception as why:
        sections.append(_kv("Scene", [("not built", str(why))]))
        return sections
    proto = getattr(task, "protocol", None)
    describe = next(d for matches, d in _TASK_DESCRIBERS if matches(task))
    sections.append(_kv("Episode", describe(task)))
    spec = getattr(task, "task_spec", None)
    if spec is not None:
        fields = asdict(spec)
        spawn = fields.pop("part_spawn", None)
        sections.append(
            _kv("Task specification", [(k, jsonable(v)) for k, v in fields.items()])
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


# -- the two task shapes the registry holds, described each its own way ------
#
# A manipulation task carries a `protocol` (steps, trials, milestones, the
# success criterion); a walk carries a `task_spec` with the episode in
# seconds and no scripted protocol. Each shape describes its own episode;
# a third shape is a row in `_TASK_DESCRIBERS`, not a branch.

Rows = list[tuple[str, Any]]


def _is_manipulation(task: Task) -> bool:
    return getattr(task, "protocol", None) is not None


def _is_walk(task: Task) -> bool:
    return (
        getattr(task, "protocol", None) is None
        and getattr(task, "task_spec", None) is not None
    )


def _episode_common(task: Task, steps: int | None, episode_s: float | None) -> Rows:
    return [
        ("robot asset", Path(str(getattr(task, "bundle_dir", ""))).name),
        ("control rate (Hz)", getattr(task, "control_hz", None)),
        ("episode length (steps)", steps),
        ("episode length (s)", _f(episode_s, 2) if episode_s else ""),
    ]


def _describe_manipulation(task: Task) -> Rows:
    proto = task.protocol
    control_hz = getattr(task, "control_hz", None)
    steps = getattr(proto, "steps", None)
    episode_s = (steps / control_hz) if (steps and control_hz) else None
    return [
        *_episode_common(task, steps, episode_s),
        ("paired trials per evaluation", getattr(proto, "trials", None)),
        ("instruction", getattr(task, "instruction", "")),
        ("cameras", list(getattr(task, "cameras", []) or [])),
        ("state dimension", getattr(task, "state_width", None)),
    ]


def _describe_walk(task: Task) -> Rows:
    spec = task.task_spec
    control_hz = getattr(task, "control_hz", None)
    episode_s = getattr(spec, "episode_s", None)
    steps = int(episode_s * control_hz) if (episode_s and control_hz) else None
    return [
        *_episode_common(task, steps, episode_s),
        ("paired trials per evaluation", getattr(spec, "trials", None)),
        ("instruction", getattr(task, "instruction", "")),
    ]


def _describe_other(task: Task) -> Rows:
    return _episode_common(task, None, None)


_TASK_DESCRIBERS: tuple[tuple[Callable[[Any], bool], Callable[[Any], Rows]], ...] = (
    (_is_manipulation, _describe_manipulation),
    (_is_walk, _describe_walk),
    (lambda _task: True, _describe_other),
)


# -- recording ---------------------------------------------------------------------


def _recording(project: Project, root: Path, artifact: Artifact) -> list[Section]:
    from rq_pipeline.robots.quality import QUALITY_KEY  # noqa: PLC0415
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
    sections = [
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
    ]
    if rec.provenance:
        sections.append(
            _kv(
                "Provenance",
                list(rec.provenance.items()),
                note="whose robot this was; a public log is real and not ours",
            )
        )
    sections.extend(_quality_sections(rec.census.get(QUALITY_KEY) or {}))
    sections.append(
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
        )
    )
    reported = {k: v for k, v in rec.census.items() if k != QUALITY_KEY}
    sections.append(
        _kv("What the robot reported", [(k, jsonable(v)) for k, v in reported.items()])
    )
    sections.append(_kv("Notes", [(f"{i + 1}", n) for i, n in enumerate(rec.notes)]))
    return sections


QUALITY_COLUMNS = [
    "channel",
    "samples",
    "rate (Hz)",
    "p50 (ms)",
    "p99 (ms)",
    "dropouts",
    "longest gap (ms)",
]


def _quality_sections(quality: dict[str, Any]) -> list[dict[str, Any]]:
    """What the recording's clock and joints did, measured (robots/quality)."""
    out: list[dict[str, Any]] = []
    clock = quality.get("clock") or {}
    if clock:
        out.append(
            _table(
                "Clock, measured",
                QUALITY_COLUMNS,
                [
                    [
                        name,
                        c.get("samples"),
                        c.get("rate_hz"),
                        c.get("interval_p50_ms"),
                        c.get("interval_p99_ms"),
                        c.get("dropouts"),
                        c.get("longest_gap_ms"),
                    ]
                    for name, c in clock.items()
                ],
                note="rate and jitter read off the samples, never a datasheet",
            )
        )
    ranges = quality.get("joint_range") or {}
    if ranges:
        out.append(
            _table(
                "Joint range covered",
                ["joint", "min", "max", "span"],
                [[j, lo, hi, _f(hi - lo)] for j, (lo, hi) in ranges.items()],
                note=(
                    f"moving {quality.get('moving_fraction', 0):.0%} of the time "
                    f"({quality.get('moving_threshold', '')})"
                ),
            )
        )
    return out


# -- batch (a generated dataset) and dataset (an exported one) -------------------------


def _batch(project: Project, root: Path, artifact: Artifact) -> list[Section]:
    from rq_pipeline.collect.datasheet import summarize  # noqa: PLC0415

    s = summarize(root)
    d = _dataclass_dict(s)
    episodes = sorted(root.glob("episode_*"))
    rows = []
    for ep in episodes[:MAX_ROWS]:
        raw = read_json(ep / EPISODE_MANIFEST, missing_ok=True)
        frames = len(list(ep.glob("frames/*.jpg")) or list(ep.glob("frames/*/*.jpg")))
        dyn = raw.get("dynamics") or {
            k: raw[k] for k in ("damping_scale", "gain_scale") if k in raw
        }
        rows.append(
            [
                ep.name,
                raw.get("seed"),
                raw.get("attempt", UNRECORDED),
                frames,
                field_words(str(raw.get("verdict", ""))),
                jsonable(dyn),
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
            field_words(read_text(root / DATASHEET_FILE)[:MAX_MARKDOWN])
            if (root / DATASHEET_FILE).is_file()
            else "",
            note="The datasheet as written at generation time; house words in an "
            "older file are shown in the field's terms.",
        ),
    ]


def _dataset(project: Project, root: Path, artifact: Artifact) -> list[Section]:
    info = read_json(root / LEROBOT_INFO, missing_ok=True)
    prov = read_json(root / PROVENANCE_FILE, missing_ok=True)
    features = info.get("features", {})
    feat_rows = [
        [
            k,
            v.get("dtype"),
            jsonable(v.get("shape")),
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
                ("robot asset", prov.get("bundle", UNRECORDED)),
                ("scripted policy", prov.get("expert", "")),
                (
                    "source dataset",
                    prov.get("source_stamp") or prov.get("source", UNRECORDED),
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
            [(k, jsonable(v)) for k, v in prov.items() if k != "manifests"],
            note=f"{len(prov.get('manifests', []))} per-episode manifests carried"
            if prov.get("manifests")
            else None,
        ),
    ]


# -- run (Experiment) and evaluation ----------------------------------------------


def _run(project: Project, root: Path, artifact: Artifact) -> list[Section]:
    if (root / RUN_MANIFEST_FILE).is_file():
        run = read_json(root / RUN_MANIFEST_FILE)
        log = root / CHAIN_LOG_FILE
        last = None
        n = 0
        if log.is_file():
            for line in read_text(log, errors="replace").splitlines():
                p = parse_train_line(line)
                if p is not None:
                    last, n = p, n + 1
        ckpts = (
            sorted(
                (RunLayout(root.parent, run["name"]).training / CHECKPOINTS_DIR).glob(
                    "*"
                )
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
                        (run.get("provenance") or {}).get("bundle", UNRECORDED),
                    ),
                    (
                        "scripted policy",
                        (run.get("provenance") or {}).get("expert", UNRECORDED),
                    ),
                    ("started", run.get("started")),
                    ("checkpoint every", run.get("checkpoint_every")),
                    ("checkpoints on disk", len(ckpts)),
                    ("metric lines logged", n),
                    ("last step", getattr(last, "step", None)),
                    (
                        "last metrics",
                        jsonable(dict(getattr(last, "metrics", {}) or {})),
                    ),
                ],
            ),
            _kv("Command", [("argv", run.get("command", ""))]),
        ]
    ident = read_json(root / IDENTITY_FILE, missing_ok=True)
    training = read_json(root / TRAINING_FILE, missing_ok=True)
    final = training.get("final") or {}
    wall = training.get("wall_seconds")
    sections = [
        _kv(
            "Experiment (reinforcement learning)",
            [
                ("version", artifact.stamp),
                ("trainer", training.get("trainer", UNRECORDED)),
                ("iterations", training.get("iterations_logged") or UNRECORDED),
                ("parallel environments", training.get("envs", UNRECORDED)),
                ("device", training.get("device", UNRECORDED)),
                ("wall time", f"{wall / 3600:.1f} h" if wall else UNRECORDED),
                ("final mean reward", final.get(COL_REWARD, UNRECORDED)),
                ("best mean reward", training.get("best_reward", UNRECORDED)),
                ("final episode length", final.get(COL_EPISODE_LENGTH, UNRECORDED)),
                ("steps per second", final.get(COL_STEPS_PER_SECOND, UNRECORDED)),
                ("robot asset", ident.get("robot", UNRECORDED)),
                ("actuator model", ident.get("actuator", UNRECORDED)),
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
                note="sampled from the trainer's record; the full log is train.log",
            )
        )
    return sections


def _certificate(project: Project, root: Path, artifact: Artifact) -> list[Section]:
    c = read_json(root / CERTIFICATE_FILE)
    n = c.get("trials")
    rows = []
    records = sorted(root.glob("records*.jsonl"))
    for rec in records:
        for line in read_text(rec).splitlines():
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
    raw_protocol = c.get("protocol")
    protocol: dict[str, Any] = raw_protocol if isinstance(raw_protocol, dict) else {}
    return [
        _kv(
            "Evaluation",
            [
                ("version", artifact.stamp),
                ("success rate", ratio_of(c)),
                ("95% confidence interval (exact)", _interval(c)),
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
                    (k2, jsonable(v))
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


# A course trial's columns (`deploy.course.CourseTrial`): the label, the
# trial-row keys the cell reads (pinned by the tests to what a trial
# writes), and the cell; a held-twist trial's row is the command and
# what it measured.
COURSE_COLUMNS: tuple[tuple[str, tuple[str, ...], Callable[..., Any]], ...] = (
    ("speed (m/s)", ("speed",), lambda speed: _f(speed, 2)),
    ("reached", ("reached", "of"), lambda reached, of: f"{reached} / {of}"),
    (
        "seconds / budget",
        ("seconds", "budget_s"),
        lambda seconds, budget: f"{_f(seconds, 1)} / {_f(budget, 1)}",
    ),
)


PREFLIGHT_MARK = {True: "passed", False: "REFUSED", None: "not measured"}


def preflight_sections(record: dict[str, Any]) -> list[Section]:
    """Before the first tick: every check with its number and its limit;
    then the ramp-in and the stops as measured."""
    sections: list[Section] = [
        _table(
            "Pre-flight (before the first tick on a robot)",
            ["check", "verdict", "measured", "limit", "note"],
            [
                [
                    c.get("name", UNRECORDED),
                    PREFLIGHT_MARK.get(c.get("passed"), UNRECORDED),
                    c.get("measured", ""),
                    c.get("limit", ""),
                    c.get("detail", ""),
                ]
                for c in record.get("checks", [])
            ],
            note=f"{card_line(record)}; the robot's state read from "
            f"{record.get('state_from', UNRECORDED)}; watchdogs: "
            f"{(record.get('watchdogs') or {}).get('source', UNRECORDED)}",
        )
    ]
    ramp = record.get("ramp_in") or {}
    stops = record.get("soft_stop") or {}
    rows: list[list[Any]] = []
    for name in ("with ramp", "without ramp"):
        r = ramp.get(name)
        if r:
            rows.append(
                [
                    f"handover {name}",
                    f"{_f(r.get('first_tick_target_step_rad'))} rad first tick, "
                    f"{_f(r.get('max_target_step_rad'))} rad max",
                    f"{_f(r.get('max_force_step_nm'))} N·m",
                    "stood" if r.get("stood") else "did not stand",
                ]
            )
    for name, s in stops.items():
        if isinstance(s, dict):
            rows.append(
                [
                    f"stop: {name}",
                    f"damping in {_f(s.get('seconds_to_damping'))} s",
                    f"{_f(s.get('max_force_step_nm'))} N·m",
                    f"body falls at most {_f(s.get('max_body_fall_mps'))} m/s, "
                    f"joints {_f(s.get('max_joint_speed_rad_s'))} rad/s",
                ]
            )
    theirs = record.get("their_stop") or {}
    if "max_body_fall_mps" in theirs:
        rows.append(
            [
                "stop: their Passive, on their simulator",
                "one press",
                "unrecorded (their controller's torques)",
                f"body falls at most {_f(theirs['max_body_fall_mps'])} m/s, "
                f"joints {_f(theirs.get('max_joint_speed_rad_s'))} rad/s",
            ]
        )
    if rows:
        sections.append(
            _table(
                "Ramp-in and stop, measured",
                ["what", "command", "largest torque step in a tick", "the body"],
                rows,
                note=f"ramp from {ramp.get('from', UNRECORDED)} over "
                f"{ramp.get('window_s', UNRECORDED)} s; stops "
                f"{stops.get('while', UNRECORDED)}, ending in "
                f"{stops.get('ends_in', UNRECORDED)}",
            )
        )
    return sections


def attribution_table(record: dict[str, Any]) -> Section:
    """Which parameter breaks it first: one row per knob in the ranking's
    order - the cliff, then every rung that ran as k/n with its interval."""
    by_name = {k["name"]: k for k in record.get("knobs", [])}
    rows: list[list[Any]] = []
    for r in record.get("ranking", []):
        entry = by_name.get(r["name"]) or {}
        unit = entry.get("unit", "")
        cliff = r.get("cliff")
        rows.append(
            [
                r["name"],
                entry.get("describe", UNRECORDED),
                SURVIVED if cliff is None else f"{cliff:g} {unit}".rstrip(),
                " · ".join(
                    f"{x['level']:g}{unit}: {x['successes']}/{x['trials']} "
                    f"{_rng(*x['ci95'])}"
                    for x in entry.get("rungs", [])
                ),
            ]
        )
    base = record.get("baseline") or {}
    cert = record.get("certificate") or {}
    return _table(
        "What would break it first (one knob turned at a time)",
        ["knob", "what is turned", "cliff", "rungs (k/n, exact 95 % interval)"],
        rows,
        note=(
            f"{record.get('sensitivity', UNRECORDED)}; baseline "
            f"{ratio_of(base)} at this draw against the certificate's lower bound "
            f"{cert.get('lower', UNRECORDED)}; "
            f"{(record.get('protocol') or {}).get('rule', '')}"
        ),
    )


def gate_trials(g: dict[str, Any], instrument: str) -> Section:
    """The gate's trials as a table in the protocol's own shape: a course
    gate's rows say how far and how fast, a twist gate's what was held."""
    protocol = g.get("protocol") or {}
    records = g.get("records", [])
    tail = ["fell", "error ratio", "success"]

    def judged(r: dict[str, Any]) -> list[Any]:
        return [
            "yes" if r.get("fell") else "no",
            _f(r.get("err_ratio")),
            "success" if r.get("success") else "failure",
        ]

    if protocol.get("course"):
        rows = [
            [
                *(cell(*(r.get(k) for k in keys)) for _, keys, cell in COURSE_COLUMNS),
                *judged(r),
            ]
            for r in records
        ]
        return _table(
            f"Gate trials: {instrument}",
            [label for label, _, _ in COURSE_COLUMNS] + tail,
            rows,
            note=f"{protocol.get('commands')}; {protocol.get('criterion')}",
        )
    return _table(
        f"Gate trials: {instrument}",
        [TWIST_LABEL, "steps", *tail],
        [
            [
                ", ".join(f"{c:.2f}" for c in r.get("command", [])),
                r.get("steps"),
                *judged(r),
            ]
            for r in records
        ],
    )


def _deploy(project: Project, root: Path, artifact: Artifact) -> list[Section]:
    """A deployment: what the manifest says a runtime needs, the joints
    and observations as tables, and the sim-to-sim gate's verdict."""
    from rq_pipeline.deploy.manifest import (  # noqa: PLC0415
        GATE_INSTRUMENTS,
        MANIFEST_FILE,
        gate_word,
        read_gates,
    )

    m = read_json(root / MANIFEST_FILE)
    control = m.get("control") or {}
    joints = m.get("joints") or {}
    onnx = m.get("onnx") or {}
    sections = [
        _kv(
            "Deployment",
            [
                ("version", artifact.stamp),
                ("policy", m.get("policy", UNRECORDED)),
                ("checkpoint", m.get("checkpoint", UNRECORDED)),
                ("experiment", m.get("run", UNRECORDED)),
                ("robot asset", m.get("robot", UNRECORDED)),
                ("actuator model", m.get("actuator", UNRECORDED)),
                ("environment", m.get("task") or UNRECORDED),
                ("domain randomization", m.get("dr_basis") or UNRECORDED),
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
                ("normalization", onnx.get("normalization", UNRECORDED)),
                (
                    "export check (max |onnx - torch|)",
                    onnx.get("export_check_max_abs_diff"),
                ),
                ("action", (m.get("action") or {}).get("kind", UNRECORDED)),
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
            ("successes", ratio_of(g)),
            ("95% confidence interval (exact)", _interval(g)),
            ("rule", verdict.get("rule", "")),
            ("tolerance", verdict.get("tolerance", "n/a")),
            ("evaluation cited", ratio_of(cert) if cert else "none cited"),
            ("runtime", (g.get("protocol") or {}).get("runtime", UNRECORDED)),
            (
                "simulator build",
                (g.get("protocol") or {}).get("instrument", UNRECORDED),
            ),
            ("judged", g.get("judged", UNRECORDED)),
        ]
        contacts = g.get("contacts") or {}
        if contacts:
            rows.append(("contact points kept", contacts.get("points", UNRECORDED)))
            site = contacts.get("site_gap") or {}
            if site:
                rows.append(
                    (
                        f"gap at the contact sites (within {site.get('radius_m')} m)",
                        _site_gap_line(site),
                    )
                )
        protocol = g.get("protocol") or {}
        rows.append(("commands", protocol.get("commands", UNRECORDED)))
        steer = protocol.get("steer") or {}
        if steer:
            rows.append(
                ("steering", f"gain {steer.get('gain')} ({steer.get('basis')})")
            )
        sections.append(
            _kv(
                f"Sim-to-sim gate: {instrument}",
                rows,
                note="The exported policy driven through its manifest alone, judged "
                f"by: {protocol.get('criterion', UNRECORDED)}.",
            )
        )
        sections.append(gate_trials(g, instrument))
    assay = read_assay(root)
    if assay:
        cliff = assay.get("cliff") or {}
        sections.append(
            _table(
                "Perturbation assay (the terrain moved, the policy not told)",
                ["perturbation", "successes", "95% confidence interval (exact)"],
                [
                    [
                        (r.get("perturbation") or {}).get("label", UNRECORDED),
                        ratio_of(r),
                        _interval(r),
                    ]
                    for r in assay.get("perturbations", [])
                ],
                note=(
                    f"cliff: nominal {cliff.get('nominal_rate')} to worst "
                    f"{cliff.get('worst_rate')} ({cliff.get('worst')}), drop "
                    f"{cliff.get('drop')}"
                    + (f"; {cliff['note']}" if cliff.get("note") else "")
                ),
            )
        )
    preflight = read_preflight(root)
    if preflight:
        sections.extend(preflight_sections(preflight))
    attribution = read_attribution(root)
    if attribution:
        sections.append(attribution_table(attribution))
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
                    jsonable(t.get("scale")),
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
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return dict(obj) if isinstance(obj, dict) else {}


def jsonable(v: Any) -> Any:
    """A value as JSON can hold it: numpy scalars and arrays as numbers
    and lists, then the task layer's own rule for the plain shapes;
    anything else as its string."""
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    plain = _plain_jsonable(v)
    if isinstance(plain, (str, int, float, bool)) or plain is None:
        return plain
    return str(plain)


def _policy(project: Project, root: Path, artifact: Artifact) -> list[Section]:
    manifest = read_json(root / POLICY_FILE, missing_ok=True)
    identity = read_json(root / IDENTITY_FILE, missing_ok=True)
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
    for cert in sorted(project.certificates.glob(f"*/{CERTIFICATE_FILE}")):
        c = read_json(cert, missing_ok=True)
        if c.get("policy") != artifact.stamp:
            continue
        rows.append(
            [
                cert.parent.name,
                ratio_of(c),
                _interval(c),
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


def _finding(project: Project, root: Path, artifact: Artifact) -> list[Section]:
    raw = read_json(root)
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
                    [name, *[jsonable(arm.get(k)) for k in keys]]
                    for name, arm in arms.items()
                ],
            )
        )
    elif isinstance(outcome, dict):
        sections.append(_kv("Outcome", [(k, jsonable(v)) for k, v in outcome.items()]))
    elif outcome is not None:
        sections.append(_markdown("Outcome", str(outcome)))
    for title, key in (
        ("Inputs", "inputs"),
        ("Artifacts", "artifacts"),
        ("Sources", "sources"),
    ):
        block = raw.get(key)
        if isinstance(block, dict) and block:
            sections.append(_kv(title, [(k, jsonable(v)) for k, v in block.items()]))
    caveats = raw.get("caveats")
    if isinstance(caveats, list) and caveats:
        sections.append(_markdown("Caveats", "\n".join(f"- {c}" for c in caveats)))
    return sections


def _drift(_project: Project, root: Path, artifact: Artifact) -> list[Section]:
    """A drift check: the verdict and what to do, then every parameter's
    fresh interval beside the reference it was judged against."""
    from rq_pipeline.fleet.drift import (  # noqa: PLC0415
        DRIFT_FILE,
        load_drift_record,
        verdict_word,
    )
    from rq_pipeline.project.index import interval_text  # noqa: PLC0415

    try:
        d = load_drift_record(root / DRIFT_FILE)
    except (OSError, ValueError, TypeError) as why:
        return [_kv("Drift check", [("unreadable", str(why))])]
    sections: list[Section] = [
        _kv(
            "Drift check",
            [
                ("version", artifact.stamp),
                ("verdict", verdict_word(d.drifted)),
                ("parameters that left", ", ".join(d.left) or "none"),
                ("unresolved", ", ".join(d.unresolved) or "none"),
                ("recommendation", d.recommendation),
                ("method", d.method),
                ("reference", f"{d.references} fit record(s): {', '.join(d.fit)}"),
                ("rule", d.rule),
                ("anchor", d.anchor),
                ("simulator build", d.instrument),
                ("code", d.code),
                ("checked", d.created_utc),
            ],
            note="Fresh telemetry identified without writing a fit record, judged "
            "against the union of the robot's identified intervals.",
        ),
        _table(
            "Parameters",
            [
                "parameter",
                "verdict",
                "reference interval",
                "fresh estimate",
                "fresh interval",
                "shift (reference half-widths)",
                "unit",
                "note",
            ],
            [
                [
                    p.name,
                    p.verdict,
                    interval_text(p.reference_lower, p.reference_upper),
                    _f(p.fresh_estimate)
                    if p.fresh_estimate is not None
                    else UNRECORDED,
                    interval_text(p.fresh_lower, p.fresh_upper),
                    _f(p.shift) if p.shift is not None else UNRECORDED,
                    p.unit,
                    p.note,
                ]
                for p in d.parameters
            ],
            note="within: overlaps the reference; left: pinned and outside it; "
            "unresolved: not pinned by this recording; anchored: fixed by the "
            "method, never judged.",
        ),
    ]
    return sections


GAP_NOTE = (
    "The gap the field never audits: altering only the collision geometry drops "
    "real success 61.7 points while the simulation looks unchanged "
    "(docs/e2e-research/75 §2)."
)
BASIS_NOTE = (
    "measured: a robot touched it and a fit record carries the interval; declared: "
    "someone said so, with the span this project randomizes over."
)


def _splat_line(splat: dict[str, Any]) -> str:
    count, degree = splat.get("count", UNRECORDED), splat.get("sh_degree", UNRECORDED)
    visible = splat.get("visible_count", UNRECORDED)
    return f"{count} gaussians, harmonics degree {degree}, {visible} visible"


def _proxy_line(proxy: dict[str, Any]) -> str:
    vertices, faces = proxy.get("vertices", UNRECORDED), proxy.get("faces", UNRECORDED)
    tight = proxy.get("watertight", UNRECORDED)
    return f"{vertices} vertices, {faces} faces, watertight {tight}"


def _alignment_line(a: Any) -> str:
    euler = tuple(round(v, 3) for v in a.rotation_euler_xyz)
    translation = tuple(round(v, 3) for v in a.translation)
    return f"scale {a.scale:.4g}, euler {euler}, translation {translation} ({a.source})"


def _scene(_project: Project, root: Path, artifact: Artifact) -> list[Section]:
    """A captured scene: the gap first (what the eye sees against what
    the solver touches), the physics by basis, the capture and the
    chain that made it, the splat's and the proxy's facts."""
    from rq_pipeline.scenes.record import SCENE_FILE, load_scene_record  # noqa: PLC0415

    try:
        s = load_scene_record(root / SCENE_FILE)
    except (OSError, ValueError, TypeError) as why:
        return [_kv("Scene", [("unreadable", str(why))])]
    g = s.gap

    def cm(v: float | None) -> Any:
        return f"{v * 100:.2f} cm" if isinstance(v, (int, float)) else UNRECORDED

    def pct(v: float | None) -> Any:
        return f"{v * 100:.1f} %" if isinstance(v, (int, float)) else UNRECORDED

    tol = f"{g.tolerance_m * 100:.0f} cm"
    sections: list[Section] = [
        _kv(
            "Scene",
            [
                ("version", artifact.stamp),
                ("source", s.source),
                ("splat", _splat_line(s.splat)),
                ("proxy", _proxy_line(s.proxy)),
                ("alignment", _alignment_line(s.alignment)),
                ("created", s.created_utc),
                ("code", s.code),
            ],
        ),
        _kv(
            "Visible surface against the collision proxy",
            [
                ("chamfer", cm(g.chamfer_m)),
                ("95th percentile, visible to proxy", cm(g.p95_m)),
                (
                    f"visible surface beyond {tol} of any collider",
                    pct(g.beyond_tolerance_fraction),
                ),
                (
                    f"proxy surface beyond {tol} of any visible gaussian",
                    pct(g.hidden_fraction),
                ),
                (
                    "visible surface inside the proxy's footprint",
                    pct(g.footprint_fraction),
                ),
                (
                    "samples",
                    f"{g.visible_samples} visible in the footprint, "
                    f"{g.proxy_samples} on the proxy",
                ),
                ("method", g.method),
            ]
            + ([("not measured", g.note)] if g.note else []),
            note=GAP_NOTE,
        ),
        _table(
            "Physics, by basis",
            ["parameter", "value", "basis", "span or interval", "cites", "unit"],
            [
                [
                    p.name,
                    p.value,
                    p.basis,
                    (
                        f"±{p.span:g}"
                        if p.span is not None
                        else (
                            f"[{p.interval[0]:.4g}, {p.interval[1]:.4g}]"
                            if p.interval
                            else ""
                        )
                    ),
                    p.cites,
                    p.unit,
                ]
                for p in s.physics
            ],
            note=BASIS_NOTE,
        ),
        _kv(
            "Capture and chain",
            [
                ("device", s.capture.device),
                ("app", s.capture.app),
                ("frames", s.capture.frames),
                ("resolution", s.capture.resolution),
                ("duration (s)", s.capture.duration_s),
                ("lighting", s.capture.lighting),
            ]
            + [
                (f"tool: {t.name}", f"{t.version} · {t.license} · {t.role}")
                for t in s.tools
            ],
        ),
    ]
    if s.notes:
        sections.append(_kv("Notes", [(f"{i + 1}", n) for i, n in enumerate(s.notes)]))
    return sections


_WRITERS: dict[str, Writer] = {
    "scene": _scene,
    "drift": _drift,
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


def _site_gap_line(site: dict[str, Any]) -> str:
    """The four numbers at the contact sites, or the reason there are none."""
    if site.get("chamfer_m") is None:
        return site.get("note") or UNRECORDED
    beyond, hidden = site["beyond_tolerance_fraction"], site["hidden_fraction"]
    return (
        f"chamfer {site['chamfer_m'] * 100:.1f} cm · p95 {site['p95_m'] * 100:.1f} cm"
        f" · {beyond * 100:.0f} % of the visible surface beyond "
        f"{site['tolerance_m'] * 100:.0f} cm · {hidden * 100:.0f} % of the proxy unseen"
    )
