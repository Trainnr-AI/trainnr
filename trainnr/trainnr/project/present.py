"""Present: stream one artifact into the Studio's embedded viewer, as itself.

The Studio can launch the viewer and index a project, but until now it
could not POINT the viewer at an artifact: a click showed a card. This is
the door docs/80 named `show_in_studio` and never built. One function
per kind, each logging the artifact into Rerun with a blueprint that fits
it — a robot as its meshes in a 3D view; a recording's channels as time
series with a scrubber; an experiment's curves; a certificate's funnel as
bars beside its interval; a dataset's episodes with their video; a task's
scene with its spawn bands drawn. Rerun's own views throughout, never a
re-implementation (docs/e2e-research/55).

Every artifact is its own Rerun recording (application id = the stamp),
so the viewer's recording list IS the project's artifact list, and
switching between them is the viewer's own affair.

Two entry points: `present(project, stamp)` for a tool call, and
`serve(project)` — the presenter loop the Studio spawns once: it watches
`<project>/.index/present.json` (the Studio writes `{"stamp": ...}` on a
click), presents, and clears it. The file is the contract, like the
index; headless callers use the function directly.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trainnr.collect.datasheet import DATASHEET_FILE
from trainnr.collect.provenance import PROVENANCE_FILE
from trainnr.envs.lerobot_train_log import (
    CHAIN_LOG_FILE,
    RUN_MANIFEST_FILE,
    parse_train_line,
)
from trainnr.envs.rsl_rl_log import COL_ITERATION, TRAINING_FILE
from trainnr.project.control import present_status_path
from trainnr.project.files import read_json, read_text, write_json
from trainnr.project.index import (
    UNRECORDED,
    Artifact,
    ProjectIndex,
    index_project,
    interval_of,
    interval_text,
    ratio_of,
)
from trainnr.project.kinds import (
    CERTIFICATE_FILE,
    IDENTITY_FILE,
    POLICY_FILE,
    TASK_FILE,
    Kind,
)
from trainnr.project.locate import INDEX_DIR, Project
from trainnr.viz import entry_name, gaussians, viewer_files

if TYPE_CHECKING:
    import rerun as rr
    from rerun.blueprint import Container, View

INTENT_FILE = "present.json"
# The intent the presenter is on, renamed away from INTENT_FILE the moment
# it is read: a show requested while a big scene streams (up to FLUSH_S)
# lands in a fresh INTENT_FILE and is served next, not deleted with the
# one before it (2026-09-27).
CLAIMED_SUFFIX = ".busy"
POLL_S = 0.25
# How long a show waits for the viewer to confirm everything arrived. The
# viewer is shared with live streams (a training's curves, the viewport's
# twin at 20 Hz), and behind them a 10 s wait ran out while the show itself
# had landed: the Studio then said "could not show" over a picture that was
# there (2026-09-26). The presenter is its own process, so it can wait.
FLUSH_S = 120.0
# The channel components drawn per recording: all of them up to this many,
# so a 14-joint arm plots as fourteen lines and not one unreadable braid.
MAX_TRACES = 16
# A viewer draws a trace, not every sample: a 12-minute 500 Hz bag is
# 368k rows per trace and its stream saved as 605 MB (2026-09-24).
# Above this many rows a trace is sent every k-th sample and the manifest
# says so; the recording on disk keeps every row.
MAX_ROWS_PER_TRACE = 20_000
PAIR = 2  # a compare view holds two artifacts
EPISODE_MANIFEST = "manifest.json"  # a generated episode's own record
MAX_VIDEOS = 8  # a dataset's cameras shown at once
FRAMES_PER_EPISODE = 60  # a generated episode is sampled down to this many
# What a presenter returns: the entity paths logged, the view's name,
# and the blueprint layout for them.
Shown = dict[str, Any]
Presenter = Callable[..., Shown]


def intent_path(project: Project) -> Path:
    return project.root / INDEX_DIR / INTENT_FILE


def present(
    project: Project, stamp: str, index: ProjectIndex | None = None
) -> dict[str, Any]:
    """Stream the artifact named by `stamp` into the viewer. Returns what
    was shown: the kind, the entity paths logged, the view it got."""
    import rerun as rr  # noqa: PLC0415 - viz extra

    from trainnr.viz import STUDIO_ADDRESS  # noqa: PLC0415

    index = index or index_project(project)
    artifact = next((a for a in index.artifacts if a.stamp == stamp), None)
    if artifact is None:
        raise KeyError(f"no artifact {stamp!r} in {project.root}")
    kind = Kind(artifact.kind)
    presenter = _PRESENTERS.get(kind)
    if presenter is None:
        raise ValueError(f"nothing to show for a {kind.value} yet")
    import rerun.blueprint as rrb  # noqa: PLC0415

    # A stable recording id per artifact: showing it again lands in the
    # same recording instead of stacking a new copy in the viewer's
    # source list (four copies of one robot, seen 2026-09-09).
    recording = rr.RecordingStream(
        application_id=entry_name(stamp), recording_id=_recording_id(stamp)
    )
    recording.connect_grpc(STUDIO_ADDRESS)
    try:
        # The saved streams first (docs/76 §10.5): each lands in the viewer
        # under its own recording id with the layout it was saved with -
        # a run that happened elsewhere opens here as it ran.
        replayed = viewer_files(project.root / artifact.path)
        for path in replayed:
            recording.log_file_from_path(path)
        shown = presenter(project, artifact, recording)
        shown["viewer_files"] = [str(path) for path in replayed]
        recording.send_blueprint(rrb.Blueprint(shown["layout"]))
        _flush(recording, stamp)
    finally:
        recording.disconnect()
    return {
        "stamp": stamp,
        "kind": kind.value,
        "paths": shown["paths"],
        "view": shown["view"],
        "viewer_files": shown["viewer_files"],
    }


def compare(
    project: Project, a: str, b: str, index: ProjectIndex | None = None
) -> dict[str, Any]:
    """Two artifacts side by side in one recording (`a` left, `b` right),
    each under its own entity root so two of a kind never collide."""
    import rerun as rr  # noqa: PLC0415 - viz extra
    import rerun.blueprint as rrb  # noqa: PLC0415

    from trainnr.viz import STUDIO_ADDRESS  # noqa: PLC0415

    index = index or index_project(project)
    halves = []
    for side, stamp in (("a", a), ("b", b)):
        artifact = next((x for x in index.artifacts if x.stamp == stamp), None)
        if artifact is None:
            raise KeyError(f"no artifact {stamp!r} in {project.root}")
        kind = Kind(artifact.kind)
        presenter = _PRESENTERS.get(kind)
        if presenter is None:
            raise ValueError(f"nothing to show for a {kind.value} yet")
        halves.append((side, artifact, kind, presenter))
    recording = rr.RecordingStream(
        application_id=entry_name(f"{a} vs {b}"),
        recording_id=_recording_id(f"{a} vs {b}"),
    )
    recording.connect_grpc(STUDIO_ADDRESS)
    shown = []
    try:
        for side, artifact, kind, presenter in halves:
            shown.append(
                presenter(project, artifact, recording, root=f"{side}/{kind.value}")
            )
        recording.send_blueprint(
            rrb.Blueprint(rrb.Horizontal(*(s["layout"] for s in shown)))
        )
        _flush(recording, f"{a} vs {b}")
    finally:
        recording.disconnect()
    return {
        "stamps": [a, b],
        "kinds": [h[2].value for h in halves],
        "paths": [p for s in shown for p in s["paths"]],
        "view": " | ".join(s["view"] for s in shown),
    }


LIVE_REFRESH_S = 15.0  # how often a training run's log becomes its record


def serve(project: Project, *, once: bool = False) -> None:
    """The presenter loop: watch the intent file, present, clear - and
    every LIVE_REFRESH_S, turn the console log of any run training in
    the project into its training record and re-index, so the Studio's
    experiment card follows the run (docs/77, the WSL machine, 2026-09-10)."""
    from trainnr.project.index import index_project, write_index  # noqa: PLC0415
    from trainnr.project.live import (  # noqa: PLC0415
        index_stale,
        refresh_project,
        refresh_verdicts,
    )

    path = intent_path(project)
    claimed = path.with_name(path.name + CLAIMED_SUFFIX)
    last_live = 0.0
    # The first tick writes the index whatever the files say: an index
    # written by an older presenter may lack what this one records (the
    # light pictures, 2026-10-03), and a cached render costs nothing.
    first = True
    while True:
        if time.time() - last_live >= LIVE_REFRESH_S:
            last_live = time.time()
            try:
                refreshed = refresh_project(project) + refresh_verdicts(project)
                if first or refreshed or index_stale(project):
                    write_index(project, index_project(project))
                if first:
                    refresh_siblings(project)
                first = False
            except Exception as why:  # a broken log must not kill the presenter
                _write_status(project, {"live_error": str(why)})
        if path.is_file() and not claimed.is_file():
            # Claimed by rename: a request written from here on is a new
            # file, kept for the next turn of the loop.
            with contextlib.suppress(OSError):
                path.replace(claimed)
        if claimed.is_file():
            stamp: str | None = None
            try:
                intent = json.loads(claimed.read_text(encoding="utf-8"))
                stamp = intent.get("stamp")
                stamps = intent.get("stamps") or []
                if len(stamps) == PAIR:
                    stamp = " vs ".join(stamps)
                    _write_status(project, {"presenting": stamp})
                    compare(project, *stamps)
                elif stamp:
                    _write_status(project, {"presenting": stamp})
                    present(project, stamp)
            except Exception as why:  # a bad intent must not kill the presenter
                _write_status(project, {"error": str(why), "stamp": stamp})
            else:
                _write_status(project, {"shown": stamp})
            claimed.unlink(missing_ok=True)
            if once:
                return
        time.sleep(POLL_S)


def refresh_siblings(project: Project) -> int:
    """Rewrite the index of every other indexed project under the same
    home once, so the Projects page has every cover in both palettes
    even for a project whose own presenter has not run under this build
    (2026-10-03: the open project's cover switched, the others stayed
    dark). Returns how many were rewritten; a failure skips that one."""
    from trainnr.project.index import index_project, write_index  # noqa: PLC0415
    from trainnr.project.locate import list_projects  # noqa: PLC0415

    done = 0
    for other in list_projects(project.root.parent):
        if other.root == project.root or not other.index_path.is_file():
            continue
        try:
            write_index(other, index_project(other))
            done += 1
        except Exception:
            continue
    return done


def _write_status(project: Project, status: dict[str, Any]) -> None:
    """The presenter's answer, with the clock it was written at — the
    door waits on that, not on a filesystem's mtime granularity."""
    write_json(present_status_path(project), {**status, "t": time.time()})


# -- per kind -------------------------------------------------------------------


def _present_robot(
    project: Project, artifact: Artifact, rr_: rr.RecordingStream, root: str = "robot"
) -> Shown:
    """The bundle's model as meshes in a 3D view, posed at its keyframe."""
    import mujoco  # noqa: PLC0415
    import rerun.blueprint as rrb  # noqa: PLC0415

    from trainnr.project.previews import _robot_model_file  # noqa: PLC0415
    from trainnr.viz import RigMirror  # noqa: PLC0415

    folder = project.root / artifact.path
    model_file = _robot_model_file(folder)
    if model_file is None:
        raise ValueError(f"{folder}: no MJCF to show")
    model = mujoco.MjModel.from_xml_path(str(model_file))
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    with _AsDefault(rr_):
        mirror = RigMirror(model, model_colors=True)
        mirror.log(data, path=root, static=True)
        rr_.log(
            f"{root}/census",
            _doc(
                f"# {artifact.stamp}\n\n"
                f"- bodies {model.nbody} · joints {model.njnt} · actuators {model.nu} "
                f"· sensors {model.nsensor} · geoms {model.ngeom}\n"
                f"- model `{model_file.name}` · timestep {model.opt.timestep:g} s\n"
                + _lineage(artifact)
            ),
            static=True,
        )
    return {
        "paths": [root],
        "view": "3D",
        "layout": rrb.Horizontal(
            rrb.Spatial3DView(origin=root, name=artifact.stamp),
            rrb.TextDocumentView(origin=f"{root}/census", name="census"),
            column_shares=[3, 1],
        ),
    }


def _present_recording(
    project: Project,
    artifact: Artifact,
    rr_: rr.RecordingStream,
    root: str = "recording",
) -> Shown:
    """Every channel as a time-series view on the recording's own clock."""
    import rerun as rr  # noqa: PLC0415
    import rerun.blueprint as rrb  # noqa: PLC0415

    from trainnr.robots.recording import Recording  # noqa: PLC0415

    recording = Recording.read(project.root / artifact.path)
    views = []
    paths = []
    decimated: dict[str, int] = {}
    with _AsDefault(rr_):
        for name, channel in recording.channels.items():
            base = f"{root}/{name}"
            paths.append(base)
            labels = channel.components or tuple(str(i) for i in range(channel.width))
            values = channel.values.reshape(len(channel.times), -1)
            traces = list(enumerate(labels[:MAX_TRACES]))
            # One column send per trace: a 12-minute 500 Hz bag is 368k
            # rows, and a log call per row per trace took minutes.
            stride = max(1, -(-len(channel.times) // MAX_ROWS_PER_TRACE))
            if stride > 1:
                decimated[name] = stride
            rows = values[::stride]
            times = rr.TimeColumn("time", duration=channel.times[::stride])
            for col, label in traces:
                rr_.log(
                    f"{base}/{label}",
                    rr.SeriesLines(names=[f"{label} [{channel.unit}]"]),
                    static=True,
                )
                rr_.send_columns(
                    f"{base}/{label}",
                    indexes=[times],
                    columns=rr.Scalars.columns(scalars=rows[:, col]),
                )
            views.append(
                rrb.TimeSeriesView(origin=base, name=f"{name} [{channel.unit}]")
            )
        rr_.log(
            f"{root}/manifest",
            _doc(
                f"# {artifact.stamp}\n\n"
                f"- source `{recording.source}` via `{recording.adapter}`"
                f" · {recording.duration_s:.1f} s · collected: {recording.collection}\n"
                + (
                    f"- provenance: {json.dumps(recording.provenance)}\n"
                    if recording.provenance
                    else ""
                )
                + f"- census: {json.dumps(recording.census)}\n"
                + "".join(f"- note: {n}\n" for n in recording.notes)
                + "".join(
                    f"- viewer shows every {k}th sample of {n} (disk keeps all)\n"
                    for n, k in decimated.items()
                )
            ),
        )
    return {
        "paths": paths,
        "view": "time series",
        "layout": rrb.Vertical(
            rrb.Grid(*views)
            if views
            else rrb.TextDocumentView(origin=f"{root}/manifest"),
            rrb.TextDocumentView(origin=f"{root}/manifest", name="manifest"),
            row_shares=[4, 1],
        ),
    }


def _present_run(
    project: Project, artifact: Artifact, rr_: rr.RecordingStream, root: str = "run"
) -> Shown:
    """A training run's curves from its chain log, plus its manifest."""
    import rerun as rr  # noqa: PLC0415
    import rerun.blueprint as rrb  # noqa: PLC0415

    folder = project.root / artifact.path
    training = folder / TRAINING_FILE
    if training.is_file():
        return _present_rl_run(artifact, folder, rr_, root)
    log = folder / CHAIN_LOG_FILE
    metrics: dict[str, list[tuple[int, float]]] = {}
    if log.is_file():
        for line in read_text(log, errors="replace").splitlines():
            parsed = parse_train_line(line)
            if parsed is None:
                continue
            for key, value in parsed.metrics.items():
                metrics.setdefault(key, []).append((parsed.step, float(value)))
    paths = []
    with _AsDefault(rr_):
        for key, points in metrics.items():
            path = f"{root}/{key}"
            paths.append(path)
            rr_.log(path, rr.SeriesLines(names=[key]), static=True)
            for step, value in points:
                rr_.set_time("step", sequence=step)
                rr_.log(path, rr.Scalars(value))
        manifest = folder / RUN_MANIFEST_FILE
        text = read_text(manifest) if manifest.is_file() else "{}"
        rr_.log(
            f"{root}/manifest",
            _doc(f"# {artifact.stamp}\n\n```json\n{text}\n```\n" + _lineage(artifact)),
        )
    views = [rrb.TimeSeriesView(origin=p, name=p.split("/", 1)[1]) for p in paths]
    return {
        "paths": paths,
        "view": "time series",
        "layout": rrb.Vertical(
            rrb.Grid(*views)
            if views
            else rrb.TextDocumentView(origin=f"{root}/manifest"),
            rrb.TextDocumentView(origin=f"{root}/manifest", name="manifest"),
            row_shares=[3, 1],
        ),
    }


# The series every reader looks at first, one tile each across the top;
# the rest of a trainer's 30-odd terms go in tabs by group underneath
# (13 tiles in one grid were unreadable, 2026-09-27).
HEADLINE_CURVES = ("reward", "episode_length", "steps_per_second", "value_loss")


def curve_layout(views: list[Any]) -> Any:
    """The run's curves: the headline series side by side, the term
    groups as tabs below; a run with only one kind of view gets that."""
    import rerun.blueprint as rrb  # noqa: PLC0415

    headline = [v for v in views if v.name in HEADLINE_CURVES]
    rest = [v for v in views if v.name not in HEADLINE_CURVES]
    if not headline:
        return rrb.Tabs(*rest)
    if not rest:
        return rrb.Horizontal(*headline)
    return rrb.Vertical(rrb.Horizontal(*headline), rrb.Tabs(*rest), row_shares=[1, 1])


def _present_rl_run(
    artifact: Artifact, folder: Path, rr_: rr.RecordingStream, root: str
) -> Shown:
    """A reinforcement-learning run: every curve the training record kept
    (the reward and its terms, losses, throughput, the curriculum) on the
    iteration timeline in grouped panels, and the run's facts as a
    document."""
    import rerun as rr  # noqa: PLC0415
    import rerun.blueprint as rrb  # noqa: PLC0415

    from trainnr.envs.tfevents import curve_groups  # noqa: PLC0415

    record = read_json(folder / TRAINING_FILE)
    columns: list[str] = record.get("columns") or []
    curve: list[list[Any]] = record.get("curve") or []
    paths = []
    with _AsDefault(rr_):
        if COL_ITERATION in columns:
            it = columns.index(COL_ITERATION)
            for c, name in enumerate(columns):
                if name == COL_ITERATION:
                    continue
                path = f"{root}/{name}"
                paths.append(path)
                rr_.log(
                    path, rr.SeriesLines(names=[name.replace("_", " ")]), static=True
                )
                for row in curve:
                    if row[it] is None or row[c] is None:
                        continue
                    rr_.set_time(COL_ITERATION, sequence=int(row[it]))
                    rr_.log(path, rr.Scalars(float(row[c])))
        stills = _log_stills(folder, rr_, root)
        identity = folder / IDENTITY_FILE
        facts = {k: v for k, v in record.items() if k not in ("columns", "curve")}
        text = json.dumps(facts, indent=1)
        ident = read_text(identity) if identity.is_file() else "{}"
        rr_.log(
            f"{root}/manifest",
            _doc(
                f"# {artifact.stamp}\n\n## training\n\n```json\n{text}\n```\n"
                f"\n## identity\n\n```json\n{ident}\n```\n" + _lineage(artifact)
            ),
            static=True,
        )
    views = [
        rrb.TimeSeriesView(
            origin=root, name=name, contents=[f"+ {root}/{c}" for c in members]
        )
        for name, members in curve_groups(columns).items()
    ]
    curves: Container | View = (
        curve_layout(views)
        if views
        else rrb.TextDocumentView(origin=f"{root}/manifest")
    )
    top = (
        rrb.Horizontal(
            rrb.Spatial2DView(origin=stills, name="checkpoint"),
            curves,
            column_shares=[2, 3],
        )
        if stills
        else curves
    )
    return {
        "paths": [*paths, *([stills] if stills else []), f"{root}/manifest"],
        "view": "time series",
        "layout": rrb.Vertical(
            top,
            rrb.TextDocumentView(origin=f"{root}/manifest", name="manifest"),
            row_shares=[3, 1],
        ),
    }


STILLS_DIR, STILLS_FILE = "stills", "stills.json"  # walk_stills writes them


def _log_stills(folder: Path, rr_: rr.RecordingStream, root: str) -> str | None:
    """A run's checkpoint stills (one rollout frame every N iterations)
    on the iteration timeline, so scrubbing the reward curve shows what
    the policy looked like there. The entity path, or None when the
    run has no stills."""
    import rerun as rr  # noqa: PLC0415

    record = folder / STILLS_DIR / STILLS_FILE
    if not record.is_file():
        return None
    path = f"{root}/checkpoint"
    for row in json.loads(read_text(record)):
        still = folder / STILLS_DIR / row["file"]
        if not still.is_file():
            continue
        rr_.set_time(COL_ITERATION, sequence=int(row[COL_ITERATION]))
        rr_.log(path, rr.EncodedImage(path=still))
    return path


def _present_policy(
    project: Project, artifact: Artifact, rr_: rr.RecordingStream, root: str = "policy"
) -> Shown:
    """A policy: the robot it drives in 3D (when the project holds it),
    its facts, and every evaluation of it in the project as bars."""
    import rerun as rr  # noqa: PLC0415
    import rerun.blueprint as rrb  # noqa: PLC0415

    index = index_project(project)
    by_stamp = {a.stamp: a for a in index.artifacts}
    robot = by_stamp.get(artifact.cites.get("robot", ""))
    folder = project.root / artifact.path
    manifest = folder / POLICY_FILE
    text = read_text(manifest) if manifest.is_file() else "{}"
    evaluations = sorted(
        (
            a
            for a in index.artifacts
            if a.kind == Kind.CERTIFICATE.value and a.stamp in artifact.cited_by
        ),
        key=lambda a: a.stamp,
    )
    paths = [f"{root}/facts"]
    panes: list[Container | View] = []
    with _AsDefault(rr_):
        if robot is not None:
            shown = _present_robot(project, robot, rr_, root=f"{root}/robot")
            paths += shown["paths"]
            panes.append(rrb.Spatial3DView(origin=f"{root}/robot", name=robot.stamp))
        lines = [f"# {artifact.stamp}", "", "```json", text, "```", _lineage(artifact)]
        if evaluations:
            rates = []
            lines += ["", "## evaluations (bars: those with a recorded rate)", ""]
            for e in evaluations:
                success = str(e.summary.get("success", ""))
                k, _, n = success.partition("/")
                name = e.stamp.split("@", 1)[0]
                judged = e.summary.get("judged at", "")
                try:
                    rates.append(float(k) / float(n))
                    lines.append(f"- {name}: **{success}** · {judged}")
                except (ValueError, ZeroDivisionError):
                    lines.append(f"- {name}: rate unrecorded · {judged}")
            if rates:
                rr_.log(f"{root}/evaluations", rr.BarChart(rates), static=True)
                paths.append(f"{root}/evaluations")
        else:
            lines += ["", "_no evaluation of this policy in the project yet_"]
        rr_.log(f"{root}/facts", _doc("\n".join(lines)), static=True)
    right: list[Container | View] = [
        rrb.TextDocumentView(origin=f"{root}/facts", name="policy")
    ]
    if evaluations:
        right.insert(
            0,
            rrb.BarChartView(
                origin=f"{root}/evaluations", name="success rate per evaluation"
            ),
        )
    panes.append(rrb.Vertical(*right))
    return {
        "paths": paths,
        "view": "3D + evaluations" if robot is not None else "evaluations",
        "layout": rrb.Horizontal(
            *panes, column_shares=[3, 2] if robot is not None else None
        ),
    }


def _present_finding(
    project: Project, artifact: Artifact, rr_: rr.RecordingStream, root: str = "finding"
) -> Shown:
    """A finding: its claim and record as a document, and its outcome by
    condition as bars (success rate per condition) when it has one."""
    import rerun as rr  # noqa: PLC0415
    import rerun.blueprint as rrb  # noqa: PLC0415

    from trainnr.project.details import outcome_of  # noqa: PLC0415

    raw = read_json(project.root / artifact.path)
    outcome = outcome_of(raw.get("outcome"))
    arms = outcome.get("arms") if isinstance(outcome, dict) else None
    lines = [f"# {raw.get('id', artifact.stamp)}", "", str(raw.get("claim", "")), ""]
    for key in ("date", "repo_commit", "instrument", "protocol", "argv"):
        if key in raw:
            lines.append(f"- {key}: `{raw[key]}`")
    rates = []
    if isinstance(arms, dict):
        lines += ["", "## outcome by condition", ""]
        for name, arm in arms.items():
            if isinstance(arm, dict) and "successes" in arm and "trials" in arm:
                k, n = arm["successes"], arm["trials"]
                if n:
                    rates.append(k / n)
                    lines.append(f"- {name}: **{k} / {n}**")
                else:
                    lines.append(f"- {name}: no trials recorded")
    if raw.get("caveats"):
        lines += ["", "## caveats", ""] + [f"- {c}" for c in raw["caveats"]]
    with _AsDefault(rr_):
        rr_.log(f"{root}/reading", _doc("\n".join(lines)), static=True)
        if rates:
            rr_.log(f"{root}/outcome", rr.BarChart(rates), static=True)
    paths = [f"{root}/reading"] + ([f"{root}/outcome"] if rates else [])
    layout = (
        rrb.Horizontal(
            rrb.BarChartView(
                origin=f"{root}/outcome", name="success rate by condition"
            ),
            rrb.TextDocumentView(origin=f"{root}/reading", name="finding"),
        )
        if rates
        else rrb.TextDocumentView(origin=f"{root}/reading", name="finding")
    )
    return {
        "paths": paths,
        "view": "outcome + reading" if rates else "reading",
        "layout": layout,
    }


def _present_batch(
    project: Project, artifact: Artifact, rr_: rr.RecordingStream, root: str = "batch"
) -> Shown:
    """A generated dataset: its successful episodes' frames on a tick
    timeline, its datasheet as the document beside them."""
    import rerun as rr  # noqa: PLC0415
    import rerun.blueprint as rrb  # noqa: PLC0415

    folder = project.root / artifact.path
    episodes = sorted(folder.glob("episode_*"))
    with _AsDefault(rr_):
        for k, episode in enumerate(episodes):
            frames = sorted(episode.glob("frames/*.jpg")) or sorted(
                episode.glob("frames/*/*.jpg")
            )
            stride = max(1, len(frames) // FRAMES_PER_EPISODE)
            for i, frame in enumerate(frames[::stride]):
                rr_.set_time("episode", sequence=k)
                rr_.set_time("frame", sequence=i)
                rr_.log(f"{root}/camera", rr.EncodedImage(path=frame))
            manifest = episode / EPISODE_MANIFEST
            if manifest.is_file():
                rr_.set_time("episode", sequence=k)
                rr_.log(
                    f"{root}/manifest", _doc(f"```json\n{read_text(manifest)}\n```")
                )
        datasheet = folder / DATASHEET_FILE
        rr_.log(
            f"{root}/datasheet",
            _doc(read_text(datasheet) if datasheet.is_file() else "no datasheet"),
            static=True,
        )
    return {
        "paths": [f"{root}/camera", f"{root}/datasheet"],
        "view": "frames + datasheet",
        "layout": rrb.Horizontal(
            rrb.Spatial2DView(origin=f"{root}/camera", name="successful episodes"),
            rrb.Vertical(
                rrb.TextDocumentView(origin=f"{root}/datasheet", name="datasheet"),
                rrb.TextDocumentView(
                    origin=f"{root}/manifest", name="episode manifest"
                ),
            ),
            column_shares=[2, 1],
        ),
    }


def _present_dataset(
    project: Project, artifact: Artifact, rr_: rr.RecordingStream, root: str = "dataset"
) -> Shown:
    """A LeRobot dataset: its videos as video assets, its provenance."""
    import rerun as rr  # noqa: PLC0415
    import rerun.blueprint as rrb  # noqa: PLC0415

    folder = project.root / artifact.path
    videos = sorted(folder.glob("videos/*/chunk-*/*.mp4"))
    paths = []
    with _AsDefault(rr_):
        for video in videos[:MAX_VIDEOS]:
            camera = video.relative_to(folder / "videos").parts[0]
            path = f"{root}/{camera}"
            paths.append(path)
            rr_.log(path, rr.AssetVideo(path=video), static=True)
        prov = folder / PROVENANCE_FILE
        prov_text = read_text(prov) if prov.is_file() else "{}"
        rr_.log(
            f"{root}/provenance",
            _doc(f"# {artifact.stamp}\n\n```json\n{prov_text}\n```"),
            static=True,
        )
    views = [
        rrb.Spatial2DView(origin=p, name=p.split("/", 1)[1])
        for p in dict.fromkeys(paths)
    ]
    return {
        "paths": paths,
        "view": "video + provenance",
        "layout": rrb.Horizontal(
            rrb.Grid(*views)
            if views
            else rrb.TextDocumentView(origin=f"{root}/provenance"),
            rrb.TextDocumentView(origin=f"{root}/provenance", name="provenance"),
            column_shares=[2, 1],
        ),
    }


def _present_task(
    project: Project, artifact: Artifact, rr_: rr.RecordingStream, root: str = "task"
) -> Shown:
    """A task: its registered spec rendered as a document, and — when the
    task builds — its scene's spawn bands as boxes in 3D over the model."""
    import mujoco  # noqa: PLC0415
    import rerun as rr  # noqa: PLC0415
    import rerun.blueprint as rrb  # noqa: PLC0415

    folder = project.root / artifact.path
    ref = read_json(folder / TASK_FILE)
    task_id = ref.get("task_id", "")
    doc = f"# {task_id}\n\n- version `{ref.get('stamp')}` · {ref.get('kind')}\n"
    paths = [f"{root}/spec"]
    with _AsDefault(rr_):
        try:
            from trainnr.tasks.overlay import build_from_reference  # noqa: PLC0415

            task = build_from_reference(ref)
            spec = getattr(task, "task_spec", None)
            if spec is not None:
                from dataclasses import asdict  # noqa: PLC0415

                fields = asdict(spec)
                doc += (
                    "\n## spec\n\n```json\n"
                    + json.dumps(fields, indent=1, default=str)
                    + "\n```\n"
                )
                spawn = fields.get("part_spawn") or {}
                for arm, band in spawn.items():
                    (x0, x1), (y0, y1) = band
                    rr_.log(
                        f"{root}/spawn/{arm}",
                        rr.Boxes3D(
                            centers=[[(x0 + x1) / 2, (y0 + y1) / 2, 0.02]],
                            half_sizes=[[(x1 - x0) / 2, (y1 - y0) / 2, 0.005]],
                            labels=[f"{arm} spawn band"],
                        ),
                        static=True,
                    )
                    paths.append(f"{root}/spawn/{arm}")
            # The task carries its scene as an MjSpec: compile it and draw
            # the rig, table and parts at the home pose (until 2026-09-09
            # this looked for a `model` attribute no task has, and the
            # view showed the spawn bands on an empty grid).
            model = getattr(task, "model", None)
            data = getattr(task, "data", None)
            if model is None and getattr(task, "spec", None) is not None:
                model = task.spec.compile()
                data = mujoco.MjData(model)
                if model.nkey > 0:
                    mujoco.mj_resetDataKeyframe(model, data, 0)
                mujoco.mj_forward(model, data)
            if model is not None and data is not None:
                from trainnr.viz import RigMirror  # noqa: PLC0415

                RigMirror(model, model_colors=True).log(
                    data, path=f"{root}/scene", static=True
                )
                paths.append(f"{root}/scene")
        except Exception as why:  # a task that will not build still shows its reference
            doc += f"\n_scene not rendered: {why}_\n"
        rr_.log(f"{root}/spec", _doc(doc), static=True)
    has_scene = any(
        p.startswith(f"{root}/scene") or p.startswith(f"{root}/spawn") for p in paths
    )
    return {
        "paths": paths,
        "view": "3D + spec" if has_scene else "spec",
        "layout": rrb.Horizontal(
            rrb.Spatial3DView(origin=root, name=task_id),
            rrb.TextDocumentView(origin=f"{root}/spec", name="spec"),
            column_shares=[2, 1],
        )
        if has_scene
        else rrb.TextDocumentView(origin=f"{root}/spec", name=task_id),
    }


def _present_certificate(
    project: Project,
    artifact: Artifact,
    rr_: rr.RecordingStream,
    root: str = "certificate",
) -> Shown:
    """An evaluation: funnel bars, the interval, every input version."""
    import rerun as rr  # noqa: PLC0415
    import rerun.blueprint as rrb  # noqa: PLC0415

    folder = project.root / artifact.path
    cert = read_json(folder / CERTIFICATE_FILE)
    with _AsDefault(rr_):
        funnel = cert.get("funnel") or {}
        counts = [v for v in funnel.values() if isinstance(v, (int, float))]
        if counts:
            rr_.log(f"{root}/funnel", rr.BarChart(counts), static=True)
        lines = [f"# {artifact.stamp}", ""]
        if ratio_of(cert) != UNRECORDED:
            lines.append(f"**{ratio_of(cert)}** trials succeeded")
        interval = interval_of(cert)
        if interval is not None:
            lines.append(f"95% CI **[{interval[0]:.2f}, {interval[1]:.2f}]**")
        if funnel:
            # A bar chart has no category labels: the reading names the
            # bars left to right, in the order they were logged.
            lines += ["", "funnel (the bars, left to right):"] + [
                f"{i}. {name}: {v}" for i, (name, v) in enumerate(funnel.items(), 1)
            ]
        lines += ["", "inputs:"] + [
            f"- {key}: `{cert.get(key)}`"
            for key in ("robot", "task", "policy", "source", "instrument", "protocol")
            if key in cert
        ]
        rr_.log(f"{root}/reading", _doc("\n".join(lines)), static=True)
    return {
        "paths": [f"{root}/funnel", f"{root}/reading"],
        "view": "funnel + reading",
        "layout": rrb.Horizontal(
            rrb.BarChartView(origin=f"{root}/funnel", name="funnel"),
            rrb.TextDocumentView(origin=f"{root}/reading", name="evaluation"),
        ),
    }


def _present_deploy(
    project: Project,
    artifact: Artifact,
    rr_: rr.RecordingStream,
    root: str = "deployment",
) -> Shown:
    """A deployment: the trained scene the policy ships with, posed at its
    home keyframe; the gate's error ratio per trial against the bound;
    the manifest's facts and the gate's verdict as a reading."""
    import mujoco  # noqa: PLC0415
    import rerun as rr  # noqa: PLC0415
    import rerun.blueprint as rrb  # noqa: PLC0415

    from trainnr.deploy.attribution import (  # noqa: PLC0415
        ladder_series,
        read_attribution,
    )
    from trainnr.deploy.gate import ERR_RATIO_BOUND  # noqa: PLC0415
    from trainnr.deploy.manifest import (  # noqa: PLC0415
        GATE_INSTRUMENTS,
        gate_word,
        load_manifest,
        read_gates,
    )
    from trainnr.deploy.runtime import assets_dir_of, load_scene  # noqa: PLC0415
    from trainnr.viz import RigMirror  # noqa: PLC0415

    folder = project.root / artifact.path
    manifest = load_manifest(folder)
    model = load_scene(manifest, assets_dir=assets_dir_of(manifest))
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    gates = read_gates(folder)
    with _AsDefault(rr_):
        RigMirror(model, model_colors=True).log(data, path=f"{root}/scene", static=True)
        unrated: dict[str, int] = {}
        for runtime, g in gates.items():
            # A trial with no recorded error ratio is left out of the bars
            # and counted, never drawn as a perfect zero.
            recorded = [t for t in g.get("records", []) if "err_ratio" in t]
            unrated[runtime] = len(g.get("records", [])) - len(recorded)
            ratios = [float(t["err_ratio"]) for t in recorded]
            if ratios:
                rr_.log(f"{root}/gate/{runtime}", rr.BarChart(ratios), static=True)
        control = manifest.control
        obs = manifest.raw.get("observations", [])
        lines = [f"# {artifact.stamp}", ""]
        for runtime, g in gates.items():
            interval = interval_of(g)
            lines.append(
                f"- gate in {GATE_INSTRUMENTS[runtime]} **{gate_word(g)}**: "
                f"{ratio_of(g)} trials"
                + (
                    f", 95% CI **[{interval[0]:.2f}, {interval[1]:.2f}]**"
                    if interval is not None
                    else ""
                )
                + (
                    f" ({unrated[runtime]} trials without a recorded error ratio)"
                    if unrated[runtime]
                    else ""
                )
                + (
                    ", reached the course's end in "
                    f"{sum(1 for t in g.get('records', []) if t.get('finished'))}"
                    if (g.get("protocol") or {}).get("course")
                    else ""
                )
            )
        if gates:
            # the judgment as each record states it: the plane's tracking
            # rule, or a staged scene's arrival (deploy/course.py)
            criteria = sorted(
                {
                    str((g.get("protocol") or {}).get("criterion", UNRECORDED))
                    for g in gates.values()
                }
            )
            lines.append(
                "bars: mean velocity error over the commanded speed per trial "
                f"(tracking bound {ERR_RATIO_BOUND}); judged by: "
                + " / ".join(criteria)
            )
        else:
            lines.append("sim-to-sim gate not run yet")
        attribution = read_attribution(folder)
        if attribution:  # what would break it first, its ladders as series
            lines.append(f"- **{attribution.get('sensitivity', UNRECORDED)}**")
            lower = float((attribution.get("certificate") or {}).get("lower", 0.0))
            for entry in attribution.get("knobs", []):
                for i, rung in enumerate(entry.get("rungs", []), start=1):
                    rr_.set_time("rung", sequence=i)
                    base = f"{root}/attribution/{entry['name']}"
                    for series, value in ladder_series(rung, lower).items():
                        rr_.log(f"{base}/{series}", rr.Scalars(value))
        lines += [
            "",
            f"- checkpoint `{manifest.raw.get('checkpoint', UNRECORDED)}` "
            f"· policy `{manifest.policy_path.name}`",
            f"- control {control.control_hz:g} Hz · physics "
            f"{control.physics_timestep_s:g} s, decimation {control.decimation}",
            f"- observations ({sum(int(o.get('width', 0)) for o in obs)}): "
            + ", ".join(f"{o.get('name')} {o.get('width')}" for o in obs),
            f"- scene `{manifest.scene_path.name}` · bodies {model.nbody} · joints "
            f"{model.njnt} · actuators {model.nu}",
            _lineage(artifact),
        ]
        rr_.log(f"{root}/reading", _doc("\n".join(lines)), static=True)
    return {
        "paths": [f"{root}/scene", f"{root}/gate", f"{root}/reading"]
        + ([f"{root}/attribution"] if attribution else []),
        "view": "scene + gate + reading" + (" + attribution" if attribution else ""),
        "layout": rrb.Horizontal(
            rrb.Spatial3DView(origin=f"{root}/scene", name=artifact.stamp),
            rrb.Vertical(
                *[
                    rrb.BarChartView(
                        origin=f"{root}/gate/{runtime}",
                        name=f"gate in {GATE_INSTRUMENTS[runtime]}: "
                        "error ratio per trial",
                    )
                    for runtime in gates
                ],
                *(
                    [
                        rrb.TimeSeriesView(
                            origin=f"{root}/attribution",
                            name="what would break it first: rate up each ladder",
                        )
                    ]
                    if attribution
                    else []
                ),
                rrb.TextDocumentView(origin=f"{root}/reading", name="deployment"),
            ),
            column_shares=[3, 2],
        ),
    }


def _present_drift(
    project: Project, artifact: Artifact, rr_: Any, root: str = "drift"
) -> Shown:
    """A drift check: each judged parameter's shift against its reference
    (in reference half-widths, so ±1 is the interval's edge) as bars, and
    the reading with the verdict and the lineage."""
    import rerun as rr  # noqa: PLC0415
    import rerun.blueprint as rrb  # noqa: PLC0415

    from trainnr.fleet.drift import (  # noqa: PLC0415
        ANCHORED,
        DRIFT_FILE,
        load_drift_record,
        verdict_word,
    )

    d = load_drift_record(project.root / artifact.path / DRIFT_FILE)
    judged = [p for p in d.parameters if p.verdict != ANCHORED]
    with _AsDefault(rr_):
        shifts = [p.shift for p in judged if p.shift is not None]
        if shifts:
            rr_.log(f"{root}/shift", rr.BarChart(shifts), static=True)
        lines = [
            f"# {artifact.stamp}",
            "",
            f"**{verdict_word(d.drifted)}** — {d.recommendation}",
            "",
            "bars: each parameter's fresh estimate against its reference "
            "interval, in reference half-widths (±1 is the edge); order: "
            + ", ".join(p.name for p in judged if p.shift is not None),
            "",
        ]
        lines += [
            f"- {p.name}: **{p.verdict}** · fresh "
            f"{interval_text(p.fresh_lower, p.fresh_upper)} · reference "
            f"{interval_text(p.reference_lower, p.reference_upper)}"
            + (f" · {p.note}" if p.note else "")
            for p in d.parameters
        ]
        lines += ["", f"method `{d.method}` · {d.references} reference record(s)"]
        lines.append(_lineage(artifact))
        rr_.log(f"{root}/reading", _doc("\n".join(lines)), static=True)
    return {
        "paths": [f"{root}/shift", f"{root}/reading"],
        "view": "shift + reading",
        "layout": rrb.Horizontal(
            rrb.BarChartView(origin=f"{root}/shift", name="shift per parameter"),
            rrb.TextDocumentView(origin=f"{root}/reading", name="drift check"),
        ),
    }


def _present_scene(
    project: Project, artifact: Artifact, rr_: Any, root: str = "scene"
) -> Shown:
    """A captured scene: the splat as Rerun's own splat archetype beside
    the collision proxy as a translucent mesh, so the gap is a picture;
    the record's facts as a reading."""
    import rerun as rr  # noqa: PLC0415
    import rerun.blueprint as rrb  # noqa: PLC0415

    from trainnr.scenes.record import (  # noqa: PLC0415
        PROXY_FILE,
        SCENE_FILE,
        SPLAT_FILE,
        load_scene_record,
    )
    from trainnr.scenes.splat import VISIBLE_OPACITY, read_ply  # noqa: PLC0415

    folder = project.root / artifact.path
    s = load_scene_record(folder / SCENE_FILE)
    splats = read_ply(folder / SPLAT_FILE)
    # the visible set (the audit's), so a capture's faint giant gaussians
    # (Brush leaves metres-wide ones at opacity 0.05) do not hide the scene
    drawn = splats.visible(VISIBLE_OPACITY)
    with _AsDefault(rr_):
        rr_.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr_.log(
            f"{root}/splat",
            gaussians(rr, drawn),
            static=True,
        )
        proxy = _obj_mesh(folder / PROXY_FILE)
        if proxy is not None:
            vertices, faces = proxy
            rr_.log(
                f"{root}/proxy",
                rr.Mesh3D(
                    vertex_positions=vertices,
                    triangle_indices=faces,
                    albedo_factor=[90, 160, 255, 90],
                ),
                static=True,
            )
        g = s.gap
        lines = [
            f"# {artifact.stamp}",
            "",
            f"source `{s.source}` · {s.splat.get('count')} gaussians, "
            f"{drawn.means.shape[0]} drawn (opacity ≥ {VISIBLE_OPACITY:g}) · "
            f"proxy {s.proxy.get('faces')} faces",
            "",
            "**the gap** (visible surface against the collision proxy): "
            + _gap_line(g),
            "",
        ]
        lines += [
            f"- {p.name}: {p.value} · **{p.basis}**"
            + (f" ±{p.span:g}" if p.span is not None else "")
            + (f" [{p.interval[0]:.4g}, {p.interval[1]:.4g}]" if p.interval else "")
            for p in s.physics
        ]
        lines.append(_lineage(artifact))
        rr_.log(f"{root}/reading", _doc("\n".join(lines)), static=True)
        # the trainer's own renders of the splat, the capture's views
        renders = [folder / r for r in s.splat.get("renders") or []]
        renders = [r for r in renders if r.is_file()]
        for i, still in enumerate(renders):
            rr_.log(f"{root}/renders/{i:02d}", rr.EncodedImage(path=still), static=True)
    side: list[Any] = [rrb.TextDocumentView(origin=f"{root}/reading", name="scene")]
    if renders:
        side.append(
            rrb.Grid(
                *[
                    rrb.Spatial2DView(
                        origin=f"{root}/renders/{i:02d}", name=f"view {i}"
                    )
                    for i in range(len(renders))
                ],
                name="renders",
            )
        )
    return {
        "paths": [f"{root}/splat", f"{root}/proxy", f"{root}/reading"]
        + [f"{root}/renders/{i:02d}" for i in range(len(renders))],
        "view": "splat + proxy + reading" + (" + renders" if renders else ""),
        "layout": rrb.Horizontal(
            rrb.Spatial3DView(origin="/", name=artifact.stamp),
            rrb.Vertical(*side, row_shares=[1, 2] if renders else [1]),
            column_shares=[3, 2] if renders else [3, 1],
        ),
    }


def _gap_line(g: Any) -> str:
    """The four numbers in one line, or why there are none."""
    if not isinstance(g.p95_m, (int, float)):
        return f"not measured ({g.note})"
    return (
        f"chamfer {g.chamfer_m * 100:.2f} cm · p95 {g.p95_m * 100:.2f} cm · "
        f"{g.beyond_tolerance_fraction * 100:.1f} % of the visible surface "
        f"beyond {g.tolerance_m * 100:.0f} cm of any collider · "
        f"{g.hidden_fraction * 100:.1f} % of the proxy unseen"
    )


def _flush(recording: Any, what: str) -> None:
    """Wait for the viewer to take the whole show; past `FLUSH_S` the
    refusal says what happened - the viewer is busy, what arrived is
    shown, the rest may be missing - not a bare SDK line."""
    try:
        recording.flush(timeout_sec=FLUSH_S)
    except Exception as why:
        raise RuntimeError(
            f"the viewer did not confirm the whole of {what} within {FLUSH_S:g} s "
            f"(live streams share it; what arrived is shown, the rest may be "
            f"missing): {why}"
        ) from why


def _obj_mesh(path: Path) -> tuple[Any, Any] | None:
    """A Wavefront OBJ's triangles, or None when the file is not there."""
    from trainnr.scenes.obj import read_obj  # noqa: PLC0415

    if not path.is_file():
        return None
    try:
        return read_obj(path)
    except ValueError:  # a file with no faces draws nothing
        return None


_PRESENTERS: dict[Kind, Presenter] = {
    Kind.SCENE: _present_scene,
    Kind.DRIFT: _present_drift,
    Kind.ROBOT: _present_robot,
    Kind.DEPLOY: _present_deploy,
    Kind.RECORDING: _present_recording,
    Kind.RUN: _present_run,
    Kind.BATCH: _present_batch,
    Kind.DATASET: _present_dataset,
    Kind.TASK: _present_task,
    Kind.CERTIFICATE: _present_certificate,
    Kind.POLICY: _present_policy,
    Kind.FINDING: _present_finding,
}


# -- helpers -------------------------------------------------------------------


class _AsDefault:
    """Make a RecordingStream the global default so `rr.log`-style calls
    inside helpers (RigMirror) land in it, then restore the previous one."""

    def __init__(self, stream: rr.RecordingStream) -> None:
        self.stream = stream

    def __enter__(self) -> rr.RecordingStream:
        import rerun as rr  # noqa: PLC0415

        # The setter returns the stream it replaced (or None when there was
        # none); it cannot take None back, so restore only what was there.
        self._prev = rr.set_global_data_recording(self.stream)
        return self.stream

    def __exit__(self, *exc: object) -> None:
        import rerun as rr  # noqa: PLC0415

        if self._prev is not None:
            rr.set_global_data_recording(self._prev)


def _recording_id(name: str) -> str:
    """Rerun wants a recording id it can put in a path: the stamp's own
    characters, with `@`, `#` and spaces made safe."""
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in name)


def _doc(markdown: str) -> rr.TextDocument:
    import rerun as rr  # noqa: PLC0415

    return rr.TextDocument(markdown, media_type=rr.MediaType.MARKDOWN)


def _lineage(artifact: Artifact) -> str:
    if not artifact.cites:
        return ""
    return "\n## lineage\n\n" + "".join(
        f"- {k}: `{v}`\n" for k, v in artifact.cites.items()
    )
