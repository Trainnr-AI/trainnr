"""Present: stream one artifact into the Studio's embedded viewer, as itself.

The Studio can launch the viewer and index a project, but until now it
could not POINT the viewer at an artifact: a click showed a card. This is
the door docs/64 §3 named `show_in_studio` and never built. One function
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

import json
import time
from pathlib import Path
from typing import Any

from rq_pipeline.project.index import Artifact, ProjectIndex, index_project
from rq_pipeline.project.kinds import Kind
from rq_pipeline.project.locate import INDEX_DIR, Project

INTENT_FILE = "present.json"
POLL_S = 0.25
FLUSH_S = 10.0
# The channel components drawn per recording: all of them up to this many,
# so a 14-joint arm plots as fourteen lines and not one unreadable braid.
MAX_TRACES = 16
INTERVAL_ENDS = 2  # a confidence interval is two numbers
PAIR = 2  # a compare view holds two artifacts


def intent_path(project: Project) -> Path:
    return project.root / INDEX_DIR / INTENT_FILE


def present(
    project: Project, stamp: str, index: ProjectIndex | None = None
) -> dict[str, Any]:
    """Stream the artifact named by `stamp` into the viewer. Returns what
    was shown: the kind, the entity paths logged, the view it got."""
    import rerun as rr  # noqa: PLC0415 - viz extra

    from rq_pipeline.viz import STUDIO_ADDRESS  # noqa: PLC0415

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
        application_id=stamp, recording_id=_recording_id(stamp)
    )
    recording.connect_grpc(STUDIO_ADDRESS)
    try:
        shown = presenter(project, artifact, recording)
        recording.send_blueprint(rrb.Blueprint(shown["layout"]))
        recording.flush(timeout_sec=FLUSH_S)
    finally:
        recording.disconnect()
    return {
        "stamp": stamp,
        "kind": kind.value,
        "paths": shown["paths"],
        "view": shown["view"],
    }


def compare(
    project: Project, a: str, b: str, index: ProjectIndex | None = None
) -> dict[str, Any]:
    """Two artifacts side by side in one recording (`a` left, `b` right),
    each under its own entity root so two of a kind never collide."""
    import rerun as rr  # noqa: PLC0415 - viz extra
    import rerun.blueprint as rrb  # noqa: PLC0415

    from rq_pipeline.viz import STUDIO_ADDRESS  # noqa: PLC0415

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
        application_id=f"{a} vs {b}", recording_id=_recording_id(f"{a} vs {b}")
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
        recording.flush(timeout_sec=FLUSH_S)
    finally:
        recording.disconnect()
    return {
        "stamps": [a, b],
        "kinds": [h[2].value for h in halves],
        "paths": [p for s in shown for p in s["paths"]],
        "view": " | ".join(s["view"] for s in shown),
    }


def serve(project: Project, *, once: bool = False) -> None:
    """The presenter loop: watch the intent file, present, clear."""
    path = intent_path(project)
    while True:
        if path.is_file():
            stamp: str | None = None
            try:
                intent = json.loads(path.read_text(encoding="utf-8"))
                stamp = intent.get("stamp")
                stamps = intent.get("stamps") or []
                if len(stamps) == PAIR:
                    stamp = " vs ".join(stamps)
                    compare(project, *stamps)
                elif stamp:
                    present(project, stamp)
            except Exception as why:  # a bad intent must not kill the presenter
                _write_status(project, {"error": str(why), "stamp": stamp})
            else:
                _write_status(project, {"shown": stamp})
            path.unlink(missing_ok=True)
            if once:
                return
        time.sleep(POLL_S)


def _write_status(project: Project, status: dict[str, Any]) -> None:
    (project.root / INDEX_DIR / "present-status.json").write_text(json.dumps(status))


# -- per kind -------------------------------------------------------------------


def _present_robot(
    project: Project, artifact: Artifact, rr_: Any, root: str = "robot"
) -> dict[str, Any]:
    """The bundle's model as meshes in a 3D view, posed at its keyframe."""
    import mujoco  # noqa: PLC0415
    import rerun.blueprint as rrb  # noqa: PLC0415

    from rq_pipeline.project.previews import _robot_model_file  # noqa: PLC0415
    from rq_pipeline.viz import RigMirror  # noqa: PLC0415

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
    project: Project, artifact: Artifact, rr_: Any, root: str = "recording"
) -> dict[str, Any]:
    """Every channel as a time-series view on the recording's own clock."""
    import rerun as rr  # noqa: PLC0415
    import rerun.blueprint as rrb  # noqa: PLC0415

    from rq_pipeline.robots.recording import Recording  # noqa: PLC0415

    recording = Recording.read(project.root / artifact.path)
    views = []
    paths = []
    with _AsDefault(rr_):
        for name, channel in recording.channels.items():
            base = f"{root}/{name}"
            paths.append(base)
            labels = channel.components or tuple(str(i) for i in range(channel.width))
            values = channel.values.reshape(len(channel.times), -1)
            traces = list(enumerate(labels[:MAX_TRACES]))
            for _, label in traces:
                rr_.log(
                    f"{base}/{label}",
                    rr.SeriesLines(names=[f"{label} [{channel.unit}]"]),
                    static=True,
                )
            for t, row in zip(channel.times, values, strict=True):
                rr_.set_time("time", duration=float(t))
                for col, label in traces:
                    rr_.log(f"{base}/{label}", rr.Scalars(float(row[col])))
            views.append(
                rrb.TimeSeriesView(origin=base, name=f"{name} [{channel.unit}]")
            )
        rr_.log(
            f"{root}/manifest",
            _doc(
                f"# {artifact.stamp}\n\n"
                f"- source `{recording.source}` via `{recording.adapter}`"
                f" · {recording.duration_s:.1f} s · collected: {recording.collection}\n"
                f"- census: {json.dumps(recording.census)}\n"
                + "".join(f"- note: {n}\n" for n in recording.notes)
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
    project: Project, artifact: Artifact, rr_: Any, root: str = "run"
) -> dict[str, Any]:
    """A training run's curves from its chain log, plus its manifest."""
    import rerun as rr  # noqa: PLC0415
    import rerun.blueprint as rrb  # noqa: PLC0415

    from rq_pipeline.envs.lerobot_train_log import parse_train_line  # noqa: PLC0415

    folder = project.root / artifact.path
    log = folder / "chain.log"
    metrics: dict[str, list[tuple[int, float]]] = {}
    if log.is_file():
        for line in log.read_text(errors="replace").splitlines():
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
        manifest = folder / "run.json"
        text = manifest.read_text() if manifest.is_file() else "{}"
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


def _present_batch(
    project: Project, artifact: Artifact, rr_: Any, root: str = "batch"
) -> dict[str, Any]:
    """A pressed batch: its kept episodes' frames on a tick timeline, its
    datasheet as the document beside them."""
    import rerun as rr  # noqa: PLC0415
    import rerun.blueprint as rrb  # noqa: PLC0415

    folder = project.root / artifact.path
    episodes = sorted(folder.glob("episode_*"))
    with _AsDefault(rr_):
        for k, episode in enumerate(episodes):
            frames = sorted(episode.glob("frames/*.jpg")) or sorted(
                episode.glob("frames/*/*.jpg")
            )
            for i, frame in enumerate(frames[:: max(1, len(frames) // 60)]):
                rr_.set_time("episode", sequence=k)
                rr_.set_time("frame", sequence=i)
                rr_.log(f"{root}/camera", rr.EncodedImage(path=frame))
            manifest = episode / "manifest.json"
            if manifest.is_file():
                rr_.set_time("episode", sequence=k)
                rr_.log(
                    f"{root}/manifest", _doc(f"```json\n{manifest.read_text()}\n```")
                )
        datasheet = folder / "datasheet.md"
        rr_.log(
            f"{root}/datasheet",
            _doc(datasheet.read_text() if datasheet.is_file() else "no datasheet"),
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
    project: Project, artifact: Artifact, rr_: Any, root: str = "dataset"
) -> dict[str, Any]:
    """A LeRobot dataset: its videos as video assets, its provenance."""
    import rerun as rr  # noqa: PLC0415
    import rerun.blueprint as rrb  # noqa: PLC0415

    folder = project.root / artifact.path
    videos = sorted(folder.glob("videos/*/chunk-*/*.mp4"))
    paths = []
    with _AsDefault(rr_):
        for video in videos[:8]:
            camera = video.relative_to(folder / "videos").parts[0]
            path = f"{root}/{camera}"
            paths.append(path)
            rr_.log(path, rr.AssetVideo(path=video), static=True)
        prov = folder / "provenance.json"
        prov_text = prov.read_text() if prov.is_file() else "{}"
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
    project: Project, artifact: Artifact, rr_: Any, root: str = "task"
) -> dict[str, Any]:
    """A task: its registered spec rendered as a document, and — when the
    task builds — its scene's spawn bands as boxes in 3D over the model."""
    import mujoco  # noqa: PLC0415
    import rerun as rr  # noqa: PLC0415
    import rerun.blueprint as rrb  # noqa: PLC0415

    folder = project.root / artifact.path
    ref = json.loads((folder / "task.json").read_text())
    task_id = ref.get("task_id", "")
    doc = f"# {task_id}\n\n- stamp `{ref.get('stamp')}` · {ref.get('kind')}\n"
    paths = [f"{root}/spec"]
    with _AsDefault(rr_):
        try:
            from rq_pipeline.tasks.registry import resolve  # noqa: PLC0415

            entry = resolve(task_id)
            task = entry.build()
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
                from rq_pipeline.viz import RigMirror  # noqa: PLC0415

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
    project: Project, artifact: Artifact, rr_: Any, root: str = "certificate"
) -> dict[str, Any]:
    """A certificate: funnel bars, the interval, every input stamp."""
    import rerun as rr  # noqa: PLC0415
    import rerun.blueprint as rrb  # noqa: PLC0415

    folder = project.root / artifact.path
    cert = json.loads((folder / "certificate.json").read_text())
    with _AsDefault(rr_):
        funnel = cert.get("funnel") or {}
        counts = [v for v in funnel.values() if isinstance(v, (int, float))]
        if counts:
            rr_.log(f"{root}/funnel", rr.BarChart(counts), static=True)
        k, n = cert.get("successes"), cert.get("trials")
        ci = cert.get("ci95") or cert.get("ci") or []
        lines = [f"# {artifact.stamp}", ""]
        if k is not None and n:
            lines.append(f"**{k} / {n}** trials succeeded")
        if len(ci) == INTERVAL_ENDS:
            lines.append(f"exact 95 % interval **[{ci[0]:.2f}, {ci[1]:.2f}]**")
        if funnel:
            lines += ["", "funnel:"] + [f"- {name}: {v}" for name, v in funnel.items()]
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


_PRESENTERS = {
    Kind.ROBOT: _present_robot,
    Kind.RECORDING: _present_recording,
    Kind.RUN: _present_run,
    Kind.BATCH: _present_batch,
    Kind.DATASET: _present_dataset,
    Kind.TASK: _present_task,
    Kind.CERTIFICATE: _present_certificate,
}


# -- helpers -------------------------------------------------------------------


class _AsDefault:
    """Make a RecordingStream the global default so `rr.log`-style calls
    inside helpers (RigMirror) land in it, then restore the previous one."""

    def __init__(self, stream: Any) -> None:
        self.stream = stream

    def __enter__(self) -> Any:
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


def _doc(markdown: str) -> Any:
    import rerun as rr  # noqa: PLC0415

    return rr.TextDocument(markdown, media_type=rr.MediaType.MARKDOWN)


def _lineage(artifact: Artifact) -> str:
    if not artifact.cites:
        return ""
    return "\n## lineage\n\n" + "".join(
        f"- {k}: `{v}`\n" for k, v in artifact.cites.items()
    )
