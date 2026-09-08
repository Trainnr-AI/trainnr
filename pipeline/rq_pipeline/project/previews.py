"""A picture for every artifact that has one — written beside the index.

The index carries facts; a platform shows pictures. This module writes a
small PNG per artifact into `<project>/.index/previews/<stamp>.png` when
the artifact's kind has an honest picture:

- robot: the bundle's model rendered once, offscreen, from a fixed
  three-quarter camera — what the robot IS, not a stock silhouette.
- batch: the first kept episode's first camera frame, downscaled.
- dataset: the same, from its source batch when the project still holds
  it (a dataset's own frames are inside video files).
- run: the loss curve from the trainer's log, drawn small.
- recording / fit / certificate / others: none yet; the kinds whose
  writers are still to come (docs/76 §5 to §9) get previews with them.

Previews are a cache like the index: keyed by the artifact's stamp, so an
unchanged artifact is never re-rendered, and deleting `.index/` loses
nothing. Rendering needs the `sim` extra (MuJoCo, PIL); with neither
available the indexer says so once and writes the index without pictures.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from rq_pipeline.project.index import ProjectIndex
from rq_pipeline.project.locate import INDEX_DIR, Project

PREVIEWS_DIR = "previews"
# Rerun's welcome-screen cards are 337x250 at 1x; we render at 2x for
# Retina and let the card scale down.
PREVIEW_SIZE = (674, 500)
ROBOT_CAMERA = {"distance_scale": 2.2, "azimuth": 135.0, "elevation": -20.0}
LOSS_LINE = re.compile(r"\bloss:\s*([0-9.eE+-]+)")
MIN_CURVE_POINTS = 2  # a line needs two


def preview_path(project: Project, stamp: str) -> Path:
    return project.root / INDEX_DIR / PREVIEWS_DIR / f"{stamp}.png"


def write_previews(project: Project, index: ProjectIndex) -> dict[str, str]:
    """Write a preview for every artifact that has one and does not yet;
    return `{stamp: relative path}` for those that exist afterwards."""
    written: dict[str, str] = {}
    for artifact in index.artifacts:
        out = preview_path(project, artifact.stamp)
        if not out.is_file():
            source = project.root / artifact.path
            try:
                ok = _RENDERERS.get(artifact.kind, lambda *_: False)(
                    source, out, artifact.summary
                )
            except Exception:  # a preview is decoration; the index is not
                ok = False
            if not ok:
                continue
        written[artifact.stamp] = str(out.relative_to(project.root))
    return written


# -- per kind -------------------------------------------------------------


def _render_robot(source: Path, out: Path, _summary: dict[str, Any]) -> bool:
    """The bundle's model, posed at its keyframe or zero, offscreen."""
    try:
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError:
        return False
    model_file = _robot_model_file(source)
    if model_file is None:
        return False
    model = mujoco.MjModel.from_xml_path(str(model_file))
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    width, height = PREVIEW_SIZE
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.distance = model.stat.extent * ROBOT_CAMERA["distance_scale"]
        camera.azimuth = ROBOT_CAMERA["azimuth"]
        camera.elevation = ROBOT_CAMERA["elevation"]
        camera.lookat[:] = model.stat.center
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
    finally:
        renderer.close()
    return _save(np.asarray(pixels), out)


def _robot_model_file(source: Path) -> Path | None:
    """The MJCF to render: a profile's `model_file`, else the one XML at
    the root that is not an include-only fragment, else the largest."""
    profile = source / "profile.json"
    if profile.is_file():
        try:
            named = json.loads(profile.read_text()).get("model_file")
        except ValueError:
            named = None
        if named and (source / named).is_file():
            return source / named
    candidates = sorted(
        source.glob("*.xml"), key=lambda p: p.stat().st_size, reverse=True
    )
    return candidates[0] if candidates else None


def _render_batch(source: Path, out: Path, _summary: dict[str, Any]) -> bool:
    """The first kept episode's first frame (the batch's own JPEG)."""
    frames = sorted(source.glob("episode_*/frames/*.jpg")) or sorted(
        source.glob("episode_*/frames/*/*.jpg")
    )
    if not frames:
        return False
    return _copy_scaled(frames[0], out)


def _render_dataset(source: Path, out: Path, _summary: dict[str, Any]) -> bool:
    """A dataset's frames live in video; its source batch, when the
    project still holds it, has the picture."""
    try:
        provenance = json.loads((source / "provenance.json").read_text())
    except (OSError, ValueError):
        return False
    name = provenance.get("source")
    if not name:
        return False
    batch = source.parent.parent / "batches" / name
    if not batch.is_dir():
        return False
    return _render_batch(batch, out, {})


def _render_run(source: Path, out: Path, _summary: dict[str, Any]) -> bool:
    """The loss curve from the chain log, as a small line plot with no
    axes — a shape, the way a platform's run tile shows one."""
    try:
        from PIL import Image, ImageDraw  # noqa: PLC0415
    except ImportError:
        return False
    log = source / "chain.log"
    if not log.is_file():
        return False
    losses = [
        float(m.group(1))
        for m in (
            LOSS_LINE.search(line)
            for line in log.read_text(errors="replace").splitlines()
        )
        if m
    ]
    if len(losses) < MIN_CURVE_POINTS:
        return False
    width, height = PREVIEW_SIZE
    image = Image.new("RGB", (width, height), (24, 26, 31))
    draw = ImageDraw.Draw(image)
    lo, hi = min(losses), max(losses)
    span = (hi - lo) or 1.0
    margin = 40
    points = [
        (
            margin + i * (width - 2 * margin) / (len(losses) - 1),
            height - margin - (v - lo) / span * (height - 2 * margin),
        )
        for i, v in enumerate(losses)
    ]
    draw.line(points, fill=(88, 166, 255), width=6, joint="curve")
    return _save_pil(image, out)


def _render_recording(source: Path, out: Path, _summary: dict[str, Any]) -> bool:
    """The recording's first joint channel as small traces — a signal's
    shape, the way a platform's log tile shows one."""
    try:
        from PIL import Image, ImageDraw  # noqa: PLC0415

        from rq_pipeline.robots.recording import Recording  # noqa: PLC0415
    except ImportError:
        return False
    recording = Recording.read(source)
    channel = next(
        (c for n, c in recording.channels.items() if "position" in n or "ticks" in n),
        next(iter(recording.channels.values()), None),
    )
    if channel is None or len(channel.times) < MIN_CURVE_POINTS:
        return False
    width, height = PREVIEW_SIZE
    image = Image.new("RGB", (width, height), (24, 26, 31))
    draw = ImageDraw.Draw(image)
    values = channel.values.reshape(len(channel.times), -1)
    finite = values[np.isfinite(values).all(axis=1)] if values.size else values
    lo = float(np.nanmin(values)) if values.size else 0.0
    hi = float(np.nanmax(values)) if values.size else 1.0
    span = (hi - lo) or 1.0
    margin = 40
    t0, t1 = float(channel.times[0]), float(channel.times[-1])
    tspan = (t1 - t0) or 1.0
    palette = [(88, 166, 255), (255, 176, 88), (120, 220, 140), (230, 120, 200)]
    for col in range(min(values.shape[1], 8)):
        pts = [
            (
                margin + (float(t) - t0) / tspan * (width - 2 * margin),
                height - margin - (float(v) - lo) / span * (height - 2 * margin),
            )
            for t, v in zip(channel.times, values[:, col], strict=True)
            if np.isfinite(v)
        ]
        if len(pts) >= MIN_CURVE_POINTS:
            draw.line(pts, fill=palette[col % len(palette)], width=4, joint="curve")
    del finite
    return _save_pil(image, out)


_RENDERERS = {
    "recording": _render_recording,
    "robot": _render_robot,
    "batch": _render_batch,
    "dataset": _render_dataset,
    "run": _render_run,
}


# -- writing ---------------------------------------------------------------


def _save(pixels: Any, out: Path) -> bool:
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        return False
    return _save_pil(Image.fromarray(pixels), out)


def _save_pil(image: Any, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.with_suffix(".png.tmp")
    image.save(staging, format="PNG", optimize=True)
    staging.replace(out)
    return True


def _copy_scaled(frame: Path, out: Path) -> bool:
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        return False
    with Image.open(frame) as opened:
        image = opened.convert("RGB")
    image.thumbnail(PREVIEW_SIZE)
    return _save_pil(image, out)
