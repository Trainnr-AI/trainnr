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

import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rq_pipeline.collect.provenance import PROVENANCE_FILE
from rq_pipeline.deploy.manifest import MANIFEST_FILE as DEPLOY_FILE
from rq_pipeline.envs.lerobot_train_log import CHAIN_LOG_FILE
from rq_pipeline.envs.rsl_rl_log import COL_ITERATION, COL_REWARD, TRAINING_FILE
from rq_pipeline.project.details import outcome_of
from rq_pipeline.project.files import read_json, read_text
from rq_pipeline.project.index import ProjectIndex, interval_of
from rq_pipeline.project.kinds import (
    CERTIFICATE_FILE,
    IDENTITY_FILE,
    POLICY_FILE,
    TASK_FILE,
)
from rq_pipeline.project.locate import INDEX_DIR, Project

if TYPE_CHECKING:
    import mujoco
    from PIL.Image import Image as PilImage
    from PIL.ImageDraw import ImageDraw
    from PIL.ImageFont import FreeTypeFont
    from PIL.ImageFont import ImageFont as ImageFontType

PREVIEWS_DIR = "previews"
# Rerun's welcome-screen cards are 337x250 at 1x; we render at 2x for
# Retina and let the card scale down.
PREVIEW_SIZE = (674, 500)
ROBOT_CAMERA = {"distance_scale": 2.2, "azimuth": 135.0, "elevation": -20.0}
SCENE_CAMERA = {"distance_scale": 1.3, "azimuth": 90.0, "elevation": -35.0}
LOSS_LINE = re.compile(r"\bloss:\s*([0-9.eE+-]+)")
MIN_CURVE_POINTS = 2  # a line needs two
# The preview palette, named once: the ground, the series in order, the
# faint reference line, the text.
GROUND = (24, 26, 31)
# The card is drawn at PREVIEW_SIZE and shown at about half that (a 337 pt
# card), so a size here is roughly twice what the eye gets: 28 px is the
# floor for anything a person must read (2026-09-28, after the drift card
# "how is this legible"). One number, one bar, at most three short lines.
PT_DISPLAY = 110  # the one number a card leads with
PT_WORD = 84  # a verdict word
PT_TITLE = 56  # a card's headline sentence
PT_BODY = 34  # a line the eye reads
PT_LABEL = 30  # a bar's label or a legend
CARD_MARGIN = 48
BAR_H = 26
ROW_STEP = 56  # a labelled bar row
SERIES = [(88, 166, 255), (255, 176, 88), (120, 220, 140), (230, 120, 200)]
REFERENCE_LINE = (60, 64, 72)
TEXT = (200, 205, 215)
PLOT_MARGIN = 40
# A tile is small: a recording's channel shows at most this many
# components here (the viewer draws up to `present.MAX_TRACES`).
TILE_TRACES = 8
# A renderer draws one artifact's picture into `out`; True when it did.
Renderer = Callable[[Project, Path, Path, dict[str, Any]], bool]


def preview_path(project: Project, stamp: str) -> Path:
    return project.root / INDEX_DIR / PREVIEWS_DIR / f"{stamp}.png"


def stale_preview(source: Path, out: Path) -> bool:
    """Whether the picture predates its artifact: a run stamped by its
    identity keeps one stamp for its whole life, and the sparkline drawn
    at its second iteration stood for the finished curve (go2-c2 showed
    "reward -1.1" over 1500 iterations, 2026-09-12)."""
    from rq_pipeline.project.index import file_times  # noqa: PLC0415

    times = file_times(source)
    return bool(times) and out.stat().st_mtime < max(times)


def write_previews(project: Project, index: ProjectIndex) -> dict[str, str]:
    """Write a preview for every artifact that has none yet or whose
    files changed since; return `{stamp: relative path}` for those that
    exist afterwards."""
    written: dict[str, str] = {}
    for artifact in index.artifacts:
        out = preview_path(project, artifact.stamp)
        source = project.root / artifact.path
        if not out.is_file() or stale_preview(source, out):
            renderer = _RENDERERS.get(artifact.kind)
            if renderer is None:
                continue
            try:
                ok = renderer(project, source, out, artifact.summary)
            except Exception:  # a preview is decoration; the index is not
                ok = False
            if not ok:
                continue
        written[artifact.stamp] = out.relative_to(project.root).as_posix()
    return written


# -- per kind -------------------------------------------------------------


def _render_robot(
    _project: Project, source: Path, out: Path, _summary: dict[str, Any]
) -> bool:
    """The bundle's model, posed at its keyframe or zero, offscreen."""
    try:
        import mujoco  # noqa: PLC0415
    except ImportError:
        return False
    model_file = _robot_model_file(source)
    if model_file is None:
        return False
    return _render_model(
        mujoco.MjModel.from_xml_path(str(model_file)), out, ROBOT_CAMERA
    )


def _render_task(
    _project: Project, source: Path, out: Path, _summary: dict[str, Any]
) -> bool:
    """The environment's scene — the registered task built and compiled —
    from a wider camera than a robot's, so the table and the parts read."""
    try:
        import mujoco  # noqa: PLC0415

        from rq_pipeline.tasks.overlay import build_from_reference  # noqa: PLC0415
    except ImportError:
        return False
    ref = read_json(source / TASK_FILE, missing_ok=True)
    task_id = ref.get("task_id")
    if not task_id:
        return False
    try:
        task = build_from_reference(ref)
        model = task.spec.compile()
    except Exception:  # an unbuildable task simply has no picture
        return False
    if not isinstance(model, mujoco.MjModel):
        return False
    return _render_model(model, out, SCENE_CAMERA)


def _render_model(
    model: mujoco.MjModel, out: Path, camera_spec: dict[str, float]
) -> bool:
    """One offscreen frame of a compiled model at its first keyframe (or
    zero), from a free camera placed by `camera_spec` around the model's
    own centre and extent."""
    import mujoco  # noqa: PLC0415

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
        camera.distance = model.stat.extent * camera_spec["distance_scale"]
        camera.azimuth = camera_spec["azimuth"]
        camera.elevation = camera_spec["elevation"]
        camera.lookat[:] = model.stat.center
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
    finally:
        renderer.close()
    return _save(np.asarray(pixels), out)


def _robot_model_file(source: Path) -> Path | None:
    """The MJCF to render: what the bundle record, the rig profile, or the
    largest-XML rule names (`bundles.bundle.model_file_of`)."""
    from rq_pipeline.bundles.bundle import model_file_of  # noqa: PLC0415

    return model_file_of(source)


def _render_batch(
    _project: Project, source: Path, out: Path, _summary: dict[str, Any]
) -> bool:
    """The first kept episode's first frame (the batch's own JPEG)."""
    frames = sorted(source.glob("episode_*/frames/*.jpg")) or sorted(
        source.glob("episode_*/frames/*/*.jpg")
    )
    if not frames:
        return False
    return _copy_scaled(frames[0], out)


def _render_dataset(
    project: Project, source: Path, out: Path, _summary: dict[str, Any]
) -> bool:
    """A dataset's frames live in video; its source batch, when the
    project still holds it, has the picture."""
    name = read_json(source / PROVENANCE_FILE, missing_ok=True).get("source")
    if not name:
        return False
    batch = project.batches / name
    if not batch.is_dir():
        return False
    return _render_batch(project, batch, out, {})


def _plot_lines(
    draw: ImageDraw,
    series: list[list[tuple[float, float]]],
    *,
    stroke: int = 4,
    zero_line: bool = False,
) -> None:
    """Every series on one shared x and y range inside the preview's
    margins — a shape, the way a platform's tile shows one; no axes."""
    width, height = PREVIEW_SIZE
    xs = [x for pts in series for x, _ in pts]
    ys = [y for pts in series for _, y in pts]
    if not xs:
        return
    x0, x1, lo, hi = min(xs), max(xs), min(ys), max(ys)
    xspan, yspan = (x1 - x0) or 1.0, (hi - lo) or 1.0
    inner_w, inner_h = width - 2 * PLOT_MARGIN, height - 2 * PLOT_MARGIN

    def at(x: float, y: float) -> tuple[float, float]:
        return (
            PLOT_MARGIN + (x - x0) / xspan * inner_w,
            height - PLOT_MARGIN - (y - lo) / yspan * inner_h,
        )

    if zero_line and lo < 0 < hi:
        y = at(x0, 0.0)[1]
        draw.line(
            [(PLOT_MARGIN, y), (width - PLOT_MARGIN, y)], fill=REFERENCE_LINE, width=2
        )
    for i, pts in enumerate(series):
        if len(pts) >= MIN_CURVE_POINTS:
            draw.line(
                [at(x, y) for x, y in pts],
                fill=SERIES[i % len(SERIES)],
                width=stroke,
                joint="curve",
            )


def _render_training_curve(source: Path, out: Path) -> bool:
    """The reward curve of an rsl_rl run from its training record: the one
    picture every RL platform shows for a run."""
    try:
        from PIL import Image, ImageDraw  # noqa: PLC0415
    except ImportError:
        return False
    record = read_json(source / TRAINING_FILE)
    columns = record.get("columns") or []
    if COL_REWARD not in columns or COL_ITERATION not in columns:
        return False
    it, rw = columns.index(COL_ITERATION), columns.index(COL_REWARD)
    points = [
        (float(row[it]), float(row[rw]))
        for row in record.get("curve", [])
        if len(row) > max(it, rw)
        and row[it] is not None
        and row[rw] is not None
        and np.isfinite(row[rw])
    ]
    if len(points) < MIN_CURVE_POINTS:
        return False
    image = Image.new("RGB", PREVIEW_SIZE, GROUND)
    draw = ImageDraw.Draw(image)
    _plot_lines(draw, [points], zero_line=True)
    # The label sits on a patch of ground so a curve that peaks early
    # never runs through it (go2-garden-c1, 2026-09-28).
    label = f"reward {points[-1][1]:.1f}"
    font = _font(PT_BODY)
    left, top, right, bottom = draw.textbbox((PLOT_MARGIN, 14), label, font=font)
    draw.rectangle([left - 10, top - 6, right + 10, bottom + 6], fill=GROUND)
    draw.text((PLOT_MARGIN, 14), label, fill=TEXT, font=font)
    return _save_pil(image, out)


def _render_run(
    _project: Project, source: Path, out: Path, _summary: dict[str, Any]
) -> bool:
    """An RL run's reward curve from its training record; an imitation
    run's loss curve from its chain log. A small line plot with no axes —
    a shape, the way a platform's run tile shows one."""
    if (source / TRAINING_FILE).is_file():
        return _render_training_curve(source, out)
    try:
        from PIL import Image, ImageDraw  # noqa: PLC0415
    except ImportError:
        return False
    log = source / CHAIN_LOG_FILE
    if not log.is_file():
        return False
    losses = [
        float(m.group(1))
        for m in (
            LOSS_LINE.search(line)
            for line in read_text(log, errors="replace").splitlines()
        )
        if m
    ]
    if len(losses) < MIN_CURVE_POINTS:
        return False
    image = Image.new("RGB", PREVIEW_SIZE, GROUND)
    draw = ImageDraw.Draw(image)
    _plot_lines(draw, [[(float(i), v) for i, v in enumerate(losses)]], stroke=6)
    return _save_pil(image, out)


def _render_recording(
    _project: Project, source: Path, out: Path, _summary: dict[str, Any]
) -> bool:
    """The recording's first channel as small traces — a signal's shape,
    the way a platform's log tile shows one. The adapter orders the
    channels; the first is its own choice of what leads."""
    try:
        from PIL import Image, ImageDraw  # noqa: PLC0415

        from rq_pipeline.robots.recording import Recording  # noqa: PLC0415
    except ImportError:
        return False
    recording = Recording.read(source)
    channel = next(iter(recording.channels.values()), None)
    if channel is None or len(channel.times) < MIN_CURVE_POINTS:
        return False
    image = Image.new("RGB", PREVIEW_SIZE, GROUND)
    draw = ImageDraw.Draw(image)
    values = channel.values.reshape(len(channel.times), -1)
    series = [
        [
            (float(t), float(v))
            for t, v in zip(channel.times, values[:, col], strict=True)
            if np.isfinite(v)
        ]
        for col in range(min(values.shape[1], TILE_TRACES))
    ]
    _plot_lines(draw, series)
    return _save_pil(image, out)


def _render_policy(
    project: Project, source: Path, out: Path, _summary: dict[str, Any]
) -> bool:
    """The robot the policy drives, from the project's own bundle (a
    checkpoint has no picture of its own; the robot it moves is the
    honest one), else nothing."""
    raw = read_json(source / POLICY_FILE, missing_ok=True) or read_json(
        source / IDENTITY_FILE, missing_ok=True
    )
    robot = str(raw.get("robot") or "")
    if "@" not in robot:
        return False
    bundle = project.robots / robot.split("@", 1)[0]
    if not bundle.is_dir():
        return False
    return _render_robot(project, bundle, out, {})


def _render_deploy(
    project: Project, source: Path, out: Path, _summary: dict[str, Any]
) -> bool:
    """A deployment's card is the robot it drives, from the project's or
    the library's bundle named by the manifest's robot version."""
    from rq_pipeline.bundles.locate import find_bundle  # noqa: PLC0415
    from rq_pipeline.deploy.attribution import STILL_FILE  # noqa: PLC0415

    if (source / STILL_FILE).is_file():  # the fall at the cliff, once pictured
        return _copy_scaled(source / STILL_FILE, out)
    robot = str(read_json(source / DEPLOY_FILE, missing_ok=True).get("robot") or "")
    bundle = find_bundle(robot.split("@", 1)[0]) if robot else None
    if bundle is None:
        return False
    return _render_robot(project, bundle, out, {})


Color = tuple[int, int, int]


def _bars(
    draw: ImageDraw, box: tuple[int, int, int, int], fraction: float, color: Color
) -> None:
    """A track and a filled share of it: `box` is (x, y, width, height)."""
    x, y, w, h = box
    draw.rounded_rectangle([x, y, x + w, y + h], radius=6, fill=(44, 48, 56))
    if fraction > 0:
        draw.rounded_rectangle(
            [x, y, x + max(12, int(w * min(1.0, fraction))), y + h],
            radius=6,
            fill=color,
        )


def _render_certificate(
    _project: Project, source: Path, out: Path, _summary: dict[str, Any]
) -> bool:
    """The evaluation at a glance: the success rate large, its exact
    interval as a bar on 0..1, the funnel as stacked bars."""
    try:
        from PIL import Image, ImageDraw  # noqa: PLC0415
    except ImportError:
        return False
    c = read_json(source / CERTIFICATE_FILE)
    k, n = c.get("successes"), c.get("trials")
    if k is None or not n:
        return False
    width, height = PREVIEW_SIZE
    image = Image.new("RGB", (width, height), GROUND)
    draw = ImageDraw.Draw(image)
    big, body, label = _font(PT_DISPLAY), _font(PT_BODY), _font(PT_LABEL)
    rate = k / n
    x0, x1 = CARD_MARGIN, width - CARD_MARGIN
    draw.text((x0, 28), f"{k} / {n}", fill=TEXT_BRIGHT, font=big)
    draw.text((x0, 152), f"{rate:.0%} success", fill=TEXT_DIM, font=body)
    interval = interval_of(c)
    if interval is not None:
        y = 224
        draw.rounded_rectangle([x0, y, x1, y + 16], radius=8, fill=(44, 48, 56))
        lo, hi = interval
        draw.rounded_rectangle(
            [x0 + int((x1 - x0) * lo), y - 3, x0 + int((x1 - x0) * hi), y + 19],
            radius=9,
            fill=(88, 166, 255),
        )
        px = x0 + int((x1 - x0) * rate)
        draw.ellipse([px - 10, y - 3, px + 10, y + 19], fill=TEXT_BRIGHT)
        draw.text(
            (x0, y + 30), f"95% CI [{lo:.2f}, {hi:.2f}]", fill=TEXT_DIM, font=label
        )
    funnel = c.get("funnel") or {}
    rows = [(k_, v) for k_, v in funnel.items() if isinstance(v, (int, float))]
    if rows:
        y = 316
        top = max(v for _, v in rows) or 1
        bar_x = x0 + 190
        for name, value in rows[:3]:
            draw.text((x0, y), name, fill=TEXT_DIM, font=label)
            _bars(
                draw,
                (bar_x, y + 6, x1 - bar_x - 96, BAR_H),
                value / top,
                (70, 200, 110),
            )
            draw.text((x1 - 84, y), f"{int(value)}", fill=TEXT_BRIGHT, font=label)
            y += ROW_STEP
    return _save_pil(image, out)


def _render_finding(
    _project: Project, source: Path, out: Path, _summary: dict[str, Any]
) -> bool:
    """A finding at a glance: when its outcome holds arms with successes
    over trials, a bar per arm; otherwise the claim, wrapped."""
    try:
        from PIL import Image, ImageDraw  # noqa: PLC0415
    except ImportError:
        return False
    raw = read_json(source)
    outcome = outcome_of(raw.get("outcome"))
    width, height = PREVIEW_SIZE
    image = Image.new("RGB", (width, height), GROUND)
    draw = ImageDraw.Draw(image)
    arms = (outcome or {}).get("arms") if isinstance(outcome, dict) else None
    rows = []
    if isinstance(arms, dict):
        for name, arm in arms.items():
            if isinstance(arm, dict) and "successes" in arm and "trials" in arm:
                rows.append((name, arm["successes"], arm["trials"]))
    if rows:
        x0, x1 = CARD_MARGIN, width - CARD_MARGIN
        body, label = _font(PT_BODY), _font(PT_LABEL)
        draw.text(
            (x0, 28), _cut(str(raw.get("id", "")), ID_CHARS), fill=TEXT_DIM, font=label
        )
        y = 92
        for name, k, n in rows[:ARM_ROWS]:
            # The label has its own column: an arm's name longer than it
            # ran under its bar ("expert-band-g", 2026-09-28).
            draw.text((x0, y), _cut(name, ARM_LABEL_CHARS), fill=TEXT_BRIGHT, font=body)
            _bars(
                draw,
                (ARM_BAR_X, y + 8, x1 - ARM_BAR_X - 112, BAR_H),
                k / n if n else 0,
                (88, 166, 255),
            )
            draw.text((x1 - 100, y), f"{k}/{n}", fill=TEXT_BRIGHT, font=body)
            y += ROW_STEP + 6
        if len(rows) > ARM_ROWS:
            draw.text(
                (x0, y), f"+{len(rows) - ARM_ROWS} more", fill=TEXT_DIM, font=label
            )
        return _save_pil(image, out)
    # A claim with no numbers to draw: its first sentence, large — the
    # headline — under the record's id; the rest waits in the drawer.
    draw.text(CARD_ORIGIN, raw.get("id", ""), fill=TEXT_DIM, font=_font(CARD_LABEL_PT))
    headline = _first_sentence(str(raw.get("claim", "")))
    y = HEADLINE_TOP
    for row in _wrap(headline, HEADLINE_CHARS)[:HEADLINE_LINES]:
        draw.text((CARD_ORIGIN[0], y), row, fill=TEXT_BRIGHT, font=_font(HEADLINE_PT))
        y += HEADLINE_LEADING
    return _save_pil(image, out)


ARM_LABEL_CHARS = 13  # an arm's name in a finding card's label column
ARM_BAR_X = 300  # where its bar starts
ARM_ROWS = 4  # bars a card holds; the rest is "+N more"
ID_CHARS = 34  # a record's id on one label line
HEADLINE_CHARS = 28  # characters per line at the headline size
HEADLINE_LINES = 5
HEADLINE_PT = 40
HEADLINE_LEADING = 58
HEADLINE_TOP = 92
CARD_ORIGIN = (CARD_MARGIN, 28)  # where a card's small label starts
CARD_LABEL_PT = PT_LABEL
TEXT_DIM = (160, 166, 178)
TEXT_BRIGHT = (236, 238, 242)


def _cut(text: str, chars: int) -> str:
    """`text` on one line of `chars`, cut with an ellipsis."""
    return text if len(text) <= chars else text[: chars - 1] + "…"


def _first_sentence(text: str) -> str:
    """Up to the first full stop that ends a sentence (not a decimal)."""
    for i, ch in enumerate(text):
        if ch in ".!?" and (i + 1 == len(text) or text[i + 1].isspace()):
            return text[: i + 1]
    return text


def _wrap(text: str, chars: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > chars:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    lines.append(line)
    return lines


def _font(size: int) -> FreeTypeFont | ImageFontType:
    """Pillow's bundled font at this size - the same face on every
    machine, no system font paths. A FreeType face where Pillow has
    FreeType; its bitmap default otherwise (Pillow's own signature)."""
    from PIL import ImageFont  # noqa: PLC0415

    return ImageFont.load_default(size=size)


def _finite_pair(lo: float | None, hi: float | None) -> tuple[float, float] | None:
    """Both ends known and finite, else nothing to draw."""
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
        return None
    return float(lo), float(hi)


def _render_drift(
    _project: Project, source: Path, out: Path, _summary: dict[str, Any]
) -> bool:
    """A drift check at a glance: the verdict word, one bar of the judged
    parameters split into out of interval, within and undetermined (each
    counted in the legend), then the names that left. Per-parameter bars
    drawn to their own scales said nothing at card size (2026-09-28)."""
    try:
        from PIL import Image, ImageDraw  # noqa: PLC0415
    except ImportError:
        return False
    from rq_pipeline.fleet.drift import (  # noqa: PLC0415
        ANCHORED,
        DRIFT_FILE,
        LEFT,
        UNRESOLVED,
        WITHIN,
        load_drift_record,
        verdict_word,
    )

    try:
        d = load_drift_record(source / DRIFT_FILE)
    except (OSError, ValueError, TypeError, KeyError):
        return False
    width, height = PREVIEW_SIZE
    image = Image.new("RGB", (width, height), GROUND)
    draw = ImageDraw.Draw(image)
    word, body, label = _font(PT_WORD), _font(PT_BODY), _font(PT_LABEL)
    x0, x1 = CARD_MARGIN, width - CARD_MARGIN
    ink = DRIFT_RED if d.drifted else TEXT_BRIGHT
    draw.text((x0, 24), verdict_word(d.drifted), fill=ink, font=word)
    judged = [p for p in d.parameters if p.verdict != ANCHORED]
    counts = {
        LEFT: sum(p.verdict == LEFT for p in judged),
        WITHIN: sum(p.verdict == WITHIN for p in judged),
        UNRESOLVED: sum(p.verdict == UNRESOLVED for p in judged),
    }
    total = sum(counts.values())
    if not total:
        draw.text((x0, 150), "no parameter judged", fill=TEXT_DIM, font=body)
        return _save_pil(image, out)
    # One bar: the judged parameters as shares, in the legend's order.
    y = 150
    x = x0
    for key, colour in DRIFT_COLOURS:
        share = counts[key] / total
        w = int((x1 - x0) * share)
        if w:
            draw.rectangle([x, y, x + w, y + BAR_H], fill=colour)
            x += w
    y += BAR_H + 22
    for key, colour in DRIFT_COLOURS:
        draw.rectangle([x0, y + 8, x0 + 18, y + 26], fill=colour)
        draw.text(
            (x0 + 32, y),
            f"{counts[key]} {DRIFT_LEGEND[key]}",
            fill=TEXT_DIM,
            font=label,
        )
        y += 40
    # Who left, by name: the reason the card is red.
    if d.left:
        y += 8
        for name in d.left[:DRIFT_NAMES]:
            draw.text(
                (x0, y), _cut(name, DRIFT_NAME_CHARS), fill=TEXT_BRIGHT, font=body
            )
            y += 42
        if len(d.left) > DRIFT_NAMES:
            draw.text(
                (x0, y), f"+{len(d.left) - DRIFT_NAMES} more", fill=TEXT_DIM, font=label
            )
    return _save_pil(image, out)


DRIFT_RED = (255, 107, 107)
DRIFT_BLUE = (88, 166, 255)
DRIFT_GREY = (70, 74, 84)
# The stacked bar's segments and legend, in order (`fleet.drift` words).
DRIFT_COLOURS = (
    ("left", DRIFT_RED),
    ("within", DRIFT_BLUE),
    ("unresolved", DRIFT_GREY),
)
DRIFT_LEGEND = {
    "left": "out of interval",
    "within": "within interval",
    "unresolved": "undetermined",
}
DRIFT_NAMES = 2  # names that left, on the card
DRIFT_NAME_CHARS = 30
TILE_SPLATS = 60_000  # points a tile draws; more is invisible at its size


def _extent_pair(extent: Any) -> tuple[list[float], list[float]] | None:
    """A recorded [[lo], [hi]] extent, both ends present, else None."""
    if isinstance(extent, list) and len(extent) == 2:  # noqa: PLR2004 - a pair
        return list(extent[0]), list(extent[1])
    return None


def _scene_still(source: Path, renders: list[str]) -> Path | None:
    """The trainer's own render of the splat, when it left any: the scene
    as a picture, the middle view of the capture."""
    present = [source / r for r in renders if (source / r).is_file()]
    return present[len(present) // 2] if present else None


def _render_scene(
    _project: Project, source: Path, out: Path, _summary: dict[str, Any]
) -> bool:
    """A scene from above: the visible gaussians as points in their own
    colour, the proxy's footprint as a box, the gap's headline number."""
    try:
        from PIL import Image, ImageDraw  # noqa: PLC0415
    except ImportError:
        return False
    from rq_pipeline.scenes.record import (  # noqa: PLC0415
        SCENE_FILE,
        SPLAT_FILE,
        load_scene_record,
    )
    from rq_pipeline.scenes.splat import read_ply  # noqa: PLC0415

    try:
        s = load_scene_record(source / SCENE_FILE)
    except (OSError, ValueError, TypeError, KeyError):
        return False
    still = _scene_still(source, s.splat.get("renders") or [])
    if still is not None:
        return _copy_scaled(still, out)
    try:
        splats = read_ply(source / SPLAT_FILE).visible()
    except (OSError, ValueError, TypeError, KeyError):
        return False
    width, height = PREVIEW_SIZE
    image = Image.new("RGB", (width, height), GROUND)
    draw = ImageDraw.Draw(image)
    extent = _extent_pair(s.proxy.get("extent_m"))
    xy = splats.means[:, :2]
    if extent:
        lo = np.array(extent[0][:2], dtype=np.float64) - 0.5
        hi = np.array(extent[1][:2], dtype=np.float64) + 0.5
    else:
        lo, hi = np.percentile(xy, 2, axis=0), np.percentile(xy, 98, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    keep = np.all((xy >= lo) & (xy <= hi), axis=1)
    pts, cols = xy[keep], (splats.colors[keep] * 255).astype(np.uint8)
    if pts.shape[0] > TILE_SPLATS:
        pick = np.random.default_rng(0).choice(pts.shape[0], TILE_SPLATS, replace=False)
        pts, cols = pts[pick], cols[pick]
    margin, top = 24, 140  # below the two text lines
    sx, sy = (width - 2 * margin) / span[0], (height - top - margin) / span[1]
    scale = min(sx, sy)
    px = (margin + (pts[:, 0] - lo[0]) * scale).astype(int)
    py = (height - margin - (pts[:, 1] - lo[1]) * scale).astype(int)
    pixels = image.load()
    if pixels is None:  # Pillow's stub: no pixel access for an image not loaded
        raise RuntimeError("the top-down tile has no pixel access")
    for x, y, c in zip(px, py, cols, strict=True):
        if 0 <= x < width and 0 <= y < height:
            pixels[x, y] = (int(c[0]), int(c[1]), int(c[2]))
    if extent:
        x0 = margin + (extent[0][0] - lo[0]) * scale
        x1 = margin + (extent[1][0] - lo[0]) * scale
        y0 = height - margin - (extent[1][1] - lo[1]) * scale
        y1 = height - margin - (extent[0][1] - lo[1]) * scale
        draw.rectangle([x0, y0, x1, y1], outline=(88, 166, 255), width=2)
    g = s.gap
    head = (
        f"gap p95 {g.p95_m * 100:.1f} cm"
        if isinstance(g.p95_m, (int, float))
        else "gap unmeasured"
    )
    # The headline's descenders reach ~y=72 at 48 px; the subtitle sits
    # below them (measured on the first scene tile, 2026-09-22).
    _scene_caption(draw, head, s.splat.get("count"))
    return _save_pil(image, out)


def _scene_caption(draw: ImageDraw, head: str, count: object) -> None:
    """The gap's number as the title, the gaussian count under it."""
    draw.text((CARD_MARGIN, 24), head, fill=TEXT_BRIGHT, font=_font(PT_TITLE))
    gaussians = (
        f"{count:,} gaussians" if isinstance(count, int) else "gaussians uncounted"
    )
    draw.text((CARD_MARGIN, 92), gaussians, fill=TEXT_DIM, font=_font(PT_LABEL))


_RENDERERS: dict[str, Renderer] = {
    "scene": _render_scene,
    "drift": _render_drift,
    "recording": _render_recording,
    "robot": _render_robot,
    "batch": _render_batch,
    "policy": _render_policy,
    "certificate": _render_certificate,
    "deploy": _render_deploy,
    "finding": _render_finding,
    "dataset": _render_dataset,
    "run": _render_run,
    "task": _render_task,
}


# -- writing ---------------------------------------------------------------


def _save(pixels: np.ndarray, out: Path) -> bool:
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        return False
    return _save_pil(Image.fromarray(pixels), out)


def _save_pil(image: PilImage, out: Path) -> bool:
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
    from PIL import ImageOps  # noqa: PLC0415

    with Image.open(frame) as opened:
        image = opened.convert("RGB")
    # Cover the card, cropped at the edges, never letterboxed: a 4:3 still
    # sat in the 337:250 card with a black band above it (2026-09-28).
    image = ImageOps.fit(image, PREVIEW_SIZE, method=Image.Resampling.LANCZOS)
    return _save_pil(image, out)
