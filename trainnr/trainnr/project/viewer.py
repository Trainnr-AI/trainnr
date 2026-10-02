"""A saved viewer recording read back with no window (docs/76 §10.5).

Two readers, both Rerun's own. The command line that ships with the SDK
answers what a file holds — every entity path, timeline and component,
chunk counts, size — and needs nothing else. The values need Rerun's
local catalog, which needs DataFusion: the `viz-query` extra. With it a
description carries the columns; without it, the inventory and one line
saying what the values would need.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from trainnr.project.index import UNRECORDED
from trainnr.viz import viewer_files

STATS_ARGV = (sys.executable, "-m", "rerun", "rrd", "stats")
STATS_TIMEOUT_S = 60.0
# The stats page is sections: a title line, a dashed rule, `key: value`
# lines, a blank line. These titles carry the inventory.
ENTITIES = "Num chunks per entity"
TIMELINES = "Num chunks per index"
COMPONENTS = "Num chunks per component"
SIZE = "Size (schema + data, compressed)"
NEEDS_VIZ = "the viz extra (Rerun's SDK and command line): uv sync --extra viz"
NEEDS_QUERY = (
    "the values need the viz-query extra (Rerun's local catalog over DataFusion): "
    "uv sync --extra viz-query"
)
SCALARS_COMPONENT = "Scalars:scalars"
WALL_CLOCK = "log_time"  # Rerun's own wall-clock timeline, beside a feed's clock


@dataclass(frozen=True)
class SeriesFacts:
    """One scalar series on one clock, every component counted: `rows`
    is the rows with a value, `width` the components per row, `minimum`
    and `maximum` over every component, `last` the last row."""

    entity: str
    timeline: str
    rows: int
    width: int
    minimum: float | None
    maximum: float | None
    last: tuple[float, ...] | None


@dataclass(frozen=True)
class ViewerRecording:
    """What one saved stream holds, as far as this machine can read."""

    path: str
    bytes: int
    entities: dict[str, int] = field(default_factory=dict)
    timelines: dict[str, int] = field(default_factory=dict)
    components: dict[str, int] = field(default_factory=dict)
    size: dict[str, str] = field(default_factory=dict)
    rows_per_timeline: dict[str, int] = field(default_factory=dict)
    series: tuple[SeriesFacts, ...] = ()
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_stats(text: str) -> dict[str, dict[str, str]]:
    """The stats page as sections of `key: value`."""
    sections: dict[str, dict[str, str]] = {}
    lines = text.splitlines()
    current: dict[str, str] | None = None
    for i, line in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if line.strip() and set(nxt.strip()) == {"-"} and nxt.strip():
            current = sections.setdefault(line.strip(), {})
            continue
        if set(line.strip()) == {"-"}:
            continue
        if not line.strip():
            current = None
            continue
        if current is not None and ": " in line:
            key, _, value = line.partition(": ")
            current[key.strip()] = value.strip()
        elif current is not None and " = " in line:
            key, _, value = line.partition(" = ")
            current[key.strip()] = value.strip()
        elif current is not None and line.strip().endswith(":"):
            current[line.strip()[:-1]] = ""
    return sections


def _counts(section: dict[str, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in section.items():
        try:
            out[key] = int(value)
        except ValueError:
            continue
    return out


def stats(path: Path) -> dict[str, dict[str, str]]:
    """Rerun's own statistics for the file; refuses by name a file its
    reader cannot load."""
    if importlib.util.find_spec("rerun") is None:
        raise ValueError(f"{path}: reading a viewer recording needs {NEEDS_VIZ}")
    done = subprocess.run(
        [*STATS_ARGV, str(path)],
        capture_output=True,
        text=True,
        timeout=STATS_TIMEOUT_S,
        check=False,
    )
    if done.returncode != 0:
        raise ValueError(
            f"{path}: rerun rrd stats failed ({done.returncode}): "
            f"{done.stderr.strip()[-400:]}"
        )
    return parse_stats(done.stdout)


def query_available() -> bool:
    try:
        import datafusion  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def _column_name(column: Any) -> str:
    """The catalog's column name for a descriptor (`/gate/command:Scalars:scalars`)."""
    name = getattr(column, "name", None)
    if isinstance(name, str) and name:
        return name
    first = str(column).splitlines()[0]
    return first.replace("Column name: ", "").strip()


def _timeline_name(column: Any) -> str:
    """The timeline a catalog index column stands for (`Index(timeline:tick)`)."""
    name = getattr(column, "name", None)
    if isinstance(name, str) and name:
        return name
    return str(column).split("timeline:")[-1].rstrip(")")


def read_columns(path: Path) -> tuple[dict[str, int], tuple[SeriesFacts, ...]]:
    """Rows per clock and the scalar series per clock, through Rerun's
    local catalog over a folder holding a copy of this one file (the
    catalog reads real files, not links). Only the scalar entities are
    read: a gate's file carries its scene's meshes per tick, and reading
    every column of a 12 MB file took the reader past the machine's
    memory (killed, 2026-09-13)."""
    import shutil  # noqa: PLC0415

    from rerun.server import Server  # noqa: PLC0415

    path = Path(path).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy2(path, Path(tmp) / path.name)
        server = Server(host="127.0.0.1", datasets={"viewer": tmp})
        try:
            dataset = server.client().get_dataset("viewer")
            schema = dataset.schema()
            timelines = [_timeline_name(c) for c in schema.index_columns()]
            scalar_columns = [
                c
                for c in schema.component_columns()
                if str(getattr(c, "component", "")) == SCALARS_COMPONENT
                or _column_name(c).endswith(SCALARS_COMPONENT)
            ]
            if not timelines or not scalar_columns:
                return {}, ()
            entities = sorted(
                {
                    str(getattr(c, "entity_path", _column_name(c).split(":", 1)[0]))
                    for c in scalar_columns
                }
            )
            view = dataset.filter_contents(entities)
            # The feed's own clocks; the wall clock only when it is all there is.
            own = [t for t in timelines if t != WALL_CLOCK] or timelines
            rows: dict[str, int] = {}
            series: list[SeriesFacts] = []
            for timeline in own:
                table = view.reader(index=timeline).to_arrow_table()
                rows[timeline] = int(table.num_rows)
                for column in scalar_columns:
                    name = _column_name(column)
                    if name not in table.column_names:
                        continue
                    values = [
                        [float(x) for x in v]
                        for v in table.column(name).to_pylist()
                        if v is not None and len(v) > 0
                    ]
                    if not values:
                        continue  # not logged on this clock
                    flat = [x for row in values for x in row]
                    series.append(
                        SeriesFacts(
                            entity=str(
                                getattr(column, "entity_path", name.split(":", 1)[0])
                            ),
                            timeline=timeline,
                            rows=len(values),
                            width=len(values[0]),
                            minimum=min(flat),
                            maximum=max(flat),
                            last=tuple(values[-1]),
                        )
                    )
        finally:
            server.shutdown()
    return rows, tuple(series)


def describe(path: Path, *, values: bool = True) -> ViewerRecording:
    """One saved stream, as far as this machine can read it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no such viewer recording")
    sections = stats(path)
    notes: list[str] = []
    rows: dict[str, int] = {}
    series: tuple[SeriesFacts, ...] = ()
    if values:
        if query_available():
            try:
                rows, series = read_columns(path)
            except Exception as why:  # a reader's own failure is a note, not a crash
                notes.append(f"columns unreadable: {why!r}"[:300])
        else:
            notes.append(NEEDS_QUERY)
    return ViewerRecording(
        path=str(path),
        bytes=path.stat().st_size,
        entities=_counts(sections.get(ENTITIES, {})),
        timelines=_counts(sections.get(TIMELINES, {})),
        components=_counts(sections.get(COMPONENTS, {})),
        size=dict(sections.get(SIZE, {})),
        rows_per_timeline=rows,
        series=series,
        notes=tuple(notes),
    )


def describe_folder(folder: Path, *, values: bool = True) -> list[dict[str, Any]]:
    """Every saved stream inside an artifact's folder, described; a file
    the reader refuses is one entry naming why."""
    out: list[dict[str, Any]] = []
    for path in viewer_files(folder):
        try:
            out.append(describe(path, values=values).as_dict())
        except (ValueError, FileNotFoundError, subprocess.TimeoutExpired) as why:
            out.append({"path": str(path), "bytes": UNRECORDED, "notes": [str(why)]})
    return out
