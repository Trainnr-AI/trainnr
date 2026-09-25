"""The Studio's cross-language constants stay mirrored: values that
exist in both the Python tools and the Rust shell are read from BOTH
sources as text and compared, so a one-sided edit becomes a red build
instead of a silently broken stream (the test_firmware_mirror idiom).
"""

from __future__ import annotations

import dataclasses
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RENDER_STREAM = (REPO / "tools" / "studio-render-stream.py").read_text(encoding="utf-8")
VIEWPORT_RS = (REPO / "crates" / "studio-shell" / "src" / "viewport.rs").read_text(
    encoding="utf-8"
)
MAIN_RS = (REPO / "crates" / "studio-shell" / "src" / "main.rs").read_text(
    encoding="utf-8"
)
VIZ = (REPO / "pipeline" / "rq_pipeline" / "viz.py").read_text(encoding="utf-8")
SIMULATOR_RS = (REPO / "crates" / "studio-shell" / "src" / "simulator.rs").read_text(
    encoding="utf-8"
)
SHELL_SRC = REPO / "crates" / "studio-shell" / "src"
MODEL_RS = (SHELL_SRC / "model.rs").read_text(encoding="utf-8")
CONTROL_RS = (SHELL_SRC / "control.rs").read_text(encoding="utf-8")
DETAIL_RS = (SHELL_SRC / "detail.rs").read_text(encoding="utf-8")
PAGES_RS = (SHELL_SRC / "pages.rs").read_text(encoding="utf-8")
RUNNING_RS = (SHELL_SRC / "running.rs").read_text(encoding="utf-8")


# The kinds the index writes that no Studio page lists by kind.
NOT_LISTED_KINDS = {"fit"}


def constant(source: str, pattern: str) -> str:
    """The one capture group of `pattern`, which must match exactly once."""
    matches = re.findall(pattern, source, flags=re.MULTILINE)
    if len(matches) != 1:
        raise AssertionError(f"{pattern!r} matched {len(matches)} times")
    return matches[0]


class RenderSideCap(unittest.TestCase):
    def test_python_and_rust_agree_on_the_render_cap(self) -> None:
        # tools/studio-render-stream.py declares itself a mirror of
        # crates/studio-shell/src/viewport.rs; nothing enforced it
        # until 2026-09-01.
        py = constant(RENDER_STREAM, r"^MAX_RENDER_SIDE = (\d+)")
        rs = constant(VIEWPORT_RS, r"const MAX_RENDER_SIDE: u32 = (\d+);")
        self.assertEqual(py, rs)


class WireTags(unittest.TestCase):
    def test_the_pan_tag_and_its_payload_agree(self) -> None:
        # The pan message (WASD/QE, 2026-09-12): the tag number and the
        # three f32 seconds on both sides of the pipe.
        py_tag = constant(RENDER_STREAM, r"^TAG_PAN = (\d+)")
        rs_tag = constant(VIEWPORT_RS, r"const TAG_PAN: u8 = (\d+);")
        self.assertEqual(py_tag, rs_tag)
        py_bytes = int(constant(RENDER_STREAM, r"^    TAG_PAN: (\d+),"))
        rs_bytes = int(constant(VIEWPORT_RS, r"fn encode_pan\(.*?\) -> \[u8; (\d+)\]"))
        self.assertEqual(py_bytes + 1, rs_bytes)  # the tag byte leads

    def test_the_group_tag_agrees(self) -> None:
        # The group-mask message (u8 kind, u8 group, u8 on): one tag,
        # four bytes on the wire, the kinds named by the stream's report.
        py_tag = constant(RENDER_STREAM, r"^TAG_GROUP = (\d+)")
        rs_tag = constant(VIEWPORT_RS, r"const TAG_GROUP: u8 = (\d+);")
        self.assertEqual(py_tag, rs_tag)
        self.assertEqual(constant(RENDER_STREAM, r"^    TAG_GROUP: (\d+),"), "3")
        self.assertIn("&[TAG_GROUP, kind, group, u8::from(on)]", VIEWPORT_RS)

    def test_the_twist_tag_and_its_payload_agree(self) -> None:
        # The commanded twist (i32 world, three f32): the tag and the
        # sixteen payload bytes on both sides.
        py_tag = constant(RENDER_STREAM, r"^TAG_TWIST = (\d+)")
        rs_tag = constant(VIEWPORT_RS, r"const TAG_TWIST: u8 = (\d+);")
        self.assertEqual(py_tag, rs_tag)
        py_bytes = int(constant(RENDER_STREAM, r"^    TAG_TWIST: (\d+),"))
        rs_bytes = int(
            constant(VIEWPORT_RS, r"fn encode_twist\(.*?\) -> \[u8; (\d+)\]")
        )
        self.assertEqual(py_bytes + 1, rs_bytes)


class TwistNames(unittest.TestCase):
    def test_the_studio_names_the_twist_axes_as_the_door_does(self) -> None:
        # manifest.TWIST_SHORT is the Python home (the door, the gate's
        # mirror); simulator.rs names the Commands tab's rows.
        from rq_pipeline.deploy.manifest import TWIST_SHORT  # noqa: PLC0415

        rows = constant(
            SIMULATOR_RS, r"const TWIST_AXES: \[\(&str, &str\); 3\] = \[(.*?)\];"
        )
        self.assertEqual(tuple(re.findall(r'\("(\w+)", "\w+"\)', rows)), TWIST_SHORT)


class StudioPort(unittest.TestCase):
    def test_the_ingest_port_is_the_one_the_shell_binds(self) -> None:
        # rq_pipeline.viz.STUDIO_ADDRESS is the one Python home; the
        # Rust shell binds the same port (documented mirror).
        address = constant(VIZ, r'STUDIO_ADDRESS = "rerun\+http://127\.0\.0\.1:(\d+)/')
        bound = constant(MAIN_RS, r'const GRPC_BIND: &str = "0\.0\.0\.0:(\d+)"')
        self.assertEqual(address, bound)


class WireConstants(unittest.TestCase):
    """Every fact the Rust shell mirrors from the render stream's wire:
    the tags, the two tokens, the ring's magic, the initial render size,
    the speed bounds, the view presets' order. Spelled once per side,
    compared here (docs/76 §10.2)."""

    TAGS = (
        "CAMERA",
        "SELECT",
        "DRAG",
        "RELEASE",
        "PAUSE",
        "RUN",
        "STEP",
        "RESET",
        "SPEED",
        "MANUAL",
        "CTRL",
        "QPOS",
        "VIS",
        "RND",
        "VIEW",
        "FOLLOW",
        "PAN",
        "GROUP",
        "TWIST",
    )

    @staticmethod
    def _python_tags() -> dict[str, int]:
        """`TAG_X = n` and `TAG_A, TAG_B = 1, 2` lines, as the stream spells them."""
        out: dict[str, int] = {}
        for line in RENDER_STREAM.splitlines():
            if not line.startswith("TAG_") or "=" not in line:
                continue
            names, _, values = line.partition("=")
            values = values.split("#", 1)[0]
            for name, value in zip(names.split(","), values.split(","), strict=False):
                if value.strip().isdigit():  # not the payload table's `{`
                    out[name.strip()] = int(value.strip())
        return out

    def test_every_tag_the_shell_names_agrees(self) -> None:
        """Every tag the Rust side spells (it sends most of the stream's
        tags, not all) has the stream's number; the stream must know every
        one the shell names."""
        python = self._python_tags()
        rust = re.findall(r"const TAG_(\w+): u8 = (\d+);", VIEWPORT_RS)
        self.assertGreaterEqual(len(rust), 15)
        for tag, number in rust:
            self.assertIn(f"TAG_{tag}", python, tag)
            self.assertEqual(int(number), python[f"TAG_{tag}"], tag)
        for tag in self.TAGS:
            self.assertIn(f"TAG_{tag}", python, tag)

    def test_the_tokens_and_the_magic_agree(self) -> None:
        for name, rust_pat, py_pat in (
            (
                "frame token",
                r"const FRAME_TOKEN: u8 = 0x([0-9A-Fa-f]+);",
                r'FRAME_TOKEN = b"\\x([0-9a-fA-F]+)"',
            ),
            (
                "status token",
                r"const STATUS_TOKEN: u8 = 0x([0-9A-Fa-f]+);",
                r'STATUS_TOKEN = b"\\x([0-9a-fA-F]+)"',
            ),
        ):
            rust = int(constant(VIEWPORT_RS, rust_pat), 16)
            python = int(constant(RENDER_STREAM, py_pat), 16)
            self.assertEqual(rust, python, name)
        rust_magic = constant(VIEWPORT_RS, r"const SHM_MAGIC: u32 = 0x([0-9A-Fa-f_]+);")
        py_magic = constant(RENDER_STREAM, r"SHM_MAGIC = 0x([0-9A-Fa-f]+)")
        self.assertEqual(int(rust_magic.replace("_", ""), 16), int(py_magic, 16))

    def test_the_render_size_and_the_speed_bounds_agree(self) -> None:
        size = re.search(
            r"const INITIAL_RENDER_SIZE: \(u32, u32\) = \((\d+), (\d+)\);", VIEWPORT_RS
        )
        wh = re.search(r"^WIDTH, HEIGHT = (\d+), (\d+)", RENDER_STREAM, re.M)
        assert size and wh
        self.assertEqual(size.groups(), wh.groups())
        speed = re.search(
            r"SPEED_RANGE: std::ops::RangeInclusive<f32> = ([\d.]+)\.\.=([\d.]+);",
            VIEWPORT_RS,
        )
        bounds = re.search(
            r"^SPEED_MIN, SPEED_MAX = ([\d.]+), ([\d.]+)", RENDER_STREAM, re.M
        )
        assert speed and bounds
        self.assertEqual(
            tuple(map(float, speed.groups())), tuple(map(float, bounds.groups()))
        )

    def test_the_view_presets_agree_by_index(self) -> None:
        """The stream's tuple position IS the wire's u8; the shell's table
        carries the index explicitly, in whatever order its menu likes."""
        block = VIEWPORT_RS[VIEWPORT_RS.index("pub const VIEW_PRESETS") :]
        block = block[: block.index("];")]
        rust = {
            int(index): name
            for name, index in re.findall(
                r'name: "(\w+)",\s*label: "[^"]*",\s*index: (\d+)', block
            )
        }
        python = re.search(r"^VIEW_PRESETS = \(([^)]*)\)", RENDER_STREAM, re.M)
        assert python
        names = [n.strip().strip('"') for n in python.group(1).split(",") if n.strip()]
        self.assertEqual(rust, dict(enumerate(names)))


if __name__ == "__main__":
    unittest.main()


class ProjectContracts(unittest.TestCase):
    """What the Studio reads off a project on disk - schemas, file names,
    the environment variable - spelled once in `rq_pipeline.project` and
    once in the shell, compared here."""

    def test_the_schemas_agree(self) -> None:
        from rq_pipeline.project import control, details, index  # noqa: PLC0415

        self.assertEqual(
            constant(MODEL_RS, r'pub const INDEX_SCHEMA: &str = "([^"]+)";'),
            index.INDEX_SCHEMA,
        )
        self.assertEqual(
            constant(DETAIL_RS, r'pub const DETAIL_SCHEMA: &str = "([^"]+)";'),
            details.SCHEMA,
        )
        self.assertEqual(
            constant(CONTROL_RS, r'pub const STATE_SCHEMA: &str = "([^"]+)";'),
            control.STATE_SCHEMA,
        )

    def test_the_pre_flight_key_agrees(self) -> None:
        from rq_pipeline.deploy.preflight import PREFLIGHT_SUMMARY_KEY  # noqa: PLC0415

        self.assertEqual(
            constant(MODEL_RS, r'pub const PREFLIGHT_KEY: &str = "([^"]+)";'),
            PREFLIGHT_SUMMARY_KEY,
        )

    def test_the_viewport_deployment_words_agree(self) -> None:
        """A deployment in the MuJoCo viewport: the scene prefix the picker
        and the stream share, and the summary key the drawer reads."""
        from rq_pipeline.deploy import viewport_source  # noqa: PLC0415

        self.assertEqual(
            constant(VIEWPORT_RS, r'pub const DEPLOY_PREFIX: &str = "([^"]+)";'),
            viewport_source.DEPLOY_PREFIX,
        )
        self.assertEqual(
            constant(MODEL_RS, r'pub const VIEWPORT_KEY: &str = "([^"]+)";'),
            viewport_source.VIEWPORT_KEY,
        )
        self.assertIn(
            viewport_source.DEPLOY_PREFIX,
            constant(RENDER_STREAM, r'^    "(deploy:)": \(open_deploy_scene'),
        )

    def test_the_state_bases_agree(self) -> None:
        from rq_pipeline.bundles.basis import BASES  # noqa: PLC0415

        listed = constant(
            MODEL_RS, r"pub const STATE_BASES: \[&str; \d+\] = \[([^\]]+)\];"
        )
        self.assertEqual(tuple(re.findall(r'"([^"]+)"', listed)), BASES)

    def test_the_hidden_summary_keys_agree(self) -> None:
        """The index's bookkeeping keys never shown on a card: one list,
        both sides (a `fit_bases` fact showed on every robot card)."""
        from rq_pipeline.project.index import HIDDEN_SUMMARY_KEYS  # noqa: PLC0415

        listed = constant(
            MODEL_RS, r"pub const HIDDEN_KEYS: &\[&str\] = &\[([^\]]+)\];"
        )
        self.assertEqual(tuple(re.findall(r'"([^"]+)"', listed)), HIDDEN_SUMMARY_KEYS)

    def test_the_kinds_agree(self) -> None:
        """Every kind the Studio spells is a kind the index writes, and the
        ones it does not list are named here (a fit is shown on its robot,
        never on a page of its own)."""
        from rq_pipeline.project.kinds import Kind  # noqa: PLC0415

        block = PAGES_RS.split("pub mod kind {", 1)[1].split("\n}\n", 1)[0]
        rust = set(re.findall(r'pub const \w+: &str = "([^"]+)";', block))
        python = {k.value for k in Kind}
        self.assertLessEqual(rust, python)
        self.assertEqual(python - rust, NOT_LISTED_KINDS)

    def test_the_files_and_the_environment_agree(self) -> None:
        from rq_pipeline.project import control, locate, present  # noqa: PLC0415

        index_dir = locate.INDEX_DIR
        self.assertEqual(
            constant(MODEL_RS, r'pub const PROJECT_ENV: &str = "([^"]+)";'),
            locate.PROJECT_ENV,
        )
        self.assertEqual(
            constant(MODEL_RS, r'const INTENT_RELATIVE: &str = "([^"]+)";'),
            f"{index_dir}/{present.INTENT_FILE}",
        )
        self.assertEqual(
            constant(MODEL_RS, r'const PRESENT_STATUS_RELATIVE: &str = "([^"]+)";'),
            f"{index_dir}/{control.PRESENT_STATUS_FILE}",
        )
        self.assertEqual(
            constant(CONTROL_RS, r'pub const STATE_RELATIVE: &str = "([^"]+)";'),
            f"{index_dir}/{control.STATE_FILE}",
        )
        self.assertEqual(
            constant(CONTROL_RS, r'pub const COMMANDS_RELATIVE: &str = "([^"]+)";'),
            f"{index_dir}/{control.COMMANDS_DIR}",
        )
        self.assertEqual(
            constant(CONTROL_RS, r'pub const EVENTS_RELATIVE: &str = "([^"]+)";'),
            f"{index_dir}/{control.EVENTS_FILE}",
        )
        self.assertEqual(
            int(constant(CONTROL_RS, r"pub const SCREENSHOT_WIDTH: u32 = (\d+);")),
            control.SCREENSHOT_WIDTH,
        )

    def test_the_state_fields_agree(self) -> None:
        """The Rust State reads every field the index writes (the basis
        word's vocabulary is pinned by `test_the_state_bases_agree`)."""
        from rq_pipeline.project import index  # noqa: PLC0415

        block = MODEL_RS.split("pub struct State {", 1)[1].split("\n}\n", 1)[0]
        rust = set(re.findall(r"pub (\w+):", block))
        python = {f.name for f in dataclasses.fields(index.State)}
        self.assertEqual(rust, python)

    def test_the_in_progress_entry_has_the_fields_the_index_writes(self) -> None:
        """The Rust entry reads every key the index writes, for both kinds
        of work in progress: a scene being captured, a capture listening
        (the entry the index itself produces, not a line of its source)."""
        import tempfile  # noqa: PLC0415

        from rq_pipeline.project import create_project, index_project  # noqa: PLC0415
        from rq_pipeline.project.locate import INDEX_DIR  # noqa: PLC0415
        from rq_pipeline.robots.capture import LISTENING, CaptureState  # noqa: PLC0415

        block = MODEL_RS.split("pub struct InProgress {", 1)[1].split("}", 1)[0]
        fields = set(re.findall(r"pub (\w+):", block))
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p", "test")
            CaptureState(state=LISTENING, name="w", source="dds:lo").write(
                project.root / INDEX_DIR
            )
            entries = index_project(project).in_progress
        self.assertTrue(entries)
        for entry in entries:
            self.assertEqual(set(entry), fields)

    def test_the_pages_and_the_verbs_agree(self) -> None:
        """Every page the rail shows is a name the door accepts, and every
        verb the door writes is one the shell's command enum parses."""
        from rq_pipeline.project import control  # noqa: PLC0415

        titles = re.findall(r"^\s+Self::\w+ => \"([A-Z][a-z]+)\",$", PAGES_RS, re.M)
        self.assertGreater(len(titles), 10)
        slugs = {t.lower().replace(" ", "-") for t in titles}
        self.assertEqual(slugs - set(control.SECTIONS), set())
        enum = CONTROL_RS.split("pub enum Command {", 1)[1].split("\n}\n", 1)[0]
        variants = {v.lower() for v in re.findall(r"^    ([A-Z]\w+)", enum, re.M)}
        self.assertEqual(set(control.VERBS), variants)


def _serde_fields(source: str, struct: str) -> set[str]:
    """The fields a Rust struct reads off JSON: its `pub` fields, less
    the `#[serde(skip)]` ones the poll fills itself."""
    body = source.split(f"pub struct {struct} {{", 1)[1].split("\n}\n", 1)[0]
    body = re.sub(r"#\[serde\(skip\)\]\s*(?:///[^\n]*\n\s*)*pub \w+:", "", body)
    return set(re.findall(r"^\s*pub (\w+):", body, re.M))


class RunningNow(unittest.TestCase):
    """The job table as the Running now panel reads it (2026-09-25):
    every word, file suffix and key, once in `rq_pipeline.mcp_jobs`,
    once in `running.rs`."""

    def test_the_table_s_words_agree(self) -> None:
        from rq_pipeline import mcp_jobs  # noqa: PLC0415

        self.assertEqual(
            constant(RUNNING_RS, r'pub const STATUS_SUFFIX: &str = "([^"]+)";'),
            mcp_jobs.STATUS_SUFFIX,
        )
        self.assertEqual(
            constant(RUNNING_RS, r'pub const STATUS_SCHEMA: &str = "([^"]+)";'),
            mcp_jobs.STATUS_SCHEMA,
        )
        listed = constant(
            RUNNING_RS, r"pub const JOB_SOURCES: \[&str; \d+\] = \[([^\]]+)\];"
        )
        self.assertEqual(tuple(re.findall(r'"([^"]+)"', listed)), mcp_jobs.JOB_SOURCES)
        self.assertEqual(
            constant(MODEL_RS, r'const JOBS_DIR: &str = "([^"]+)";'),
            mcp_jobs.JOBS_DIR_NAME,
        )
        self.assertEqual(
            constant(VIEWPORT_RS, r'pub const WALK_TASK: &str = "([^"]+)";'),
            mcp_jobs.VIEWPORT_WALK,
        )
        for word in (mcp_jobs.STATE_RUNNING, mcp_jobs.STATE_DIED):
            self.assertIn(f'"{word}"', RUNNING_RS)

    def test_the_keys_rust_reads_are_written(self) -> None:
        from rq_pipeline import mcp_jobs  # noqa: PLC0415

        record = {f.name for f in dataclasses.fields(mcp_jobs.JobRecord)}
        status = {f.name for f in dataclasses.fields(mcp_jobs.RunStatus)}
        job_keys = _serde_fields(RUNNING_RS, "Job")
        status_keys = _serde_fields(RUNNING_RS, "RunStatus")
        self.assertIn("source", job_keys)
        self.assertLessEqual(job_keys, record)
        self.assertIn("stage", status_keys)
        self.assertLessEqual(status_keys, status)
