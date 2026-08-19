"""Reader for the rig's `.wire` recordings — the Python mirror of
`crates/hil-protocol`.

Two implementations of one format, in two languages, is exactly the
repo's recurring bug shape — so this module's conformance test replays a
committed recording and asserts the Rust gate's exact published counts
(`tools/verify.sh` pins the same numbers). The two parsers are allowed
to exist only because something now compares them.

Format facts mirrored from the Rust source, not from memory:
- Status lines are `key=value` tokens; all nine fields must parse or the
  line is rejected WHOLE — half a pose treated as complete is worse than
  a dropped frame, and a cancelled USB write genuinely truncates lines.
- Image rows are `# ` + exactly width*4 hex digits (RGB565, 4 per pixel);
  a picture is complete after exactly `height` parsed rows, and a new
  header while one is open means the old picture was torn.
- Classification order matters: header, then in-flight image row, then
  `# `-note (servo or prose), then status, else unparsable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

STATUS_FIELD_COUNT = 9
IMAGE_HEADER = "# IMG"
NOTE_PREFIX = "# "
SERVO_PAYLOAD_PREFIX = "servo us "
_HEX_DIGITS_PER_PIXEL = 4
_SERVO_CHANNELS = 3
# A header carries width and height; the trailing "rgb565" tag is ignored.
_HEADER_DIMENSIONS = 2


@dataclass(frozen=True)
class StatusFrame:
    seq: int
    x: float
    y: float
    heading: float
    ticks_left: int
    ticks_right: int
    errors_left: int
    errors_right: int
    duty_percent: int
    stalled: bool


@dataclass(frozen=True)
class CameraNote:
    """A `# camera …` note; blob is None when the line says `no blob`."""

    raw: str
    blob: tuple[float, float, int] | None  # (x, y, area), window-relative


@dataclass(frozen=True)
class Image:
    width: int
    height: int
    rows: tuple[tuple[int, ...], ...]  # RGB565 pixels


@dataclass
class Recording:
    """One parsed session, with the same census the Rust gate prints."""

    statuses: list[StatusFrame] = field(default_factory=list)
    images: list[Image] = field(default_factory=list)
    servo_pulses: list[tuple[int, int, int]] = field(default_factory=list)
    camera_notes: list[CameraNote] = field(default_factory=list)
    plain_notes: list[str] = field(default_factory=list)
    torn_images: int = 0
    unparsable: int = 0

    @property
    def stalled_count(self) -> int:
        return sum(1 for status in self.statuses if status.stalled)

    def summary(self) -> str:
        """Byte-identical to `rig_replay`'s output line — the gate greps it."""
        note_count = len(self.camera_notes) + len(self.plain_notes)
        return (
            f"{len(self.statuses)} status ({self.stalled_count} stalled), "
            f"{len(self.images)} images ({self.torn_images} torn), "
            f"{len(self.servo_pulses)} servo, {note_count} notes, "
            f"{self.unparsable} unparsable"
        )


def parse_status(line: str) -> StatusFrame | None:
    values: dict[str, str] = {}
    for token in line.split():
        key, _, value = token.partition("=")
        if value:
            values[key] = value
    try:
        frame = StatusFrame(
            seq=int(values["n"]),
            x=float(values["x"]),
            y=float(values["y"]),
            heading=float(values["th"]),
            ticks_left=int(values["L"]),
            ticks_right=int(values["R"]),
            errors_left=int(values["errL"]),
            errors_right=int(values["errR"]),
            duty_percent=int(values["duty"].rstrip("%")),
            stalled="STALLED" in line,
        )
    except (KeyError, ValueError):
        return None
    return frame


def parse_image_header(line: str) -> tuple[int, int] | None:
    if not line.startswith(IMAGE_HEADER):
        return None
    parts = line[len(IMAGE_HEADER) :].split()
    if len(parts) < _HEADER_DIMENSIONS:
        return None
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    return (width, height) if width > 0 and height > 0 else None


def parse_image_row(line: str, width: int) -> tuple[int, ...] | None:
    if not line.startswith(NOTE_PREFIX):
        return None
    hex_digits = line[len(NOTE_PREFIX) :]
    expected_length = width * _HEX_DIGITS_PER_PIXEL
    if len(hex_digits) != expected_length:
        return None
    try:
        return tuple(
            int(hex_digits[i : i + _HEX_DIGITS_PER_PIXEL], 16)
            for i in range(0, expected_length, _HEX_DIGITS_PER_PIXEL)
        )
    except ValueError:
        return None


def parse_servo_note(payload: str) -> tuple[int, int, int] | None:
    if not payload.startswith(SERVO_PAYLOAD_PREFIX):
        return None
    parts = payload[len(SERVO_PAYLOAD_PREFIX) :].split()
    if len(parts) != _SERVO_CHANNELS:
        return None
    try:
        first, second, third = (int(part) for part in parts)
    except ValueError:
        return None
    return first, second, third


def parse_camera_note(payload: str) -> CameraNote | None:
    if not payload.startswith("camera "):
        return None
    blob: tuple[float, float, int] | None = None
    marker = "blob x"
    position = payload.find(marker)
    if position >= 0:
        try:
            parts = payload[position + len("blob ") :].split()
            blob = (
                float(parts[0].lstrip("x")),
                float(parts[1].lstrip("y")),
                int(parts[3]),
            )
        except (IndexError, ValueError):
            blob = None
    return CameraNote(raw=payload, blob=blob)


def _record_note(recording: Recording, payload: str) -> None:
    """Classify a `# `-note: servo command, camera report, or prose."""
    pulses = parse_servo_note(payload)
    if pulses is not None:
        recording.servo_pulses.append(pulses)
        return
    camera = parse_camera_note(payload)
    if camera is not None:
        recording.camera_notes.append(camera)
    else:
        recording.plain_notes.append(payload)


def parse_recording(path: Path) -> Recording:
    """Classify every line of a `.wire` file, mirroring `rig_replay` exactly."""
    recording = Recording()
    open_image: tuple[int, int, list[tuple[int, ...]]] | None = None

    for raw_line in Path(path).read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        dimensions = parse_image_header(line)
        if dimensions is not None:
            if open_image is not None:
                recording.torn_images += 1
            open_image = (dimensions[0], dimensions[1], [])
            continue

        if open_image is not None:
            row = parse_image_row(line, open_image[0])
            if row is not None:
                open_image[2].append(row)
                if len(open_image[2]) >= open_image[1]:
                    recording.images.append(
                        Image(
                            width=open_image[0],
                            height=open_image[1],
                            rows=tuple(open_image[2]),
                        )
                    )
                    open_image = None
                continue

        if line.startswith(NOTE_PREFIX):
            _record_note(recording, line[len(NOTE_PREFIX) :])
            continue

        status = parse_status(line)
        if status is not None:
            recording.statuses.append(status)
        else:
            recording.unparsable += 1

    if open_image is not None:
        recording.torn_images += 1
    return recording
