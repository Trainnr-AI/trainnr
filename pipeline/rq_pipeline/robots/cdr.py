"""CDR — the byte layout every ROS 2 message travels in — read from a
LAYOUT, not from code per message.

A ROS 2 message on a bag is the message's fields in declaration order,
little-endian, each primitive aligned to its own size relative to the
start of the body (after a 4-byte encapsulation header), fixed arrays
inline, sequences as a u32 count then the items, strings as a u32 length
then the bytes with a NUL. That is the whole of what this module knows;
WHICH fields a message has is a `Layout`: a tuple of `Field`s copied
from the vendor's `.msg` file, nested layouts by name. Unitree's
`LowState` is 22 lines of data in `robots.adapters.rosbag2`; the next
vendor's message is another table entry, never another decoder.

Standard library only, like the MCAP adapter: a recording enters with
nothing installed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

ENCAPSULATION_HEADER = 4  # representation id + options
NS_PER_S = 1e9

# The primitive types a `.msg` file spells, with their struct code and
# size. `string` and nested messages are handled by name.
PRIMITIVES: dict[str, tuple[str, int]] = {
    "bool": ("<?", 1),
    "int8": ("<b", 1),
    "uint8": ("<B", 1),
    "int16": ("<h", 2),
    "uint16": ("<H", 2),
    "int32": ("<i", 4),
    "uint32": ("<I", 4),
    "int64": ("<q", 8),
    "uint64": ("<Q", 8),
    "float32": ("<f", 4),
    "float64": ("<d", 8),
}
STRING = "string"
SEQUENCE = -1  # a `type[]` field: a u32 count precedes the items


@dataclass(frozen=True)
class Field:
    """One line of a `.msg` file: `type[count] name`. `count` is None for a
    scalar, an int for a fixed array, SEQUENCE for an unbounded one."""

    name: str
    type: str
    count: int | None = None


Layout = tuple[Field, ...]


class Reader:
    """A little-endian CDR reader over one message's bytes."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = ENCAPSULATION_HEADER

    def align(self, n: int) -> None:
        rel = self.pos - ENCAPSULATION_HEADER
        self.pos += (-rel) % n

    def primitive(self, kind: str) -> Any:
        code, size = PRIMITIVES[kind]
        self.align(size)
        (value,) = struct.unpack_from(code, self.data, self.pos)
        self.pos += size
        return value

    def u32(self) -> int:
        return int(self.primitive("uint32"))

    def i32(self) -> int:
        return int(self.primitive("int32"))

    def f64(self) -> float:
        return float(self.primitive("float64"))

    def string(self) -> str:
        n = self.u32()
        s = self.data[self.pos : self.pos + n - 1].decode("utf-8") if n else ""
        self.pos += n
        return s

    def f64_seq(self) -> list[float]:
        n = self.u32()
        return [self.f64() for _ in range(n)]

    def string_seq(self) -> list[str]:
        n = self.u32()
        return [self.string() for _ in range(n)]

    def header(self) -> float:
        """std_msgs/Header: stamp (sec i32, nanosec u32), frame_id — the
        stamp in seconds."""
        sec = self.i32()
        nsec = self.u32()
        self.string()
        return sec + nsec / NS_PER_S

    def message(self, layout: Layout, layouts: dict[str, Layout]) -> dict[str, Any]:
        """A whole message by its layout; nested types resolved through
        `layouts` by name; refuses an unknown type by name."""
        out: dict[str, Any] = {}
        for field in layout:
            out[field.name] = self._field(field, layouts)
        return out

    def _field(self, field: Field, layouts: dict[str, Layout]) -> Any:
        if field.count is None:
            return self._one(field.type, layouts)
        count = self.u32() if field.count == SEQUENCE else field.count
        return [self._one(field.type, layouts) for _ in range(count)]

    def _one(self, kind: str, layouts: dict[str, Layout]) -> Any:
        if kind in PRIMITIVES:
            return self.primitive(kind)
        if kind == STRING:
            return self.string()
        nested = layouts.get(kind)
        if nested is None:
            raise ValueError(f"no CDR layout for message type {kind!r}")
        return self.message(nested, layouts)


class Writer:
    """The reader's inverse, for tests and fixtures: a message from a
    dict by the same layout, so a layout is pinned from both ends."""

    def __init__(self) -> None:
        self.out = bytearray(b"\x00\x01\x00\x00")  # CDR_LE, no options

    def align(self, n: int) -> None:
        rel = len(self.out) - ENCAPSULATION_HEADER
        self.out.extend(b"\x00" * ((-rel) % n))

    def primitive(self, kind: str, value: Any) -> None:
        code, size = PRIMITIVES[kind]
        self.align(size)
        self.out.extend(struct.pack(code, value))

    def u32(self, value: int) -> None:
        self.primitive("uint32", value)

    def string(self, value: str) -> None:
        raw = value.encode("utf-8") + b"\x00"
        self.u32(len(raw))
        self.out.extend(raw)

    def message(
        self, layout: Layout, layouts: dict[str, Layout], values: dict[str, Any]
    ) -> bytes:
        for field in layout:
            self._field(field, layouts, values[field.name])
        return bytes(self.out)

    def _field(self, field: Field, layouts: dict[str, Layout], value: Any) -> None:
        if field.count is None:
            self._one(field.type, layouts, value)
            return
        if field.count == SEQUENCE:
            self.u32(len(value))
        elif len(value) != field.count:
            raise ValueError(f"{field.name}: {len(value)} items for [{field.count}]")
        for item in value:
            self._one(field.type, layouts, item)

    def _one(self, kind: str, layouts: dict[str, Layout], value: Any) -> None:
        if kind in PRIMITIVES:
            self.primitive(kind, value)
        elif kind == STRING:
            self.string(value)
        else:
            self.message(layouts[kind], layouts, value)


def parse_msg(text: str) -> Layout:
    """A `.msg` file's field lines as a layout — comments and constants
    skipped — so a vendor's definition is copied, not transcribed."""
    fields: list[Field] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" in line:
            continue
        kind, name = line.split()[:2]
        count: int | None = None
        if kind.endswith("[]"):
            kind, count = kind[:-2], SEQUENCE
        elif kind.endswith("]"):
            kind, size = kind[:-1].split("[")
            count = int(size)
        fields.append(Field(name=name, type=kind.rsplit("/", 1)[-1], count=count))
    return tuple(fields)
