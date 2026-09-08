"""Live capture: a robot's stream into a recording, with the Studio watching.

The "robot operation" collection source (docs/76 §5): the robot is on and
moving — driven by an operator, a script, or a policy — and its telemetry
is captured into a recording the loop can use. Three things happen at
once, each already proven separately in this repo and composed here:

1. a **listener** appends every datagram the robot sends to a growing
   raw file (this repo's UDP wire bridge, `tools/udp-wire-bridge.py`);
2. a **live view** tails that file into the Studio so the operator sees
   the capture as it happens (the pattern of `tools/rig-rerun.py`);
3. at the end, the raw file goes through its **adapter** and lands in the
   project as a stamped recording (`robots/ingest`).

The listener is the only part that speaks a wire protocol, and there is
one today: this repo's Pico rig over UDP. A ROS 2 graph is captured by
`ros2 bag record` into an MCAP file and ingested afterwards — a live ROS
listener would need a ROS installation, which is exactly what the seam
avoids. A Unitree listener waits on its SDK research (docs/76 §5).

`Capture` is deliberately a small state machine (idle → listening →
ingested) with its state on disk (`<project>/.index/capture.json`), so
the Studio can show it and a tool can poll it, the way jobs work.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rq_pipeline.project.locate import INDEX_DIR, Project

STATE_FILE = "capture.json"
# The firmware's telemetry port (firmware/pico-odom, TELEMETRY_PORT); the
# UDP bridge carries the same second copy on purpose.
WIRE_UDP_PORT = 9870
DATAGRAM_BYTES = 4096
QUIET_S = 5.0
DEFAULT_WINDOW_S = 600.0

IDLE = "idle"
LISTENING = "listening"
INGESTED = "ingested"
FAILED = "failed"


@dataclass
class CaptureState:
    state: str = IDLE
    source: str = ""  # "udp:9870", later "ros2:<topics>", "unitree:<iface>"
    raw_file: str = ""
    started: float | None = None
    datagrams: int = 0
    last_datagram: float | None = None
    stamp: str | None = None  # set when ingested
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    def write(self, project: Project) -> Path:
        out = project.root / INDEX_DIR / STATE_FILE
        out.parent.mkdir(parents=True, exist_ok=True)
        staging = out.with_suffix(".json.tmp")
        staging.write_text(json.dumps(asdict(self), indent=1))
        staging.replace(out)
        return out

    @classmethod
    def read(cls, project: Project) -> CaptureState:
        path = project.root / INDEX_DIR / STATE_FILE
        if not path.is_file():
            return cls()
        raw = json.loads(path.read_text())
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


class WireUdpCapture:
    """Listen on the rig's UDP port; append datagrams to a `.wire` file;
    on `stop()` ingest it. One capture at a time per project."""

    def __init__(
        self, project: Project, name: str, *, port: int = WIRE_UDP_PORT
    ) -> None:
        self.project = project
        self.name = name
        self.port = port
        self.raw = project.folder("recordings") / f".capture-{name}.wire"
        self.state = CaptureState(source=f"udp:{port}", raw_file=str(self.raw))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, window_s: float = DEFAULT_WINDOW_S) -> CaptureState:
        if (self.project.folder("recordings") / self.name).exists():
            raise FileExistsError(f"a recording named {self.name!r} already exists")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", self.port))
        except OSError as error:
            raise OSError(
                f"cannot bind UDP {self.port} — another capture holds it"
            ) from error
        sock.settimeout(QUIET_S)
        self.state.state = LISTENING
        self.state.started = time.time()
        self.state.write(self.project)
        self._thread = threading.Thread(
            target=self._listen,
            args=(sock, window_s),
            daemon=True,
            name=f"capture-{self.name}",
        )
        self._thread.start()
        return self.state

    def _listen(self, sock: socket.socket, window_s: float) -> None:
        start = time.time()
        with self.raw.open("a", encoding="utf-8", newline="\n") as out:
            while not self._stop.is_set() and time.time() - start < window_s:
                try:
                    datagram, _addr = sock.recvfrom(DATAGRAM_BYTES)
                except TimeoutError:
                    self.state.write(
                        self.project
                    )  # quiet is visible, not a dead listener
                    continue
                except OSError:
                    break
                out.write(datagram.decode(errors="replace").strip() + "\n")
                out.flush()
                self.state.datagrams += 1
                self.state.last_datagram = time.time()
                if self.state.datagrams % 50 == 1:
                    self.state.write(self.project)
        sock.close()

    def stop(self) -> CaptureState:
        """Stop listening and ingest what was captured."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=QUIET_S + 1.0)
        return self.finish()

    def finish(self) -> CaptureState:
        from rq_pipeline.robots.ingest import ingest  # noqa: PLC0415

        if self.state.datagrams == 0 or not self.raw.is_file():
            self.state.state = FAILED
            self.state.error = "no datagrams received — nothing to ingest"
            self.raw.unlink(missing_ok=True)
        else:
            try:
                record = ingest(self.project, self.raw, name=self.name, adapter="wire")
                self.state.stamp = record["stamp"]
                self.state.state = INGESTED
                self.state.notes = list(record.get("notes", []))
            except Exception as why:
                self.state.state = FAILED
                self.state.error = str(why)
            finally:
                self.raw.unlink(missing_ok=True)
        self.state.write(self.project)
        return self.state


def status(project: Project) -> dict[str, Any]:
    return asdict(CaptureState.read(project))
