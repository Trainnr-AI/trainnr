"""Live capture from Unitree's bus: `rt/lowstate` and `rt/lowcmd` over
CycloneDDS, as one recording.

The pair no public log holds. A bag of a robot walking under someone
else's controller carries the state and not the command (docs/77 §2,
2026-09-24), so the torque the joints were asked for and the gains the
motors ran under are unknown, and an identification has to guess them.
On the vendor's bus both messages pass, and this listener takes both,
stamped with the time each arrived (`time.time_ns()` in the SDK's
reader thread — rosbag2's own convention, a `LowState` carrying no
header stamp), so the command's lead over the state is a measurable
number of the recording (`robots.quality.command_lead`).

What lands is a rosbag2 sqlite store (`adapters/rosbag2.Store`): the
SDK's message objects serialize to the CDR bytes a bag holds, so the
capture is ingested by the bag adapter and its layouts, one decoder
for a bag and a live capture alike. Whose robot it was is DECLARED
(`basis`): the operator's own by default, `simulation` when the source
is Unitree's simulator (the DDS gate's stand-in, which publishes the
same topics). Nothing on the bus tells them apart: this docstring once
said the simulator's zero serial number would be noted, but a real
Go2's `LowState` carries `sn = [0, 0]` too (the public leg-odometry
bag, read 2026-09-24), so no such check exists and the basis stays the
caller's declaration.

The interface is named, never defaulted: `lo` is where the stand-in
meets its controller, a robot is on its own Ethernet interface, and a
capture that silently listened on loopback beside a real robot would
record nothing and say so only when stopped. The SDK binds ONE
interface per process (`ChannelFactory` is a singleton that returns
early on a second `Init`), so a second capture on another interface in
the same process is refused by name rather than silently kept on the
first; the interface actually bound is what the provenance records.

Linux only, like everything on their SDK (`deploy.runtimes` refuses it
by name elsewhere); the SDK and CycloneDDS come with the `dds` extra
(docs/77 §7). The state machine, the state file the Studio reads and
the ingest at the end are `robots.capture`'s: this module is one
listener in its registry.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Any

from rq_pipeline.bundles.basis import BASES, BASIS_OWN, BASIS_SIMULATION
from rq_pipeline.deploy.runtimes import RUNTIMES, require_platform
from rq_pipeline.robots.adapters.rosbag2 import (
    NAME as ROSBAG2,
)
from rq_pipeline.robots.adapters.rosbag2 import (
    TOPIC_LOW_CMD,
    TOPIC_LOW_STATE,
    TOPIC_TYPES,
    Store,
)
from rq_pipeline.robots.capture import (
    DEFAULT_WINDOW_S,
    FAILED,
    INGESTED,
    LISTENING,
    CaptureState,
    Ingest,
    OnState,
    ingest_into,
)
from rq_pipeline.scenes.tooling import install_hint

SOURCE = "dds"
STORE_DIR = "capture"  # the rosbag2 store, under the recording's raw/
STANDIN_NETWORK = "lo"  # their simulator and controller meet on loopback
DOMAIN_ID = 0
# No rt/lowstate this long after the listener starts: nothing is
# publishing on this interface and domain (a wrong interface, a robot
# not yet in low-level mode). Refused then, not after the whole window.
FIRST_STATE_TIMEOUT_S = 5.0
# How long `stop()` waits for the writer to land the last batch and
# close the store before it refuses by name. Only a bound against a hung
# writer, never a performance target: at 2 s a loaded box (the full test
# suite beside it, 2026-09-25) made a healthy stop report the writer
# stuck and ingest nothing - on a robot, a lost recording.
WRITER_JOIN_S = 30.0
# The two topics of the pair, and the SDK class each carries.
TOPICS: tuple[str, ...] = (TOPIC_LOW_STATE, TOPIC_LOW_CMD)
SDK_CLASSES: dict[str, str] = {TOPIC_LOW_STATE: "LowState_", TOPIC_LOW_CMD: "LowCmd_"}
SDK_MODULE = "unitree_sdk2py"
QUEUE_LEN = 4096  # the SDK's handler queue per topic; 8 s of a 500 Hz stream
DRAIN_S = 0.2  # how often the writer thread lands what arrived
STATE_EVERY_S = 0.5  # how often the state file is rewritten while listening
QUIET_NOTE = "the bus stayed silent for the whole window: nothing to ingest"
NO_NETWORK = (
    "name the interface the robot is on (e.g. eth0; `ip link` lists them), "
    f"or {STANDIN_NETWORK!r} for Unitree's simulator: a capture never "
    "defaults to one"
)
NO_STATE = (
    "no rt/lowstate within {timeout:.1f} s on {network!r} (domain {domain}): "
    "nothing is publishing there - check the interface, the domain, and "
    "that the robot is in low-level mode"
)
WRITER_STUCK = (
    "the capture's writer did not finish within {timeout:.1f} s: the store "
    "may be incomplete; nothing was ingested"
)
REBIND = (
    "this process's Unitree SDK is bound to {bound!r} (domain {bound_domain}); "
    "it cannot rebind to {network!r} (domain {domain}) - the SDK's channel "
    "factory is a per-process singleton. Restart the MCP server (or the "
    "tool's process) to capture on another interface"
)

# The (interface, domain) this process's SDK is bound to, once bound.
_BOUND: dict[str, Any] = {}


def bind_interface(network: str, domain_id: int) -> tuple[str, int]:
    """Bind this process's SDK channel factory to `network`/`domain_id`
    once; a later call with the same pair is a no-op, another pair is
    refused by name (the SDK would silently keep the first). Returns the
    pair bound."""
    bound = _BOUND.get("pair")
    if bound is None:
        _BOUND["pair"] = (network, domain_id)
        return network, domain_id
    if bound != (network, domain_id):
        raise RuntimeError(
            REBIND.format(
                bound=bound[0],
                bound_domain=bound[1],
                network=network,
                domain=domain_id,
            )
        )
    return bound


STAND_IN_NOTE = (
    "basis declared as simulation: the source is Unitree's simulator over "
    "DDS, not a robot"
)
# A message-in-flight: (topic, receive ns since the epoch, CDR bytes).
Row = tuple[str, int, bytes]


class DdsCapture:
    """Subscribe to the pair; append to a rosbag2 store the capture state
    machine tracks; on `stop()` ingest through the bag adapter."""

    def __init__(  # noqa: PLR0913 - the listener's own knobs, each named
        self,
        recordings: Path,
        state_dir: Path,
        name: str,
        *,
        network: str | None = None,
        domain_id: int = DOMAIN_ID,
        basis: str = BASIS_OWN,
        ingest: Ingest = ingest_into,
        on_state: OnState | None = None,
        subscribe: Any = None,
        first_state_timeout_s: float = FIRST_STATE_TIMEOUT_S,
    ) -> None:
        require_platform(RUNTIMES["dds"])
        if not network:
            raise ValueError(NO_NETWORK)
        if basis not in BASES:
            raise ValueError(f"capture basis {basis!r}; known: {BASES}")
        self.recordings = Path(recordings)
        self.state_dir = Path(state_dir)
        self.name = name
        self.network = network
        self.domain_id = domain_id
        self.basis = basis
        self.ingest = ingest
        self.on_state = on_state
        self.first_state_timeout_s = float(first_state_timeout_s)
        self._subscribe = subscribe or _sdk_subscribe
        self.raw = self.recordings / f".capture-{name}"
        self.state = CaptureState(
            name=name, source=f"{SOURCE}:{network}", raw_file=str(self.raw)
        )
        self._rows: queue.Queue[Row] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closers: list[Any] = []

    def start(self, window_s: float = DEFAULT_WINDOW_S) -> CaptureState:
        if (self.recordings / self.name).exists():
            raise FileExistsError(f"a recording named {self.name!r} already exists")
        if self.raw.exists():
            raise FileExistsError(f"a capture is already writing {self.raw}")
        self._closers = [
            self._subscribe(topic, self._handler(topic), self.network, self.domain_id)
            for topic in TOPICS
        ]
        self.state.state = LISTENING
        self.state.started = time.time()
        self.state.topics = {topic: 0 for topic in TOPICS}
        if self.basis == BASIS_SIMULATION:
            self.state.notes = [STAND_IN_NOTE]
        self._write_state()
        self._thread = threading.Thread(
            target=self._write,
            args=(window_s,),
            daemon=True,
            name=f"capture-{self.name}",
        )
        self._thread.start()
        return self.state

    def _handler(self, topic: str) -> Any:
        def on_message(msg: Any) -> None:
            self._rows.put((topic, time.time_ns(), bytes(msg.serialize())))

        return on_message

    def _write(self, window_s: float) -> None:
        store = Store(self.raw)
        start = time.monotonic()
        last_state = start
        try:
            while not self._stop.is_set() and time.monotonic() - start < window_s:
                self._land(store, _drain(self._rows, DRAIN_S))
                if (
                    not self.state.topics.get(TOPIC_LOW_STATE)
                    and time.monotonic() - start >= self.first_state_timeout_s
                ):
                    self._refuse_silent()
                    return
                if time.monotonic() - last_state >= STATE_EVERY_S:
                    self._write_state()
                    last_state = time.monotonic()
            # What arrived between the stop and the subscriptions closing
            # lands and is counted like the rest (found by the test: five
            # messages a topic were stored and not counted, 2026-09-24).
            self._land(store, _drain(self._rows, 0.0))
        finally:
            store.close()

    def _refuse_silent(self) -> None:
        """No state arrived in time: close the subscriptions and fail by
        name now (the Studio shows it at once); `stop()` leaves nothing."""
        for closer in self._closers:
            closer()
        self._closers = []
        self.state.state = FAILED
        self.state.error = NO_STATE.format(
            timeout=self.first_state_timeout_s,
            network=self.network,
            domain=self.domain_id,
        )
        self._write_state()

    def _land(self, store: Store, batch: list[Row]) -> None:
        """Append a batch to the store and count it, per topic and in all."""
        if not batch:
            return
        store.append(batch)
        for topic, _ns, _data in batch:
            self.state.topics[topic] = self.state.topics.get(topic, 0) + 1
        self.state.datagrams = store.count
        self.state.last_datagram = time.time()

    def stop(self) -> CaptureState:
        """Stop listening and ingest what was captured."""
        self._stop.set()
        for closer in self._closers:
            closer()
        self._closers = []
        if self._thread is not None:
            self._thread.join(timeout=WRITER_JOIN_S)
            if self._thread.is_alive():
                self.state.state = FAILED
                self.state.error = WRITER_STUCK.format(timeout=WRITER_JOIN_S)
                self._write_state()
                return self.state
        if self.state.state == FAILED:  # refused while listening (no state)
            _remove(self.raw)
            self._write_state()
            return self.state
        return self.finish()

    def finish(self) -> CaptureState:
        if self.state.datagrams == 0:
            self.state.state = FAILED
            self.state.error = QUIET_NOTE
            _remove(self.raw)
        else:
            try:
                # The store travels with the recording, as a bag's file does
                # (the adapter can be re-run on the bytes that arrived), and
                # lands inside it BEFORE the stamp is taken: moved in after,
                # it gave the recording a stamp its files never had
                # (review 2026-09-24).
                record = self.ingest(
                    self.recordings,
                    self.raw,
                    name=self.name,
                    adapter=ROSBAG2,
                    basis=self.basis,
                    provenance=self.provenance(),
                    keep_as=STORE_DIR,
                )
                self.state.stamp = record.get("stamp")
                self.state.state = INGESTED
                self.state.notes = list(record.get("notes", [])) + self.state.notes
            except Exception as why:
                self.state.state = FAILED
                self.state.error = str(why)
            finally:
                _remove(self.raw)
        self._write_state()
        return self.state

    def provenance(self) -> dict[str, Any]:
        """The facts behind the basis: which bus, which topics, when."""
        return {
            "source": f"Unitree DDS on {self.network} (domain {self.domain_id})",
            "topics": list(TOPICS),
            "started": self.state.started,
            "messages": dict(self.state.topics),
        }

    def _write_state(self) -> None:
        self.state.write(self.state_dir)
        if self.on_state is not None:
            self.on_state(self.state)


def _drain(rows: queue.Queue[Row], wait_s: float) -> list[Row]:
    """Everything queued, waiting up to `wait_s` for the first row."""
    out: list[Row] = []
    try:
        out.append(rows.get(timeout=wait_s) if wait_s > 0 else rows.get_nowait())
    except queue.Empty:
        return out
    while True:
        try:
            out.append(rows.get_nowait())
        except queue.Empty:
            return out


def _remove(path: Path) -> None:
    import shutil  # noqa: PLC0415

    shutil.rmtree(path, ignore_errors=True)


def _sdk_subscribe(topic: str, handler: Any, network: str, domain_id: int) -> Any:
    """Unitree's SDK subscriber for one topic, its messages handed to
    `handler`; returns the closer. The SDK (and CycloneDDS under it) is
    the `dds` extra; absent, the refusal names the install line."""
    try:
        from unitree_sdk2py.core.channel import (  # noqa: PLC0415
            ChannelFactoryInitialize,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.unitree_go.msg import dds_ as messages  # noqa: PLC0415
    except ImportError as why:
        raise RuntimeError(
            f"Unitree's SDK is not installed ({why}); {install_hint(SDK_MODULE)}"
        ) from why
    bind_interface(network, domain_id)
    ChannelFactoryInitialize(domain_id, network)
    subscriber = ChannelSubscriber(topic, getattr(messages, SDK_CLASSES[topic]))
    subscriber.Init(handler, QUEUE_LEN)
    return subscriber.Close


# -- the stand-in: Unitree's simulator and controller on loopback --------------

STANDIN_SETTLE_S = 1.0  # after the stack is up, before the capture starts


def standin_capture(  # noqa: PLR0913 - the run's knobs, each named
    deployment: Path,
    recordings: Path,
    state_dir: Path,
    name: str,
    *,
    seconds: float,
    reference: Path | None = None,
    ingest: Ingest = ingest_into,
    on_state: OnState | None = None,
) -> CaptureState:
    """The hardware-day capture rehearsed on the stand-in: their simulator
    and their controller for `deployment` brought up on loopback (the DDS
    gate's stack, with no virtual pad: their controller stays passive and
    still commands the motors), the pair captured for `seconds` with the
    basis `simulation`, then everything brought down. On the robot the
    same capture runs with the robot's interface and basis `own robot`;
    nothing else changes."""
    from rq_pipeline.deploy.manifest import load_manifest  # noqa: PLC0415
    from rq_pipeline.deploy.unitree_stage import UnitreeStack  # noqa: PLC0415

    manifest = load_manifest(Path(deployment))
    with UnitreeStack(manifest, reference=reference, with_pad=False):
        time.sleep(STANDIN_SETTLE_S)
        listener = DdsCapture(
            recordings,
            state_dir,
            name,
            network=STANDIN_NETWORK,
            basis=BASIS_SIMULATION,
            ingest=ingest,
            on_state=on_state,
        )
        listener.start(window_s=seconds + DEFAULT_WINDOW_S)
        time.sleep(seconds)
        return listener.stop()


__all__ = [
    "SOURCE",
    "STANDIN_NETWORK",
    "TOPICS",
    "TOPIC_TYPES",
    "DdsCapture",
    "bind_interface",
    "standin_capture",
]
