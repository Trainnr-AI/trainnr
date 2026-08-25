"""Bridge the robot's WiFi telemetry into a growing .wire file.

    python3 tools/udp-wire-bridge.py recordings/wifi-session.wire [seconds]

Listens on UDP 9870 (the firmware's TELEMETRY_PORT) and appends each
datagram as one line — the datagrams already ARE wire-format status
lines, so everything that reads a wire file works unchanged: the live
dashboard (`tools/rig-rerun.py <file>`) tails it as it grows, and the
finished file replays like any recording.

Written for the untethered choreography (2026-08-25): joining the
robot's AP costs the Mac its internet, so the bridge and the dashboard
run as local background processes through the offline window — the
assistant arms them, the operator joins pico2w, the dashboard fills
live, and the capture is on disk when connectivity returns. Flushes
every line; quiet periods are logged so an empty window is
distinguishable from a dead listener.
"""

import socket
import sys
import time
from pathlib import Path

TELEMETRY_PORT = 9870

if len(sys.argv) < 2:
    sys.exit("usage: udp-wire-bridge.py <out.wire> [seconds=600]")
out_path = Path(sys.argv[1])
window = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(("0.0.0.0", TELEMETRY_PORT))
except OSError:
    sys.exit(f"cannot bind UDP {TELEMETRY_PORT} — another bridge probably holds it")
sock.settimeout(5.0)

count = 0
start = time.time()
print(f"bridging UDP {TELEMETRY_PORT} -> {out_path} for {window:.0f}s", flush=True)
with out_path.open("a") as out:
    while time.time() - start < window:
        try:
            datagram, addr = sock.recvfrom(4096)
        except socket.timeout:
            print(f"[{time.time() - start:6.1f}s] quiet ({count} so far)", flush=True)
            continue
        line = datagram.decode(errors="replace").strip()
        out.write(line + "\n")
        out.flush()
        count += 1
        if count == 1:
            print(f"[{time.time() - start:6.1f}s] first datagram from {addr[0]}", flush=True)
print(f"done: {count} datagrams -> {out_path}", flush=True)
