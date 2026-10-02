"""A policy in one venv, an environment in another: the bridge.

The RL stack (mjlab, warp, torch 2.13) and LeRobot (its own torch pin)
cannot share an interpreter, and a vision student is a LeRobot
checkpoint that must be JUDGED in the mjlab env (docs/66 D2, the
distill half). So the student runs as a subprocess in the train venv
and the env talks to it over a pipe — which is also the shape of the
real deployment (a policy server, a robot loop), not a workaround.

Wire (both directions): u32 little-endian length, then an `.npz` body.
Request keys: `state` (N, width) float32, `image` (N, H, W, 3) uint8,
`reset` (N,) bool — a world whose episode just began drops its held
chunk. Response: `action` (N, nu) float32. One request per control
tick for all N worlds; each world owns an `ActionScheduler` so chunked
policies (ACT) replan on their own executed horizon.

    # server, in the train venv (the client spawns it):
    python -m trainnr.envs.policy_bridge --checkpoint <pretrained_model>
        --camera chase --horizon 20 --device cuda

The loop (`serve`) takes the policy as a callable so the tests drive it
with a fake through in-memory pipes; only `main` touches LeRobot.
"""

from __future__ import annotations

import argparse
import io
import struct
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import IO, Any

import numpy as np

from trainnr.envs.contract import ObservationKeys

HEADER = struct.Struct("<I")
PerWorldPolicy = Callable[[int, Mapping[str, Any], bool], Any]
"""(world, observation, reset) -> action (nu,) - the server's per-world seam."""


def write_frame(stream: IO[bytes], **arrays: Any) -> None:
    body = io.BytesIO()
    np.savez(body, **arrays)
    payload = body.getvalue()
    stream.write(HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


MAX_FRAME_BYTES = 512 * 1024 * 1024


def read_frame(stream: IO[bytes]) -> dict[str, Any] | None:
    """The next frame, or None at a clean EOF (the peer closed)."""
    head = stream.read(HEADER.size)
    if len(head) < HEADER.size:
        return None
    (length,) = HEADER.unpack(head)
    if length > MAX_FRAME_BYTES:
        # ASCII read as a length: something wrote TEXT onto the frame
        # channel. Loud, never a silent hang (measured 2026-09-02: the
        # first student run sat deadlocked for ten minutes).
        raise RuntimeError(
            f"frame channel corrupted: header {head!r} is not a frame length "
            "(a stray print on the pipe?)"
        )
    payload = stream.read(length)
    if len(payload) < length:
        return None
    with np.load(io.BytesIO(payload)) as archive:
        return {key: archive[key] for key in archive.files}


def observation_for(camera: str, state: Any, image: Any) -> dict[str, Any]:
    """One world's observation in the env contract LeRobot's
    preprocess_observation reads (`pixels` per camera key, `agent_pos`)."""
    return {
        ObservationKeys.PIXELS: {camera: np.ascontiguousarray(image, dtype=np.uint8)},
        ObservationKeys.AGENT_POS: np.asarray(state, dtype=np.float32),
    }


def serve(
    policy: PerWorldPolicy, camera: str, inbound: IO[bytes], outbound: IO[bytes]
) -> int:
    """Answer requests until the peer closes; returns the count served."""
    served = 0
    while (request := read_frame(inbound)) is not None:
        states, images = request["state"], request["image"]
        resets = request.get("reset", np.zeros(len(states), dtype=bool))
        actions = [
            policy(
                world,
                observation_for(camera, states[world], images[world]),
                bool(resets[world]),
            )
            for world in range(len(states))
        ]
        write_frame(outbound, action=np.asarray(actions, dtype=np.float32))
        served += 1
    return served


class BridgePolicy:
    """The client: spawn the server in `python` (the train venv's), then
    `act(state, image, reset)` per tick. Closing the stdin ends it."""

    def __init__(  # noqa: PLR0913 - the server's command line, named
        self,
        python: Path | str,
        checkpoint: Path | str,
        *,
        camera: str,
        horizon: int,
        device: str,
        cwd: Path | None = None,
        latency: int = 0,
    ) -> None:
        argv = [
            str(python),
            "-m",
            "trainnr.envs.policy_bridge",
            "--checkpoint",
            str(checkpoint),
            "--camera",
            camera,
            "--horizon",
            str(horizon),
            "--device",
            device,
            "--latency",
            str(latency),
        ]
        self.process = subprocess.Popen(
            argv,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
        )
        assert self.process.stdin is not None and self.process.stdout is not None
        self._inbound = self.process.stdout
        self._outbound = self.process.stdin

    def act(self, state: Any, image: Any, reset: Any) -> Any:
        write_frame(
            self._outbound,
            state=np.asarray(state, dtype=np.float32),
            image=np.asarray(image, dtype=np.uint8),
            reset=np.asarray(reset, dtype=bool),
        )
        reply = read_frame(self._inbound)
        if reply is None:
            raise RuntimeError("the policy server closed the pipe (see its stderr)")
        return reply["action"]

    def close(self) -> None:
        self._outbound.close()
        self.process.wait(timeout=30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--camera", default="chase")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--latency",
        type=int,
        default=0,
        help="inference budget in control ticks: a chunk asked at t takes over at "
        "t + latency (docs/e2e-research/71); 0 = the synchronous loop",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--instruction", default="")
    args = parser.parse_args()

    # The frame channel is OURS alone: keep a private handle to the
    # real stdout, then point fd 1 at stderr so every library print,
    # progress bar or warning that follows lands where text belongs.
    # Without this the first student run deadlocked - a stray line was
    # read as a frame length (2026-09-02).
    import os  # noqa: PLC0415

    frames_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    from trainnr.envs.lerobot_policy import load_policy  # noqa: PLC0415
    from trainnr.evaluate.scheduler import ActionScheduler  # noqa: PLC0415

    loaded = load_policy(
        args.checkpoint, instruction=args.instruction, device=args.device
    )
    chunk_policy = loaded.as_chunk_policy()
    schedulers: dict[int, ActionScheduler] = {}
    nu: int | None = None

    def per_world(world: int, observation: Mapping[str, Any], reset: bool) -> Any:
        nonlocal nu
        if nu is None:
            nu = int(chunk_policy.predict(observation).shape[-1])
        scheduler = schedulers.get(world)
        if scheduler is None or reset:
            scheduler = ActionScheduler(
                chunk_policy, executed_horizon=args.horizon, nu=nu, latency=args.latency
            )
            scheduler.reset()
            schedulers[world] = scheduler
        return scheduler.act(observation)

    print(f"[bridge] {loaded.name} serving on stdio", file=sys.stderr, flush=True)
    served = serve(per_world, args.camera, sys.stdin.buffer, frames_out)
    print(f"[bridge] served {served} ticks", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
