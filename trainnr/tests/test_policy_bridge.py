"""The policy bridge: a LeRobot student in one venv, an
mjlab env in another, talking over a pipe. The framing round-trips,
the server answers one action per world, and a world's reset reaches
the per-world policy - all through in-memory pipes, no LeRobot."""

from __future__ import annotations

import io
import unittest

import numpy as np

from trainnr.envs.contract import ObservationKeys
from trainnr.envs.policy_bridge import (
    observation_for,
    read_frame,
    serve,
    write_frame,
)


class Framing(unittest.TestCase):
    def test_a_frame_round_trips_every_array_and_dtype(self) -> None:
        pipe = io.BytesIO()
        write_frame(
            pipe,
            state=np.arange(6, dtype=np.float32).reshape(2, 3),
            image=np.zeros((2, 4, 5, 3), dtype=np.uint8),
            reset=np.array([True, False]),
        )
        pipe.seek(0)
        frame = read_frame(pipe)
        assert frame is not None
        self.assertEqual(frame["state"].dtype, np.float32)
        self.assertEqual(frame["image"].shape, (2, 4, 5, 3))
        self.assertEqual(frame["reset"].tolist(), [True, False])
        self.assertIsNone(read_frame(pipe))  # clean EOF, never an exception

    def test_text_on_the_frame_channel_is_refused_not_awaited(self) -> None:
        # A stray print read as a length: the first student run sat
        # deadlocked ten minutes on exactly this (2026-09-02).
        polluted = io.BytesIO(b"Loading policy...\n" + b"\x00" * 8)
        with self.assertRaises(RuntimeError) as ctx:
            read_frame(polluted)
        self.assertIn("stray print", str(ctx.exception))

    def test_a_truncated_frame_reads_as_eof(self) -> None:
        pipe = io.BytesIO()
        write_frame(pipe, state=np.zeros(3, dtype=np.float32))
        truncated = io.BytesIO(pipe.getvalue()[:-5])
        self.assertIsNone(read_frame(truncated))


class TheObservationContract(unittest.TestCase):
    def test_pixels_by_camera_and_agent_pos(self) -> None:
        obs = observation_for("chase", [1.0, 2.0], np.ones((4, 5, 3)))
        self.assertEqual(list(obs[ObservationKeys.PIXELS]), ["chase"])
        self.assertEqual(obs[ObservationKeys.PIXELS]["chase"].dtype, np.uint8)
        self.assertEqual(obs[ObservationKeys.AGENT_POS].dtype, np.float32)


class TheServer(unittest.TestCase):
    def test_one_action_per_world_and_resets_reach_the_policy(self) -> None:
        seen: list[tuple[int, bool]] = []

        def policy(world: int, observation, reset: bool):
            seen.append((world, reset))
            # The action echoes the world so the reply order is pinned.
            return np.full(2, float(world), dtype=np.float32)

        inbound = io.BytesIO()
        write_frame(
            inbound,
            state=np.zeros((3, 4), dtype=np.float32),
            image=np.zeros((3, 4, 5, 3), dtype=np.uint8),
            reset=np.array([True, False, True]),
        )
        write_frame(
            inbound,
            state=np.zeros((3, 4), dtype=np.float32),
            image=np.zeros((3, 4, 5, 3), dtype=np.uint8),
        )
        inbound.seek(0)
        outbound = io.BytesIO()
        self.assertEqual(serve(policy, "chase", inbound, outbound), 2)
        outbound.seek(0)
        first = read_frame(outbound)
        assert first is not None
        self.assertEqual(first["action"].tolist(), [[0, 0], [1, 1], [2, 2]])
        self.assertEqual(seen[:3], [(0, True), (1, False), (2, True)])
        self.assertEqual(seen[3:], [(0, False), (1, False), (2, False)])


if __name__ == "__main__":
    unittest.main()
