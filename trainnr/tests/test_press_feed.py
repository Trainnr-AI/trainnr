"""The press's Studio feed: a keeper's frames ride a tick timeline
(thinned to a cap, compressed), a discard logs no frames, and the
tick clock never leaks into the next attempt's rows."""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from trainnr.collect.press import PressResult
from trainnr.collect.press_feed import (
    FRAMES_PER_KEEPER,
    TICK_TIMELINE,
    StudioPressFeed,
    thinned,
)


class _Image:
    def __init__(self, image: Any) -> None:
        self.image = image
        self.compressed = False

    def compress(self, jpeg_quality: int) -> _Image:
        self.compressed = True
        return self


class _FakeRerun:
    """What the feed calls on `rerun`, recorded."""

    Image = _Image

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def set_time(self, timeline: str, **kwargs: Any) -> None:
        self.calls.append(("set_time", (timeline, kwargs)))

    def disable_timeline(self, timeline: str) -> None:
        self.calls.append(("disable_timeline", timeline))

    def log(self, entity: str, payload: Any) -> None:
        self.calls.append(("log", (entity, payload)))

    def Scalars(self, value: float) -> Any:  # noqa: N802 - rerun's spelling
        return ("scalars", value)

    def TextLog(self, text: str) -> Any:  # noqa: N802
        return ("text", text)


def _feed(fake: _FakeRerun) -> StudioPressFeed:
    feed = StudioPressFeed.__new__(StudioPressFeed)
    feed._rr = fake
    return feed


class Thinning(unittest.TestCase):
    def test_keeps_everything_under_the_cap_and_the_last_frame_over_it(self) -> None:
        frames = [(t, t) for t in range(10)]
        self.assertEqual(thinned(frames, 20), frames)
        picked = thinned(frames, 4)
        self.assertEqual(len(picked), 4)
        self.assertEqual(picked[-1], frames[-1])
        self.assertEqual([t for t, _ in picked][:3], [0, 2, 5])


class TheKeeperOnScreen(unittest.TestCase):
    def test_a_keeper_logs_every_camera_by_tick_then_drops_the_clock(self) -> None:
        fake = _FakeRerun()
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        result = PressResult(
            True,
            camera_frames={"front": [(0, image), (5, image)], "wrist": [(0, image)]},
        )
        _feed(fake).attempt(3, 1, 4, result)
        ticks = [
            payload[1]["sequence"]
            for kind, payload in fake.calls
            if kind == "set_time" and payload[0] == TICK_TIMELINE
        ]
        self.assertEqual(ticks, [0, 5, 0])
        images = [
            payload
            for kind, payload in fake.calls
            if kind == "log" and isinstance(payload[1], _Image)
        ]
        self.assertEqual(
            [entity for entity, _ in images],
            ["press/front", "press/front", "press/wrist"],
        )
        self.assertTrue(all(payload.compressed for _, payload in images))
        self.assertEqual(fake.calls[-1], ("disable_timeline", TICK_TIMELINE))

    def test_a_discard_logs_its_verdict_and_no_frames(self) -> None:
        fake = _FakeRerun()
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        result = PressResult(False, camera_frames={"front": [(0, image)]})
        _feed(fake).attempt(1, 0, 4, result)
        logged = [payload for kind, payload in fake.calls if kind == "log"]
        self.assertFalse(any(isinstance(payload[1], _Image) for payload in logged))
        self.assertIn(("press/log", ("text", "attempt 1: discard")), logged)

    def test_the_cap_is_a_constant_a_reader_can_find(self) -> None:
        self.assertGreaterEqual(FRAMES_PER_KEEPER, 100)


if __name__ == "__main__":
    unittest.main()
