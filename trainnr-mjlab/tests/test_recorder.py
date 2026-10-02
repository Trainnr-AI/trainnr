"""The Rerun recorder against a memory sink and a duck env — the hooks'
contract without a viewer or a GPU. The live proof (mjlab's cartpole
streaming into the Studio) is `python -m trainnr_mjlab.demo_recorder`."""

from __future__ import annotations

import unittest

import torch


class _Duck:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _env(steps: int = 0):
    import mujoco  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    wp.init()
    xml = (
        '<mujoco><worldbody><body><joint name="slider" type="slide" axis="1 0 0"/>'
        '<geom type="box" size="0.05 0.05 0.05" mass="1"/></body></worldbody></mujoco>'
    )
    model = mujoco.MjModel.from_xml_string(xml)
    import mujoco_warp as mjw  # noqa: PLC0415

    with wp.ScopedDevice("cpu"):
        data = mjw.put_data(model, mujoco.MjData(model), nworld=2)
    return _Duck(
        sim=_Duck(mj_model=model, data=data),
        common_step_counter=steps,
        episode_length_buf=torch.tensor([7, 3]),
        reward_buf=torch.tensor([0.5, 0.1]),
    )


try:
    import rerun as _rerun
except ImportError:  # pragma: no cover - the clean clone without --extra viz
    _rerun = None


@unittest.skipUnless(
    _rerun is not None, "viz extra not installed (uv sync --extra viz)"
)
class TheRecorder(unittest.TestCase):
    def _term(self, env):
        import rerun as rr  # noqa: PLC0415

        from trainnr_mjlab.recorder import (  # noqa: PLC0415
            RerunRecorder,
            RerunRecorderCfg,
        )

        cfg = RerunRecorderCfg(every=1)
        # A memory sink instead of a socket: rr.init + memory recording,
        # then the term's connect is redirected by binding first.
        rr.init(cfg.app_id, spawn=False)
        memory = rr.memory_recording()
        term = RerunRecorder.__new__(RerunRecorder)
        # Wire by hand what __init__ does after the connect, against the
        # memory sink — the connect itself is the demo's job.
        term._rr = rr
        term._cfg = cfg
        term._env = env
        # (name, qpos address) pairs — the post-review shape; this rig's
        # one slider is global joint 0, qpos 0.
        term._joints = [("slider", 0)]
        term._skipped_joints = []
        term._said_no_reward = False
        term._qpos = None
        term._geom_views = None
        term._mirror = None
        import time  # noqa: PLC0415

        term._began = time.time()
        return term, memory

    def test_post_step_logs_the_watched_worlds_story(self) -> None:
        env = _env(steps=10)
        term, memory = self._term(env)
        term.record_post_step()
        self.assertGreater(memory.num_msgs(), 0)

    def test_post_step_logs_every_reward_term(self) -> None:
        env = _env(steps=10)
        env.reward_manager = _Duck(
            get_active_iterable_terms=lambda i: [("pose", [0.25]), ("upright", [0.9])]
        )
        term, memory = self._term(env)
        before = memory.num_msgs()
        term.record_post_step()
        # the total, two terms, the joint, the episode length, and the clock
        self.assertGreaterEqual(memory.num_msgs() - before, 5)

    def test_the_throttle_skips_off_beat_steps(self) -> None:
        env = _env(steps=11)
        term, memory = self._term(env)
        term._cfg.every = 10
        before = memory.num_msgs()
        term.record_post_step()
        self.assertEqual(memory.num_msgs(), before)  # 11 % 10 != 0: skipped

    def test_pre_reset_marks_only_the_watched_world(self) -> None:
        env = _env(steps=20)
        term, memory = self._term(env)
        before = memory.num_msgs()
        term.record_pre_reset(torch.tensor([1]))  # not the watched world
        self.assertEqual(memory.num_msgs(), before)
        term.record_pre_reset(torch.tensor([0, 1]))
        self.assertGreater(memory.num_msgs(), before)

    def test_the_mirror_draws_the_watched_worlds_geoms(self) -> None:
        from trainnr.viz import RigMirror  # noqa: PLC0415

        env = _env(steps=10)
        term, memory = self._term(env)
        term._mirror = RigMirror(env.sim.mj_model, model_colors=True)
        before = memory.num_msgs()
        term.record_post_step()
        with_mirror = memory.num_msgs() - before
        term._mirror = None
        before = memory.num_msgs()
        term.record_post_step()
        without = memory.num_msgs() - before
        # The mirror adds the 3D leg on top of the scalar story.
        self.assertGreater(with_mirror, without)

    def test_the_cfg_carries_the_saved_stream_and_opens_through_the_seam(self) -> None:
        from trainnr import viz  # noqa: PLC0415

        from trainnr_mjlab import recorder  # noqa: PLC0415
        from trainnr_mjlab.recorder import RerunRecorderCfg  # noqa: PLC0415

        self.assertIsNone(
            RerunRecorderCfg().file, "live only unless a run names its folder"
        )
        self.assertEqual(
            RerunRecorderCfg(file="runs/x/.viewer/train.rrd").file,
            "runs/x/.viewer/train.rrd",
        )
        self.assertIs(
            recorder.open_stream, viz.open_stream, "one seam opens every stream"
        )

    def test_the_cfg_names_its_term_class(self) -> None:
        from trainnr_mjlab.recorder import (  # noqa: PLC0415
            RerunRecorder,
            RerunRecorderCfg,
        )

        self.assertIs(RerunRecorderCfg(every=1).func, RerunRecorder)


if __name__ == "__main__":
    unittest.main()
