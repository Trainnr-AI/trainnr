"""Camera rigs as data: what a vision policy sees, by name and resolution.

The 2026-08-25 census found all 56 ArmnetBench checkpoints public, every
one observing pixels; `ARMNETBENCH_CAMERAS` is that rig exactly, matching
the dataset's observation keys and resolutions. The vision rollout itself,
the policy adapter and the vision harness that lived here until
2026-08-26 are gone (docs/32 step 5): a pixel policy is evaluated through
the gymnasium env (rq_pipeline.envs) by `lerobot-eval`, which loads any
LeRobot family through its own processors; the harness folds the
resulting records.

The instrument caveat, kept where the data lives: a vision policy can
fail on COSMETICS (lighting, colors, viewpoint) and that failure would be
read as a dynamics gap. Camera placement is therefore calibration, not
decoration — sim renders are matched by eye against the released real
videos (tools/camera-match.py) before any correlation is trusted.
"""

from __future__ import annotations

from rq_pipeline.protocol import CameraSpec

__all__ = ["ARMNETBENCH_CAMERAS", "CameraSpec"]

# The rig every ArmnetBench policy was trained on (dataset README):
# front/top 576x1024, wrist 720x1280, all at 20 fps.
ARMNETBENCH_CAMERAS: tuple[CameraSpec, ...] = (
    CameraSpec("front", "front", width=1024, height=576),
    CameraSpec("top", "top", width=1024, height=576),
    CameraSpec("wrist", "wrist", width=1280, height=720),
)
