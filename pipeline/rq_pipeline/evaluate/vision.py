"""Vision evaluation: released checkpoints meet the simulator's cameras.

The 2026-08-25 census found all 56 ArmnetBench checkpoints public —
Paper 2's sim side becomes *evaluating* them, and every one of them
observes pixels, not sensor vectors. This module carries the three
pieces the sensor-only harness lacks:

- the ArmnetBench camera rig as data (`ARMNETBENCH_CAMERAS`), matching
  the dataset's observation keys and resolutions exactly;
- `evaluate_vision_policies`, the same paired-trial, census-gated
  protocol as `harness.evaluate_policies` but over the vision rollout;
- `LeRobotCheckpointPolicy`, the adapter that loads a released
  checkpoint through LeRobot and steps it in the loop. Heavy imports
  are lazy (`--extra train`); inference runs are the WSL card's job.

The instrument caveat, stated where the code lives: a vision policy can
fail on COSMETICS (lighting, colors, viewpoint) and that failure would
be read as a dynamics gap. Camera placement is therefore calibration,
not decoration — sim renders are matched by eye against the released
real videos (tools/camera-match.py) before any correlation is trusted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from rq_pipeline.evaluate.harness import EpisodeProtocol, SimScore, home_state
from rq_pipeline.robot.model_checks import assert_model_alive


@dataclass(frozen=True)
class CameraSpec:
    """One camera: the LeRobot key it feeds and the render geometry."""

    key: str  # observation.images.<key>
    camera_name: str  # the MuJoCo camera
    width: int
    height: int


# The rig every ArmnetBench policy was trained on (dataset README):
# front/top 576x1024, wrist 720x1280, all at 20 fps.
ARMNETBENCH_CAMERAS: tuple[CameraSpec, ...] = (
    CameraSpec("front", "front", width=1024, height=576),
    CameraSpec("top", "top", width=1024, height=576),
    CameraSpec("wrist", "wrist", width=1280, height=720),
)


@dataclass(frozen=True)
class VisionPolicy:
    """A named pixels-in controller: (step, observation dict) → controls.

    `reset` runs before every episode — checkpoint policies carry
    action-chunk queues that must not leak between trials.
    """

    name: str
    act: Callable[[int, dict[str, Any]], Any]
    reset: Callable[[], None] = lambda: None


def evaluate_vision_policies(
    backend: Any,
    policies: Sequence[VisionPolicy],
    protocol: EpisodeProtocol,
    *,
    cameras: Sequence[CameraSpec] = ARMNETBENCH_CAMERAS,
    source: str,
) -> tuple[SimScore, ...]:
    """`harness.evaluate_policies`, over pixels. Same rules, same pairing."""
    names = [policy.name for policy in policies]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate policy names: {sorted(names)}")
    if "@" not in source:
        raise ValueError(
            f"source must be a name@hash stamp, got {source!r} — the same "
            "rule certify() enforces, applied before episodes are spent"
        )
    counts = backend.counts()
    assert_model_alive(counts.actuators, counts.sensors, counts.geoms, source=source)
    home = home_state(backend, protocol)
    scores = []
    for policy in policies:
        successes = 0
        for trial in range(protocol.trials):
            policy.reset()
            states, sensors = backend.closed_loop_vision_rollout(
                protocol.perturb(trial, home),
                policy.act,
                protocol.steps,
                protocol.control_interval,
                cameras,
            )
            if protocol.success(states, sensors):
                successes += 1
        scores.append(
            SimScore(name=policy.name, successes=successes, trials=protocol.trials)
        )
    return tuple(scores)


def lerobot_checkpoint_policy(
    repo_id: str,
    *,
    task_instruction: str,
    device: str = "cuda",
) -> VisionPolicy:
    """A released checkpoint as a VisionPolicy, via LeRobot.

    Covers the five LeRobot-native families (act, diffusion, smolvla,
    grootn1.7, molmoact2 — the census's safetensors column). pi0/pi0.5
    are openpi-format and need their own adapter later. Requires the
    `train` extra and a working torch; exercised on the WSL card.
    """
    try:
        import torch  # noqa: PLC0415
        from lerobot.policies.factory import get_policy_class  # noqa: PLC0415
        from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: PLC0415
    except ImportError as error:
        raise ImportError(
            "checkpoint policies need the 'train' extra: uv sync --extra train"
        ) from error

    del get_policy_class  # resolution happens inside from_pretrained
    policy: Any = PreTrainedPolicy.from_pretrained(repo_id)
    policy.to(device)
    policy.eval()

    def act(step: int, observation: dict[str, Any]) -> Any:
        del step
        batch: dict[str, Any] = {"task": task_instruction}
        for key, value in observation.items():
            if key.startswith("observation.images."):
                # uint8 HWC -> float32 CHW in [0,1], batched — LeRobot's
                # preprocessing contract for camera features.
                tensor = torch.from_numpy(value).to(device)
                batch[key] = (
                    tensor.permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255.0
                )
            else:
                batch[key] = (
                    torch.from_numpy(value).to(device).unsqueeze(0).to(torch.float32)
                )
        with torch.no_grad():
            action = policy.select_action(batch)
        return action.squeeze(0).cpu().numpy()

    return VisionPolicy(
        name=repo_id.rsplit("/", maxsplit=1)[-1],
        act=act,
        reset=policy.reset,
    )
