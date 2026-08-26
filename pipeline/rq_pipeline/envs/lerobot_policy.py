"""A LeRobot checkpoint as a function over the env's raw observation.

The path `lerobot-eval` takes — the checkpoint's config selects the
policy class, its pre/post-processors normalise and de-normalise, and
`preprocess_observation` renames our raw keys — packaged for tools that
drive the env themselves (train-watch's live viewers, notebooks) so none
of them hand-converts a tensor. Rig-specific translations (gym-aloha's
normalised grippers for the public checkpoints) stay with the rig
(`tasks.aloha2.act_sim_state` / `ctrl_from_act_sim_action`) and wrap
`act` from outside. Needs the `train` extra.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LoadedPolicy:
    name: str
    act: Callable[[dict[str, Any]], Any]  # raw env observation -> action (nu,)
    reset: Callable[[], None]  # before every episode: the chunk queue must not leak


def best_device(requested: str | None = None) -> str:
    """The best device this machine has, unless told otherwise."""
    if requested:
        return requested
    import torch  # noqa: PLC0415 - train extra

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_policy(path: Path | str, *, instruction: str, device: str) -> LoadedPolicy:
    try:
        import torch  # noqa: PLC0415
        from lerobot.configs.policies import PreTrainedConfig  # noqa: PLC0415
        from lerobot.envs.utils import preprocess_observation  # noqa: PLC0415
        from lerobot.policies.factory import (  # noqa: PLC0415
            get_policy_class,
            make_pre_post_processors,
        )
    except ImportError as error:
        raise ImportError(
            "checkpoint policies need the 'train' extra on Python >= 3.12 "
            "(uv sync --python 3.12 --extra train)"
        ) from error

    path = str(path)
    config = PreTrainedConfig.from_pretrained(path)
    config.device = device
    policy = get_policy_class(config.type).from_pretrained(path, config=config)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    def act(observation: dict[str, Any]) -> Any:
        batch = preprocess_observation(dict(observation))
        batch["task"] = [instruction]
        with torch.inference_mode():
            action = postprocessor(policy.select_action(preprocessor(batch)))
        return action.squeeze(0).cpu().numpy()

    return LoadedPolicy(
        name=Path(path).rstrip("/") if isinstance(path, str) else str(path),
        act=act,
        reset=policy.reset,
    )
