"""LeRobot's door to our tasks: the `EnvConfig` that `lerobot-eval` and
`lerobot-train` load.

    # any OS:
    lerobot-eval --env.type=trainnr --env.task=kitting \\
        --env.discover_packages_path=trainnr.envs \\
        --policy.path=runs/t5-act-kitting/checkpoints/020000/pretrained_model \\
        --seed=1000 --eval.n_episodes=4 --eval.batch_size=1 \\
        --eval.use_async_envs=false
    # on WSL, under trainnr/wsl.env (tools/wsl-run.sh): llvmpipe otherwise
    # renders at 317 ms a frame (the log 2026-08-26)

`--env.discover_packages_path` (lerobot/configs/parser.py) imports every
module of the named PACKAGE before the CLI is parsed — a module path is
refused, it walks `__path__` — which runs the registration decorator
below; a distribution named `lerobot_env_*` is imported the same way with
no flag. Everything LeRobot needs beyond gymnasium lives here and nowhere
else: the feature declarations a policy config reads (`features`,
`features_map`) and the vector-env factory. Physics, cameras, referee and
pairing are the gymnasium env's (trainnr.envs.gymnasium_env); the task
comes from the registry (`trainnr.tasks.registry`), so a plugin
package's `acme/pour` is `--env.task=acme/pour` with nothing here edited.

Pairing survives the runner: `lerobot-eval` seeds episode i with
`seed + i`, and the env starts trial `seed % trials` — so every policy's
episode i begins from the same state, and the recorded `seed` lets the
per-episode rows fold into paired counts.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium.vector import AutoresetMode
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from trainnr.envs.contract import (
    DEFAULT_POLICY_NAME,
    ENV_TYPE,
    RGB_CHANNELS,
    ObservationKeys,
)
from trainnr.envs.gymnasium_env import make_env
from trainnr.envs.hold import ActionHold
from trainnr.evaluate.variations import parse_variation
from trainnr.tasks.registry import resolve, tasks

# Worker processes must each build their own GL context; `spawn` exists
# on every OS (`forkserver` does not on Windows) and gives that for free.
WORKER_START_METHOD = "spawn"
# LeRobot's CLI namespaces every env option under this prefix, and
# discovers a plugin by importing the PACKAGE that registers it.
ENV_FLAG_PREFIX = "--env."
PLUGIN_PACKAGE = __name__.rsplit(".", 1)[0]


def held_env(*, frame_every: int = 1, **kwargs: Any) -> gym.Env:
    """`make_env`, holding each action `frame_every` ticks (envs/hold.py)."""
    env = make_env(**kwargs)
    return ActionHold(env, frame_every) if frame_every > 1 else env


@EnvConfig.register_subclass(ENV_TYPE)
@dataclass
class TrainnrEnvConfig(EnvConfig):
    """`--env.type=trainnr --env.task=<id>`; features and fps come from
    the task. `task` has no default on purpose — the repo's rule is no
    silent default where a wrong value is possible, and evaluating the
    wrong task silently is exactly that."""

    task: str | None = None
    # `--env.record_to=<file>.jsonl --env.policy_name=<name>`: the env
    # appends our full per-episode row (verdict, milestones, seed) that
    # eval_info.json lacks.
    record_to: str | None = None
    policy_name: str = DEFAULT_POLICY_NAME
    # Distinct paired starts; None keeps the task spec's count. An
    # evaluation of N episodes wants N (see `make_env`).
    trials: int | None = None
    # `--env.variations='["joints.damping_scale=0.7:1.3",
    #                     "top.offset_m=-0.03,-0.03,-0.03:0.03,0.03,0.03"]'`
    # — each drawn by trial index (paired across policies), applied at
    # reset, written into the row.
    variations: list[str] = field(default_factory=list)
    # The dataset's cadence in control ticks (the press's --frame-every):
    # the env holds each action this many ticks and reports the frame
    # rate the policy was trained at. 1 is the harness's native 50 Hz.
    frame_every: int = 1
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task is None:
            raise ValueError(
                f"--env.task is required; the registry knows {sorted(tasks())}"
            )
        # Building the task (an MjSpec, no compile) is cheap; its state
        # width, cameras and control rate are the features. The action is
        # one command per servo and the state one reading per servo: the
        # same width.
        try:
            built = resolve(self.task).build()
        except KeyError as error:  # a config error is a ValueError, like the rest
            raise ValueError(str(error)) from error
        if self.frame_every < 1 or built.control_hz % self.frame_every:
            raise ValueError(
                f"--env.frame_every={self.frame_every} does not divide the task's "
                f"{built.control_hz} Hz control rate"
            )
        self.fps = built.control_hz // self.frame_every
        width = built.state_width
        self.features = {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(width,)),
            ObservationKeys.AGENT_POS: PolicyFeature(
                type=FeatureType.STATE, shape=(width,)
            ),
        }
        self.features_map = {ACTION: ACTION, ObservationKeys.AGENT_POS: OBS_STATE}
        for camera in built.cameras:
            key = f"{ObservationKeys.PIXELS}/{camera.key}"
            self.features[key] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(camera.height, camera.width, RGB_CHANNELS),
            )
            self.features_map[key] = f"{OBS_IMAGES}.{camera.key}"

    @classmethod
    def cli_flags(  # noqa: PLR0913 - the env's knobs on a command line, each named
        cls,
        task: str,
        *,
        record_to: str | Path | None = None,
        policy_name: str | None = None,
        variations: Sequence[str] = (),
        trials: int | None = None,
        frame_every: int | None = None,
    ) -> list[str]:
        """The `--env.*` arguments that select this plugin on LeRobot's
        command lines (`lerobot-train`, `lerobot-eval`): the type, the
        task, the package LeRobot must import to find the registration,
        and the optional record / policy-name / variation knobs — spelled
        here once, so no tool restates a flag name."""
        flags = [
            f"{ENV_FLAG_PREFIX}type={ENV_TYPE}",
            f"{ENV_FLAG_PREFIX}task={task}",
            f"{ENV_FLAG_PREFIX}discover_packages_path={PLUGIN_PACKAGE}",
        ]
        if record_to is not None:
            flags.append(f"{ENV_FLAG_PREFIX}record_to={record_to}")
        if policy_name is not None:
            flags.append(f"{ENV_FLAG_PREFIX}policy_name={policy_name}")
        if variations:
            flags.append(f"{ENV_FLAG_PREFIX}variations={json.dumps(list(variations))}")
        if trials is not None:
            flags.append(f"{ENV_FLAG_PREFIX}trials={trials}")
        if frame_every is not None:
            flags.append(f"{ENV_FLAG_PREFIX}frame_every={frame_every}")
        return flags

    @property
    def gym_kwargs(self) -> dict:
        return {
            "task": self.task,
            "record_to": self.record_to,
            "policy_name": self.policy_name,
            "variations": tuple(parse_variation(text) for text in self.variations),
            "trials": self.trials,
        }

    def create_envs(
        self, n_envs: int, use_async_envs: bool = False
    ) -> dict[str, dict[int, gym.vector.VectorEnv]]:
        """`{suite: {task_id: VectorEnv}}`, the shape `lerobot.envs.make_env`
        documents. Async copies are spawned so each worker builds its own
        renderer (the env defers renderer creation to first use)."""
        factories = [
            partial(held_env, frame_every=self.frame_every, **self.gym_kwargs)
        ] * n_envs
        if use_async_envs and n_envs > 1:
            vec: gym.vector.VectorEnv = gym.vector.AsyncVectorEnv(
                factories,
                context=WORKER_START_METHOD,
                autoreset_mode=AutoresetMode.SAME_STEP,
            )
        else:
            vec = gym.vector.SyncVectorEnv(
                factories, autoreset_mode=AutoresetMode.SAME_STEP
            )
        return {self.type: {0: vec}}
