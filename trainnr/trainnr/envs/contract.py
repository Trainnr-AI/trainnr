"""The env's contract with the ecosystem, as constants — spelled once.

Observation keys are LeRobot's raw names (`lerobot.envs.utils.
preprocess_observation` renames them to `observation.images.<key>` and
`observation.state`); the success flag is emitted under both the name
LeRobot reads and the one GR00T, SimplerEnv and robomimic read
(docs/e2e-research/46 §3). Tools and tests import these instead of
retyping them: a contract string that exists in one place cannot drift.
Standard library only.
"""

from __future__ import annotations

# Re-exported: the env's consumers read the channel count here, the
# exporters (a tier below envs) from protocol, and it is one number.
from trainnr.protocol import RGB_CHANNELS
from trainnr.tasks.registry import BUILTIN_NAMESPACE

__all__ = ["BUILTIN_NAMESPACE", "RGB_CHANNELS", "InfoKeys", "ObservationKeys"]


class ObservationKeys:
    PIXELS = "pixels"  # {camera key: uint8 (H, W, 3)}
    AGENT_POS = "agent_pos"  # float32 (state_width,)


class InfoKeys:
    IS_SUCCESS = "is_success"  # what LeRobot's rollout reads
    SUCCESS = "success"  # what GR00T, SimplerEnv, robomimic read
    TRIAL = "trial"  # the pairing key


ENV_TYPE = BUILTIN_NAMESPACE  # `--env.type=trainnr`; also the gym id namespace
GYM_ID_VERSION = "v0"
RENDER_MODE = "rgb_array"
DEFAULT_POLICY_NAME = "policy"  # what a row calls a policy the env never sees
PIXEL_MAX = 255
XYZ = 3


def gym_id(task_id: str) -> str:
    """`trainnr/kitting` → `trainnr/kitting-v0`: gymnasium's `namespace/name-vN`."""
    return f"{task_id}-{GYM_ID_VERSION}"
