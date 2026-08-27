"""A policy served over openpi's websocket protocol, as a `ChunkPolicy`.

openpi (Physical Intelligence) serves pi0 / pi0.5 from its own process:
`serve_policy.py` opens a websocket, sends its metadata, then answers
each msgpack-packed observation with `{"actions": (horizon, nu), ...}`.
Their client (`openpi_client.WebsocketClientPolicy`, the `remote` extra)
speaks that protocol; this module only packs OUR observation into
their request and hands the chunk to `evaluate.scheduler.ActionScheduler`,
so the executed horizon is the protocol's, not the server's, and pi0.5
evaluates through the same env as a LeRobot checkpoint, paired trial for
paired trial (docs/e2e-research/43 §3 item 3).

The rig adapter is the camera mapping: which env camera feeds which of
openpi's image names (`OpenpiRequest.cameras`), and the state and the
prompt. Joint conventions are NOT translated here — a checkpoint
trained on a real ALOHA expects that robot's state; the bundle's own
jointpos is what our env observes, and a translation, when a checkpoint
needs one, belongs beside `tasks.aloha2.act_sim_state`, wrapped around
`predict` from outside (the ActionSpace pattern).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from rq_pipeline.envs.contract import ObservationKeys
from rq_pipeline.evaluate.scheduler import ChunkPolicy

DEFAULT_PORT = 8000  # openpi's serve_policy.py default


class OpenpiKeys:
    """openpi's observation dict, as `AlohaInputs` reads it: images by
    their names (CHW uint8), a state vector, a prompt; the reply's chunk."""

    IMAGES = "images"
    STATE = "state"
    PROMPT = "prompt"
    ACTIONS = "actions"
    # The ALOHA image names openpi's policy accepts; a missing one is
    # masked out on their side.
    ALOHA_CAMERAS = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")


@dataclass(frozen=True)
class OpenpiRequest:
    """The rig adapter: env camera key -> openpi image name, plus the
    prompt. `pack(observation)` is the request their server reads."""

    cameras: Mapping[str, str] = field(default_factory=dict)
    prompt: str = ""

    def pack(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np  # noqa: PLC0415

        pixels = observation[ObservationKeys.PIXELS]
        missing = sorted(set(self.cameras) - set(pixels))
        if missing:
            raise KeyError(
                f"openpi request names env cameras {missing}; the env has "
                f"{sorted(pixels)}"
            )
        images = {
            name: np.ascontiguousarray(
                np.transpose(pixels[key], (2, 0, 1))
            )  # HWC -> CHW
            for key, name in self.cameras.items()
        }
        request = {
            OpenpiKeys.IMAGES: images,
            OpenpiKeys.STATE: np.asarray(
                observation[ObservationKeys.AGENT_POS], dtype=np.float32
            ),
        }
        if self.prompt:
            request[OpenpiKeys.PROMPT] = self.prompt
        return request


def _require_client() -> Any:
    try:
        from openpi_client import websocket_client_policy  # noqa: PLC0415
    except ImportError as error:
        raise ImportError(
            "openpi's client is the 'remote' extra: uv sync --extra remote "
            "(it pins numpy<2; the extra overrides that pin, measured working "
            "with numpy 2 on 2026-08-27)"
        ) from error
    return websocket_client_policy.WebsocketClientPolicy


def openpi_chunk_policy(
    host: str,
    port: int = DEFAULT_PORT,
    *,
    request: OpenpiRequest,
    name: str | None = None,
    api_key: str | None = None,
) -> ChunkPolicy:
    """Connect (their client retries every 5 s until the server answers
    and reads its metadata), and return a `ChunkPolicy` whose `predict`
    is one round trip. A server-side exception arrives as a string and
    their client raises `RuntimeError("Error in inference server: ...")`
    — the run stops with the traceback, never with a zero action."""
    client_class = _require_client()
    client = client_class(host=host, port=port, api_key=api_key)

    def predict(observation: Mapping[str, Any]) -> Any:
        import numpy as np  # noqa: PLC0415

        reply = client.infer(request.pack(observation))
        if OpenpiKeys.ACTIONS not in reply:
            raise KeyError(
                f"openpi server replied without {OpenpiKeys.ACTIONS!r}: {sorted(reply)}"
            )
        return np.asarray(reply[OpenpiKeys.ACTIONS], dtype=float)

    return ChunkPolicy(
        name=name or f"openpi@{host}:{port}", predict=predict, reset=client.reset
    )
