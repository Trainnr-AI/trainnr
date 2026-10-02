"""Stage ② — the robot as a measured artifact, not a downloaded file."""

from trainnr.robot.identify import (
    ExcitationData,
    IdentificationResult,
    IdentifiedParameter,
    ParameterSpec,
    identify,
    staged_excitation,
)
from trainnr.robot.model_checks import DeadModelError, assert_model_alive

__all__ = [
    "DeadModelError",
    "ExcitationData",
    "IdentificationResult",
    "IdentifiedParameter",
    "ParameterSpec",
    "assert_model_alive",
    "identify",
    "staged_excitation",
]
