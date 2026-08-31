"""rq_mjlab: identified actuator physics for mjlab, with refusals.

The maintained mjlab consumer of docs/e2e-research/58 §2 — BAM's law
from the certified bundle store, the field-expansion event made
unforgettable, and a linter that turns silent DR no-ops into errors.
"""

from rq_mjlab.actuator import BamActuator, BamActuatorCfg
from rq_mjlab.bundle import BundleRefused, verified_bundle
from rq_mjlab.dr import bam_param_dr_event, dr_from_bundle
from rq_mjlab.entity import entity_from_bundle
from rq_mjlab.events import bam_expansion_event, expand_bam_fields
from rq_mjlab.kernel import LawParams, duty, external_torque, friction_budget, torque
from rq_mjlab.linter import SilentNoOp, lint

__all__ = [
    "BamActuator",
    "BamActuatorCfg",
    "BundleRefused",
    "LawParams",
    "SilentNoOp",
    "bam_expansion_event",
    "bam_param_dr_event",
    "dr_from_bundle",
    "duty",
    "entity_from_bundle",
    "expand_bam_fields",
    "external_torque",
    "friction_budget",
    "lint",
    "torque",
    "verified_bundle",
]

__version__ = "0.1.0"
