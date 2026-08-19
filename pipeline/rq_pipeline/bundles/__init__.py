"""Hash-versioned artifacts: nothing in the pipeline is nameable without one.

A demo, a policy or an evaluation is meaningless without the scene, robot
and calibration versions it was produced under — calibration is device
state, and an episode that does not record which calibration produced it
silently poisons every later fine-tune.
"""

from rq_pipeline.bundles.hashing import bundle_hash, stamp

__all__ = ["bundle_hash", "stamp"]
