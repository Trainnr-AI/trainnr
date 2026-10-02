"""Hash-versioned artifacts: nothing in the pipeline is nameable without one.

A demo, a policy or an evaluation is meaningless without the scene, robot
and calibration versions it was produced under — calibration is device
state, and an episode that does not record which calibration produced it
silently poisons every later fine-tune.
"""

from trainnr.bundles.hashing import bundle_hash, stamp
from trainnr.bundles.profile import PROFILE_FILE, RobotProfile, load_profile

__all__ = ["PROFILE_FILE", "RobotProfile", "bundle_hash", "load_profile", "stamp"]
