"""The servo-DR rule, once: how to scale a position servo's stiffness.

The hard-won half is the ACTUATOR rule — `gainprm[0]` (kp on the
command) and `biasprm[1]` (-kp on the position) scaled TOGETHER.
Scaling `gainprm[0]` alone multiplies the SETPOINT, not the stiffness:
force = kp'*ctrl - kp*q settles at q = (kp'/kp)*ctrl. The demo
generator did exactly that until 2026-08-26 — a 5 % "gain" change moved
every joint target 5 % (3 deg on the elbow), and the scripted expert
"only worked at nominal" (2/10 at +-10 %, 0/10 at +-30 %). With both
terms scaled it keeps 8/10 at +-10 % and 10/10 at +-30 % with no
retries.

`tasks/aloha2/rig.py` said "One function, so a tool cannot get it wrong
again" — and then a second copy grew in `collect/scripted_demos.py`
with a different joint filter (review, 2026-09-01). This module is that
one function; the rigs keep only their own JOINT PREDICATE, which is
the only thing they ever disagreed about.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Which joints carry the rig's damping — the one rig-specific choice.
JointPredicate = Callable[[Any], bool]


def damped_joints(joint: Any) -> bool:
    """Every joint that declares damping — the rig-neutral predicate
    (a free object's joints declare none, so they are left alone)."""
    return bool(joint.damping[0] > 0)


def named_joints(prefixes: tuple[str, ...]) -> JointPredicate:
    """The joints whose names start with one of `prefixes` — for a rig
    whose scene holds damped joints it must NOT randomise."""

    def predicate(joint: Any) -> bool:
        return bool(joint.name.startswith(prefixes))

    return predicate


def scale_servo_dynamics(
    spec: Any,
    *,
    damping_scale: float,
    gain_scale: float,
    joints: JointPredicate = damped_joints,
) -> None:
    """Domain randomisation on an UNCOMPILED spec, in place: the
    selected joints' damping, and EVERY actuator's stiffness under the
    both-terms rule above."""
    for joint in spec.joints:
        if joints(joint):
            joint.damping[0] = joint.damping[0] * damping_scale
    for actuator in spec.actuators:
        actuator.gainprm[0] = actuator.gainprm[0] * gain_scale
        actuator.biasprm[1] = actuator.biasprm[1] * gain_scale
