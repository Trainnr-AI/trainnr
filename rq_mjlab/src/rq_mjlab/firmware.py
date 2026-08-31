"""The servo firmwares' own constants — the one copy on this branch.

A certified actuator bundle (robots/actuator-bundles, schema
robotiq-actuator-bundle/1) carries BAM's IDENTIFIED parameters: kt, R,
armature, the friction budget, the command delay. It deliberately does
not carry the FIRMWARE's constants — the supply voltage, the gain
register, the register-to-duty gain, the pwm ceiling, the current
limiter — because those are facts about a servo family's controller,
not about one unit's fit. BAM measured them with an oscilloscope and
hard-codes them in its actuator classes (*bam/feetech/actuator.py*,
*bam/dynamixel/actuator.py*, 1.0.2); this module is that table as data,
with the numbers BAM published and nothing invented.

`kp` and `vin` are DEFAULTS, not truths: they are register values a
deployment chooses (microduck runs the XL330 at kp 200 where the family
constant is 400; LeRobot configures the STS3215 at P=16 where BAM's
bench ran 32). `BamActuatorCfg` therefore takes both as explicit
overrides, and the value used is part of the config — never silent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Firmware:
    """One servo family's controller, as BAM measured it.

    `error_gain` turns `kp_register * position_error [rad]` into a duty
    cycle in [-1, 1]; `max_pwm` is the duty ceiling; `max_current` an
    optional firmware current limiter [A] applied as a duty window
    before the pwm clip; `max_velocity` the firmware's internal
    rate limit on its target [rad/s] where the family has one (the
    STS3215 always does; Dynamixels do not).
    """

    family: str
    vin: float
    kp: float
    error_gain: float
    max_pwm: float
    max_current: float | None = None
    max_velocity: float | None = None


FIRMWARE: dict[str, Firmware] = {
    # *bam/feetech/actuator.py*: vin 7.4, kp 32, error_gain 0.166
    # ("determined using an oscilloscope"), max_pwm 0.97 (their own
    # comment asks "can we assume 1.0?" — a placeholder, kept as
    # published); the rate limit is 3400 counts/s on a 4096 encoder.
    "sts3215": Firmware(
        family="sts3215",
        vin=7.4,
        kp=32.0,
        error_gain=0.166,
        max_pwm=0.97,
        max_velocity=3400 * 2 * math.pi / 4096,
    ),
    # *bam/dynamixel/actuator.py*: vin 7.5, kp 400,
    # error_gain = (4096/2pi) / (256 * 885) (counts-per-radian over the
    # kp divisor times the pwm limit), max_pwm 1.0, current limit 1.75 A.
    "xl330": Firmware(
        family="xl330",
        vin=7.5,
        kp=400.0,
        error_gain=(4096 / (2 * math.pi)) / (256 * 885),
        max_pwm=1.0,
        max_current=1.75,
    ),
}


def firmware_for(actuator_slug: str) -> Firmware:
    """The firmware table entry for a bundle's `params.actuator` slug.

    Refuses an unknown family by name: a bundle whose fit exists but
    whose controller constants were never measured cannot run the
    voltage law honestly, and a made-up vin/kp would be exactly the
    silent guess this package exists to forbid.
    """
    if actuator_slug not in FIRMWARE:
        raise KeyError(
            f"no firmware constants for actuator family {actuator_slug!r}; "
            f"known: {sorted(FIRMWARE)}. The bundle's fit is usable only "
            "with its controller's measured vin/kp/error_gain/max_pwm — "
            "add them here with their provenance, never inline."
        )
    return FIRMWARE[actuator_slug]
