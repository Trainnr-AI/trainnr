"""Variations as data: what an evaluation sweeps, drawn by trial index.

Written 2026-08-26 (docs/30 §7 row 41, docs/32). Arena's shape — a named
knob on a scene host with a sampler, a dotted key `host.name`, off by
default, listable before a trial is spent — with one change that
matters: **the draw takes the trial index, never an RNG.** Arena's
samplers hit torch's global generator, so nothing is paired; here the
value for trial *t* is a hash of (protocol, key, t), so every policy sees
the identical factor vector on trial *t* and a sensitivity comparison
between policies is paired too. `Choice` labels go round-robin by trial:
exact balance, where Arena can only warn at 1.5x.

Stdlib only: the certificate's sensitivity table (stats/effects.py) must
recompute the same draws on an auditor's laptop. Applying a value to a
simulator is the env's job (rq_pipeline.envs); this module knows nothing
about MuJoCo.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rq_pipeline.stats.effects import SplitLabel, main_effect, split_continuous
from rq_pipeline.stats.intervals import DEFAULT_CONFIDENCE

_HASH_BITS = 53  # a float's mantissa: the largest exact integer fraction
_HASH_BYTES = 8  # the digest prefix the bits are taken from
KEY_SEPARATOR = "."


class CliGrammar:
    """`host.name=low:high` / `host.name=a|b` — the CLI form, spelled once
    for `parse_variation` and `describe`."""

    ASSIGN = "="
    RANGE = ":"
    COMPONENT = ","
    CHOICE = "|"


class VariationKeys:
    """The vocabulary, spelled once. Hosts that are fixed words, and the
    knob names an engine's appliers register under (physics/variations.py):
    `joints.damping_scale`, `actuators.gain_scale`, `lights.diffuse_scale`,
    `<body>.mass_scale`, `<camera>.offset_m`."""

    JOINTS = "joints"
    ACTUATORS = "actuators"
    LIGHTS = "lights"
    # MuJoCo's built-in camera-attached light (model.vis.headlight) —
    # the ONLY light a scene without <light> elements has, and so the
    # brightness knob that actually works there (measured 2026-09-02:
    # the so101 scenes compile with nlight == 0).
    HEADLIGHT = "headlight"
    DAMPING_SCALE = "damping_scale"
    GAIN_SCALE = "gain_scale"
    DIFFUSE_SCALE = "diffuse_scale"
    MASS_SCALE = "mass_scale"
    OFFSET_M = "offset_m"

    # The purely-visual knobs — what a generation-time visual-DR spec
    # may draw (docs/66 §4). collect/scripted_demos.py refuses the
    # rest: dynamics draws have their own contract and their own basis.
    VISUAL_NAMES = frozenset({DIFFUSE_SCALE, OFFSET_M})

    @staticmethod
    def key(host: str, name: str) -> str:
        return f"{host}{KEY_SEPARATOR}{name}"

    @staticmethod
    def split(key: str) -> tuple[str, str]:
        host, _, name = key.rpartition(KEY_SEPARATOR)
        return host, name


@dataclass(frozen=True)
class Uniform:
    """Uniform over a box; `low`/`high` per component (one for scalars)."""

    low: tuple[float, ...]
    high: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.low) != len(self.high) or not self.low:
            raise ValueError(f"low and high must match and be non-empty: {self}")
        if any(h < lo for lo, h in zip(self.low, self.high, strict=True)):
            raise ValueError(f"high must be >= low: {self}")

    @property
    def midpoint(self) -> tuple[float, ...]:
        return tuple((lo + h) / 2.0 for lo, h in zip(self.low, self.high, strict=True))


@dataclass(frozen=True)
class Choice:
    """A finite set of labels, assigned round-robin by trial."""

    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.labels)) != len(self.labels) or not self.labels:
            raise ValueError(f"labels must be unique and non-empty: {self.labels}")


@dataclass(frozen=True)
class Variation:
    """One knob: where it lives (`host`), what it is (`name`), how it is
    drawn. The key `host.name` is the override and record key; the env
    maps it to a simulator write (a joint damping scale, a body mass, a
    camera offset, a light level) and snapshots the nominal value first
    so draws never compound across resets."""

    host: str
    name: str
    sampler: Uniform | Choice
    enabled: bool = True

    @property
    def key(self) -> str:
        return VariationKeys.key(self.host, self.name)


def _unit(protocol_hash: str, key: str, trial: int, component: int) -> float:
    """A deterministic point in [0, 1) for (protocol, key, trial, component)."""
    digest = hashlib.sha256(
        f"{protocol_hash}|{key}|{trial}|{component}".encode()
    ).digest()
    return (
        int.from_bytes(digest[:_HASH_BYTES], "big") >> (8 * _HASH_BYTES - _HASH_BITS)
    ) / float(1 << _HASH_BITS)


def draw(variation: Variation, trial: int, protocol_hash: str) -> Any:
    """The value trial `trial` sees — the same for every policy."""
    if trial < 0:
        raise ValueError(f"trial must be >= 0, got {trial}")
    sampler = variation.sampler
    if isinstance(sampler, Choice):
        return sampler.labels[trial % len(sampler.labels)]
    values = tuple(
        lo + _unit(protocol_hash, variation.key, trial, i) * (h - lo)
        for i, (lo, h) in enumerate(zip(sampler.low, sampler.high, strict=True))
    )
    return values[0] if len(values) == 1 else values


def draw_random(variation: Variation, rng: Any) -> Any:
    """The value one PRESS attempt sees, drawn from `rng` (a numpy
    Generator) — generation wants the run seed's stream, where `draw`'s
    trial-hash pairing is an evaluation contract. Sampler semantics
    have ONE home either way: Choice picks a label, Uniform draws each
    component of its box."""
    sampler = variation.sampler
    if isinstance(sampler, Choice):
        return sampler.labels[int(rng.integers(len(sampler.labels)))]
    values = tuple(
        float(rng.uniform(lo, high))
        for lo, high in zip(sampler.low, sampler.high, strict=True)
    )
    return values[0] if len(values) == 1 else values


def draw_all(
    variations: Sequence[Variation], trial: int, protocol_hash: str
) -> dict[str, Any]:
    """Every ENABLED variation's value for the trial, keyed `host.name`."""
    keys = [v.key for v in variations]
    if len(set(keys)) != len(keys):
        raise ValueError(f"variation keys must be unique, got {keys}")
    return {v.key: draw(v, trial, protocol_hash) for v in variations if v.enabled}


def describe(variations: Sequence[Variation]) -> list[str]:
    """`--list-variations`: the whole space, one line each, before a
    trial is spent."""
    lines = []
    for v in variations:
        state = "on" if v.enabled else "off"
        if isinstance(v.sampler, Choice):
            span = "choice " + CliGrammar.CHOICE.join(v.sampler.labels)
        else:
            span = f"uniform {list(v.sampler.low)}..{list(v.sampler.high)}"
        lines.append(f"{v.key} [{state}] {span}")
    return lines


def by_key(variations: Sequence[Variation]) -> Mapping[str, Variation]:
    return {v.key: v for v in variations}


def parse_variation(text: str) -> Variation:
    """`host.name=low:high` (comma-separated components for vectors) or
    `host.name=a|b|c` — the CLI form (`--env.variations=...`)."""
    key, _, spec = text.partition(CliGrammar.ASSIGN)
    host, name = VariationKeys.split(key.strip())
    if not host or not name or not spec:
        raise ValueError(f"expected host.name=low:high or host.name=a|b, got {text!r}")
    if CliGrammar.CHOICE in spec:
        labels = tuple(s.strip() for s in spec.split(CliGrammar.CHOICE))
        return Variation(host, name, Choice(labels))
    low_text, colon, high_text = spec.partition(CliGrammar.RANGE)
    if not colon:
        raise ValueError(f"a uniform variation needs low:high, got {text!r}")
    low = tuple(float(x) for x in low_text.split(CliGrammar.COMPONENT))
    high = tuple(float(x) for x in high_text.split(CliGrammar.COMPONENT))
    return Variation(host, name, Uniform(low, high))


def sensitivity_table(
    records: Sequence[Any],
    variations: Sequence[Variation],
    *,
    alpha: float,
    delta: float,
    confidence: float = DEFAULT_CONFIDENCE,
) -> list[Any]:
    """One main effect per enabled factor (per component for vectors,
    per label for choices) over the records' recorded draws — the table
    the certificate prints (stats/effects.py). Records that lack a
    factor are refused: an unrecorded draw is a protocol bug, not a
    missing value."""
    effects = []
    outcomes = [bool(r.success) for r in records]
    for variation in variations:
        if not variation.enabled:
            continue
        key = variation.key
        missing = [r for r in records if key not in r.variations]
        if missing:
            raise ValueError(f"{len(missing)} record(s) carry no draw for {key!r}")
        values = [r.variations[key] for r in records]
        sampler = variation.sampler
        if isinstance(sampler, Choice):
            for label in sampler.labels:
                inside = [
                    o for v, o in zip(values, outcomes, strict=True) if v == label
                ]
                outside = [
                    o for v, o in zip(values, outcomes, strict=True) if v != label
                ]
                effects.append(
                    main_effect(
                        key,
                        f"not {label}",
                        label,
                        outside,
                        inside,
                        alpha=alpha,
                        delta=delta,
                        confidence=confidence,
                    )
                )
            continue
        width = len(sampler.low)
        for component, midpoint in enumerate(sampler.midpoint):
            scalars = [float(v) if width == 1 else float(v[component]) for v in values]
            below, above = split_continuous(scalars, outcomes, midpoint)
            name = key if width == 1 else f"{key}[{component}]"
            effects.append(
                main_effect(
                    name,
                    SplitLabel.LOW,
                    SplitLabel.HIGH,
                    below,
                    above,
                    alpha=alpha,
                    delta=delta,
                    confidence=confidence,
                )
            )
    return effects
