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

_HASH_BITS = 53  # a float's mantissa: the largest exact integer fraction


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
        return f"{self.host}.{self.name}"


def _unit(protocol_hash: str, key: str, trial: int, component: int) -> float:
    """A deterministic point in [0, 1) for (protocol, key, trial, component)."""
    digest = hashlib.sha256(
        f"{protocol_hash}|{key}|{trial}|{component}".encode()
    ).digest()
    return (int.from_bytes(digest[:8], "big") >> (64 - _HASH_BITS)) / float(
        1 << _HASH_BITS
    )


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
            span = "choice " + "|".join(v.sampler.labels)
        else:
            span = f"uniform {list(v.sampler.low)}..{list(v.sampler.high)}"
        lines.append(f"{v.key} [{state}] {span}")
    return lines


def by_key(variations: Sequence[Variation]) -> Mapping[str, Variation]:
    return {v.key: v for v in variations}


def parse_variation(text: str) -> Variation:
    """`host.name=low:high` (comma-separated components for vectors) or
    `host.name=a|b|c` — the CLI form (`--env.variations=...`)."""
    key, _, spec = text.partition("=")
    host, dot, name = key.strip().rpartition(".")
    if not dot or not host or not name or not spec:
        raise ValueError(f"expected host.name=low:high or host.name=a|b, got {text!r}")
    if "|" in spec:
        return Variation(host, name, Choice(tuple(s.strip() for s in spec.split("|"))))
    low_text, colon, high_text = spec.partition(":")
    if not colon:
        raise ValueError(f"a uniform variation needs low:high, got {text!r}")
    low = tuple(float(x) for x in low_text.split(","))
    high = tuple(float(x) for x in high_text.split(","))
    return Variation(host, name, Uniform(low, high))


def sensitivity_table(
    records: Sequence[Any],
    variations: Sequence[Variation],
    *,
    alpha: float,
    delta: float,
    confidence: float = 0.95,
) -> list[Any]:
    """One main effect per enabled factor (per component for vectors,
    per label for choices) over the records' recorded draws — the table
    the certificate prints (stats/effects.py). Records that lack a
    factor are refused: an unrecorded draw is a protocol bug, not a
    missing value."""
    from rq_pipeline.stats.effects import (  # noqa: PLC0415 - stats is the lower layer
        SPLIT_LABEL_HIGH,
        SPLIT_LABEL_LOW,
        main_effect,
        split_continuous,
    )

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
                    SPLIT_LABEL_LOW,
                    SPLIT_LABEL_HIGH,
                    below,
                    above,
                    alpha=alpha,
                    delta=delta,
                    confidence=confidence,
                )
            )
    return effects
