"""The lowest-friction on-ramp: a CSV of your robot's log → a fit.

The 2026-08-25 product review found the aperture problem: the proven
recording chain assumes this repo's exact rig (Pico wire format, 50 Hz
status, sweep-shaped duty), while the fit machinery itself only ever
needed three row-aligned arrays. Any robot with any logger can produce
those. This module is that missing bridge — the entry point an outside
practitioner reaches first.

The contract it inherits from `identify()` (stated here so the CSV
author never has to source-dive):

- `times` in seconds, strictly increasing;
- `controls` in the model's actuator units, one column per actuator,
  in the model's actuator order;
- `measurements` = the model's sensor outputs, one column per sensor,
  in the model's sensor order;
- the run starts at rest at the model's home state — trim anything
  recorded mid-motion.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from trainnr.robot.identify import ExcitationData


def excitation_from_csv(
    path: Path | str,
    *,
    time: str,
    controls: Sequence[str],
    measurements: Sequence[str],
) -> ExcitationData:
    """Read named columns from a headed CSV into identification data."""
    import numpy as np  # noqa: PLC0415 - keeps the fast import path light

    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: no header row")
        missing = [
            name
            for name in (time, *controls, *measurements)
            if name not in reader.fieldnames
        ]
        if missing:
            raise ValueError(
                f"{path}: missing column(s) {missing}; header has {reader.fieldnames}"
            )
        rows = list(reader)
    minimum_rows = 2  # any fewer and no dynamics are visible at all
    if len(rows) < minimum_rows:
        raise ValueError(f"{path}: need at least {minimum_rows} rows, got {len(rows)}")

    def column(name: str) -> np.ndarray:
        try:
            return np.array([float(row[name]) for row in rows])
        except ValueError as error:
            raise ValueError(f"{path}: non-numeric value in column {name!r}") from error

    times = column(time)
    if not np.all(np.diff(times) > 0):
        raise ValueError(f"{path}: column {time!r} must be strictly increasing")
    return ExcitationData(
        times=times,
        controls=np.column_stack([column(name) for name in controls]),
        measurements=np.column_stack([column(name) for name in measurements]),
    )
