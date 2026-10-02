"""Whose robot the data came from — the word the loop's Sys ID stage
shows beside its light (2026-09-24).

A fit on a public log is a real fit of a real robot, and not ours; a
fit on simulation is a method check; a fit on the operator's own robot
is the thesis. Recordings carry the word from their adapter, fit
records copy it, the index takes the strongest among the proving
records, the Studio prints it. Spelled once here, at the bottom of the
tiers, so a record and an ingest seam read the same list.
"""

from __future__ import annotations

BASIS_OWN = "own robot"
BASIS_PUBLIC = "public log"
BASIS_SIMULATION = "simulation"
BASIS_UNKNOWN = "unknown"
# Strongest first: an own-robot fit outranks a public log, which
# outranks a simulation.
BASES = (BASIS_OWN, BASIS_PUBLIC, BASIS_SIMULATION, BASIS_UNKNOWN)
