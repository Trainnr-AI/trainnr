"""The honesty layer: every number the pipeline reports passes through here.

No leaderboard in this field publishes confidence intervals and no shipping
framework computes them — that absence is the product's opening, so this
module is deliberately dependency-free: an audited report must be
recomputable on any Python, anywhere.
"""

from trainnr.stats.intervals import clopper_pearson, wilson
from trainnr.stats.ranking import (
    bootstrap_rank_ci,
    fisher_rank_ci,
    spearman,
    top_pick_probability,
)

__all__ = [
    "bootstrap_rank_ci",
    "clopper_pearson",
    "fisher_rank_ci",
    "spearman",
    "top_pick_probability",
    "wilson",
]
