"""The package's previous import name, kept for one release.

`import rq_pipeline.project` resolves to the very same module object as
`import trainnr.project` (a meta-path alias, not a copy), with a
DeprecationWarning once per process. Removed in the release after the
rename (docs/80 §5).
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings
from types import ModuleType

OLD = "rq_pipeline"
NEW = "trainnr"


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Every `rq_pipeline[.x]` import becomes the `trainnr[.x]` module."""

    def find_spec(self, name, path=None, target=None):  # type: ignore[override]
        if name == OLD or name.startswith(OLD + "."):
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):  # type: ignore[override]
        return importlib.import_module(NEW + spec.name[len(OLD) :])

    def exec_module(self, module: ModuleType) -> None:
        pass


if not any(isinstance(f, _Alias) for f in sys.meta_path):
    sys.meta_path.insert(0, _Alias())
warnings.warn(
    f"`{OLD}` is now `{NEW}`: update the import (the alias goes away next release)",
    DeprecationWarning,
    stacklevel=2,
)
_real = importlib.import_module(NEW)
sys.modules[__name__] = _real
