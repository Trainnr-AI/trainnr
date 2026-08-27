"""The plugin door both registries share: import the built-in modules,
then every entry point of a group — and when a plugin fails to import,
say WHICH one, instead of one stranger's broken package taking the
listing down with a bare traceback. Standard library only."""

from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module
from importlib.metadata import entry_points


def load_group(group: str, builtin_modules: Iterable[str]) -> None:
    """Import the built-ins (a checkout works before its entry points
    are installed), then load every installed entry point of `group`."""
    for module in builtin_modules:
        import_module(module)
    load_entries(group, entry_points(group=group))


def load_entries(group: str, entries: Iterable) -> None:
    """Load entry points; a failure names the plugin, then re-raises."""
    for entry in entries:
        try:
            entry.load()
        except Exception as error:
            raise ImportError(
                f"plugin {group}:{entry.name} ({entry.value}) failed to import: {error}"
            ) from error
