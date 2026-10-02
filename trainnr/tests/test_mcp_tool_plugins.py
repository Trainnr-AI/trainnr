"""The tool extension seam: an installed package registers tools on the
same server through the `trainnr.mcp_tools` entry-point group (docs/83)."""

import unittest
from importlib.metadata import EntryPoint
from unittest import mock

from trainnr import mcp_server


class FakeServer:
    def __init__(self) -> None:
        self.tools: list[str] = []

    def tool(self, description: str = ""):
        def deco(fn):
            self.tools.append(fn.__name__)
            return fn

        return deco


def register_ok(server) -> None:
    @server.tool(description="a cloud door")
    def submit_job() -> dict:
        return {}


def register_bad(server) -> None:
    raise ValueError("misconfigured")


class ToolPlugins(unittest.TestCase):
    def _entries(self, name: str, target: str) -> list[EntryPoint]:
        return [EntryPoint(name=name, value=target, group=mcp_server.TOOL_PLUGIN_GROUP)]

    def test_a_plugin_adds_its_tools_after_the_builtins(self) -> None:
        server = FakeServer()
        with mock.patch(
            "importlib.metadata.entry_points",
            return_value=self._entries("cloud", f"{__name__}:register_ok"),
        ):
            names = mcp_server.register_plugin_tools(server)
        self.assertEqual(names, ["cloud"])
        self.assertEqual(server.tools, ["submit_job"])

    def test_a_failing_plugin_is_named(self) -> None:
        with (
            mock.patch(
                "importlib.metadata.entry_points",
                return_value=self._entries("broken", f"{__name__}:register_bad"),
            ),
            self.assertRaisesRegex(RuntimeError, "broken"),
        ):
            mcp_server.register_plugin_tools(FakeServer())

    def test_no_plugins_is_the_default(self) -> None:
        with mock.patch("importlib.metadata.entry_points", return_value=[]):
            self.assertEqual(mcp_server.register_plugin_tools(FakeServer()), [])


if __name__ == "__main__":
    unittest.main()
