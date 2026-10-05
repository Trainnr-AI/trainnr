"""The Studio's launcher checks the viewer port the way the Studio's own
server will bind it: the same host, and on Linux past connections still
closing. Checking 0.0.0.0 without SO_REUSEADDR, a training job retrying
its stream kept an empty port "held" and the launch was refused
(2026-10-04)."""

import os
import socket
import sys
import unittest
from unittest import mock

from trainnr.project.control import (
    VIEWER_BIND_ENV,
    VIEWER_BIND_HOST,
    viewer_bind_host,
    viewer_port_free,
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind((VIEWER_BIND_HOST, 0))
        return int(s.getsockname()[1])


class TheProbe(unittest.TestCase):
    def test_a_live_listener_is_held(self) -> None:
        with socket.socket() as server:
            server.bind((VIEWER_BIND_HOST, 0))
            server.listen()
            self.assertFalse(viewer_port_free(server.getsockname()[1]))

    def test_an_empty_port_is_free(self) -> None:
        self.assertTrue(viewer_port_free(_free_port()))

    @unittest.skipUnless(sys.platform.startswith("linux"), "TIME_WAIT rule is Linux's")
    def test_a_closed_connection_does_not_hold_the_port(self) -> None:
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((VIEWER_BIND_HOST, 0))
            server.listen()
            port = server.getsockname()[1]
            client = socket.create_connection((VIEWER_BIND_HOST, port))
            accepted, _ = server.accept()
            accepted.close()  # the server side closes first: its TIME_WAIT
            client.close()
        self.assertTrue(viewer_port_free(port))

    def test_the_host_follows_the_studios_setting(self) -> None:
        with mock.patch.dict(os.environ, {VIEWER_BIND_ENV: "0.0.0.0:9876"}):
            self.assertEqual(viewer_bind_host(), "0.0.0.0")
        with mock.patch.dict(os.environ, {VIEWER_BIND_ENV: ""}):
            self.assertEqual(viewer_bind_host(), VIEWER_BIND_HOST)


if __name__ == "__main__":
    unittest.main()
