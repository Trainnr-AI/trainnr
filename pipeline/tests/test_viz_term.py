"""A process streaming into the Studio leaves on TERM with its stream
closed and TERM's exit status, so a stopped job never reads as a
finished one (2026-09-13 review: the first handler exited 0)."""

from __future__ import annotations

import signal
import subprocess
import sys
import threading
import unittest

from rq_pipeline.viz import TERM_EXIT_STATUS, leave_cleanly_on_term

CHILD = """
import sys, time
from rq_pipeline.viz import leave_cleanly_on_term
class Rr:
    def disconnect(self):
        sys.stdout.write("disconnected\\n"); sys.stdout.flush()
leave_cleanly_on_term(Rr())
sys.stdout.write("ready\\n"); sys.stdout.flush()
time.sleep(30)
"""


STUCK_CHILD = """
import sys, time
from rq_pipeline.viz import leave_cleanly_on_term
class Rr:
    def disconnect(self):
        time.sleep(60)  # a viewer that stopped answering
leave_cleanly_on_term(Rr())
sys.stdout.write("ready\\n"); sys.stdout.flush()
time.sleep(60)
"""


class LeavingOnTerm(unittest.TestCase):
    @unittest.skipIf(sys.platform == "win32", "SIGTERM is a POSIX signal")
    def test_a_stream_that_never_closes_does_not_hold_term(self) -> None:
        # A disconnect toward a viewer that stopped answering never returns;
        # the handler gives it TERM_FLUSH_S and leaves (2026-09-22: a smoke
        # train ignored TERM and needed KILL).
        import time  # noqa: PLC0415

        from rq_pipeline.viz import TERM_FLUSH_S  # noqa: PLC0415

        child = subprocess.Popen(
            [sys.executable, "-c", STUCK_CHILD], stdout=subprocess.PIPE, text=True
        )
        try:
            self.assertEqual(child.stdout.readline().strip(), "ready")
            began = time.monotonic()
            child.send_signal(signal.SIGTERM)
            child.wait(timeout=TERM_FLUSH_S + 10)
            waited = time.monotonic() - began
        finally:
            if child.poll() is None:
                child.kill()
            child.stdout.close()
        self.assertEqual(child.returncode, TERM_EXIT_STATUS)
        self.assertLess(waited, TERM_FLUSH_S + 5)

    @unittest.skipIf(sys.platform == "win32", "SIGTERM is a POSIX signal")
    def test_term_closes_the_stream_and_exits_as_term(self) -> None:
        child = subprocess.Popen(
            [sys.executable, "-c", CHILD], stdout=subprocess.PIPE, text=True
        )
        try:
            self.assertEqual(child.stdout.readline().strip(), "ready")
            child.send_signal(signal.SIGTERM)
            out, _ = child.communicate(timeout=10)
        finally:
            if child.poll() is None:
                child.kill()
        self.assertIn("disconnected", out)
        self.assertEqual(child.returncode, TERM_EXIT_STATUS)

    def test_off_the_main_thread_the_default_action_stays(self) -> None:
        before = signal.getsignal(signal.SIGTERM)
        worker = threading.Thread(target=leave_cleanly_on_term, args=(object(),))
        worker.start()
        worker.join()
        self.assertIs(signal.getsignal(signal.SIGTERM), before)


if __name__ == "__main__":
    unittest.main()
