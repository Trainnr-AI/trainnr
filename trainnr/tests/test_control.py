"""The Studio's control surface (docs/76 §10.1): commands are files the agent
writes and the Studio answers; state carries a heartbeat the agent checks;
events are what the human did. A dead or absent Studio is refused by name
in every door, never waited on."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from trainnr.project import control, create_project
from trainnr.project.control import (
    KEEP_COMMANDS,
    SCHEMA,
    SECTIONS,
    STALE_S,
    ack,
    command,
    commands_dir,
    events,
    events_path,
    launch,
    quit,
    screenshot,
    send,
    state,
    state_path,
    wait,
)
from trainnr.project.locate import INDEX_DIR


def _write_state(project, *, pid=None, age_s=0.0, **more):
    body = {
        "schema": "trainnr-studio-state/1",
        "pid": os.getpid() if pid is None else pid,
        "heartbeat": time.time() - age_s,
        "project": str(project.root),
        "section": "overview",
        "selected": None,
        "live": {},
        "presenter_running": False,
        "jobs_running": 0,
    }
    body.update(more)
    state_path(project).parent.mkdir(parents=True, exist_ok=True)
    state_path(project).write_text(json.dumps(body))


class Commands(unittest.TestCase):
    def test_a_command_is_one_time_ordered_file_the_studio_answers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            cid = send(project, "open", section="robots", artifact=None)
            path = commands_dir(project) / f"{cid}.json"
            self.assertTrue(path.is_file())
            body = json.loads(path.read_text())
            self.assertEqual(body["schema"], SCHEMA)
            self.assertEqual(body["verb"], "open")
            self.assertEqual(body["section"], "robots")
            self.assertNotIn("artifact", body, "None fields are not written")
            self.assertIsNone(ack(project, cid))
            later = send(project, "quit")
            self.assertLess(cid, later, "ids sort by time")
            with self.assertRaisesRegex(ValueError, "unknown verb"):
                send(project, "dance")

    def test_wait_returns_the_answer_or_says_nobody_answered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            cid = send(project, "show", artifact="a@000000000000")
            self.assertEqual(wait(project, cid, timeout_s=0.1)["status"], "no answer")

            def studio() -> None:
                time.sleep(0.05)
                (commands_dir(project) / f"{cid}.ack.json").write_text(
                    json.dumps({"id": cid, "status": "done"})
                )

            threading.Thread(target=studio).start()
            self.assertEqual(wait(project, cid, timeout_s=2.0)["status"], "done")

    def test_old_commands_are_pruned_with_their_answers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            folder = commands_dir(project)
            folder.mkdir(parents=True)
            for i in range(KEEP_COMMANDS + 5):
                (folder / f"{i:04d}-open.json").write_text("{}")
                (folder / f"{i:04d}-open.ack.json").write_text("{}")
            send(project, "quit")
            live = [
                p for p in folder.glob("*.json") if not p.name.endswith(".ack.json")
            ]
            self.assertEqual(len(live), KEEP_COMMANDS)
            self.assertFalse((folder / "0000-open.ack.json").exists())


class State(unittest.TestCase):
    def test_no_studio_and_a_stale_or_dead_one_are_named_not_waited_on(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            self.assertFalse(state(project)["alive"])
            self.assertIn("no Studio has run", state(project)["reason"])
            _write_state(project, age_s=STALE_S + 1)
            self.assertFalse(state(project)["alive"])
            self.assertIn("stale", state(project)["reason"])
            _write_state(project, pid=2**22 + 12345)  # no such process
            self.assertFalse(state(project)["alive"])
            self.assertIn("gone", state(project)["reason"])
            answer = command(project, "open", section="live")
            self.assertEqual(answer["status"], "refused")
            self.assertIn("launch_studio", answer["reason"])
            self.assertEqual(list(commands_dir(project).glob("*.json")), [])

    def test_a_fresh_heartbeat_from_a_live_pid_is_alive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            _write_state(project, section="robots", selected="a@000000000000")
            current = state(project)
            self.assertTrue(current["alive"])
            self.assertEqual(current["section"], "robots")
            self.assertLess(current["heartbeat_age_s"], 1.0)

    def test_a_studio_that_moved_is_found_through_its_pointer(self) -> None:
        from trainnr.project.control import home  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            old = create_project(Path(tmp) / "old", "old")
            new = create_project(Path(tmp) / "new", "new")
            # The pointer the switch left behind: a fresh heartbeat and a
            # live pid, yet never alive on its own.
            _write_state(old, moved_to=str(new.root))
            self.assertFalse(state(old)["alive"])
            self.assertIn("moved from", state(old)["reason"])
            _write_state(new, section="deployments")
            found = state(old)
            self.assertTrue(found["alive"])
            self.assertTrue(found["elsewhere"])
            self.assertEqual(found["asked"], str(old.root))
            self.assertEqual(found["project"], str(new.root))
            self.assertEqual(found["section"], "deployments")
            self.assertEqual(home(old).root, new.root)
            self.assertNotIn("elsewhere", state(new))
            # A command about the old project's artifacts is refused with
            # the door that brings the window back; a page or a screenshot
            # goes where the Studio is.
            answer = command(old, "show", artifact="a@000000000000")
            self.assertEqual(answer["status"], "refused")
            self.assertIn("open_in_studio(project=", answer["reason"])
            answer = command(old, "open", artifact="a@000000000000")
            self.assertEqual(answer["status"], "refused")
            self.assertEqual(list(commands_dir(new).glob("*.json")), [])
            answer = command(old, "open", section="robots", timeout_s=0.05)
            self.assertEqual(answer["status"], "no answer")
            self.assertEqual(answer["project"], str(new.root))
            self.assertEqual(len(list(commands_dir(new).glob("*-open.json"))), 1)
            self.assertFalse(commands_dir(old).exists())
            # Launching on the old project moves the running window.
            answer = launch(old)
            self.assertEqual(answer["status"], "failed")  # nobody answered
            self.assertEqual(answer["switched_from"], str(new.root))
            sent = sorted(commands_dir(new).glob("*-open.json"))
            self.assertEqual(len(sent), 2)
            self.assertEqual(json.loads(sent[-1].read_text())["project"], str(old.root))

    def test_a_pointer_ring_ends_dead_not_forever(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = create_project(Path(tmp) / "a", "a")
            b = create_project(Path(tmp) / "b", "b")
            _write_state(a, moved_to=str(b.root))
            _write_state(b, moved_to=str(a.root))
            self.assertFalse(state(a)["alive"])

    def test_a_rebuilt_binary_is_named_and_a_current_or_vanished_one_is_not(
        self,
    ) -> None:
        from trainnr.project.control import binary_rebuilt_since  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "trainnr-studio"
            exe.write_bytes(b"")
            started = exe.stat().st_mtime + 10.0  # launched after the build

            class Proc:
                def __init__(self, _pid: int) -> None:
                    pass

                def create_time(self) -> float:
                    return started

                def exe(self) -> str:
                    return str(exe)

            fake = mock.MagicMock(
                Process=Proc, NoSuchProcess=KeyError, AccessDenied=KeyError
            )
            with mock.patch("trainnr.project.control._psutil", return_value=fake):
                self.assertIsNone(binary_rebuilt_since(1))
                os.utime(exe, (started + 5, started + 5))  # rebuilt after launch
                word = binary_rebuilt_since(1)
                self.assertIsNotNone(word)
                self.assertIn("rebuilt", word or "")
                self.assertIn("launch_studio", word or "")
                exe.unlink()
                self.assertIsNone(binary_rebuilt_since(1))

    def test_events_read_after_a_time_newest_last(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            self.assertEqual(events(project), [])
            path = events_path(project)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                '{"t": 10, "kind": "open", "section": "robots"}\n'
                "not json\n"
                '{"t": 20, "kind": "select", "artifact": "a@000000000000"}\n'
            )
            self.assertEqual([e["kind"] for e in events(project)], ["open", "select"])
            self.assertEqual([e["t"] for e in events(project, since_ns=10)], [20])
            self.assertEqual(len(events(project, limit=1)), 1)


class Screenshot(unittest.TestCase):
    def test_a_screenshot_navigates_first_and_carries_the_answer_through(self) -> None:
        """With a fake Studio answering on a thread: the open command lands
        before the capture, and the capture's path comes back verbatim."""
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            _write_state(project)
            seen: list[str] = []
            expected = 2  # the open, then the screenshot

            def studio() -> None:
                deadline = time.time() + 5
                while time.time() < deadline and len(seen) < expected:
                    for path in sorted(commands_dir(project).glob("*.json")):
                        if path.name.endswith(".ack.json") or path.name in seen:
                            continue
                        seen.append(path.name)
                        body = json.loads(path.read_text())
                        answer = {"id": body["id"], "status": "done"}
                        if body["verb"] == "screenshot":
                            answer.update(
                                {"path": "/x/shot.png", "width": 800, "height": 500}
                            )
                        (commands_dir(project) / f"{body['id']}.ack.json").write_text(
                            json.dumps(answer)
                        )
                    time.sleep(0.01)

            threading.Thread(target=studio).start()
            answer = screenshot(project, section="robots", width=800)
            self.assertEqual(answer["status"], "done")
            self.assertEqual(answer["path"], "/x/shot.png")
            self.assertIn("read", answer)
            self.assertEqual(
                [n.split("-", 1)[1] for n in seen], ["open.json", "screenshot.json"]
            )

    def test_a_failed_navigation_stops_the_screenshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            answer = screenshot(project, section="robots")
            self.assertEqual(answer["status"], "refused")
            self.assertEqual(list(commands_dir(project).glob("*.json")), [])


class Lifecycle(unittest.TestCase):
    def test_launch_refuses_while_the_viewer_port_is_held(self) -> None:
        import socket  # noqa: PLC0415

        from trainnr.project import control as ctl  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp, socket.socket() as holder:
            holder.bind(("127.0.0.1", 0))
            holder.listen(1)
            port = holder.getsockname()[1]
            project = create_project(Path(tmp) / "p", "p")
            fake = Path(tmp) / "trainnr-studio"
            fake.write_text("#!/bin/sh\nsleep 1\n")
            fake.chmod(0o755)
            old = (ctl.VIEWER_PORT, ctl.PORT_FREE_TIMEOUT_S)
            ctl.VIEWER_PORT, ctl.PORT_FREE_TIMEOUT_S = port, 0.2
            try:
                answer = launch(project, binary=fake)
            finally:
                ctl.VIEWER_PORT, ctl.PORT_FREE_TIMEOUT_S = old
            self.assertEqual(answer["status"], "refused")
            self.assertIn("viewer server", answer["reason"])

    def test_launch_refuses_a_running_studio_and_a_missing_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            _write_state(project)
            self.assertEqual(launch(project)["status"], "refused")
            (state_path(project)).unlink()
            answer = launch(project, binary=Path(tmp) / "nowhere" / "trainnr-studio")
            self.assertEqual(answer["status"], "refused")
            self.assertIn("nowhere", answer["reason"])

    def test_a_fake_studio_that_exits_at_once_is_reported_with_its_log(self) -> None:
        from trainnr.project import control as ctl  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            fake = Path(tmp) / "trainnr-studio"
            fake.write_text("#!/bin/sh\necho 'no display'\nexit 3\n")
            fake.chmod(0o755)
            # A real Studio may hold the real port on this machine; the
            # fake one needs no port, so the check runs on a free one.
            old = ctl.VIEWER_PORT
            ctl.VIEWER_PORT = 0
            try:
                answer = launch(project, binary=fake)
            finally:
                ctl.VIEWER_PORT = old
            self.assertEqual(answer["status"], "failed")
            self.assertIn("code 3", answer["reason"])
            log = project.root / INDEX_DIR / "studio.log"
            self.assertIn("no display", log.read_text())

    def test_quit_with_no_studio_is_done_and_says_so(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            answer = quit(project)
            self.assertEqual(answer["status"], "done")
            self.assertIn("no Studio", answer["reason"])

    def test_wait_presented_reports_shown_failed_and_pending(self) -> None:
        import json as _json  # noqa: PLC0415

        from trainnr.project.control import (  # noqa: PLC0415
            present_status_path,
            wait_presented,
        )

        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            path = present_status_path(project)
            path.parent.mkdir(parents=True, exist_ok=True)
            since = time.time()
            path.write_text(_json.dumps({"shown": "a@1", "t": since + 1}))
            self.assertEqual(
                wait_presented(project, "a@1", since=since, timeout_s=0.5)["status"],
                "shown",
            )
            path.write_text(
                _json.dumps({"error": "no", "stamp": "a@1", "t": since + 1})
            )
            self.assertEqual(
                wait_presented(project, "a@1", since=since, timeout_s=0.5)["status"],
                "failed",
            )
            # An answer older than the request is not this request's answer.
            path.write_text(_json.dumps({"shown": "a@1", "t": since - 1}))
            self.assertEqual(
                wait_presented(project, "a@1", since=since, timeout_s=0.3)["status"],
                "pending",
            )
            # Taken up but not landed at the timeout: presenting, not pending.
            path.write_text(_json.dumps({"presenting": "a@1", "t": since + 1}))
            answer = wait_presented(project, "a@1", since=since, timeout_s=0.3)
            self.assertEqual(answer["status"], "presenting")
            self.assertIn("describe_studio", answer["reason"])

    def test_a_show_requested_during_a_presentation_is_kept(self) -> None:
        """The presenter claims the intent file by rename before it streams;
        a request written meanwhile is a new file, served next (2026-09-27)."""
        import json as _json  # noqa: PLC0415
        from unittest import mock  # noqa: PLC0415

        from trainnr.project import present as pr  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            intent = pr.intent_path(project)
            intent.parent.mkdir(parents=True, exist_ok=True)
            intent.write_text(_json.dumps({"stamp": "first@1"}))
            shown: list[str] = []

            def slow_present(project_, stamp):
                shown.append(stamp)
                if len(shown) == 1:  # the Studio asks for another meanwhile
                    intent.write_text(_json.dumps({"stamp": "second@2"}))
                self.assertEqual(status()["presenting"], stamp)

            def status():
                return _json.loads(pr.present_status_path(project).read_text())

            with (
                mock.patch.object(pr, "present", slow_present),
                mock.patch.object(pr, "refresh_project", lambda p: 0, create=True),
            ):
                pr.serve(project, once=True)
                self.assertEqual(shown, ["first@1"])
                self.assertEqual(status()["shown"], "first@1")
                self.assertTrue(intent.is_file(), "the second request survived")
                pr.serve(project, once=True)
            self.assertEqual(shown, ["first@1", "second@2"])
            self.assertFalse(intent.is_file())
            self.assertFalse(intent.with_name(intent.name + pr.CLAIMED_SUFFIX).exists())

    def test_the_page_names_are_the_rails(self) -> None:
        self.assertIn("evaluations", SECTIONS)
        self.assertNotIn("certificates", SECTIONS)


if __name__ == "__main__":
    unittest.main()


class WslGpuEnvironment(unittest.TestCase):
    def test_wsl_puts_the_gpu_libraries_first_and_keeps_the_callers_driver(
        self,
    ) -> None:
        with (
            mock.patch.object(control, "on_wsl", return_value=True),
            mock.patch("pathlib.Path.is_dir", return_value=True),
        ):
            env = control.wsl_gpu_environment({"LD_LIBRARY_PATH": "/opt/cuda/lib"})
            self.assertEqual(env["LD_LIBRARY_PATH"], "/usr/lib/wsl/lib:/opt/cuda/lib")
            self.assertEqual(env["GALLIUM_DRIVER"], "d3d12")
            env = control.wsl_gpu_environment(
                {"LD_LIBRARY_PATH": "/usr/lib/wsl/lib:/x", "GALLIUM_DRIVER": "llvmpipe"}
            )
            self.assertEqual(env["LD_LIBRARY_PATH"], "/usr/lib/wsl/lib:/x")
            self.assertEqual(env["GALLIUM_DRIVER"], "llvmpipe")

    def test_elsewhere_the_environment_is_untouched(self) -> None:
        with mock.patch.object(control, "on_wsl", return_value=False):
            self.assertEqual(control.wsl_gpu_environment({"A": "b"}), {"A": "b"})


class TheSessionToken(unittest.TestCase):
    """The Studio applies a command only when it carries its session's
    token (security review, 2026-10-05); the token is in its state file."""

    def test_a_command_carries_the_running_studios_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            path = control.state_path(project)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"session": "abc123"}))
            cid = control.send(project, "quit")
            body = json.loads(
                (control.commands_dir(project) / f"{cid}.json").read_text()
            )
            self.assertEqual(body["session"], "abc123")


class WslDisplayEnvironment(unittest.TestCase):
    """A WSL boot without /run/user/<uid>: the Studio found no Wayland
    compositor and exited (2026-10-05)."""

    def test_a_missing_socket_points_at_wslgs_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wslg = Path(tmp) / "wslg"
            wslg.mkdir()
            (wslg / "wayland-0").touch()
            env = {"WAYLAND_DISPLAY": "wayland-0", "XDG_RUNTIME_DIR": f"{tmp}/gone"}
            with mock.patch.object(control, "on_wsl", return_value=True):
                fixed = control.wsl_display_environment(dict(env), wslg)
                self.assertEqual(fixed["XDG_RUNTIME_DIR"], str(wslg))
                (Path(tmp) / "gone").mkdir()
                (Path(tmp) / "gone" / "wayland-0").touch()
                self.assertEqual(control.wsl_display_environment(dict(env), wslg), env)

    def test_elsewhere_the_environment_is_untouched(self) -> None:
        env = {"WAYLAND_DISPLAY": "wayland-0", "XDG_RUNTIME_DIR": "/nowhere"}
        with mock.patch.object(control, "on_wsl", return_value=False):
            self.assertEqual(control.wsl_display_environment(dict(env)), env)


class APlantedStateFile(unittest.TestCase):
    """A shared project's studio-state.json named a pid with a heartbeat in
    the future, and quit_studio signalled that process (security review,
    2026-10-04)."""

    def test_a_heartbeat_from_the_future_is_not_alive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            _write_state(project, age_s=-3600.0)
            self.assertFalse(control.state(project)["alive"])

    def test_quit_signals_only_a_studio(self) -> None:
        import subprocess  # noqa: PLC0415
        import sys  # noqa: PLC0415

        victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(victim.kill)
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            _write_state(project, pid=victim.pid)
            with mock.patch.object(control, "command", return_value={}):
                out = control.quit(project, timeout_s=0.1)
            self.assertEqual(out["status"], "failed", out)
            self.assertIn("not a Studio", out["reason"])
            self.assertIsNone(victim.poll(), "the process was not signalled")
