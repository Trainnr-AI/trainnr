"""The openpi door: our observation packed the way their server reads
it, their chunk executed on our horizon — against a fake server that
speaks their protocol (metadata first, msgpack-numpy, a string on error).
Runs where the `remote` extra is installed (the train venv)."""

import importlib.util
import threading
import unittest

REMOTE_PRESENT = importlib.util.find_spec("openpi_client") is not None
SIM_PRESENT = importlib.util.find_spec("mujoco") is not None
HORIZON, NU = 8, 14
EXECUTED = 3
LOCALHOST = "127.0.0.1"


def _serve(handler):
    """A fake openpi server on a free port: yields (port, stop)."""
    from websockets.sync.server import serve  # noqa: PLC0415

    server = serve(handler, LOCALHOST, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.socket.getsockname()[1]
    return port, server.shutdown


@unittest.skipUnless(REMOTE_PRESENT and SIM_PRESENT, "needs the remote and sim extras")
class ThroughOpenpi(unittest.TestCase):
    def test_a_chunk_round_trips_and_is_executed_on_our_horizon(self) -> None:
        import numpy as np  # noqa: PLC0415
        from openpi_client import msgpack_numpy  # noqa: PLC0415

        from rq_pipeline.envs.openpi_policy import (  # noqa: PLC0415
            OpenpiKeys,
            OpenpiRequest,
            openpi_chunk_policy,
        )
        from rq_pipeline.envs.robotiq import make_env  # noqa: PLC0415
        from rq_pipeline.evaluate.scheduler import ActionScheduler  # noqa: PLC0415

        requests = []

        def handler(connection):
            connection.send(msgpack_numpy.packb({"served": "fake"}))  # metadata first
            for message in connection:
                request = msgpack_numpy.unpackb(message)
                requests.append(request)
                chunk = np.full((HORIZON, NU), len(requests), dtype=np.float32)
                connection.send(msgpack_numpy.packb({OpenpiKeys.ACTIONS: chunk}))

        port, stop = _serve(handler)
        env = make_env("kitting")
        try:
            observation, _ = env.reset(seed=1000)
            top = env.cameras[0].key
            policy = openpi_chunk_policy(
                LOCALHOST,
                port,
                request=OpenpiRequest(
                    cameras={top: "cam_high"}, prompt=env.task_description
                ),
            )
            scheduler = ActionScheduler(policy, executed_horizon=EXECUTED, nu=NU)
            scheduler.reset()
            actions = [scheduler.act(observation) for _ in range(EXECUTED + 1)]
            self.assertEqual(
                [float(a[0]) for a in actions], [1, 1, 1, 2]
            )  # replanned once
            self.assertEqual(len(requests), 2)
            sent = requests[0]
            self.assertEqual(sent[OpenpiKeys.PROMPT], env.task_description)
            self.assertEqual(sent[OpenpiKeys.STATE].shape, (NU,))
            image = sent[OpenpiKeys.IMAGES]["cam_high"]
            self.assertEqual(image.dtype, np.uint8)
            self.assertEqual(image.shape[0], 3)  # CHW, as AlohaInputs expects
        finally:
            env.close()
            stop()

    def test_a_server_error_stops_the_run_with_the_traceback(self) -> None:
        from openpi_client import msgpack_numpy  # noqa: PLC0415

        from rq_pipeline.envs.openpi_policy import (  # noqa: PLC0415
            OpenpiRequest,
            openpi_chunk_policy,
        )

        def handler(connection):
            connection.send(msgpack_numpy.packb({}))
            for _message in connection:
                connection.send(
                    "Traceback: the fake server refused"
                )  # a string, their error frame
                return

        port, stop = _serve(handler)
        try:
            policy = openpi_chunk_policy(LOCALHOST, port, request=OpenpiRequest())
            observation = {"pixels": {}, "agent_pos": [0.0] * NU}
            with self.assertRaises(RuntimeError) as caught:
                policy.predict(observation)
            self.assertIn("inference server", str(caught.exception))
        finally:
            stop()

    def test_a_request_naming_a_missing_camera_is_refused(self) -> None:
        from rq_pipeline.envs.openpi_policy import OpenpiRequest  # noqa: PLC0415

        with self.assertRaises(KeyError) as caught:
            OpenpiRequest(cameras={"wrist": "cam_left_wrist"}).pack(
                {"pixels": {"top": None}, "agent_pos": []}
            )
        self.assertIn("wrist", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
