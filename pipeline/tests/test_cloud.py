"""The GPU-provider seam: Runpod's mapping through a canned transport,
the registry by name, and the credential rule (a missing key is refused
by name; the key travels in one header and nowhere else)."""

import json
import os
import unittest
from unittest import mock

from rq_pipeline.cloud import (
    Action,
    MachineSpec,
    MissingCredentialError,
    ProviderError,
    Tier,
    providers,
    resolve,
)
from rq_pipeline.cloud.runpod import RunpodApi, RunpodProvider

KEY = "rpa_test_key_not_real"
GPU_4090 = "NVIDIA GeForce RTX 4090"
IMAGE = "runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404"

CATALOG = {
    "gpus": [
        {
            "id": GPU_4090,
            "name": "RTX 4090",
            "memory": 24,
            "secure": True,
            "community": True,
            "price": {"secure": 0.74, "community": 0.34},
            "maxCount": {"secure": 8, "community": 4},
            "availability": "LOW",
            "cudaVersions": [{"version": "13.0"}, {"version": "13.2"}],
        },
        {
            "id": "NVIDIA A40",
            "name": "A40",
            "memory": 48,
            "price": {"secure": 0.44, "community": 0},
            "availability": "NONE",
        },
    ]
}
PROVISIONING = {
    "id": "pod123",
    "name": "t5-cloud",
    "status": "PROVISIONING",
    "gpu": {"id": GPU_4090, "count": 1},
    "cloud": "COMMUNITY",
    "cost": 0.34,
    "actions": ["terminate"],
    "ssh": {"proxy": None, "direct": None},
}
RUNNING = {
    **PROVISIONING,
    "status": "RUNNING",
    "actions": ["stop", "restart", "terminate"],
    "dataCenterId": "EU-RO-1",
    "cudaVersion": "13.0",
    "ssh": {
        "proxy": {
            "host": "ssh.runpod.io",
            "port": 22,
            "username": "pod123-abc",
            "command": "ssh pod123-abc@ssh.runpod.io",
        },
        "direct": {
            "host": "203.0.113.7",
            "port": 12345,
            "username": "root",
            "command": "ssh root@203.0.113.7 -p 12345",
        },
    },
}
PROBLEM = {
    "title": "Request validation failed.",
    "status": 422,
    "detail": "gpu.id is not a known GPU type",
    "errors": [{"field": "gpu.id", "message": "unknown"}],
}


class CannedTransport:
    """Answers by (method, path); records every call for the assertions."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url, dict(headers), body))
        path = url[len(RunpodApi.BASE_URL) :].split("?")[0]
        status, doc = self.answers[(method, path)]
        return status, doc if isinstance(doc, str) else json.dumps(doc)


class TheRunpodMapping(unittest.TestCase):
    def test_offers_are_priced_for_the_tier_with_live_stock(self) -> None:
        transport = CannedTransport({("GET", RunpodApi.CATALOG_GPUS): (200, CATALOG)})
        offers = RunpodProvider(key=KEY, transport=transport).offers(
            tier=Tier.COMMUNITY, min_cuda="13.0"
        )
        by_gpu = {o.gpu: o for o in offers}
        self.assertEqual(by_gpu[GPU_4090].price_per_hour, 0.34)
        self.assertEqual(by_gpu[GPU_4090].availability, "LOW")
        self.assertEqual(by_gpu[GPU_4090].cuda_versions, ("13.0", "13.2"))
        self.assertEqual(by_gpu[GPU_4090].max_count, 4)
        self.assertIsNone(by_gpu["NVIDIA A40"].price_per_hour)  # 0 = not offered
        url = transport.calls[0][1]
        for expected in (
            "include=AVAILABILITY",
            "product=POD",
            "cloud=COMMUNITY",
            "minCudaVersion=13.0",
        ):
            self.assertIn(expected, url)

    def test_launch_sends_the_pod_request_and_reads_the_machine(self) -> None:
        transport = CannedTransport({("POST", RunpodApi.PODS): (201, PROVISIONING)})
        machine = RunpodProvider(key=KEY, transport=transport).launch(
            MachineSpec(
                name="t5-cloud",
                gpu=GPU_4090,
                image=IMAGE,
                tier=Tier.COMMUNITY,
                disk_gb=60,
                env={"RQ_SCALE": "cloud"},
                data_centers=("EU-RO-1",),
            )
        )
        method, _url, headers, body = transport.calls[0]
        sent = json.loads(body)
        self.assertEqual(method, "POST")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(sent["gpu"], {"id": GPU_4090, "count": 1})
        self.assertEqual(sent["image"], IMAGE)
        self.assertEqual(sent["disk"], 60)
        self.assertEqual(sent["cloud"], "COMMUNITY")
        self.assertEqual(sent["ports"], ["22/tcp"])
        self.assertTrue(sent["startSsh"])
        self.assertEqual(sent["env"], {"RQ_SCALE": "cloud"})
        self.assertEqual(sent["dataCenterIds"], ["EU-RO-1"])
        self.assertEqual(machine.id, "pod123")
        self.assertEqual(machine.status, "PROVISIONING")
        self.assertFalse(machine.running)
        self.assertIsNone(machine.ssh_direct)
        self.assertEqual(machine.actions, (Action.TERMINATE,))
        self.assertEqual(machine.cost_per_hour, 0.34)

    def test_a_running_machine_carries_both_ssh_doors(self) -> None:
        transport = CannedTransport(
            {("GET", f"{RunpodApi.PODS}/pod123"): (200, RUNNING)}
        )
        machine = RunpodProvider(key=KEY, transport=transport).machine("pod123")
        self.assertTrue(machine.running)
        self.assertEqual(machine.ssh_direct.host, "203.0.113.7")
        self.assertEqual(machine.ssh_direct.port, 12345)
        self.assertEqual(machine.ssh_proxy.username, "pod123-abc")
        self.assertEqual(machine.data_center, "EU-RO-1")
        self.assertEqual(machine.cuda_version, "13.0")
        self.assertIn(Action.STOP, machine.actions)

    def test_actions_go_to_the_action_door_and_terminate_deletes(self) -> None:
        transport = CannedTransport(
            {
                ("POST", f"{RunpodApi.PODS}/pod123/action"): (200, ""),
                ("DELETE", f"{RunpodApi.PODS}/pod123"): (204, ""),
            }
        )
        runpod = RunpodProvider(key=KEY, transport=transport)
        runpod.act("pod123", Action.STOP)
        runpod.act("pod123", Action.TERMINATE)
        self.assertEqual(json.loads(transport.calls[0][3]), {"action": "stop"})
        self.assertEqual(transport.calls[1][0], "DELETE")

    def test_logs_are_read_out_of_the_event_stream(self) -> None:
        stream = "event: log\ndata: step 100 loss 0.5\n\ndata: step 200 loss 0.4\n\n"
        transport = CannedTransport(
            {("GET", f"{RunpodApi.PODS}/pod123/logs"): (200, stream)}
        )
        text = RunpodProvider(key=KEY, transport=transport).logs("pod123", tail=2)
        self.assertEqual(text, "step 100 loss 0.5\nstep 200 loss 0.4")

    def test_a_problem_document_is_quoted_in_the_refusal(self) -> None:
        transport = CannedTransport({("POST", RunpodApi.PODS): (422, PROBLEM)})
        with self.assertRaisesRegex(
            ProviderError, "422 Request validation failed"
        ) as caught:
            RunpodProvider(key=KEY, transport=transport).launch(
                MachineSpec(name="x", gpu="NVIDIA Nonesuch", image=IMAGE)
            )
        self.assertIn("gpu.id: unknown", str(caught.exception))

    def test_the_key_travels_in_one_header_only(self) -> None:
        transport = CannedTransport({("GET", RunpodApi.PODS): (200, {"pods": []})})
        RunpodProvider(key=KEY, transport=transport).machines()
        _method, url, headers, body = transport.calls[0]
        self.assertEqual(headers["Authorization"], f"Bearer {KEY}")
        # Cloudflare refuses urllib's default agent with a 403 (2026-08-27).
        self.assertEqual(headers["User-Agent"], RunpodApi.USER_AGENT)
        self.assertNotIn(KEY, url)
        self.assertIsNone(body)


class TheCredentialRule(unittest.TestCase):
    def test_a_missing_key_is_refused_by_variable_name(self) -> None:
        with (
            mock.patch.dict(os.environ, {RunpodApi.KEY_ENV: ""}),
            self.assertRaisesRegex(MissingCredentialError, RunpodApi.KEY_ENV),
        ):
            RunpodProvider(transport=lambda *a: (200, "{}"))

    def test_the_key_comes_from_the_environment(self) -> None:
        with mock.patch.dict(os.environ, {RunpodApi.KEY_ENV: KEY}):
            runpod = RunpodProvider(transport=lambda *a: (200, '{"pods": []}'))
        self.assertEqual(runpod.machines(), [])


class TheRegistry(unittest.TestCase):
    def test_runpod_is_the_built_in_and_unknown_names_are_refused(self) -> None:
        self.assertIn("runpod", providers())
        self.assertIs(resolve("runpod").build, RunpodProvider)
        with self.assertRaisesRegex(KeyError, "known: \\['runpod'\\]"):
            resolve("acme-gpus")


if __name__ == "__main__":
    unittest.main()
