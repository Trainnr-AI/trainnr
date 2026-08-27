"""Runpod behind the provider seam — REST API v2, standard library only.

Read from the OpenAPI schema at https://api.runpod.io/v2/openapi.json
on 2026-08-27 (v1 retires 2026-11-15): bearer auth, `/v2/catalog/gpus`
with `include=AVAILABILITY&product=POD` for live stock, `/v2/pods` to
rent and list, `/v2/pods/{id}/action` for start/stop/restart, DELETE
to terminate, `/v2/pods/{id}/logs` as an event stream. No SDK: the
surface we use is six calls, and a dependency would pin us to its pace.

The API key comes from `RUNPOD_API_KEY` (`RunpodApi.KEY_ENV`); it goes
into one header and nowhere else. A `transport` can be injected — the
tests speak canned JSON through it, the tool speaks HTTPS.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from rq_pipeline.cloud.provider import (
    Action,
    GpuOffer,
    Machine,
    MachineSpec,
    ProviderError,
    SshEndpoint,
    Tier,
    credential,
    provider,
)

# transport(method, url, headers, body_bytes_or_None) -> (status, body_text)
Transport = Callable[[str, str, Mapping[str, str], bytes | None], tuple[int, str]]
# stream(url, headers, seconds) -> the text that arrived within `seconds`
Stream = Callable[[str, Mapping[str, str], float], str]


@dataclass(frozen=True)
class RunpodApi:
    """The constants of Runpod's REST API v2, in one place."""

    NAME = "runpod"
    BASE_URL = "https://api.runpod.io/v2"
    KEY_ENV = "RUNPOD_API_KEY"
    CATALOG_GPUS = "/catalog/gpus"
    PODS = "/pods"
    PRODUCT_POD = "POD"
    INCLUDE_AVAILABILITY = "AVAILABILITY"
    TIMEOUT_S = 60.0
    # The logs endpoint is an event stream that never closes: read it
    # for this long, then return what arrived (measured 2026-08-27: a
    # reader waiting for EOF hung the tool).
    LOG_READ_S = 5.0
    # Cloudflare fronts the API and answers urllib's default agent string
    # with 403 "Error 1010: Access denied — browser signature" (measured
    # 2026-08-27; curl and any named product token pass). Every request
    # carries this.
    USER_AGENT = "rq-pipeline/cloud-gpu"
    # The catalog's price object is keyed by tier, lower-case.
    PRICE_KEY: ClassVar[Mapping[Tier, str]] = {
        Tier.SECURE: "secure",
        Tier.COMMUNITY: "community",
    }
    # A pod's terminal states, per the schema's PodStatus.
    GONE = ("TERMINATED",)


def https_transport(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None
) -> tuple[int, str]:
    """The real thing: one request over urllib, errors returned as
    (status, body) rather than raised, so the caller reads the
    provider's problem document."""
    request = urllib.request.Request(
        url, data=body, method=method, headers=dict(headers)
    )
    try:
        with urllib.request.urlopen(request, timeout=RunpodApi.TIMEOUT_S) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", errors="replace")


def https_stream(url: str, headers: Mapping[str, str], seconds: float) -> str:
    """Read an event stream for `seconds` of silence at most, then return
    the text so far — a stream has no EOF to wait for."""
    request = urllib.request.Request(url, headers=dict(headers))
    chunks: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=seconds) as response:
            if response.status >= 400:  # noqa: PLR2004 - HTTP's own line
                return response.read().decode("utf-8", errors="replace")
            while True:
                line = response.readline()
                if not line:
                    break
                chunks.append(line.decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as error:
        return error.read().decode("utf-8", errors="replace")
    except TimeoutError:
        pass  # the stream went quiet: what arrived is the answer
    return "".join(chunks)


@provider(RunpodApi.NAME, doc="Runpod pods over REST API v2 (RUNPOD_API_KEY)")
class RunpodProvider:
    """Runpod pods over REST API v2."""

    name = RunpodApi.NAME

    def __init__(
        self,
        *,
        key: str | None = None,
        transport: Transport | None = None,
        stream: Stream | None = None,
    ):
        self._key = key or credential(RunpodApi.KEY_ENV, provider=self.name)
        self._transport = transport or https_transport
        self._stream = stream or https_stream

    # ---- the seam -------------------------------------------------------

    def offers(
        self, *, tier: Tier = Tier.SECURE, min_cuda: str | None = None
    ) -> list[GpuOffer]:
        params: dict[str, str] = {
            "include": RunpodApi.INCLUDE_AVAILABILITY,
            "product": RunpodApi.PRODUCT_POD,
            "cloud": tier.value,
        }
        if min_cuda:
            params["minCudaVersion"] = min_cuda
        raw = self._request("GET", RunpodApi.CATALOG_GPUS, params=params)
        return [self._offer(gpu, tier) for gpu in raw["gpus"]]

    def launch(self, spec: MachineSpec) -> Machine:
        body: dict[str, Any] = {
            "name": spec.name,
            "image": spec.image,
            "gpu": {"id": spec.gpu, "count": spec.count},
            "disk": spec.disk_gb,
            "cloud": spec.tier.value,
            "ports": list(spec.ports),
            "startSsh": spec.ssh,
            "env": dict(spec.env),
        }
        if spec.data_centers:
            body["dataCenterIds"] = list(spec.data_centers)
        return self._machine(self._request("POST", RunpodApi.PODS, body=body))

    def machine(self, machine_id: str) -> Machine:
        return self._machine(self._request("GET", f"{RunpodApi.PODS}/{machine_id}"))

    def machines(self) -> list[Machine]:
        raw = self._request("GET", RunpodApi.PODS)
        return [self._machine(pod) for pod in raw["pods"]]

    def act(self, machine_id: str, action: Action) -> None:
        if action is Action.TERMINATE:
            self._request("DELETE", f"{RunpodApi.PODS}/{machine_id}")
            return
        self._request(
            "POST",
            f"{RunpodApi.PODS}/{machine_id}/action",
            body={"action": action.value},
        )

    def logs(self, machine_id: str, *, tail: int = 100) -> str:
        # An event stream that never closes; each event's data line is
        # one log line. Read for a bounded time, then return.
        url = f"{RunpodApi.BASE_URL}{RunpodApi.PODS}/{machine_id}/logs?" + (
            urllib.parse.urlencode({"tail": str(tail)})
        )
        text = self._stream(url, self._headers(), RunpodApi.LOG_READ_S)
        return "\n".join(_log_lines(text)) or text

    # ---- the wire -------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        body: Mapping[str, Any] | None = None,
        raw: bool = False,
    ) -> Any:
        url = RunpodApi.BASE_URL + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        headers = self._headers()
        payload = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            payload = json.dumps(body).encode("utf-8")
        status, text = self._transport(method, url, headers, payload)
        if status >= 400:  # noqa: PLR2004 - HTTP's own line
            raise ProviderError(_problem(self.name, method, path, status, text))
        if raw or not text.strip():
            return text
        return json.loads(text)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._key}",
            "Accept": "application/json",
            "User-Agent": RunpodApi.USER_AGENT,
        }

    # ---- the mapping ----------------------------------------------------

    def _offer(self, gpu: Mapping[str, Any], tier: Tier) -> GpuOffer:
        price = (gpu.get("price") or {}).get(RunpodApi.PRICE_KEY[tier])
        count = (gpu.get("maxCount") or {}).get(RunpodApi.PRICE_KEY[tier])
        return GpuOffer(
            provider=self.name,
            gpu=gpu["id"],
            name=gpu.get("name", gpu["id"]),
            memory_gb=int(gpu.get("memory", 0)),
            tier=tier,
            price_per_hour=float(price) if price not in (None, 0) else None,
            availability=str(gpu.get("availability", "")),
            cuda_versions=tuple(
                str(entry["version"]) for entry in gpu.get("cudaVersions") or []
            ),
            max_count=int(count) if count else None,
        )

    def _machine(self, pod: Mapping[str, Any]) -> Machine:
        gpu = pod.get("gpu") or {}
        ssh = pod.get("ssh") or {}
        return Machine(
            provider=self.name,
            id=pod["id"],
            name=pod.get("name", ""),
            status=pod.get("status", ""),
            gpu=gpu.get("id"),
            count=int(gpu.get("count", 1) or 1),
            tier=Tier(pod.get("cloud", Tier.SECURE.value)),
            cost_per_hour=float(pod.get("cost", 0.0) or 0.0),
            actions=tuple(
                Action(a)
                for a in pod.get("actions", [])
                if a in Action._value2member_map_
            ),
            data_center=pod.get("dataCenterId"),
            cuda_version=pod.get("cudaVersion"),
            ssh_direct=_endpoint(ssh.get("direct")),
            ssh_proxy=_endpoint(ssh.get("proxy")),
        )


def _log_lines(text: str) -> list[str]:
    """The stream's lines as `ts source: line`. Measured 2026-08-27: each
    event's payload is a JSON object `{"source", "line", "ts"}`, with or
    without an SSE `data:` prefix; anything else is passed through."""
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.removeprefix("data:").strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            out.append(line)
            continue
        if isinstance(event, dict) and "line" in event:
            out.append(
                f"{event.get('ts', '')} {event.get('source', '')}: {event['line']}"
            )
        else:
            out.append(line)
    return out


def _endpoint(raw: Mapping[str, Any] | None) -> SshEndpoint | None:
    if not raw:
        return None
    return SshEndpoint(
        host=raw["host"],
        port=int(raw["port"]),
        username=raw["username"],
        command=raw.get("command", ""),
    )


def _problem(name: str, method: str, path: str, status: int, text: str) -> str:
    """Runpod answers errors as RFC 7807 problem documents; quote them."""
    try:
        doc = json.loads(text)
    except ValueError:
        doc = {}
    title = doc.get("title") or "request refused"
    detail = doc.get("detail") or text.strip()[:200]
    errors = doc.get("errors") or []
    fields = "; ".join(
        f"{e.get('field', e.get('path', '?'))}: {e.get('message', e)}" for e in errors
    )
    return f"{name} {method} {path}: {status} {title} — {detail}" + (
        f" ({fields})" if fields else ""
    )
