"""A RunPod network volume over its S3 API: what is on it, by prefix,
and what a delete would take.

The door that works when the pod is STOPPED: ssh needs a running
container, the volume outlives it, and the S3 endpoint is the only
way to read or free it without paying for a boot (2026-09-02, the
35 GB quota at 100 %). The accounting is pure and tested; boto3 is
touched only inside `VolumeDoor`, so the listing logic never needs a
network to be checked. Credentials come from boto3's own chain
(`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, a RunPod S3 API key
pair) - `read_env_file` lifts exactly the named keys out of an env
file into the process and prints nothing.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CREDENTIAL_NAMES = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
DELETE_BATCH = 1000  # the S3 DeleteObjects limit
UNITS = ("B", "KB", "MB", "GB", "TB")
KILO = 1024


@dataclass(frozen=True)
class Entry:
    key: str
    size: int


@dataclass(frozen=True)
class Usage:
    bytes: int = 0
    objects: int = 0

    def plus(self, size: int) -> Usage:
        return Usage(self.bytes + size, self.objects + 1)


def human(n: int) -> str:
    """1536 -> '1.5 KB'; exact bytes below a kilobyte."""
    value = float(n)
    for unit in UNITS:
        if value < KILO or unit == UNITS[-1]:
            return f"{int(value)} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= KILO
    raise AssertionError("unreachable")


def prefix_of(key: str, depth: int) -> str:
    """The first `depth` path segments of `key` (a directory-like
    prefix, trailing slash when the key goes deeper)."""
    parts = key.split("/")
    if len(parts) <= depth:
        return key
    return "/".join(parts[:depth]) + "/"


def usage_by_prefix(entries: Iterable[Entry], depth: int = 1) -> dict[str, Usage]:
    """Bytes and object counts grouped by the first `depth` segments,
    largest first."""
    totals: dict[str, Usage] = {}
    for entry in entries:
        prefix = prefix_of(entry.key, depth)
        totals[prefix] = totals.get(prefix, Usage()).plus(entry.size)
    return dict(sorted(totals.items(), key=lambda item: -item[1].bytes))


def under(entries: Iterable[Entry], prefix: str) -> list[Entry]:
    """The entries a delete of `prefix` would take - never a wider
    match than the prefix itself."""
    return [entry for entry in entries if entry.key.startswith(prefix)]


def total(entries: Iterable[Entry]) -> Usage:
    usage = Usage()
    for entry in entries:
        usage = usage.plus(entry.size)
    return usage


def read_env_file(path: Path, names: Sequence[str]) -> dict[str, str]:
    """Exactly the `names` present in a KEY=VALUE file, nothing else -
    the credentials door for a file that also holds keys nobody should
    load (the RunPod API key)."""
    wanted: dict[str, str] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        if key in names:
            wanted[key] = value.strip().strip("'\"")
    return wanted


def load_credentials(path: Path, environ: Mapping[str, str] | None = None) -> None:
    """Put the credential names from `path` into the process
    environment where boto3 reads them; never overrides what is set."""
    target: Any = os.environ if environ is None else environ
    for key, value in read_env_file(path, CREDENTIAL_NAMES).items():
        target.setdefault(key, value)


@dataclass(frozen=True)
class VolumeDoor:
    """One network volume's S3 door."""

    endpoint: str
    region: str
    bucket: str

    def client(self) -> Any:
        try:
            import boto3  # noqa: PLC0415 - optional, the box's own
        except ImportError as error:
            raise ImportError(
                "the volume door needs boto3 (pip install boto3, or run under "
                "a python that has it)"
            ) from error
        return boto3.client("s3", endpoint_url=self.endpoint, region_name=self.region)

    def entries(self, prefix: str = "") -> Iterator[Entry]:
        """Every object under `prefix`, paginated."""
        paginator = self.client().get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", ()):
                yield Entry(item["Key"], int(item["Size"]))

    def delete(self, keys: Sequence[str]) -> int:
        """Delete the given keys in S3-sized batches; returns the count."""
        client = self.client()
        deleted = 0
        for start in range(0, len(keys), DELETE_BATCH):
            batch = [{"Key": key} for key in keys[start : start + DELETE_BATCH]]
            client.delete_objects(
                Bucket=self.bucket, Delete={"Objects": batch, "Quiet": True}
            )
            deleted += len(batch)
        return deleted
