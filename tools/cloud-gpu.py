#!/usr/bin/env python3
"""A rented GPU as a runbook: offers → launch → push → bootstrap → run
→ pull → terminate, every step one subcommand, every vendor behind
the provider seam (`rq_pipeline/cloud`). The first vendor is Runpod.

    cd pipeline && uv run --extra sim python ../tools/cloud-gpu.py offers \\
        --tier COMMUNITY
    cd pipeline && uv run --extra sim python ../tools/cloud-gpu.py launch \\
        --name t5-cloud --gpu "NVIDIA GeForce RTX 4090" --tier COMMUNITY --wait
    cd pipeline && uv run --extra sim python ../tools/cloud-gpu.py push <id>
    cd pipeline && uv run --extra sim python ../tools/cloud-gpu.py bootstrap <id>
    cd pipeline && uv run --extra sim python ../tools/cloud-gpu.py run <id> -- \\
        ../tools/e2e-smoke.py --scale cloud --name t5-cloud
    cd pipeline && uv run --extra sim python ../tools/cloud-gpu.py pull <id> t5-cloud
    cd pipeline && uv run --extra sim python ../tools/cloud-gpu.py terminate <id>

The API key is read from the repo's `.env` (`RUNPOD_API_KEY`) into the
environment and travels in one request header; this tool never prints
it and `push` never ships it (the `.env` file is excluded from the
rsync). Billing starts at `launch` and stops at `terminate` — `stop`
keeps the disk and a smaller bill; nothing here terminates on your
behalf.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from _lab import REPO, bootstrap, load_dotenv

bootstrap()

from rq_pipeline.cloud import (  # noqa: E402
    DEFAULT_PROVIDER,
    Action,
    GpuProvider,
    Machine,
    MachineSpec,
    MissingCredentialError,
    ProviderError,
    Tier,
    providers,
    resolve,
)


@dataclass(frozen=True)
class Defaults:
    """What a launch takes when the command line says nothing else."""

    # Runpod's PyTorch image on CUDA 13 (our train venv is torch+cu130:
    # the HOST driver must be that new, the image only needs to run sshd
    # and apt). Tag read from Docker Hub 2026-08-27.
    IMAGE = "runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404"
    DISK_GB = 60  # the T5 batch is ~90 MB/episode; 50 episodes + a dataset
    MIN_CUDA = "13.0"
    TIER = Tier.SECURE
    POLL_S = 10.0
    WAIT_S = 900.0


@dataclass(frozen=True)
class Remote:
    """Where the repo lives on the machine and how its venv is built —
    the WSL box's own recipe (docs/07 2026-08-26), verbatim."""

    DIR = "/workspace/robotiq"
    VENV = ".venv-train"
    PYTHON = "3.12.8"
    EXTRAS = ("sim", "viz", "train")
    # EGL is the offscreen renderer on a bare Linux GPU box; the WSL
    # variables (Mesa's D3D12 path) do not apply there.
    RUN_ENV = "MUJOCO_GL=egl OMP_NUM_THREADS=1"
    APT = "libegl1 libgl1 libglib2.0-0 rsync"

    @classmethod
    def bootstrap_script(cls) -> str:
        extras = " ".join(f"--extra {e}" for e in cls.EXTRAS)
        return "\n".join(
            [
                "set -euo pipefail",
                "export DEBIAN_FRONTEND=noninteractive",
                "apt-get update -qq && apt-get install -y -qq "
                f"--no-install-recommends {cls.APT}",
                "command -v uv >/dev/null "
                "|| curl -LsSf https://astral.sh/uv/install.sh | sh",
                'export PATH="$HOME/.local/bin:$PATH"',
                f"cd {cls.DIR}/pipeline",
                f"UV_PROJECT_ENVIRONMENT={cls.VENV} uv sync "
                f"--python {cls.PYTHON} {extras}",
                f"{cls.VENV}/bin/python -c 'import torch, mujoco; "
                'print("torch", torch.__version__, "cuda", torch.cuda.is_available(), '
                'torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-", '
                '"mujoco", mujoco.__version__)\'',
                "nvidia-smi --query-gpu=name,driver_version,memory.total "
                "--format=csv,noheader",
            ]
        )


@dataclass(frozen=True)
class Rsync:
    """What `push` ships: the working tree minus what must never leave
    or need not travel."""

    EXCLUDES = (
        ".env",  # the credentials — never
        ".git",
        "claude-sync",  # memory and transcripts: not the pod's business
        "pipeline/.venv",
        "pipeline/.venv-train",
        "pipeline/runs",
        "docs/photos",
        "target",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".coverage",
    )
    PULL_SUBDIR = "pipeline/runs"


@dataclass(frozen=True)
class Ssh:
    KEY = Path.home() / ".ssh" / "id_ed25519"
    OPTIONS = ("-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--provider", default=DEFAULT_PROVIDER, help=f"one of {sorted(providers())}"
    )
    parser.add_argument(
        "--dotenv", type=Path, default=REPO / ".env", help="where the API key lives"
    )
    parser.add_argument("--ssh-key", type=Path, default=Ssh.KEY)
    sub = parser.add_subparsers(dest="command", required=True)

    offers = sub.add_parser(
        "offers", help="the catalog with live stock, priced per card-hour"
    )
    offers.add_argument("--tier", type=Tier, choices=list(Tier), default=Defaults.TIER)
    offers.add_argument(
        "--min-cuda", default=Defaults.MIN_CUDA, help="host driver floor"
    )
    offers.add_argument(
        "--all", action="store_true", help="include types with no stock"
    )

    launch = sub.add_parser("launch", help="rent a machine (billing starts)")
    launch.add_argument("--name", required=True)
    launch.add_argument("--gpu", required=True, help="the id column of `offers`")
    launch.add_argument("--tier", type=Tier, choices=list(Tier), default=Defaults.TIER)
    launch.add_argument("--image", default=Defaults.IMAGE)
    launch.add_argument("--disk", type=int, default=Defaults.DISK_GB)
    launch.add_argument("--count", type=int, default=1)
    launch.add_argument(
        "--datacenter", action="append", default=[], help="preferred; repeatable"
    )
    launch.add_argument(
        "--wait", action="store_true", help="block until it runs and sshd is reachable"
    )

    sub.add_parser("machines", help="every machine the account has")
    status = sub.add_parser("status", help="one machine")
    status.add_argument("id")
    status.add_argument("--wait", action="store_true")
    for name, text in (
        ("ssh", "print the ssh commands"),
        ("push", "rsync the repo onto the machine"),
        ("bootstrap", "apt + uv + the train venv, then a torch/CUDA/mujoco check"),
        ("logs", "the container's recent output"),
        ("stop", "stop (keeps the disk)"),
        ("start", "resume a stopped machine"),
        ("restart", "restart the container"),
        ("terminate", "delete it — permanent"),
    ):
        one = sub.add_parser(name, help=text)
        one.add_argument("id")
        if name == "logs":
            one.add_argument("--tail", type=int, default=100)
    run = sub.add_parser(
        "run", help="ssh: run a command in the repo's pipeline/ with the train venv"
    )
    run.add_argument("id")
    run.add_argument(
        "--deadline-min",
        type=int,
        default=None,
        help="the machine kills the command after this many minutes (`timeout`, "
        "TERM then KILL a minute later); checkpoints written before survive",
    )
    run.add_argument(
        "argv", nargs=argparse.REMAINDER, help="after --: a tool and its flags"
    )
    pull = sub.add_parser("pull", help="rsync runs/<name>-* back")
    pull.add_argument("id")
    pull.add_argument("name", help="the run name (the --name of the chain)")
    return parser.parse_args(argv)


# ---- the steps ---------------------------------------------------------


def show_offers(gpu: GpuProvider, args: argparse.Namespace) -> None:
    offers = gpu.offers(tier=args.tier, min_cuda=args.min_cuda)
    rows = [
        o
        for o in offers
        if args.all or (o.availability not in ("NONE", "") and o.price_per_hour)
    ]
    rows.sort(key=lambda o: (o.price_per_hour or 1e9, o.memory_gb))
    print(
        f"{gpu.name} {args.tier.value}, host CUDA >= {args.min_cuda}: "
        f"{len(rows)} of {len(offers)} types in stock"
    )
    print(f"{'gpu (the --gpu id)':38s} {'GB':>4} {'$/h':>6} {'stock':7s} cuda")
    for o in rows:
        price = f"{o.price_per_hour:.2f}" if o.price_per_hour else "-"
        cuda = ",".join(o.cuda_versions)
        print(f"{o.gpu:38s} {o.memory_gb:>4} {price:>6} {o.availability:7s} {cuda}")


def describe(machine: Machine) -> str:
    ssh = machine.ssh_direct or machine.ssh_proxy
    door = f"{ssh.username}@{ssh.host}:{ssh.port}" if ssh else "no ssh yet"
    return (
        f"{machine.id}  {machine.name:16s} {machine.status:12s} "
        f"{machine.gpu or '-'} x{machine.count} {machine.tier.value.lower():9s} "
        f"${machine.cost_per_hour:.2f}/h  {machine.data_center or '-':8s} "
        f"cuda {machine.cuda_version or '-':5s} {door}"
    )


def wait_running(gpu: GpuProvider, machine_id: str) -> Machine:
    """Poll until the machine runs AND its own sshd is published — the
    door rsync needs. A machine that errors is reported, not waited on."""
    deadline = time.monotonic() + Defaults.WAIT_S
    while True:
        machine = gpu.machine(machine_id)
        print(describe(machine), flush=True)
        if machine.running and machine.ssh_direct is not None:
            return machine
        if machine.status in ("ERROR", "TERMINATED", "EXITED"):
            raise SystemExit(
                f"cloud-gpu: {machine_id} is {machine.status}; not waiting"
            )
        if time.monotonic() > deadline:
            raise SystemExit(
                f"cloud-gpu: {machine_id} not running after {Defaults.WAIT_S:.0f} s"
            )
        time.sleep(Defaults.POLL_S)


def direct_door(gpu: GpuProvider, machine_id: str) -> Machine:
    machine = gpu.machine(machine_id)
    if machine.ssh_direct is None:
        raise SystemExit(
            f"cloud-gpu: {machine_id} has no direct ssh door ({machine.status}); "
            "rsync needs one — `status --wait` first"
        )
    return machine


def ssh_argv(machine: Machine, key: Path) -> list[str]:
    door = machine.ssh_direct
    assert door is not None
    return [
        "ssh",
        "-i",
        str(key),
        "-p",
        str(door.port),
        *Ssh.OPTIONS,
        f"{door.username}@{door.host}",
    ]


def rsync_argv(
    machine: Machine, key: Path, source: str, target: str, *, filters: Sequence[str]
) -> list[str]:
    door = machine.ssh_direct
    assert door is not None
    ssh = " ".join(
        shlex.quote(p)
        for p in ["ssh", "-i", str(key), "-p", str(door.port), *Ssh.OPTIONS]
    )
    return ["rsync", "-az", "--info=progress2", "-e", ssh, *filters, source, target]


def push_filters() -> list[str]:
    return [f"--exclude={pattern}" for pattern in Rsync.EXCLUDES]


def pull_filters(name: str) -> list[str]:
    """Only this run's directories: rsync reads includes before the
    final exclude, and a directory needs itself and its contents."""
    return [f"--include={name}-*/", f"--include={name}-*/**", "--exclude=*"]


def run_remote(machine: Machine, key: Path, script: str) -> None:
    argv = [*ssh_argv(machine, key), "bash", "-s"]
    print("$", " ".join(argv), "<<script", flush=True)
    completed = subprocess.run(argv, input=script, text=True, check=False)
    if completed.returncode:
        raise SystemExit(f"cloud-gpu: remote command exited {completed.returncode}")


def run_local(argv: list[str]) -> None:
    print("$", " ".join(argv), flush=True)
    completed = subprocess.run(argv, check=False)
    if completed.returncode:
        raise SystemExit(f"cloud-gpu: {argv[0]} exited {completed.returncode}")


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    load_dotenv(args.dotenv)
    try:
        gpu: GpuProvider = resolve(args.provider).build()
    except MissingCredentialError as error:
        raise SystemExit(f"cloud-gpu: {error}") from None
    try:
        dispatch(gpu, args)
    except ProviderError as error:
        raise SystemExit(f"cloud-gpu: {error}") from None


def dispatch(gpu: GpuProvider, args: argparse.Namespace) -> None:  # noqa: PLR0912 - one branch per subcommand
    if args.command == "offers":
        show_offers(gpu, args)
    elif args.command == "launch":
        spec = MachineSpec(
            name=args.name,
            gpu=args.gpu,
            image=args.image,
            count=args.count,
            disk_gb=args.disk,
            tier=args.tier,
            data_centers=tuple(args.datacenter),
        )
        machine = gpu.launch(spec)
        print(describe(machine))
        if args.wait:
            wait_running(gpu, machine.id)
    elif args.command == "machines":
        for machine in gpu.machines():
            print(describe(machine))
    elif args.command == "status":
        print(
            describe(wait_running(gpu, args.id) if args.wait else gpu.machine(args.id))
        )
    elif args.command == "ssh":
        machine = gpu.machine(args.id)
        for label, door in (
            ("direct", machine.ssh_direct),
            ("proxy", machine.ssh_proxy),
        ):
            print(
                f"{label:7s}",
                (door.command + f" -i {args.ssh_key}") if door else "not yet",
            )
    elif args.command == "push":
        machine = direct_door(gpu, args.id)
        run_remote(machine, args.ssh_key, f"mkdir -p {Remote.DIR}")
        run_local(
            rsync_argv(
                machine,
                args.ssh_key,
                f"{REPO}/",
                f"{machine.ssh_direct.username}@{machine.ssh_direct.host}:{Remote.DIR}/",
                filters=push_filters(),
            )
        )
    elif args.command == "bootstrap":
        run_remote(direct_door(gpu, args.id), args.ssh_key, Remote.bootstrap_script())
    elif args.command == "run":
        words = [w for w in args.argv if w != "--"]
        if not words:
            raise SystemExit("cloud-gpu run: give the command after --")
        command = " ".join(shlex.quote(w) for w in words)
        clock = (
            f"timeout --kill-after=60 {args.deadline_min * 60} "
            if args.deadline_min
            else ""
        )
        run_remote(
            direct_door(gpu, args.id),
            args.ssh_key,
            f"set -euo pipefail\ncd {Remote.DIR}/pipeline\n"
            f"{clock}env {Remote.RUN_ENV} {Remote.VENV}/bin/python {command}",
        )
    elif args.command == "pull":
        machine = direct_door(gpu, args.id)
        local = REPO / Rsync.PULL_SUBDIR
        local.mkdir(parents=True, exist_ok=True)
        door = machine.ssh_direct
        remote = f"{door.username}@{door.host}:{Remote.DIR}/{Rsync.PULL_SUBDIR}/"
        run_local(
            rsync_argv(
                machine,
                args.ssh_key,
                remote,
                f"{local}/",
                filters=pull_filters(args.name),
            )
        )
    elif args.command == "logs":
        print(gpu.logs(args.id, tail=args.tail))
    else:  # stop / start / restart / terminate
        gpu.act(args.id, Action(args.command))
        print(f"{args.command}: {args.id}")


if __name__ == "__main__":
    main(sys.argv[1:])
