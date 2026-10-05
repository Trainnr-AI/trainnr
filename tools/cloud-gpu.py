#!/usr/bin/env python3
"""A rented GPU as a runbook: offers → launch → push → bootstrap → run
→ pull → terminate, every step one subcommand, every vendor behind
the provider seam (`trainnr/cloud`). The first vendor is Runpod.

    cd trainnr && uv run --extra sim python ../tools/cloud-gpu.py offers \\
        --tier COMMUNITY
    cd trainnr && uv run --extra sim python ../tools/cloud-gpu.py launch \\
        --name t5-cloud --gpu "NVIDIA GeForce RTX 4090" --tier COMMUNITY --wait
    cd trainnr && uv run --extra sim python ../tools/cloud-gpu.py push <id>
    cd trainnr && uv run --extra sim python ../tools/cloud-gpu.py bootstrap <id>
    cd trainnr && uv run --extra sim python ../tools/cloud-gpu.py run <id> -- \\
        ../tools/e2e-smoke.py --scale cloud --name t5-cloud --from train
    cd trainnr && uv run --extra sim python ../tools/cloud-gpu.py pull <id> t5-cloud
    cd trainnr && uv run --extra sim python ../tools/cloud-gpu.py terminate <id>

The API key is read from the repo's `.env` (`RUNPOD_API_KEY`) into the
environment and travels in one request header; this tool never prints
it and `push` never ships it (`trainnr/cloud/transfer.py` names
what leaves this machine). Billing starts at `launch` and stops at
`terminate` — `stop` keeps the disk and a smaller bill; nothing here
terminates on your behalf.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from _lab import REPO, TOOLS, bootstrap, load_dotenv

bootstrap()

from trainnr.cloud import (  # noqa: E402
    DEFAULT_PROVIDER,
    NETWORK_VOLUME_MOUNT,
    Action,
    GpuProvider,
    Machine,
    MachineSpec,
    MissingCredentialError,
    ProviderError,
    SshEndpoint,
    Tier,
    providers,
    resolve,
)
from trainnr.cloud.transfer import (  # noqa: E402
    Rsync,
    follow_filters,
    pull_filters,
    push_filters,
    remote_path,
    rsync_argv,
    ssh_argv,
)
from trainnr.envs.lerobot_train_log import RunLayout  # noqa: E402


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
    FOLLOW_EVERY_S = 20


class Remote:
    """Where the repo lives on the machine and how its venv is built — the
    values `tools/_pod-bootstrap.sh` takes as TRAINNR_* variables."""

    DIR = "/workspace/trainnr"
    VENV = ".venv-train"
    PYTHON = "3.12.8"
    EXTRAS = ("sim", "viz", "train")
    APT = "libegl1 libgl1 libglib2.0-0 rsync"
    # EGL is the offscreen renderer on a bare Linux GPU box; the WSL
    # variables (Mesa's D3D12 path) do not apply there.
    RUN_ENV = "MUJOCO_GL=egl OMP_NUM_THREADS=1"
    # A detached `run` writes its command here and its output there;
    # `tail` reads the log. The script file is the record of what ran.
    RUN_SH = f"{DIR}/run.sh"
    RUN_LOG = f"{DIR}/run.log"
    BOOTSTRAP = TOOLS / "_pod-bootstrap.sh"

    @classmethod
    def bootstrap_script(cls) -> str:
        extras = " ".join(f"--extra {e}" for e in cls.EXTRAS)
        exports = (
            f"export TRAINNR_DIR={cls.DIR} TRAINNR_VENV={cls.VENV} "
            f"TRAINNR_PYTHON={cls.PYTHON} "
            f"TRAINNR_EXTRAS={shlex.quote(extras)} TRAINNR_APT={shlex.quote(cls.APT)}\n"
        )
        return exports + cls.BOOTSTRAP.read_text(encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    # Everything after `--` is the remote command, verbatim. Split by
    # hand: argparse's REMAINDER swallows this tool's own flags after the
    # machine id (measured 2026-08-27: `--detach` reached python on the pod).
    remote: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, remote = argv[:cut], argv[cut + 1 :]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--provider", default=DEFAULT_PROVIDER, help=f"one of {sorted(providers())}"
    )
    parser.add_argument(
        "--dotenv", type=Path, default=REPO / ".env", help="where the API key lives"
    )
    parser.add_argument(
        "--ssh-key",
        type=Path,
        default=None,
        help="an identity file for ssh/rsync; default: what ~/.ssh/config and "
        "the agent decide",
    )
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
    launch.add_argument(
        "--volume",
        default="",
        help="a network volume id to mount (its data center must be among "
        "--datacenter; the repo and venvs of earlier pods live there)",
    )
    launch.add_argument(
        "--mount",
        default=NETWORK_VOLUME_MOUNT,
        help="where the volume appears in the container",
    )
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
        ("bootstrap", "apt + uv + the train venv, then a torch/CUDA/mujoco check"),
        ("logs", "the container's recent output"),
        ("tail", "the last lines of a detached run's log"),
        ("stop", "stop (keeps the disk)"),
        ("start", "resume a stopped machine"),
        ("restart", "restart the container"),
        ("terminate", "delete it — permanent"),
    ):
        one = sub.add_parser(name, help=text)
        one.add_argument("id")
    sub.choices["logs"].add_argument("--tail", type=int, default=100)
    sub.choices["tail"].add_argument("--lines", type=int, default=40)
    push = sub.add_parser("push", help="rsync the repo onto the machine")
    push.add_argument("id")
    push.add_argument(
        "--with-runs",
        default=None,
        metavar="NAME",
        help="also ship runs/NAME-lerobot (a converted dataset, not its frames)",
    )
    follow = sub.add_parser(
        "follow", help="mirror a run's light files every few seconds (for train-watch)"
    )
    follow.add_argument("id")
    follow.add_argument("name", help="the run name (the --name of the chain)")
    follow.add_argument("--every", type=int, default=Defaults.FOLLOW_EVERY_S)
    follow.add_argument(
        "--watch",
        action="store_true",
        help="also open the dashboard: train-watch --follow on the mirrored run",
    )
    run = sub.add_parser(
        "run", help="ssh: run a command in the repo's trainnr/ with the train venv"
    )
    run.add_argument("id")
    run.add_argument(
        "--detach",
        action="store_true",
        help=f"start it under nohup and return; it writes {Remote.RUN_LOG}, "
        "which `tail` reads",
    )
    run.add_argument(
        "--deadline-min",
        type=int,
        default=None,
        help="the machine kills the command after this many minutes (`timeout`, "
        "TERM then KILL a minute later); checkpoints written before survive",
    )
    pull = sub.add_parser("pull", help="rsync runs/<name>-* back")
    pull.add_argument("id")
    pull.add_argument("name", help="the run name (the --name of the chain)")
    args = parser.parse_args(argv)
    args.argv = remote
    return args


# ---- the steps ---------------------------------------------------------


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


def door_of(gpu: GpuProvider, machine_id: str) -> SshEndpoint:
    """The machine's own sshd — what rsync and a script need."""
    machine = gpu.machine(machine_id)
    if machine.ssh_direct is None:
        raise SystemExit(
            f"cloud-gpu: {machine_id} has no direct ssh door ({machine.status}); "
            "rsync needs one — `status --wait` first"
        )
    return machine.ssh_direct


def run_remote(door: SshEndpoint, key: Path | None, script: str) -> None:
    argv = [*ssh_argv(door, key), "bash", "-s"]
    print("$", " ".join(argv), "<<script", flush=True)
    completed = subprocess.run(argv, input=script, text=True, check=False)
    if completed.returncode:
        raise SystemExit(f"cloud-gpu: remote command exited {completed.returncode}")


def run_local(argv: Sequence[str]) -> None:
    if shutil.which(argv[0]) is None:
        raise SystemExit(f"cloud-gpu: {argv[0]} is not installed on this machine")
    print("$", " ".join(argv), flush=True)
    completed = subprocess.run(argv, check=False)
    if completed.returncode:
        raise SystemExit(f"cloud-gpu: {argv[0]} exited {completed.returncode}")


def mirror(
    door: SshEndpoint,
    key: Path | None,
    name: str,
    *,
    filters: Sequence[str],
    progress: bool,
) -> list[str]:
    """The rsync that brings `runs/<name>-*` from the machine into this
    box's runs/ (what `pull` runs once and `follow` runs in a loop)."""
    local = REPO / Rsync.RUNS_SUBDIR
    local.mkdir(parents=True, exist_ok=True)
    remote = remote_path(door, f"{Remote.DIR}/{Rsync.RUNS_SUBDIR}/")
    del name  # the filters carry it; kept in the signature for the reader
    return rsync_argv(
        door, key, remote, f"{local}/", filters=filters, progress=progress
    )


# ---- the subcommands ---------------------------------------------------------


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


def launch_machine(gpu: GpuProvider, args: argparse.Namespace) -> None:
    machine = gpu.launch(
        MachineSpec(
            name=args.name,
            gpu=args.gpu,
            image=args.image,
            count=args.count,
            disk_gb=args.disk,
            tier=args.tier,
            data_centers=tuple(args.datacenter),
            network_volume=args.volume,
            volume_mount=args.mount,
        )
    )
    print(describe(machine))
    if args.wait:
        wait_running(gpu, machine.id)


def show_status(gpu: GpuProvider, args: argparse.Namespace) -> None:
    print(describe(wait_running(gpu, args.id) if args.wait else gpu.machine(args.id)))


def show_ssh(gpu: GpuProvider, args: argparse.Namespace) -> None:
    machine = gpu.machine(args.id)
    identity = f" -i {args.ssh_key}" if args.ssh_key else ""
    for label, door in (("direct", machine.ssh_direct), ("proxy", machine.ssh_proxy)):
        print(f"{label:7s}", door.command + identity if door else "not yet")


def push_repo(gpu: GpuProvider, args: argparse.Namespace) -> None:
    door = door_of(gpu, args.id)
    run_remote(door, args.ssh_key, f"mkdir -p {Remote.DIR}")
    run_local(
        rsync_argv(
            door,
            args.ssh_key,
            f"{REPO}/",
            remote_path(door, f"{Remote.DIR}/"),
            filters=push_filters(),
        )
    )
    if not args.with_runs:
        return
    dataset = RunLayout(REPO / Rsync.RUNS_SUBDIR, args.with_runs).dataset
    if not dataset.is_dir():
        raise SystemExit(f"cloud-gpu push: no dataset at {dataset}")
    relative = dataset.relative_to(REPO).as_posix()
    run_remote(door, args.ssh_key, f"mkdir -p {Remote.DIR}/{relative}")
    run_local(
        rsync_argv(
            door,
            args.ssh_key,
            f"{dataset}/",
            remote_path(door, f"{Remote.DIR}/{relative}/"),
            filters=[],
        )
    )


def run_command(gpu: GpuProvider, args: argparse.Namespace) -> None:
    if not args.argv:
        raise SystemExit("cloud-gpu run: give the command after --")
    command = " ".join(shlex.quote(w) for w in args.argv)
    clock = (
        f"timeout --kill-after=60 {args.deadline_min * 60} "
        if args.deadline_min
        else ""
    )
    line = f"{clock}env {Remote.RUN_ENV} {Remote.VENV}/bin/python {command}"
    if args.detach:
        # The command goes into a file and nohup runs the file: no nested
        # quoting through the ssh stdin pipe (which broke a backgrounded
        # `bash -c` on the pod, 2026-08-27), and the file stays as the
        # record of what ran.
        script = (
            f"cat > {Remote.RUN_SH} <<'TRAINNR_RUN'\n"
            f"cd {Remote.DIR}/pipeline\n{line}\nRQ_RUN\n"
            f"nohup bash {Remote.RUN_SH} > {Remote.RUN_LOG} 2>&1 < /dev/null &\n"
            f"echo started: $(tail -n 1 {Remote.RUN_SH})"
        )
    else:
        script = f"set -euo pipefail\ncd {Remote.DIR}/pipeline\n{line}"
    run_remote(door_of(gpu, args.id), args.ssh_key, script)
    if args.detach:
        print(f"detached; `tail {args.id}` reads {Remote.RUN_LOG}")


def tail_log(gpu: GpuProvider, args: argparse.Namespace) -> None:
    run_remote(
        door_of(gpu, args.id),
        args.ssh_key,
        f"tail -n {args.lines} {Remote.RUN_LOG}; echo; "
        "pgrep -fa 'e2e-smoke|lerobot' | head -3 || true",
    )


def pull_runs(gpu: GpuProvider, args: argparse.Namespace) -> None:
    door = door_of(gpu, args.id)
    run_local(
        mirror(
            door,
            args.ssh_key,
            args.name,
            filters=pull_filters(args.name),
            progress=True,
        )
    )


def follow_run(gpu: GpuProvider, args: argparse.Namespace) -> None:
    """Mirror `runs/<name>-*` minus the weights every `--every` seconds
    until interrupted; with --watch, the dashboard runs alongside on the
    mirrored files (the sim venv suffices: it reads files, not torch)."""
    door = door_of(gpu, args.id)
    argv = mirror(
        door, args.ssh_key, args.name, filters=follow_filters(args.name), progress=False
    )
    dashboard = None
    if args.watch:
        training = RunLayout(REPO / Rsync.RUNS_SUBDIR, args.name).training
        watch = [
            sys.executable,
            str(TOOLS / "train-watch.py"),
            "--follow",
            str(training),
        ]
        print("$", " ".join(watch), flush=True)
        dashboard = subprocess.Popen(watch)
    print(f"following {args.name} on {args.id} every {args.every} s; Ctrl-C to stop")
    try:
        while dashboard is None or dashboard.poll() is None:
            subprocess.run(argv, check=False)
            time.sleep(args.every)
    except KeyboardInterrupt:
        pass
    finally:
        if dashboard is not None and dashboard.poll() is None:
            dashboard.terminate()


def act(gpu: GpuProvider, args: argparse.Namespace) -> None:
    gpu.act(args.id, Action(args.command))
    print(f"{args.command}: {args.id}")


HANDLERS = {
    "offers": show_offers,
    "launch": launch_machine,
    "machines": lambda gpu, _args: print(*map(describe, gpu.machines()), sep="\n"),
    "status": show_status,
    "ssh": show_ssh,
    "push": push_repo,
    "bootstrap": lambda gpu, args: run_remote(
        door_of(gpu, args.id), args.ssh_key, Remote.bootstrap_script()
    ),
    "run": run_command,
    "tail": tail_log,
    "follow": follow_run,
    "pull": pull_runs,
    "logs": lambda gpu, args: print(gpu.logs(args.id, tail=args.tail)),
    **{action.value: act for action in Action},
}


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    load_dotenv(args.dotenv)
    try:
        gpu: GpuProvider = resolve(args.provider).build()
        HANDLERS[args.command](gpu, args)
    except (MissingCredentialError, ProviderError) as error:
        raise SystemExit(f"cloud-gpu: {error}") from None


if __name__ == "__main__":
    main(sys.argv[1:])
