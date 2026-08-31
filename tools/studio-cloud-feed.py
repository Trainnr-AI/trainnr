"""The rented card's training, live in the Studio window.

    cd pipeline && uv run --env-file wsl.env --extra sim --extra viz \\
        python ../tools/studio-cloud-feed.py <ssh-door> <remote-log> \\
        [--name cloud] [--every 2]

One window for everything (the operator's ask, 2026-08-31): the Studio
already owns the Rerun ingest port; this tool completes the cloud leg —
it tails the remote run.log over ssh, parses the trainer's own metric
lines with the pipeline's parsers (`envs/lerobot_train_log`), and
streams them to :9876:

    cloud/train/loss, l1, kld, grdn, lr     on the `train_step` timeline
    cloud/gpu/utilization, memory_gb        the card itself
    cloud/stage                             arm starts and checkpoints
                                            as TextLog events
    cloud/status                            a live card: which arm, the
                                            step, throughput, ETA, GPU

The ssh door is e.g. `root@216.243.220.136:13337` (`cloud-gpu.py
machines` prints it). The feed is read-only and reconnects on drops.
"""

import argparse
import pathlib
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from _lab import bootstrap

bootstrap()

from rq_pipeline.envs.lerobot_train_log import parse_train_line  # noqa: E402

STAGE = re.compile(
    r"== training [a-z]+|== evaluating [a-z]+"
    r"|[a-z]+: checkpoint under \S+|verdict -> \S+"
)


def parse_poll(
    rr: Any, name: str, raw: str, state: tuple
) -> tuple[str, str, Any, str, int]:
    """One ssh poll's lines into Rerun: sentinel-prefixed arm/GPU/eval
    lines, stages, per-trial eval records, and train metrics."""
    seen_stages, seen_records, last_step = state
    arm, gpu, latest = "?", "?", None
    eval_progress, record_file = "", None
    for line in raw.splitlines():
        if line.startswith("RQEVAL "):
            eval_progress = line.removeprefix("RQEVAL ").strip()
            continue
        if line.startswith("RQREC "):
            record_file = pathlib.PurePosixPath(
                line.removeprefix("RQREC ").strip()
            ).name.removesuffix("-records.jsonl")
            continue
        if record_file and line.startswith("{"):
            emit_record(rr, name, record_file, line, seen_records)
            continue
        if line.startswith("RQARM "):
            arm = line.removeprefix("RQARM ").removeprefix("== training ").strip()
            continue
        if line.startswith("RQGPU "):
            gpu = line.removeprefix("RQGPU ").strip()
            continue
        for stage in STAGE.findall(line):
            if stage not in seen_stages:
                seen_stages.add(stage)
                rr.log(f"{name}/stage", rr.TextLog(stage))
                print(f"stage: {stage}", flush=True)
        metrics = parse_train_line(line)
        if metrics is None or metrics.step <= last_step:
            continue
        last_step = metrics.step
        latest = metrics
        rr.set_time("train_step", sequence=metrics.step)
        for key, value in metrics.metrics.items():
            rr.log(f"{name}/train/{key}", rr.Scalars(value))
    return arm, gpu, latest, eval_progress, last_step


def emit_record(
    rr: Any, name: str, arm: str, line: str, seen: set[tuple[str, int]]
) -> None:
    """One per-trial eval record onto the `trial` timeline: success as a
    0/1 series per arm, and a TextLog naming the trial's outcome."""
    import json  # noqa: PLC0415

    try:
        record = json.loads(line)
        trial, success = int(record["trial"]), bool(record["success"])
    except (ValueError, KeyError):
        return
    if (arm, trial) in seen:
        return
    seen.add((arm, trial))
    rr.set_time("trial", sequence=trial)
    rr.log(f"{name}/eval/{arm}/success", rr.Scalars(1.0 if success else 0.0))
    rr.log(
        f"{name}/stage",
        rr.TextLog(f"eval {arm} trial {trial}: {'KEEP' if success else 'fail'}"),
    )


def status_card(
    rr: Any,
    name: str,
    state: tuple,
    *,
    eval_progress: str = "",
    trials_seen: int = 0,
) -> tuple[float, int] | None:
    """Render the live card; returns the (wall clock, step) pair the next
    poll's ETA is computed against."""
    arm, gpu, latest, last_step, last_seen, stages = state
    now = time.time()
    rate = eta = "?"
    if last_step >= 0 and last_seen is not None and last_step > last_seen[1]:
        per_s = (last_step - last_seen[1]) / max(now - last_seen[0], 1e-9)
        rate = f"{per_s:.1f} steps/s"
        eta = f"~{(10000 - last_step) / max(per_s, 1e-9) / 60:.0f} min"
    losses = (
        "  ".join(
            f"{key} {value:.3f}"
            for key, value in latest.metrics.items()
            if "loss" in key
        )
        if latest
        else "(between arms - the next trainer is starting up)"
    )
    rr.log(
        f"{name}/status",
        rr.TextDocument(
            f"## cloud training - {arm}\n\n"
            f"- step **{max(last_step, 0)} / 10000** ({rate}, ETA {eta})\n"
            f"- {losses}\n"
            f"- GPU {gpu}\n"
            f"- stages so far: {stages}\n"
            + (f"- {eval_progress}\n" if eval_progress else "")
            + (f"- eval trials recorded: {trials_seen}\n" if trials_seen else ""),
            media_type=rr.MediaType.MARKDOWN,
        ),
        # Static: the card answers "right now" regardless of where the
        # operator has scrubbed the timeline.
        static=True,
    )
    if last_step >= 0 and (last_seen is None or last_step > last_seen[1]):
        return (now, last_step)
    return last_seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("door", help="user@host:port (cloud-gpu machines prints it)")
    parser.add_argument("log", help="remote log path, e.g. /workspace/robotiq/run.log")
    parser.add_argument("--name", default="cloud")
    parser.add_argument("--every", type=float, default=2.0, help="poll seconds")
    parser.add_argument("--key", type=Path, default=Path.home() / ".ssh" / "id_ed25519")
    parser.add_argument("--address", default="rerun+http://127.0.0.1:9876/proxy")
    parser.add_argument(
        "--records",
        default="",
        help="remote glob of per-trial records.jsonl files to stream as cloud/eval/*",
    )
    args = parser.parse_args()

    import rerun as rr  # noqa: PLC0415 - viz extra

    user_host, _, port = args.door.rpartition(":")
    rr.init(f"rq-{args.name}-feed", spawn=False)
    rr.connect_grpc(args.address)
    print(f"feeding {args.door}:{args.log} -> {args.address} as {args.name}/")

    seen_stages: set[str] = set()
    seen_records: set[tuple[str, int]] = set()
    last_step = -1
    last_seen: tuple[float, int] | None = None  # (wall clock, step) for ETA
    while True:
        try:
            raw = subprocess.run(
                [
                    "ssh",
                    "-p",
                    port,
                    "-i",
                    str(args.key),
                    "-o",
                    "ConnectTimeout=15",
                    user_host,
                    f"tr '\\r' '\\n' < {args.log} 2>/dev/null | tail -c 262144; "
                    f"echo; tr '\\r' '\\n' < {args.log} 2>/dev/null | "
                    "grep -a '== training' | tail -1 | sed 's/^/RQARM /'; "
                    "nvidia-smi --query-gpu=utilization.gpu,memory.used "
                    "--format=csv,noheader | sed 's/^/RQGPU /'; "
                    f"tr '\\r' '\\n' < {args.log} 2>/dev/null | "
                    "grep -a 'Stepping through eval batches' | tail -1 "
                    "| sed 's/^/RQEVAL /'"
                    + (
                        f'; for f in {args.records}; do echo "RQREC $f"; '
                        'cat "$f" 2>/dev/null; done'
                        if args.records
                        else ""
                    ),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=False,
            ).stdout
        except subprocess.TimeoutExpired:
            time.sleep(args.every * 5)
            continue
        arm, gpu, latest, eval_progress, last_step = parse_poll(
            rr, args.name, raw, (seen_stages, seen_records, last_step)
        )
        if "%" in gpu:
            util, memory = (part.strip() for part in gpu.split(","))
            rr.log(f"{args.name}/gpu/utilization", rr.Scalars(float(util.split()[0])))
            rr.log(
                f"{args.name}/gpu/memory_gb",
                rr.Scalars(float(memory.split()[0]) / 1024.0),
            )
        last_seen = status_card(
            rr,
            args.name,
            (arm, gpu, latest, last_step, last_seen, len(seen_stages)),
            eval_progress=eval_progress,
            trials_seen=len(seen_records),
        )
        time.sleep(args.every)


if __name__ == "__main__":
    sys.exit(main())
