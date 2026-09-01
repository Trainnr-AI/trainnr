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

SEEN_LINES_CAP = 20000

# rsl-rl's console block (the RL trainers' format, vs lerobot's k:v
# line): an ANSI-bold "Learning iteration N/M" header followed by
# aligned "Name: value" rows, including per-term reward breakdowns.
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RSL_ITER_RE = re.compile(r"Learning iteration +(\d+)/(\d+)")
RSL_ROW_RE = re.compile(r"^\s*([A-Za-z_/ ]+[A-Za-z_/]): +(-?[0-9.]+)")
RSL_SERIES = {
    "Mean reward": "rl/reward",
    "Mean episode length": "rl/episode_length",
    "Mean value loss": "rl/value_loss",
    "Mean surrogate loss": "rl/surrogate_loss",
    "Mean entropy loss": "rl/entropy_loss",
    "Mean action std": "rl/action_std",
    "Steps per second": "rl/steps_per_second",
}


def parse_rsl(rr: Any, name: str, raw: str, seen_iters: set) -> tuple | None:
    """rsl-rl blocks into Rerun: each unseen iteration's rows become
    series on the `iteration` timeline (reward terms and metrics get
    their own families). Returns the latest (iteration, total, reward)
    for the status card."""
    latest = None
    iteration = None
    total = 0
    emit = False
    reward = None
    for line in ANSI_RE.sub("", raw).splitlines():
        header = RSL_ITER_RE.search(line)
        if header:
            iteration = int(header.group(1))
            total = int(header.group(2))
            emit = iteration not in seen_iters
            if emit:
                seen_iters.add(iteration)
                rr.set_time("wall", timestamp=time.time())
                rr.set_time("iteration", sequence=iteration)
            latest = (iteration, total, reward)
            continue
        if iteration is None or not emit:
            continue
        row = RSL_ROW_RE.match(line)
        if not row:
            continue
        key, value = row.group(1).strip(), float(row.group(2))
        if key in RSL_SERIES:
            rr.log(f"{name}/{RSL_SERIES[key]}", rr.Scalars(value))
            if key == "Mean reward":
                reward = value
                latest = (iteration, total, reward)
        elif key.startswith("Episode_Reward/"):
            term = key.removeprefix("Episode_Reward/")
            rr.log(f"{name}/rl/reward_terms/{term}", rr.Scalars(value))
        elif key.startswith("Metrics/"):
            metric = key.removeprefix("Metrics/")
            rr.log(f"{name}/rl/metrics/{metric}", rr.Scalars(value))
    return latest


# Only a default: every run states its own budget with --steps.
DEFAULT_TOTAL_STEPS = 10000

STAGE = re.compile(
    r"== training [a-z]+|== evaluating [a-z]+"
    r"|[a-z]+: checkpoint under \S+|verdict -> \S+"
)


def parse_poll(
    rr: Any, name: str, raw: str, state: tuple
) -> tuple[str, str, Any, str, int]:
    """One ssh poll's lines into Rerun: sentinel-prefixed arm/GPU/eval
    lines, stages, per-trial eval records, and train metrics."""
    seen_stages, seen_records, seen_lines = state
    arm, gpu, latest = "?", "?", None
    last_step = -1
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
        if metrics is None:
            continue
        latest = metrics
        if line in seen_lines:
            last_step = metrics.step
            continue
        seen_lines[line] = None
        if 0 <= metrics.step < last_step - 1000:
            # The counter jumped far backward: a NEW arm starting over
            # (the paired study trains twice) - mark it once.
            rr.log(f"{name}/stage", rr.TextLog("new arm: step counter restarted"))
        last_step = metrics.step
        # Two clocks on every point: train_step overlays the arms for
        # comparison; wall keeps live data at the END of a timeline the
        # operator can follow (the "it stops at 10000" lesson,
        # 2026-09-01 - a restarted counter streams BEHIND a shared
        # sequence timeline's end, invisibly).
        rr.reset_time()  # the trial clock must not ride along (see emit_record)
        rr.set_time("wall", timestamp=time.time())
        rr.set_time("train_step", sequence=metrics.step)
        # The step itself as a series: on the wall timeline this plot IS
        # "which step are we at right now" (the operator's ask).
        rr.log(f"{name}/train/step", rr.Scalars(float(metrics.step)))
        for key, value in metrics.metrics.items():
            rr.log(f"{name}/train/{key}", rr.Scalars(value))
    return arm, gpu, latest, eval_progress, last_step


def emit_record(
    rr: Any, name: str, arm: str, line: str, seen: set[tuple[str, int]]
) -> bool:
    """One per-trial eval record onto the `trial` timeline: success as a
    0/1 series per arm, and a TextLog naming the trial's outcome.
    Returns False for a line this feed could not read — a JSON array or
    a null field raises TypeError, which the old two-exception catch let
    through to kill an overnight feed (review 2026-09-01)."""
    import json  # noqa: PLC0415

    try:
        record = json.loads(line)
        trial, success = int(record["trial"]), bool(record["success"])
    except (ValueError, KeyError, TypeError):
        return False
    if (arm, trial) in seen:
        return True
    seen.add((arm, trial))
    # Only the trial timeline for this family: `set_time` persists on the
    # thread, so leaving wall/train_step set here stamped the NEXT
    # metric line with this record's stale clocks (review 2026-09-01).
    rr.reset_time()
    rr.set_time("trial", sequence=trial)
    rr.log(f"{name}/eval/{arm}/success", rr.Scalars(1.0 if success else 0.0))
    rr.log(
        f"{name}/stage",
        rr.TextLog(f"eval {arm} trial {trial}: {'KEEP' if success else 'fail'}"),
    )
    return True


def status_card(  # noqa: PLR0913 - the card's inputs, each named
    rr: Any,
    name: str,
    state: tuple,
    *,
    total_steps: int,
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
        eta = f"~{(total_steps - last_step) / max(per_s, 1e-9) / 60:.0f} min"
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
            f"- step **{max(last_step, 0)} / {total_steps}** ({rate}, ETA {eta})\n"
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
    parser.add_argument(
        # The run's own step budget: baked in as 10000 once, which made
        # the card lie for every other run (review 2026-09-01).
        "--steps",
        type=int,
        default=DEFAULT_TOTAL_STEPS,
        help="the run's total training steps, for the card's progress and ETA",
    )
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
    seen_iters: set[int] = set()
    # Insertion-ordered so eviction can drop the OLDEST half; a plain
    # set has no age and the cap could only clear (review 2026-09-01).
    seen_lines: dict[str, None] = {}
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
                    "echo RQPOLL_OK; "
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
            rr.log(
                f"{args.name}/stage",
                rr.TextLog("feed: ssh poll timed out; retrying"),
            )
            time.sleep(args.every * 5)
            continue
        if "RQPOLL_OK" not in raw:
            # The sentinel is echoed by the remote shell itself, so its
            # absence means the LINK failed — not an idle trainer or a
            # log that does not exist yet (which the first cut conflated
            # with a dead link; second review, 2026-09-01).
            rr.log(
                f"{args.name}/stage",
                rr.TextLog("feed: ssh poll returned nothing (link down?)"),
            )
            time.sleep(args.every * 5)
            continue
        if len(seen_lines) > SEEN_LINES_CAP:
            # Evict the OLDEST half, never clear: a clear makes the whole
            # 256 KB tail "new" next poll, re-logging every historical
            # point at wall=now — a vertical replay cliff on the very
            # timeline the wall clock exists to keep honest, and the new-arm
            # detector re-fires on any restart still in the tail
            # (review 2026-09-01).
            for old in list(seen_lines)[: SEEN_LINES_CAP // 2]:
                del seen_lines[old]
        arm, gpu, latest, eval_progress, last_step = parse_poll(
            rr, args.name, raw, (seen_stages, seen_records, seen_lines)
        )
        rsl = parse_rsl(rr, args.name, raw, seen_iters)
        if rsl is not None and rsl[0] is not None:
            card = f"## RL training\n\n- iteration **{rsl[0]} / {rsl[1]}**\n"
            if rsl[2] is not None:
                card += f"- mean reward **{rsl[2]:.2f}**\n"
            card += f"- GPU {gpu}\n"
            rr.log(
                f"{args.name}/status",
                rr.TextDocument(card, media_type=rr.MediaType.MARKDOWN),
                static=True,
            )
        if "%" in gpu:
            rr.set_time("wall", timestamp=time.time())
            util, memory = (part.strip() for part in gpu.split(","))
            rr.log(f"{args.name}/gpu/utilization", rr.Scalars(float(util.split()[0])))
            rr.log(
                f"{args.name}/gpu/memory_gb",
                rr.Scalars(float(memory.split()[0]) / 1024.0),
            )
        if rsl is None or rsl[0] is None:
            # The lerobot-style card only when no rsl-rl blocks are in
            # the log - it was clobbering the RL card every poll.
            last_seen = status_card(
                rr,
                args.name,
                (arm, gpu, latest, last_step, last_seen, len(seen_stages)),
                total_steps=args.steps,
                eval_progress=eval_progress,
                trials_seen=len(seen_records),
            )
        time.sleep(args.every)


if __name__ == "__main__":
    sys.exit(main())
