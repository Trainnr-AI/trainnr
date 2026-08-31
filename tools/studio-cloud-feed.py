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
    cloud/stage                             arm starts and checkpoints
                                            as TextLog events

The ssh door is e.g. `root@216.243.220.136:13337` (`cloud-gpu.py
machines` prints it). The feed is read-only and reconnects on drops.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from _lab import bootstrap

bootstrap()

from rq_pipeline.envs.lerobot_train_log import parse_train_line  # noqa: E402

STAGE = re.compile(r"== training [a-z]+|[a-z]+: checkpoint under \S+")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("door", help="user@host:port (cloud-gpu machines prints it)")
    parser.add_argument("log", help="remote log path, e.g. /workspace/robotiq/run.log")
    parser.add_argument("--name", default="cloud")
    parser.add_argument("--every", type=float, default=2.0, help="poll seconds")
    parser.add_argument("--key", type=Path, default=Path.home() / ".ssh" / "id_ed25519")
    parser.add_argument("--address", default="rerun+http://127.0.0.1:9876/proxy")
    args = parser.parse_args()

    import rerun as rr  # noqa: PLC0415 - viz extra

    user_host, _, port = args.door.rpartition(":")
    rr.init(f"rq-{args.name}-feed", spawn=False)
    rr.connect_grpc(args.address)
    print(f"feeding {args.door}:{args.log} -> {args.address} as {args.name}/")

    seen_stages: set[str] = set()
    last_step = -1
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
                    f"tr '\\r' '\\n' < {args.log} 2>/dev/null | tail -c 262144",
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
        for line in raw.splitlines():
            for stage in STAGE.findall(line):
                if stage not in seen_stages:
                    seen_stages.add(stage)
                    rr.log(f"{args.name}/stage", rr.TextLog(stage))
                    print(f"stage: {stage}", flush=True)
            metrics = parse_train_line(line)
            if metrics is None or metrics.step <= last_step:
                continue
            last_step = metrics.step
            rr.set_time("train_step", sequence=metrics.step)
            for key, value in metrics.metrics.items():
                rr.log(f"{args.name}/train/{key}", rr.Scalars(value))
        time.sleep(args.every)


if __name__ == "__main__":
    sys.exit(main())
