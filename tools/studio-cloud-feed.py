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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from _lab import bootstrap

bootstrap()

from rq_pipeline.cloud.provider import SshEndpoint  # noqa: E402
from rq_pipeline.cloud.transfer import ssh_argv  # noqa: E402
from rq_pipeline.envs.lerobot_train_log import (  # noqa: E402
    METRIC_NAMES,
    parse_train_line,
)
from rq_pipeline.viz import STUDIO_ADDRESS  # noqa: E402

SEEN_LINES_CAP = 20000

# rsl-rl's console block (the RL trainers' format, vs lerobot's k:v
# line): an ANSI-bold "Learning iteration N/M" header followed by
# aligned "Name: value" rows, including per-term reward breakdowns.
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RSL_ITER_RE = re.compile(r"Learning iteration +(\d+)/(\d+)")
RSL_ROW_RE = re.compile(r"^\s*([A-Za-z_/ ]+[A-Za-z_/]): +(-?[0-9.]+)")
# "Time elapsed: 0:40:04" / "ETA: 0:52:06" - H:MM:SS, not a scalar;
# checked before RSL_ROW_RE, which would read them as the bare hour.
RSL_CLOCK_RE = re.compile(r"^\s*(Time elapsed|ETA): +([0-9:]+)")
RSL_SERIES = {
    "Mean reward": "rl/reward",
    "Mean episode length": "rl/episode_length",
    "Mean value loss": "rl/value_loss",
    "Mean surrogate loss": "rl/surrogate_loss",
    "Mean entropy loss": "rl/entropy_loss",
    "Mean action std": "rl/action_std",
    "Total steps": "rl/total_steps",
    "Steps per second": "rl/steps_per_second",
    "Collection time": "rl/collection_time",
    "Learning time": "rl/learning_time",
    "Iteration time": "rl/iteration_time",
}
RSL_FAMILIES = {
    "Episode_Reward/": "rl/reward_terms/",
    "Episode_Termination/": "rl/terminations/",
    "Episode_Metrics/": "rl/metrics/",
    "Metrics/": "rl/metrics/",
}


@dataclass
class RslStatus:
    """The latest rsl-rl block's headline, for the status card."""

    iteration: int
    total: int
    reward: float | None = None
    elapsed: str | None = None
    eta: str | None = None


# Slugs routed to rl/more/ - the layout only grows its "more" pane
# once this is non-empty, so the catch-all never sits as an empty
# plot (the operator's rule: no pane without data, 2026-09-01).
MORE_SEEN: set[str] = set()


def route_rsl_row(rr: Any, name: str, key: str, value: float) -> None:
    """Every numeric row lands somewhere: known singles by their given
    names, the families under theirs, anything unrecognised under
    rl/more/ - the whitelist was silently dropping terminations and
    the timing rows (2026-09-01)."""
    if key in RSL_SERIES:
        rr.log(f"{name}/{RSL_SERIES[key]}", rr.Scalars(value))
        return
    for prefix, family in RSL_FAMILIES.items():
        if key.startswith(prefix):
            path = f"{name}/{family}{key.removeprefix(prefix)}"
            rr.log(path, rr.Scalars(value))
            return
    slug = key.lower().replace(" ", "_").replace("/", "_")
    MORE_SEEN.add(slug)
    rr.log(f"{name}/rl/more/{slug}", rr.Scalars(value))


def parse_rsl(rr: Any, name: str, raw: str, seen_iters: set[int]) -> RslStatus | None:
    """rsl-rl blocks into Rerun: each unseen iteration's rows become
    series on the `iteration` timeline. Returns the latest block's
    status for the card."""
    latest: RslStatus | None = None
    emit = False
    for line in ANSI_RE.sub("", raw).splitlines():
        header = RSL_ITER_RE.search(line)
        if header:
            iteration = int(header.group(1))
            emit = iteration not in seen_iters
            if emit:
                seen_iters.add(iteration)
                rr.set_time("wall", timestamp=time.time())
                rr.set_time("iteration", sequence=iteration)
            latest = RslStatus(iteration, int(header.group(2)))
            continue
        if latest is None:
            continue
        clock = RSL_CLOCK_RE.match(line)
        if clock:
            if clock.group(1) == "Time elapsed":
                latest.elapsed = clock.group(2)
            else:
                latest.eta = clock.group(2)
            continue
        row = RSL_ROW_RE.match(line)
        if not row:
            continue
        key, value = row.group(1).strip(), float(row.group(2))
        if emit:
            route_rsl_row(rr, name, key, value)
        if key == "Mean reward":
            latest.reward = value
    return latest


# The trainer's series by scale, one pane each: on a shared axis the
# step counter (thousands) and the sample rate (hundreds) flatten the
# losses (units) into the floor - seen on the campaign feed 2026-09-02.
TRAINER_PANES = (
    ("loss", ("loss", "l1_loss", "kld_loss")),
    ("optimizer", ("lr", "grad_norm")),
    ("throughput", ("samples_per_s", "dataloading_s", "mem_gb")),
    # The step on the wall clock is "are we moving" at a glance (the
    # operator's ask); alone, so it flattens nothing else.
    ("progress", ("step",)),
)


def trainer_views(name: str) -> list[Any]:
    import rerun.blueprint as rrb  # noqa: PLC0415

    return [
        rrb.TimeSeriesView(
            origin=f"{name}/train",
            name=pane,
            contents=[f"$origin/{series}" for series in members],
        )
        for pane, members in TRAINER_PANES
    ]


# The engine layout grows with the campaign: press panes first, the
# trainer's once it logs, the verdict's once a row lands - a pane never
# sits empty waiting for its stage (the operator's rule).
ENGINE_PRESS, ENGINE_TRAINING, ENGINE_VERDICT = 0, 1, 2


def send_layout(
    rr: Any, name: str, kind: str, *, eval_pane: bool = False, level: int = 0
) -> None:
    """The feed names its own panes - reward front and centre, one
    view per series family - instead of the viewer's auto-layout,
    which panes whichever entities it notices first and buried
    rl/reward entirely (2026-09-01). Sent only once the first poll
    has said which kind of run this is: rsl-rl runs get the rl/*
    panes, lerobot-style runs get train/eval - never both, so no
    pane sits empty for the run's whole life."""
    try:
        import rerun.blueprint as rrb  # noqa: PLC0415 - viz extra

        if kind == "engine":
            # The data engine's log (press, campaign stages, the student's
            # verdict): counters and text, the trainer's series once that
            # stage runs - never the RL panes, which would sit empty.
            views = [
                rrb.TimeSeriesView(origin=f"{name}/press", name="press"),
                rrb.TextLogView(origin=f"{name}/stage", name="stages"),
            ]
            if level >= ENGINE_TRAINING:
                views += trainer_views(name)
            if level >= ENGINE_VERDICT:
                views.append(rrb.TextLogView(origin=f"{name}/verdict", name="verdict"))
        elif kind == "rl":
            views = [
                rrb.TimeSeriesView(origin=f"{name}/rl/reward", name="reward"),
                rrb.TimeSeriesView(
                    origin=f"{name}/rl",
                    name="losses",
                    contents=[
                        "$origin/value_loss",
                        "$origin/surrogate_loss",
                        "$origin/entropy_loss",
                        "$origin/action_std",
                    ],
                ),
                rrb.TimeSeriesView(
                    origin=f"{name}/rl/reward_terms", name="reward terms"
                ),
                rrb.TimeSeriesView(origin=f"{name}/rl/metrics", name="metrics"),
                rrb.TimeSeriesView(
                    origin=f"{name}/rl/terminations", name="terminations"
                ),
                rrb.TimeSeriesView(
                    origin=f"{name}/rl",
                    name="pace",
                    contents=[
                        "$origin/episode_length",
                        "$origin/steps_per_second",
                    ],
                ),
                rrb.TimeSeriesView(
                    origin=f"{name}/rl",
                    name="times",
                    contents=[
                        "$origin/collection_time",
                        "$origin/learning_time",
                        "$origin/iteration_time",
                    ],
                ),
            ]
            if MORE_SEEN:
                views.append(rrb.TimeSeriesView(origin=f"{name}/rl/more", name="more"))
        else:
            views = [
                *trainer_views(name),
                *(
                    [rrb.TimeSeriesView(origin=f"{name}/eval", name="eval")]
                    if eval_pane
                    else []
                ),
            ]
        views += [
            rrb.TimeSeriesView(origin=f"{name}/gpu", name="gpu"),
            rrb.TextDocumentView(origin=f"{name}/status", name="status"),
        ]
        rr.send_blueprint(rrb.Blueprint(rrb.Grid(*views), collapse_panels=False))
    except Exception as error:
        print(f"layout not sent: {error}")


def choose_layout(  # noqa: PLR0913 - the signals and the two pane flags, named
    rr: Any,
    name: str,
    sent: tuple[str, int] | None,
    signals: tuple[bool, bool, bool],
    *,
    eval_seen: bool = False,
    engine_level: int = ENGINE_PRESS,
) -> tuple[str, int] | None:
    """Send (or resend) the layout whenever what SHOULD be on screen
    changes: the first data reveals the run's kind, and a first
    unrecognised row makes the "more" pane earn its place."""
    rl, lerobot, engine = signals
    if engine:
        want = ("engine", engine_level)  # the flag: how far the campaign got
    elif rl:
        want = ("rl", bool(MORE_SEEN))
    elif lerobot:
        want = ("lerobot", eval_seen)  # the flag: the eval pane earned its place
    else:
        return sent
    if want != sent:
        send_layout(
            rr,
            name,
            want[0],
            eval_pane=want[0] == "lerobot" and bool(want[1]),
            level=int(want[1]) if want[0] == "engine" else 0,
        )
    return want


# The press's and the certificate's own console lines (walk_press,
# press.press, walk_verdict, tools/campaign-distill.sh) - a rented
# card's data engine is watched from here, not from a pod's screen.
PRESS_BATCH_RE = re.compile(r"^batch seed (\d+): (\d+)/(\d+) pass")
PRESS_ATTEMPT_RE = re.compile(r"^attempt (\d+): (KEEP|discard)")
# `kept N/M episodes -> <dir>`: the dir names the ARM of a multi-arm
# study (tools/study.py prints one line per arm; the card once showed
# the last arm's 128/128 for a 248-episode study, 2026-09-04).
PRESS_KEPT_RE = re.compile(r"^kept (\d+)/(\d+) episodes(?: -> (\S+))?")
VERDICT_RE = re.compile(
    r"^\[verdict\] survived (\d+)/(\d+), tracked (\d+)/(\d+) -> success (\d+)/(\d+), "
    r"CP95 \[([0-9.]+), ([0-9.]+)\]"
)
STAGE_RE = re.compile(r"^== (\d\d):(\d\d):(\d\d) (.+)$")


def stage_instant(hh: int, mm: int, ss: int, now: float) -> float:
    """A stage line's own clock (`== HH:MM:SS`, the campaign script's
    `date -u`) as a wall timestamp: today's date at that UTC time, or
    yesterday's when that would be in the future. Stage rows used to be
    stamped at the poll that first READ them, so a feed restarted at
    14:35 re-listed the 12:59 and 14:28 stages at 14:35 (2026-09-04)."""
    import datetime as dt  # noqa: PLC0415

    day = dt.datetime.fromtimestamp(now, tz=dt.UTC).replace(
        hour=hh, minute=mm, second=ss, microsecond=0
    )
    if day.timestamp() > now + 60:
        day -= dt.timedelta(days=1)
    return day.timestamp()


TRAIN_BUDGET_RE = re.compile(r"train (\d+) steps")  # the campaign header
# The trainer's own progress bar: exact step, its budget, elapsed, ETA
# and rate - "7399/60000 [11:20<1:06:49, 13.12step/s]". The INFO lines
# print the step rounded to a kilo ("step:7K"), which froze the card's
# rate at "?" for 80 s at a time (measured 2026-09-03).
TRAIN_BAR_RE = re.compile(r"(\d+)/(\d+) \[([\d:]+)<([\d:?]+), ([\d.]+)step/s\]")


@dataclass(frozen=True)
class TrainBar:
    step: int
    total: int
    elapsed: str
    eta: str
    rate: float

    @property
    def line(self) -> str:
        return (
            f"step **{self.step} / {self.total}** ({self.rate:.1f} step/s, "
            f"ETA {self.eta}, elapsed {self.elapsed})"
        )


def train_bar(raw: str) -> TrainBar | None:
    """The newest progress bar in the poll, or None."""
    last = None
    for match in TRAIN_BAR_RE.finditer(raw):
        last = match
    if last is None:
        return None
    step, total, elapsed, eta, rate = last.groups()
    return TrainBar(int(step), int(total), elapsed, eta, float(rate))


DIGEST_PATTERN = r"^== |^batch seed |^attempt [0-9]+: |^kept |^\[verdict\]"
DIGEST_LINES = 400


@dataclass
class EngineStatus:
    """What the data engine's log says right now, for the status card."""

    stage: str | None = None
    kept: int = 0
    attempts: int = 0
    last_batch: str | None = None
    # Per-arm press counts, keyed by the batch directory's name; the
    # totals above are their sums once any arm is known.
    arms: dict[str, tuple[int, int]] = field(default_factory=dict)
    verdict: str | None = None
    done: bool = False
    train_steps: int | None = None  # the campaign header's budget


def parse_engine(rr: Any, name: str, raw: str, seen: set[str]) -> EngineStatus | None:
    """Press counters on the `attempt` timeline, every campaign stage
    and verdict as text, the newest facts for the card. `seen` holds
    the lines already logged (the log is re-read whole each poll)."""
    status = EngineStatus()
    found = False
    kept_attempts: set[int] = set()
    # Text rows ride the attempt axis too: logged with no time set they
    # miss the pane (the operator's screenshot, 2026-09-03).
    rr.set_time("attempt", sequence=0)
    for raw_line in ANSI_RE.sub("", raw).splitlines():
        line = raw_line.strip()
        if attempt := PRESS_ATTEMPT_RE.match(line):
            found = True
            number = int(attempt.group(1))
            status.attempts = max(status.attempts, number)
            if attempt.group(2) == "KEEP" and number not in kept_attempts:
                # Distinct attempts only: the poll carries the digest AND
                # the tail, so a line can appear twice (the card once read
                # "84 kept of 48 attempts", 2026-09-03).
                kept_attempts.add(number)
                status.kept = len(kept_attempts)
            rr.set_time("attempt", sequence=int(attempt.group(1)))
            if line not in seen:
                seen.add(line)
                rr.log(f"{name}/press/kept", rr.Scalars(float(status.kept)))
        elif batch := PRESS_BATCH_RE.match(line):
            found = True
            status.last_batch = (
                f"{batch.group(2)}/{batch.group(3)} of batch {batch.group(1)} pass"
            )
            if line not in seen:
                seen.add(line)
                rr.log(f"{name}/press/log", rr.TextLog(line))
        elif kept := PRESS_KEPT_RE.match(line):
            found = True
            arm = pathlib.PurePosixPath(kept.group(3)).name if kept.group(3) else ""
            status.arms[arm] = (int(kept.group(1)), int(kept.group(2)))
            status.kept = sum(k for k, _ in status.arms.values())
            status.attempts = sum(a for _, a in status.arms.values())
        elif verdict := VERDICT_RE.match(line):
            found = True
            status.verdict = (
                f"survived {verdict.group(1)}/{verdict.group(2)}, tracked "
                f"{verdict.group(3)}/{verdict.group(4)} -> **{verdict.group(5)}/"
                f"{verdict.group(6)}** CP95 [{verdict.group(7)}, {verdict.group(8)}]"
            )
            if line not in seen:
                seen.add(line)
                rr.log(f"{name}/verdict/log", rr.TextLog(line))
        elif stage := STAGE_RE.match(line):
            found = True
            status.stage = stage.group(4)
            budget = TRAIN_BUDGET_RE.search(line)
            if budget:
                status.train_steps = int(budget.group(1))
            status.done = status.done or stage.group(4) == "CAMPAIGN DONE"
            if line not in seen:
                seen.add(line)
                # The row lands at the stage's OWN clock, then the poll's
                # clock is restored for everything logged after it.
                now = time.time()
                hh, mm, ss = (int(stage.group(i)) for i in (1, 2, 3))
                rr.set_time("wall", timestamp=stage_instant(hh, mm, ss, now))
                rr.log(f"{name}/stage", rr.TextLog(line))
                rr.set_time("wall", timestamp=now)
    return status if found else None


def train_progress(
    latest: Any, last_step: int, last_seen: tuple[float, int] | None, total: int
) -> tuple[str, tuple[float, int] | None]:
    """'step k / N (rate, ETA), losses' for a card, and the updated
    (wall, step) pair the ETA is measured from."""
    if last_step < 0:
        return "", last_seen
    now = time.time()
    rate, eta = "?", "?"
    if last_seen is not None and last_step > last_seen[1] and now > last_seen[0]:
        per_second = (last_step - last_seen[1]) / (now - last_seen[0])
        rate = f"{per_second:.1f} step/s"
        eta = f"{(total - last_step) / per_second / 60:.0f} min"
    losses = ""
    if latest is not None:
        losses = ", " + " ".join(
            f"{METRIC_NAMES.get(k, k)} {v:.3f}"
            for k, v in latest.metrics.items()
            if "loss" in k
        )
    line = f"step **{last_step} / {total}** ({rate}, ETA {eta}){losses}"
    if last_seen is None or last_step > last_seen[1]:
        return line, (now, last_step)
    return line, last_seen


def engine_train_line(
    rr: Any,
    name: str,
    raw: str,
    train: tuple[Any, int, tuple[float, int] | None],
    total: int,
) -> tuple[str, tuple[float, int] | None]:
    """The card's training line: the trainer's own progress bar when
    the poll has one (exact step, its ETA and rate; the step logged on
    the wall clock for the progress pane), else the INFO-line estimate."""
    latest, last_step, last_seen = train
    bar = train_bar(raw)
    if bar is None:
        return train_progress(latest, last_step, last_seen, total)
    rr.reset_time()
    rr.set_time("wall", timestamp=time.time())
    rr.log(f"{name}/train/step", rr.Scalars(float(bar.step)))
    losses = ""
    if latest is not None:
        losses = ", " + " ".join(
            f"{METRIC_NAMES.get(k, k)} {v:.3f}"
            for k, v in latest.metrics.items()
            if "loss" in k
        )
    return bar.line + losses, last_seen


def engine_stage(engine: Any, latest: Any, last_step: int) -> int:
    """How far the campaign got, for the layout: press, training, verdict."""
    if engine is None:
        return ENGINE_PRESS
    if engine.verdict or engine.done:
        return ENGINE_VERDICT
    if latest is not None or last_step >= 0:
        return ENGINE_TRAINING
    return ENGINE_PRESS


def write_engine_card(
    rr: Any, name: str, status: EngineStatus, gpu: str, train_line: str = ""
) -> None:
    """The data engine's 'right now' card."""
    card = "## data engine\n\n"
    if status.stage:
        card += f"- stage **{status.stage}**\n"
    card += f"- pressed **{status.kept}** kept of {status.attempts} attempts"
    if len(status.arms) > 1:
        per_arm = ", ".join(f"{arm} {k}/{a}" for arm, (k, a) in status.arms.items())
        card += f" ({per_arm})"
    card += "\n"
    if status.last_batch:
        card += f"- last batch {status.last_batch}\n"
    if train_line:
        card += f"- {train_line}\n"
    if status.verdict:
        card += f"- student certificate: {status.verdict}\n"
    if status.done:
        card += "- **CAMPAIGN DONE - stop the pod**\n"
    card += f"- GPU {gpu}\n"
    rr.log(
        f"{name}/status",
        rr.TextDocument(card, media_type=rr.MediaType.MARKDOWN),
        static=True,
    )


def write_rl_card(rr: Any, name: str, rsl: RslStatus, gpu: str) -> None:
    """The rsl-rl run's 'right now' card."""
    card = f"## RL training\n\n- iteration **{rsl.iteration} / {rsl.total}**\n"
    if rsl.reward is not None:
        card += f"- mean reward **{rsl.reward:.2f}**\n"
    if rsl.elapsed is not None:
        card += f"- elapsed {rsl.elapsed}" + (
            f", ETA **{rsl.eta}**\n" if rsl.eta else "\n"
        )
    card += f"- GPU {gpu}\n"
    rr.log(
        f"{name}/status",
        rr.TextDocument(card, media_type=rr.MediaType.MARKDOWN),
        static=True,
    )


def log_gpu(rr: Any, name: str, gpu: str) -> None:
    """The sentinel GPU line into utilization and memory series."""
    if "%" not in gpu:
        return
    rr.set_time("wall", timestamp=time.time())
    util, memory = (part.strip() for part in gpu.split(","))
    rr.log(f"{name}/gpu/utilization", rr.Scalars(float(util.split()[0])))
    rr.log(f"{name}/gpu/memory_gb", rr.Scalars(float(memory.split()[0]) / 1024.0))


# Only a default: every run states its own budget with --steps.
DEFAULT_TOTAL_STEPS = 10000

STAGE = re.compile(
    r"== training [a-z]+|== evaluating [a-z]+"
    r"|[a-z]+: checkpoint under \S+|verdict -> \S+"
)


def parse_poll(
    rr: Any,
    name: str,
    raw: str,
    state: tuple[set[str], set[tuple[str, int]], dict[str, None]],
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
            rr.log(f"{name}/train/{METRIC_NAMES.get(key, key)}", rr.Scalars(value))
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


def recording_id_for(door: str, log: str) -> str:
    """The recording a (machine, log) pair always streams into."""
    import uuid  # noqa: PLC0415

    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rq-feed://{door}/{log}"))


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--address", default=STUDIO_ADDRESS)
    parser.add_argument(
        "--records",
        default="",
        help="remote glob of per-trial records.jsonl files to stream as cloud/eval/*",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import rerun as rr  # noqa: PLC0415 - viz extra

    user_host, _, port = args.door.rpartition(":")
    username, _, host = user_host.partition("@")
    # The shared ssh options (transfer.Ssh): keepalive included — a
    # hand-rolled argv without ServerAliveInterval is why an overnight
    # feed could silently drop (review 2026-09-01).
    door = SshEndpoint(host=host, port=int(port), username=username, command=args.door)
    poll_argv = ssh_argv(door, args.key)
    # One recording per remote log, stable across feed restarts: a
    # restarted feed CONTINUES the run's recording instead of opening
    # another, so the Studio's sources list never fills with stale
    # copies of the same campaign (the operator's screenshot,
    # 2026-09-03, showed the pre-fix recording beside the live one).
    rr.init(
        f"rq-{args.name}-feed",
        recording_id=recording_id_for(args.door, args.log),
        spawn=False,
    )
    rr.connect_grpc(args.address)
    print(f"feeding {args.door}:{args.log} -> {args.address} as {args.name}/")
    layout_sent: tuple[str, int] | None = None

    seen_stages: set[str] = set()
    seen_records: set[tuple[str, int]] = set()
    seen_iters: set[int] = set()
    seen_engine: set[str] = set()
    # Insertion-ordered so eviction can drop the OLDEST half; a plain
    # set has no age and the cap could only clear (review 2026-09-01).
    seen_lines: dict[str, None] = {}
    last_seen: tuple[float, int] | None = None  # (wall clock, step) for ETA
    while True:
        try:
            raw = subprocess.run(
                [
                    *poll_argv,
                    # The structural lines of the WHOLE log first: stage
                    # markers, press batches, verdict rows, the campaign
                    # header. A trainer's progress bars push them out of
                    # the 256 KB tail within seconds, and a feed started
                    # mid-run then never learns the run's kind (seen
                    # 2026-09-02: the lerobot layout on a campaign).
                    f"tr '\\r' '\\n' < {args.log} 2>/dev/null | "
                    f"grep -aE '{DIGEST_PATTERN}' | tail -n {DIGEST_LINES}; echo; "
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
        engine = parse_engine(rr, args.name, raw, seen_engine)
        layout_sent = choose_layout(
            rr,
            args.name,
            layout_sent,
            (
                rsl is not None,
                latest is not None or last_step >= 0 or bool(eval_progress),
                engine is not None,
            ),
            eval_seen=bool(eval_progress) or bool(seen_records),
            engine_level=engine_stage(engine, latest, last_step),
        )
        if engine is not None:
            train_line, last_seen = engine_train_line(
                rr,
                args.name,
                raw,
                (latest, last_step, last_seen),
                engine.train_steps or args.steps,
            )
            write_engine_card(rr, args.name, engine, gpu, train_line)
        elif rsl is not None:
            write_rl_card(rr, args.name, rsl, gpu)
        log_gpu(rr, args.name, gpu)
        if rsl is None and engine is None:
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
