"""The data press: the success-gated generation loop, task-agnostic.

The generalization of the kitting demo generator (docs/e2e-research/60
§2: "the plan generalizes, it does not invent"): draw, run, and keep an
episode ONLY when the task's own referee scores it — with the loop's
contract owned here, once, and every task contributing exactly one
callable. `kitting_demos.generate_demos` is now an adapter over this
loop; the loop itself has never met a kitting-specific name.

Two rules the field's factories break, enforced by shape:
- "N episodes" always means N SUCCESSFUL episodes (Arena/Mimic got this
  right; it is the one part of their contract worth copying verbatim —
  59 §5), with an attempts ceiling so an impossible draw regime fails
  loudly instead of spinning.
- Every kept episode records the dynamics that produced it. The
  go-forward sidecar is `EpisodeManifest`: the task's content stamp,
  the engine's instrument stamp, the expert's stamp, and the NAMED
  dynamics values actually applied — the provenance no other data
  factory in the arc can write (60 §5). The kitting adapter still
  writes its legacy manifest so every committed batch and reader stays
  valid; new tasks start on `EpisodeManifest` from day one.

The sim extra is imported by attempt callables, not here — the loop is
pure accounting and imports without it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from trainnr.bundles.json_record import JsonRecord
from trainnr.collect.kitting_export import DemoLayout, write_episode

if TYPE_CHECKING:  # library types as annotations only — the loop stays
    # importable without the sim extra (numpy arrives with it).
    from numpy.random import Generator
    from numpy.typing import NDArray

# One retry, as the kitting expert records it: (arm, physics_step, part_z).
Retry = tuple[str, int, float]

# The sampler can in principle draw only unreachable dynamics: give up
# after this many draws per wanted episode (overridable per press run).
ATTEMPTS_PER_EPISODE = 20

Say = Callable[[str], None]


@dataclass(frozen=True)
class PressResult:
    """One attempt's outcome, however the task produced it.

    `dynamics` are the NAMED dynamics values this attempt actually ran
    under (e.g. {"damping_scale": 1.12}) — the per-episode vector the
    sidecar exists to carry. `draws` are the start-condition draws
    (spawn positions, goal picks). `note` names a discard's reason."""

    succeeded: bool
    dynamics: dict[str, float] = field(default_factory=dict)
    draws: dict[str, Any] = field(default_factory=dict)
    retries: list[Retry] = field(default_factory=list)
    # Where `dynamics` came from — A3's basis string ("identified-
    # interval", or a caller-declared span that says so). Recorded on
    # the sidecar; the datasheet groups episodes by it.
    dynamics_basis: str = ""
    states: NDArray | None = None
    sensors: NDArray | None = None
    actions: NDArray | None = None
    frames: list[tuple[int, NDArray]] | None = None
    note: str = ""
    # Generation-time visual draws (lighting, camera pose — keys the
    # engine's appliers know; a note) and where their ranges came
    # from. Empty when the adapter drew none.
    visuals: dict[str, Any] = field(default_factory=dict)
    visual_basis: str = ""
    # Frames per camera key, for adapters that render the task's whole
    # declared rig; `frames` stays the single-camera path (kitting's
    # committed layout).
    camera_frames: dict[str, list[tuple[int, NDArray]]] | None = None


class AttemptFn(Protocol):
    """One attempt: draw everything, run the expert, judge with the
    referee."""

    def __call__(self, rng: Generator, *, frame_every: int) -> PressResult: ...


class EpisodeSidecar(Protocol):
    """Anything that can write itself beside an episode — this task's
    legacy `Manifest` or the go-forward `EpisodeManifest`."""

    def write_to(self, episode_dir: Path) -> Path: ...


# Builds the sidecar record for a KEPT episode.
ManifestFn = Callable[[PressResult, int], EpisodeSidecar]


@dataclass(frozen=True)
class EpisodeManifest(JsonRecord):
    """The go-forward per-episode sidecar (docs/e2e-research/60 §3):
    which task (content stamp), which engine (instrument stamp), which
    expert, and which dynamics — the questions no dataset in the arc
    can answer about itself."""

    seed: int
    attempt: int
    task: str
    expert: str
    instrument: str
    dynamics: dict[str, float]
    draws: dict[str, Any]
    retries: list[Retry]
    control_hz: int
    frame_every_control_ticks: int
    dynamics_basis: str = ""
    action_semantics: str = "commanded actuator positions, ctrl order"
    verdict: str = "success (task referee)"
    visuals: dict[str, Any] = field(default_factory=dict)
    visual_basis: str = ""

    def write_to(self, episode_dir: Path) -> Path:
        return self.write(Path(episode_dir) / DemoLayout.MANIFEST_FILE)

    @classmethod
    def read_from(cls, episode_dir: Path) -> EpisodeManifest:
        return cls.read(Path(episode_dir) / DemoLayout.MANIFEST_FILE)


@dataclass(frozen=True)
class DemoBatch:
    """What a press run produced: the accounting a chain log and a docs
    entry quote."""

    out: Path
    wanted: int
    kept: int
    attempts: int
    first_episode: int
    expert: str
    task: str

    @property
    def complete(self) -> bool:
        return self.kept >= self.wanted

    @property
    def last_episode(self) -> int:
        return self.first_episode + self.kept - 1


class PressFeed(Protocol):
    """Where a press run streams (docs/66 §0: everything that happens
    streams to the Studio). The loop reports every attempt and the
    final accounting; what listens — the Studio via
    `collect/press_feed.py`, a test's list — is the caller's choice."""

    def attempt(
        self, attempts: int, kept: int, wanted: int, result: PressResult
    ) -> None: ...

    def done(self, batch: DemoBatch) -> None: ...


def press(  # noqa: PLR0913 - every knob of the loop, named
    out: Path,
    *,
    task: str,
    expert: str,
    instrument: str,
    control_hz: int,
    attempt_fn: AttemptFn,
    episodes: int,
    seed: int,
    frame_every: int = 1,
    first_episode: int = 0,
    max_attempts: int | None = None,
    attempts_per_episode: int = ATTEMPTS_PER_EPISODE,
    manifest_fn: ManifestFn | None = None,
    feed: PressFeed | None = None,
    say: Say = print,
) -> DemoBatch:
    """Keep `episodes` referee-passing demonstrations under `out`,
    numbered from `first_episode`; stop early after `max_attempts`
    draws. `task`/`expert`/`instrument` are stamps, recorded on every
    kept episode's sidecar."""
    import numpy as np  # noqa: PLC0415 - sim extra (attempt callables need it too)

    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    say(f"expert {expert}, task {task}, instrument {instrument}")

    if manifest_fn is None:

        def manifest_fn(result: PressResult, attempt: int) -> EpisodeManifest:
            return EpisodeManifest(
                seed=seed,
                attempt=attempt,
                task=task,
                expert=expert,
                instrument=instrument,
                dynamics=result.dynamics,
                draws=result.draws,
                retries=result.retries,
                control_hz=control_hz,
                frame_every_control_ticks=frame_every,
                dynamics_basis=result.dynamics_basis,
                visuals=result.visuals,
                visual_basis=result.visual_basis,
            )

    limit = attempts_per_episode * episodes if max_attempts is None else max_attempts
    kept = attempts = 0
    while kept < episodes and attempts < limit:
        attempts += 1
        result = attempt_fn(rng, frame_every=frame_every)
        dynamics = ", ".join(f"{k} x{v:.2f}" for k, v in result.dynamics.items())
        verdict = "KEEP" if result.succeeded else "discard"
        if result.note:
            # Notes ride on kept attempts too (a retry story, a device
            # disagreement); dropping them there hid information the
            # adapter deliberately wrote (review 2026-09-01).
            verdict += f" ({result.note})"
        say(
            f"attempt {attempts}: {verdict} ({dynamics}, retries {len(result.retries)})"
        )
        if not result.succeeded:
            if feed is not None:
                feed.attempt(attempts, kept, episodes, result)
            continue
        write_episode(
            out,
            first_episode + kept,
            states=result.states,
            sensors=result.sensors,
            actions=result.actions,
            frames=result.frames or [],
            camera_frames=result.camera_frames,
            manifest=manifest_fn(result, attempts),
        )
        kept += 1
        if feed is not None:
            feed.attempt(attempts, kept, episodes, result)

    if kept:
        # The batch describes itself before anyone asks (docs/
        # e2e-research/60 §3: nobody in the arc ships a datasheet).
        from trainnr.collect.datasheet import write_datasheet  # noqa: PLC0415

        say(f"datasheet -> {write_datasheet(out)}")
    batch = DemoBatch(out, episodes, kept, attempts, first_episode, expert, task)
    if feed is not None:
        feed.done(batch)
    say(
        f"kept {kept}/{attempts} episodes -> {out} "
        f"(episodes {first_episode}..{batch.last_episode})"
        if kept
        else f"gave up after {attempts} attempts: kept 0/{episodes} -> {out}"
    )
    return batch
