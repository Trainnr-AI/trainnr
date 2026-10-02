"""The docs/e2e-research/52 probe: is MJX-Warp bit-repeatable, and at what cost?

    # the deciding run: the WSL card, the train venv (mujoco 3.12 + mujoco_warp
    # 3.12 + warp 1.16, the DeterministicMode lockstep), wsl.env for CUDA:
    cd trainnr && ../tools/wsl-run.sh .venv-train/bin/python \\
        ../tools/determinism-probe.py [--budget-s 900]
    # the Mac / CPU plumbing smoke (warp 1.14: RUN_TO_RUN reports UNAVAILABLE):
    cd trainnr && uv run --extra sim --extra mjx python ../tools/determinism-probe.py

Runs the kitting scene twice per Warp determinism mode (the default
NOT_GUARANTEED, then RUN_TO_RUN) and reports bit-equality and wall
time. Each mode runs in its OWN subprocess because determinism is a
module-level COMPILE option — `wp.config.deterministic` must be set
before mujoco_warp's kernels are imported, and an imported module
keeps its setting for the life of the process.

On a CPU device both modes should already be bit-equal (Warp CPU is
sequential) — the Mac run is a plumbing smoke. Budget: a COLD kernel
compile for this scene is ~10 minutes on the WSL box (2026-08-27), and
each mode compiles its own build; run it under `timeout` and in the
background. The verdict that
matters comes from the WSL box's CUDA device, where NOT_GUARANTEED is
expected to differ across runs and RUN_TO_RUN to match; the wall-time
column prices the mode (their 4090 data says contention-heavy scenes
can even get faster).
"""

import json
import os
import subprocess
import sys
import time

from _lab import bootstrap

bootstrap()

MODE_ENV = "TRAINNR_DET_MODE"
MODES = ("NOT_GUARANTEED", "RUN_TO_RUN")
REFUSAL_MARKER = "Deterministic mode does not support"  # warp/_src/deterministic.py
DEFAULT_BUDGET_S = (
    900.0  # a cold mujoco_warp compile is ~10 min on the RTX (2026-08-27)
)
TAIL_CHARS = 300
LOG_TAIL_CHARS = 2000


class Probe:
    """The rollout every mode runs twice, and the contact sizing the
    kitting scene needs at this batch (measured: 128/512 for four
    worlds — naconmax is the TOTAL across worlds)."""

    STEPS = 1000
    WORLDS = 4
    NACONMAX = 128
    NJMAX = 512


def child(mode: str) -> None:
    # Deliberately late imports throughout: determinism is a COMPILE
    # option, so wp.config must be set before the engine's kernels are
    # imported — the module docstring is about exactly this.
    import warp as wp  # noqa: PLC0415

    if mode != "NOT_GUARANTEED":
        if not hasattr(wp, "DeterministicMode"):
            # Measured 2026-08-27: warp 1.14 (the mujoco 3.11 lockstep)
            # predates the feature; it ships with warp >= 1.16, which
            # arrives with the mujoco 3.12 pin — a re-identification
            # event, so the two decisions are COUPLED.
            print(json.dumps({"mode": mode, "unavailable": wp.__version__}))
            return
        wp.config.deterministic = getattr(wp.DeterministicMode, mode)
    import numpy as np  # noqa: PLC0415
    from trainnr.physics.mjx_backend import MJXWarpBackend  # noqa: PLC0415
    from trainnr.physics.mujoco_backend import (  # noqa: PLC0415
        MuJoCoBackend,
        keyframe_state,
    )
    from trainnr.tasks.aloha2 import NEUTRAL_CTRL, build_kitting  # noqa: PLC0415

    cpu = MuJoCoBackend()
    cpu.load_spec(build_kitting().spec)
    warp_backend = MJXWarpBackend(naconmax=Probe.NACONMAX, njmax=Probe.NJMAX)
    warp_backend.load_model(cpu.model)
    initial = np.tile(keyframe_state(cpu.model, "neutral_pose"), (Probe.WORLDS, 1))
    controls = np.tile(
        np.asarray(NEUTRAL_CTRL, dtype=float), (Probe.WORLDS, Probe.STEPS, 1)
    )

    def run() -> tuple:
        start = time.perf_counter()
        states = warp_backend.rollout(initial, controls)
        return states, time.perf_counter() - start

    try:
        first, _ = run()  # includes JIT
    except Exception as error:
        # Measured 2026-08-27 on the RTX 3090 Ti: RUN_TO_RUN cannot compile
        # mujoco_warp 3.12.0 — its _sensor_tactile kernel mixes max and add
        # reductions on sensordata_out, which deterministic codegen refuses.
        # That is a verdict about the engine build, reported as one.
        text = str(error)
        if REFUSAL_MARKER in text:  # the engine's verdict, reported as one
            reason = text[text.index(REFUSAL_MARKER) :].split("\n", maxsplit=1)[0]
            print(json.dumps({"mode": mode, "refused": reason}))
        else:  # anything else is a tool failure, and says so
            print(json.dumps({"mode": mode, "error": text[-TAIL_CHARS:]}))
        return
    second, t_second = run()
    print(
        json.dumps(
            {
                "mode": mode,
                "instrument": warp_backend.instrument,
                "bit_equal": bool(np.array_equal(first, second)),
                "max_gap": float(np.max(np.abs(first - second))),
                "wall_s_warm": t_second,
            }
        )
    )


def parent(budget_s: float) -> None:
    print(
        f"kitting scene, {Probe.WORLDS} worlds x {Probe.STEPS} physics steps, twice "
        f"per mode; each mode is its own process and may compile kernels cold "
        f"(~10 min on the RTX 3090 Ti) — budget {budget_s:.0f} s per mode"
    )
    failed = False
    for mode in MODES:
        env = dict(os.environ, **{MODE_ENV: mode})
        try:
            result = subprocess.run(
                [sys.executable, __file__],
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
                timeout=budget_s,
            )
        except subprocess.TimeoutExpired:
            print(
                f"{mode:>16}  over the {budget_s:.0f} s budget — a warm cache or a "
                "bigger machine"
            )
            sys.exit(2)
        verdicts = [line for line in result.stdout.splitlines() if line.startswith("{")]
        if result.returncode != 0 or not verdicts:
            print(
                f"{mode}: FAILED\n{result.stdout[-LOG_TAIL_CHARS:]}\n"
                f"{result.stderr[-LOG_TAIL_CHARS:]}"
            )
            sys.exit(1)
        r = json.loads(verdicts[-1])
        if "unavailable" in r:
            print(
                f"{r['mode']:>16}  UNAVAILABLE on warp {r['unavailable']} — "
                "needs warp >= 1.16 (the mujoco 3.12 lockstep)"
            )
            continue
        if "refused" in r:  # a verdict about the engine build: exit 0
            print(f"{r['mode']:>16}  REFUSED at compile: {r['refused']}")
            continue
        if "error" in r:  # a tool failure: every mode reported, then exit 1
            print(f"{r['mode']:>16}  ERROR: {r['error']}")
            failed = True
            continue
        print(
            f"{r['mode']:>16}  bit_equal={r['bit_equal']}  "
            f"max_gap={r['max_gap']:.2e}  warm_wall={r['wall_s_warm']:.1f}s  "
            f"[{r['instrument']}]"
        )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    if MODE_ENV in os.environ:
        child(os.environ[MODE_ENV])
    else:
        import argparse

        cli = argparse.ArgumentParser(description=__doc__.split("\n")[0])
        cli.add_argument("--budget-s", type=float, default=DEFAULT_BUDGET_S)
        parent(cli.parse_args().budget_s)
