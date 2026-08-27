"""The docs/e2e-research/52 probe: is MJX-Warp bit-repeatable, and at what cost?

    cd pipeline && uv run --extra sim --extra mjx python ../tools/determinism-probe.py

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

MODE_ENV = "ROBOTIQ_DET_MODE"
STEPS = 1000
WORLDS = 4


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
    from rq_pipeline.physics.mjx_backend import MJXWarpBackend  # noqa: PLC0415
    from rq_pipeline.physics.mujoco_backend import (  # noqa: PLC0415
        MuJoCoBackend,
        keyframe_state,
    )
    from rq_pipeline.tasks.aloha2 import NEUTRAL_CTRL, build_kitting  # noqa: PLC0415

    cpu = MuJoCoBackend()
    cpu.load_spec(build_kitting().spec)
    warp_backend = MJXWarpBackend(naconmax=128, njmax=512)
    warp_backend.load_model(cpu.model)
    initial = np.tile(keyframe_state(cpu.model, "neutral_pose"), (WORLDS, 1))
    controls = np.tile(np.asarray(NEUTRAL_CTRL, dtype=float), (WORLDS, STEPS, 1))

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
        marker = "Deterministic mode does not support"
        reason = (
            text[text.index(marker) :].split("\n", maxsplit=1)[0]
            if marker in text
            else text[-300:]
        )
        print(json.dumps({"mode": mode, "refused": reason}))
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


def parent() -> None:
    print(f"kitting scene, {WORLDS} worlds x {STEPS} physics steps, twice per mode")
    for mode in ("NOT_GUARANTEED", "RUN_TO_RUN"):
        env = dict(os.environ, **{MODE_ENV: mode})
        result = subprocess.run(
            [sys.executable, __file__],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        verdicts = [line for line in result.stdout.splitlines() if line.startswith("{")]
        if result.returncode != 0 or not verdicts:
            print(f"{mode}: FAILED\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
            sys.exit(1)
        r = json.loads(verdicts[-1])
        if "unavailable" in r:
            print(
                f"{r['mode']:>16}  UNAVAILABLE on warp {r['unavailable']} — "
                "needs warp >= 1.16 (the mujoco 3.12 lockstep)"
            )
            continue
        print(
            f"{r['mode']:>16}  bit_equal={r['bit_equal']}  "
            f"max_gap={r['max_gap']:.2e}  warm_wall={r['wall_s_warm']:.1f}s  "
            f"[{r['instrument']}]"
        )


if __name__ == "__main__":
    if MODE_ENV in os.environ:
        child(os.environ[MODE_ENV])
    else:
        parent()
