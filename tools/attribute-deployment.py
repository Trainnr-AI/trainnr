#!/usr/bin/env python3
"""Attribution: which parameter would break this deployment first. A
passing plane gate re-run with one dynamics knob turned at a time up its
ladder (latency, friction, payload, gains, encoder noise, tilt, pushes);
the cliff per knob, the knobs ranked, the record beside the manifest,
the fall pictured, the ladders streamed. Spawned by the MCP door
`attribute_deployment`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from _lab import bootstrap, deployment_args, resolve_deployment, running

bootstrap()

from rq_pipeline.deploy.attribution import (  # noqa: E402
    STREAM,
    SURVIVED,
    attribute,
    knob,
    load_fit,
    log_knob,
    log_ranking,
    open_live,
    still_at_cliff,
    write_attribution,
)
from rq_pipeline.deploy.gate import DEFAULT_TOLERANCE  # noqa: E402
from rq_pipeline.deploy.manifest import Key  # noqa: E402
from rq_pipeline.deploy.viewport_source import scene_text  # noqa: E402
from rq_pipeline.project import index_project, write_index  # noqa: E402
from rq_pipeline.project.cited import cited_certificate  # noqa: E402
from rq_pipeline.viz import viewer_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    deployment_args(parser)
    parser.add_argument(
        "--trials", type=int, default=None, help="the passing gate's own when unset"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="the passing gate's own when unset"
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--workers", type=int, default=None, help="knobs climbed at once (0: here)"
    )
    parser.add_argument(
        "--no-still", action="store_true", help="skip the picture of the fall"
    )
    parser.add_argument(
        "--fit",
        type=Path,
        default=None,
        help="a joints fit record (robots/<robot>/fits/<recording>.json): its "
        "armature, damping and friction set AT the fitted values, one term "
        "at a time and all together",
    )
    args = parser.parse_args()
    try:
        project, deployment, manifest, assets = resolve_deployment(args)
        certificate = cited_certificate(project, manifest.raw.get(Key.CERTIFICATE))
        fit = load_fit(args.fit) if args.fit is not None else None
        live = _live(deployment)
        with running(
            project.root,
            name=args.name,
            viewport=scene_text(args.name),
            viewer=viewer_file(deployment, STREAM),
        ) as run:
            run.stage("the baseline: the passing gate's own trials, untouched")
            record = attribute(
                deployment,
                assets_dir=assets,
                certificate=certificate,
                runtime=args.runtime,
                trials=args.trials,
                seed=args.seed,
                tolerance=args.tolerance,
                fit=fit,
                workers=args.workers,
                on_knob=None if live is None else _on_knob(live),
                on_progress=_climbed(run),
            )
            run.stage(record["sensitivity"])
    except (FileNotFoundError, ValueError, ImportError) as exc:
        raise SystemExit(str(exc)) from exc
    base = record["baseline"]
    print(
        f"[attribution] baseline {base['successes']}/{base['trials']} ci95 "
        f"{base['ci95']}; cliff rule {record['protocol']['cliff_rule']}: the "
        f"gate's floor {record['certificate']['floor']}, the certificate's "
        f"lower bound {record['certificate']['lower']}",
        flush=True,
    )
    for entry in record["knobs"]:
        rungs = " · ".join(
            f"{r['level']:g}{entry['unit']}: {r['successes']}/{r['trials']} "
            f"[{r['ci95'][0]}, {r['ci95'][1]}]"
            for r in entry["rungs"]
        )
        cliff = entry["cliff"]
        word = (
            SURVIVED if cliff is None else f"cliff at {cliff['level']:g}{entry['unit']}"
        )
        print(f"[attribution] {entry['name']:>16}: {word} — {rungs}", flush=True)
    print(f"[attribution] {record['sensitivity']}", flush=True)
    fit = record.get("fit") or {}
    for r in fit.get("rungs") or []:
        word = "past the cliff" if r["past_cliff"] else "holds"
        print(
            f"[attribution] {r['name']:>16} ({fit['label']}, "
            f"{'+'.join(r['terms'])}): {r['successes']}/{r['trials']} "
            f"[{r['ci95'][0]}, {r['ci95'][1]}] {word}",
            flush=True,
        )
    if not args.no_still:
        still = still_at_cliff(deployment, record, assets_dir=assets)
        record["still"] = still
        write_attribution(deployment, record)
        print(
            "[attribution] still: "
            + (
                still["unrendered"]
                if "unrendered" in still
                else f"{still['file']} ({knob(still['knob']).label(still['level'])}, "
                f"trial {still['trial']}, tick {still['tick']})"
            ),
            flush=True,
        )
    if live is not None:  # the ladders are already in; the ranking closes it
        log_ranking(live, record, deployment.name)
        print("[attribution] streamed live to the Studio and saved", flush=True)
    if record.get("replay"):
        print(
            f"[attribution] replays: {record['replay']['poses']} "
            f"({record['replay']['segments']} rungs; the Studio's MuJoCo viewport "
            f"shows them as deploy:{deployment.name}:attribution:<knob>:<rung>)",
            flush=True,
        )
    write_index(project, index_project(project))
    sys.exit(0)


def _climbed(run: Any) -> Any:
    """Each knob's ladder, as it lands, told to the Running now panel."""

    def told(done: int, total: int, name: str, rungs: list[Any]) -> None:
        last = rungs[-1] if rungs else None
        tail = f": {last.successes}/{last.trials} at its last rung" if last else ""
        run.progress(
            done, total, "knobs", f"knob {done} of {total} climbed, {name}{tail}"
        )

    return told


def _live(deployment: Path) -> Any:
    """The attribution's stream, opened before the sweep so each knob's
    ladder reaches a listening Studio as it lands; None without the viz
    extra (the record stands alone)."""
    try:
        return open_live(deployment)
    except ImportError as missing:
        print(f"[attribution] no live stream: {missing}", flush=True)
        return None


def _on_knob(rr: Any) -> Any:
    def log(entry: dict[str, Any], certificate_lower: float) -> None:
        log_knob(rr, entry, certificate_lower)
        cliff = entry["cliff"]
        word = SURVIVED if cliff is None else f"cliff at {cliff['level']:g}"
        print(f"[attribution] live: {entry['name']} landed, {word}", flush=True)

    return log


if __name__ == "__main__":
    main()
