"""The instrument's records, one shot, into the Studio's embedded viewer.

    cd pipeline && uv run --extra sim --extra viz \\
        python ../tools/studio-instrument-view.py

Logs three families of existing artifacts — nothing is produced here —
as native Rerun views with a curated blueprint, into whatever viewer
holds the standard port (the Studio itself, or standalone Rerun):

- every evaluation run's milestone funnel and success count
  (`describe_eval`, the same fold the certificate uses);
- every fit-carrying bundle's parameters as estimate and interval bounds
  per sweep, with the SPREAD verdicts as a document beside them;
- the STS3215's friction budget at M1 and M6, unloaded and under load,
  on a `velocity_mrad_s` timeline — the x-axis IS velocity (mrad/s),
  never a sample index dressed up as one.

All numbers come through `rq_pipeline.mcp_server`'s query functions —
the same surface the agents' MCP tools serve, so these views and an
agent's answers can never disagree.
"""

from _lab import bootstrap

bootstrap()

import rerun as rr  # noqa: E402
import rerun.blueprint as rrb  # noqa: E402
from rq_pipeline.mcp_server import (  # noqa: E402
    describe_bundle,
    describe_bundles,
    describe_eval,
    friction_curve,
    list_eval_records,
)

FRICTION_SERVO = "feetech_sts3215_7_4V"
FRICTION_TIERS = ("m1", "m6")


def log_evals() -> list[str]:
    origins = []
    for entry in list_eval_records():
        run = entry["run"]
        detail = describe_eval(run)
        for file, data in detail["files"].items():
            milestones = data["milestones"]
            for policy, counts in data["funnel"].items():
                origin = f"evals/{run}/{policy}"
                rr.log(f"{origin}/funnel", rr.BarChart(counts), static=True)
                lines = [
                    f"# {run} — {policy}",
                    f"**{data['successes']} / {data['records']}** trials succeeded"
                    f" ({file})",
                    "",
                    "Funnel bars, left to right:",
                ]
                lines += [
                    f"{i + 1}. {name} — {count} trial(s) reached it"
                    for i, (name, count) in enumerate(
                        zip(milestones, counts, strict=False)
                    )
                ]
                rr.log(
                    f"{origin}/reading",
                    rr.TextDocument("\n".join(lines), media_type=rr.MediaType.MARKDOWN),
                    static=True,
                )
                origins.append(origin)
    return origins


def log_fits() -> list[str]:
    param_origins: list[str] = []
    for bundle in describe_bundles():
        if not bundle["has_fits"]:
            continue
        name = bundle["name"]
        detail = describe_bundle(name)
        records = [
            (file, record)
            for file, record in detail["fits"].items()
            if file.startswith("sweep-")
        ]
        for sweep, (_, record) in enumerate(records):
            rr.set_time("sweep", sequence=sweep)
            for param in record["parameters"]:
                origin = f"fits/{name}/{param['name']}"
                estimate = float(param["estimate"])
                half = float(param["half_width"])
                rr.log(f"{origin}/estimate", rr.Scalars(estimate))
                rr.log(f"{origin}/interval_low", rr.Scalars(estimate - half))
                rr.log(f"{origin}/interval_high", rr.Scalars(estimate + half))
                if origin not in param_origins:
                    param_origins.append(origin)
        spread = detail["fits"].get("SPREAD.json")
        if spread:
            lines = [f"# {name} — cross-run SPREAD verdicts", ""]
            lines += [
                f"- **{param}**: {info['verdict']}"
                for param, info in spread.get("spread", {}).items()
            ]
            rr.log(
                f"fits/{name}/verdicts",
                rr.TextDocument("\n".join(lines), media_type=rr.MediaType.MARKDOWN),
                static=True,
            )
    return param_origins


def log_friction() -> list[str]:
    origins = []
    for tier in FRICTION_TIERS:
        curve = friction_curve(FRICTION_SERVO, tier)
        origin = f"friction/{FRICTION_SERVO}/{tier}"
        for velocity, unloaded, loaded in zip(
            curve["velocity"], curve["unloaded"], curve["loaded"], strict=True
        ):
            rr.set_time("velocity_mrad_s", sequence=int(velocity * 1000))
            rr.log(f"{origin}/unloaded", rr.Scalars(unloaded))
            rr.log(f"{origin}/under_{curve['tau_external']}Nm_load", rr.Scalars(loaded))
        origins.append(origin)
    return origins


def blueprint(
    eval_origins: list[str], fit_origins: list[str], friction_origins: list[str]
) -> rrb.Blueprint:
    eval_views = [
        view
        for origin in eval_origins
        for view in (
            rrb.BarChartView(origin=f"{origin}/funnel", name=origin.split("/")[1]),
            rrb.TextDocumentView(origin=f"{origin}/reading", name="reading"),
        )
    ]
    fit_views = [
        rrb.TimeSeriesView(origin=origin, name=origin.rsplit("/", 1)[1])
        for origin in fit_origins
    ] + [rrb.TextDocumentView(origin="fits/rig-drivetrain/verdicts", name="verdicts")]
    friction_views = [
        rrb.TimeSeriesView(origin=origin, name=f"{origin.rsplit('/', 1)[1]} model")
        for origin in friction_origins
    ]
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(*eval_views, name="evaluations"),
            rrb.Horizontal(*fit_views, name="fit records"),
            rrb.Horizontal(*friction_views, name="friction (x: velocity, mrad/s)"),
        ),
        collapse_panels=True,
    )


def main() -> None:
    rr.init("robotiq-instrument", spawn=False)
    rr.connect_grpc()  # the Studio's embedded viewer on the standard port
    eval_origins = log_evals()
    fit_origins = log_fits()
    friction_origins = log_friction()
    rr.send_blueprint(blueprint(eval_origins, fit_origins, friction_origins))
    print(
        f"logged {len(eval_origins)} eval funnel(s), "
        f"{len(fit_origins)} fit parameter(s), "
        f"{len(friction_origins)} friction tier(s)"
    )


if __name__ == "__main__":
    main()
