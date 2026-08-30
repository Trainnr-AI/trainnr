"""One query from the instrument, as JSON on stdout — the Studio's data
panel calls this per selection instead of linking Python in.

    cd pipeline && uv run --extra sim python ../tools/studio-data.py <query> [args…]

Queries map 1:1 onto `rq_pipeline.mcp_server`'s functions — the same
surface the MCP tools serve, so the native panels and an agent's answers
can never disagree about what the instrument holds:

    describe_bundles | describe_bundle NAME | describe_actuators
    describe_actuator SLUG [TIER] | list_eval_records
    describe_eval RUN | friction_curve SLUG [TIER]
"""

import json
import sys

from _lab import bootstrap

bootstrap()

from rq_pipeline import mcp_server  # noqa: E402

QUERIES = {
    "describe_bundles": 0,
    "describe_bundle": 1,
    "describe_actuators": 0,
    "describe_actuator": 2,
    "list_eval_records": 0,
    "describe_eval": 1,
    "friction_curve": 2,
}


def main() -> None:
    name, args = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if name not in QUERIES:
        sys.exit(f"usage: studio-data.py <query> [args…]; one of {sorted(QUERIES)}")
    if len(args) > QUERIES[name]:
        sys.exit(f"{name} takes at most {QUERIES[name]} argument(s), got {len(args)}")
    result = getattr(mcp_server, name)(*args)
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
