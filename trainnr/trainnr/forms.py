"""The Studio's forms: which tools a person runs from which page, and what
each argument is beyond its type (an artifact of the project to pick, a
file to choose, one of a few words).

The arguments themselves come from the tool's own input schema, the one
an agent reads (`mcp_server.tool_api`), so a form never disagrees with
the tool it runs. The hints live here, not in the schema: the schema is
what every agent session pays for in tokens, and a picker is the app's
business (the desktop app plan, docs/85, 2026-10-11). `trainnr forms`
prints `describe()` for the Studio, which runs each submitted form as
`trainnr call <tool> --args '<json>'`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from trainnr.project.kinds import Kind

SCHEMA = "trainnr-forms/1"


@dataclass(frozen=True)
class Hint:
    """What an argument is beyond its type. `kind`: an artifact of the
    open project of this kind, picked from a list. `files`: a file to
    choose, with these extensions (`*`: any file); `folder`: a folder may
    be chosen too (a LeRobot dataset, a ROS 2 bag). `choices`: one of these
    words; `choices_from`: the words a registry holds when the forms are
    described (task families, public logs, runtimes: plugins add to them)."""

    kind: Kind | None = None
    files: tuple[str, ...] = ()
    folder: bool = False
    choices: tuple[str, ...] = ()
    choices_from: Callable[[], tuple[str, ...]] | None = None
    label: str = ""


@dataclass(frozen=True)
class Form:
    """A tool as a button on a page: `page` names the Studio page (its
    rail title), `label` the button, `hints` the arguments that need more
    than their type. Arguments not named in `fields` stay out of the form
    and take the tool's defaults; the agent still has them all."""

    tool: str
    page: str
    label: str
    fields: tuple[str, ...]
    hints: dict[str, Hint] = field(default_factory=dict)


def _task_families() -> tuple[str, ...]:
    from trainnr.tasks.overlay import families  # noqa: PLC0415

    return tuple(sorted(families()))


def _public_logs() -> tuple[str, ...]:
    from trainnr.robots.public_logs import PUBLIC_LOGS  # noqa: PLC0415

    return tuple(sorted(PUBLIC_LOGS))


def _runtimes() -> tuple[str, ...]:
    from trainnr.deploy.runtimes import runtime_names  # noqa: PLC0415

    return tuple(runtime_names())


ROBOT_FILES = ("xml", "mjcf", "urdf", "usd", "usda", "usdc", "usdz")
# Any file: a recording's reader is chosen by what the file holds
# (`robots.adapter.detect`), and plugins add readers, so no list of
# extensions here could stay right; the reader refuses what it cannot read.
ANY_FILE = ("*",)

FORMS: tuple[Form, ...] = (
    Form(
        "onboard_robot",
        "robots",
        "Add robot",
        ("model_path", "name"),
        {"model_path": Hint(files=ROBOT_FILES, label="Model file")},
    ),
    Form(
        "identify_system",
        "robots",
        "Identify dynamics",
        ("robot", "recording"),
        {"robot": Hint(kind=Kind.ROBOT), "recording": Hint(kind=Kind.RECORDING)},
    ),
    Form(
        "ingest_recording",
        "recordings",
        "Add recording",
        ("source", "name"),
        {"source": Hint(files=ANY_FILE, folder=True, label="Telemetry")},
    ),
    Form(
        "ingest_public_log",
        "recordings",
        "Add public log",
        ("log", "name", "accept_unlicensed"),
        {"log": Hint(choices_from=_public_logs, label="Public log")},
    ),
    Form(
        "create_task",
        "environments",
        "Define environment",
        ("family", "name"),
        {"family": Hint(choices_from=_task_families, label="Task family")},
    ),
    Form(
        "check_task",
        "environments",
        "Check environment",
        ("task",),
        {"task": Hint(kind=Kind.TASK, label="Environment")},
    ),
    Form(
        "train_walk",
        "experiments",
        "Train",
        ("task", "recipe", "iterations"),
        {
            "task": Hint(kind=Kind.TASK, label="Environment"),
            "recipe": Hint(choices=("smoke", "full")),
        },
    ),
    Form(
        "evaluate_walk",
        "evaluations",
        "Evaluate",
        ("checkpoint", "trials"),
        {"checkpoint": Hint(kind=Kind.RUN, label="Experiment")},
    ),
    Form(
        "export_deployment",
        "deployments",
        "Export",
        ("experiment", "checkpoint", "name"),
        {"experiment": Hint(kind=Kind.RUN)},
    ),
    Form(
        "gate_deployment",
        "deployments",
        "Run gate",
        ("deployment", "runtime", "trials"),
        {
            "deployment": Hint(kind=Kind.DEPLOY),
            "runtime": Hint(choices_from=_runtimes),
        },
    ),
    Form(
        "preflight_deployment",
        "deployments",
        "Pre-flight",
        ("deployment", "runtime"),
        {
            "deployment": Hint(kind=Kind.DEPLOY),
            "runtime": Hint(choices_from=_runtimes),
        },
    ),
    Form(
        "check_drift",
        "monitoring",
        "Check for drift",
        ("robot", "recording"),
        {"robot": Hint(kind=Kind.ROBOT), "recording": Hint(kind=Kind.RECORDING)},
    ),
)


def _called(source: Callable[[], tuple[str, ...]] | None) -> list[str]:
    """A registry's words, or none when it cannot be read (the field is
    then free text, and the tool refuses a wrong word by name)."""
    if source is None:
        return []
    try:
        return list(source())
    except Exception:  # a registry that fails leaves a text field, never no form
        return []


def _first_sentence(text: str) -> str:
    head = text.strip().split("\n")[0]
    return head.split(". ")[0].rstrip(".") + "."


def _label(name: str) -> str:
    return name.replace("_", " ").capitalize()


def describe(api: list[dict[str, Any]]) -> dict[str, Any]:
    """Every form with its fields filled in from the tools' schemas
    (`api`: `mcp_server.tool_api`). A form whose tool is missing is
    left out, so a build without it shows no dead button."""
    tools = {tool["name"]: tool for tool in api}
    forms = []
    for form in FORMS:
        tool = tools.get(form.tool)
        if tool is None:
            continue
        schema = tool["inputSchema"]
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        fields = []
        for name in form.fields:
            prop = properties[name]  # a form naming no argument is a bug
            hint = form.hints.get(name, Hint())
            fields.append(
                {
                    "name": name,
                    "label": hint.label or _label(name),
                    "type": prop.get("type", "string"),
                    "required": name in required,
                    "default": prop.get("default"),
                    "description": prop.get("description", ""),
                    "kind": hint.kind.value if hint.kind else None,
                    "files": list(hint.files),
                    "folder": hint.folder,
                    "choices": list(hint.choices) or _called(hint.choices_from),
                }
            )
        forms.append(
            {
                "tool": form.tool,
                "page": form.page,
                "label": form.label,
                "summary": _first_sentence(tool["description"]),
                "fields": fields,
            }
        )
    return {"schema": SCHEMA, "forms": forms}
