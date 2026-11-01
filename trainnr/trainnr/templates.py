"""Project templates: a named plan for one path from raw video to a robot
policy, with NVIDIA's models in each step by default (docs/86 §6).

A template is chosen when a project is made (`create_project`'s
`template`): its name goes into `project.json` and its loop sets the
project's. Nothing is downloaded and nothing runs. The index then carries
its steps, each done when its stage is (`project/index.py`), and the
Studio's Overview walks them in place of the generic first steps.

A step names a trainnr tool only when that tool exists; a step without
one is planned (`test_templates` checks every named tool is served). The
Studio mirrors the list for its New project picker
(`crates/trainnr-studio/src/onboarding.rs`, `TEMPLATES`; checked equal by
`test_studio_mirrors`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# What a step runs on, in the words the Overview shows.
THIS_MACHINE = "this machine"
NVIDIA_GPU = "an NVIDIA GPU"
CLOUD_GPU = "a rented GPU, 40 GB or more"


@dataclass(frozen=True)
class Step:
    # The loop state the step proves (`project/index.py`, STATES).
    stage: str
    title: str
    body: str
    runs_on: str
    # The trainnr tool that does it, or None while it is planned.
    tool: str | None = None
    # The model or program the step runs, when it is someone else's.
    model: str | None = None


@dataclass(frozen=True)
class Template:
    name: str
    title: str
    summary: str
    robot: str
    loop: str  # `project/locate.py`, LOOPS
    vendor: str
    steps: tuple[Step, ...]


ADD_SO101 = Step(
    "asset onboarded",
    "Add the SO-101",
    "trainnr's so101-nominal bundle, or your own arm's MJCF or URDF.",
    THIS_MACHINE,
    tool="onboard_robot",
)
SO101_TASK = Step(
    "environment defined",
    "Define the task",
    "The objects, where they start, and what counts as success.",
    THIS_MACHINE,
    tool="create_task",
)
GR00T = "GR00T N1.7"

TEMPLATES: dict[str, Template] = {
    t.name: t
    for t in (
        Template(
            name="humanoid-motion-from-video",
            title="Humanoid motion from video",
            summary=(
                "Film a person moving; a humanoid learns the motion in simulation, "
                "tracking it without falling."
            ),
            robot="Unitree G1",
            loop="reinforcement",
            vendor="NVIDIA",
            steps=(
                Step(
                    "asset onboarded",
                    "Add the Unitree G1",
                    "Its MJCF from MuJoCo Menagerie or mjlab.",
                    THIS_MACHINE,
                    tool="onboard_robot",
                ),
                Step(
                    "telemetry recorded",
                    "Turn a video into motion",
                    "GEM-X estimates the person's 3D motion from one camera; the SOMA "
                    "Retargeter maps it onto the G1's joints.",
                    NVIDIA_GPU,
                    model="GEM-X, SOMA Retargeter",
                ),
                Step(
                    "environment defined",
                    "Make the tracking task",
                    "Success is tracking the whole clip without falling.",
                    THIS_MACHINE,
                ),
                Step(
                    "policy trained",
                    "Train it to track the motion",
                    "Reinforcement learning in mjlab's tracking task: the actions come "
                    "from simulated physics, the video only says what the motion is.",
                    NVIDIA_GPU,
                    model="mjlab",
                ),
                Step(
                    "policy evaluated",
                    "Evaluate it with pushes and delays",
                    "Paired trials with an interval on the success rate.",
                    NVIDIA_GPU,
                ),
                Step(
                    "deployment exported",
                    "Export the policy",
                    "An ONNX policy and its deployment manifest, checked in simulation "
                    "first.",
                    THIS_MACHINE,
                ),
            ),
        ),
        Template(
            name="manipulation-from-human-video",
            title="Manipulation from human video",
            summary=(
                "Film hands doing a task; the arm replays each motion in simulation "
                "and keeps the ones that succeed."
            ),
            robot="SO-101",
            loop="imitation",
            vendor="NVIDIA",
            steps=(
                ADD_SO101,
                Step(
                    "telemetry recorded",
                    "Turn hand video into hand motion",
                    "GEM-X's hand pose estimates the wrist and fingers from video, "
                    "egocentric or from the side.",
                    NVIDIA_GPU,
                    model="GEM-X",
                ),
                SO101_TASK,
                Step(
                    "data generated",
                    "Replay the motions in simulation, keep the successes",
                    "Each hand path is retargeted to the gripper and tried; the task's "
                    "success criterion decides which become episodes.",
                    NVIDIA_GPU,
                ),
                Step(
                    "policy trained",
                    f"Fine-tune {GR00T}",
                    "The episodes exported in GR00T's LeRobot format with its modality "
                    "file, then a fine-tune of the 3B model.",
                    CLOUD_GPU,
                    model=GR00T,
                ),
                Step(
                    "policy evaluated",
                    "Evaluate in simulation",
                    "Paired trials with an interval on the success rate.",
                    NVIDIA_GPU,
                ),
            ),
        ),
        Template(
            name="more-from-your-demonstrations",
            title="More from your own demonstrations",
            summary=(
                "Vary how your teleoperated episodes look, then measure whether the "
                "variations help."
            ),
            robot="SO-101",
            loop="imitation",
            vendor="NVIDIA",
            steps=(
                ADD_SO101,
                Step(
                    "telemetry recorded",
                    "Bring in your teleoperated episodes",
                    "A LeRobot dataset from your arm, with its camera video.",
                    THIS_MACHINE,
                ),
                SO101_TASK,
                Step(
                    "data generated",
                    "Vary how the episodes look",
                    "Cosmos Transfer changes the scene's appearance; the actions stay "
                    "the ones the robot took.",
                    NVIDIA_GPU,
                    model="Cosmos Transfer 2.5",
                ),
                Step(
                    "policy trained",
                    f"Fine-tune {GR00T}, with and without the variations",
                    "Two policies from the same episodes, so the variations are judged "
                    "by what they change.",
                    CLOUD_GPU,
                    model=GR00T,
                ),
                Step(
                    "policy evaluated",
                    "Compare the two",
                    "Paired trials: the variations stay only if they measurably help.",
                    NVIDIA_GPU,
                ),
            ),
        ),
    )
}


def resolve(name: str) -> Template:
    try:
        return TEMPLATES[name]
    except KeyError:
        known = ", ".join(sorted(TEMPLATES))
        raise KeyError(f"no template {name!r}; the templates are {known}") from None


def describe(template: Template) -> dict[str, Any]:
    """A template as the tools and the index give it."""
    return asdict(template) | {"steps": [asdict(s) for s in template.steps]}
