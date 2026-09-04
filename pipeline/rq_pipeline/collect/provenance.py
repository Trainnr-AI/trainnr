"""What every exporter writes beside a dataset: the provenance sidecar
name and the image feature's axis names — spelled once for the two
exporters (kitting_export, lerobot_export) and their tests."""

PROVENANCE_FILE = "provenance.json"
# An expert stamp is `<labeler>[+dagger:<driver>]`: the labeler produced
# the actions the dataset learns from; the driver, when present, chose
# the states (DAgger, docs/66 §6). Two datasets are one story when their
# LABELERS agree — the driver is part of the story, not a second story.
DAGGER_SEP = "+dagger:"


def labeler(expert: str) -> str:
    """The part of an expert stamp that produced the actions."""
    return expert.split(DAGGER_SEP, 1)[0]


def dagger_stamp(expert: str, driver: str) -> str:
    return f"{expert}{DAGGER_SEP}{driver}"


IMAGE_AXES = ("height", "width", "channels")
