"""What every exporter writes beside a dataset: the provenance sidecar
name and the image feature's axis names — spelled once for the two
exporters (kitting_export, lerobot_export) and their tests."""

PROVENANCE_FILE = "provenance.json"
IMAGE_AXES = ("height", "width", "channels")
