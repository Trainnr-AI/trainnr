"""Render the sim's three ArmnetBench cameras for eyeball calibration.

    cd trainnr && uv run --extra sim python ../tools/camera-match.py [task]

Writes data/camera-match/<task>-<key>.png for front/top/wrist. The
calibration protocol: put these beside real frames from the released
dataset videos and iterate camera placement until a squint can't tell
the viewpoint apart — a vision policy failing on COSMETICS must never
be read as a dynamics gap. (Real-frame extraction needs a video
decoder; run that half on the WSL box where torchcodec works.)
"""

import sys
from pathlib import Path

from _lab import bootstrap

bootstrap()

import mujoco  # noqa: E402
from PIL import Image  # noqa: E402
from trainnr.evaluate.vision import ARMNETBENCH_CAMERAS  # noqa: E402
from trainnr.tasks.registry import tasks  # noqa: E402

# Every SO-101 task the registry knows, by name.
BUILDERS = {
    entry.name: entry.build for entry in tasks().values() if entry.rig == "so101"
}

task_name = sys.argv[1] if len(sys.argv) > 1 else "block_stack"
if task_name not in BUILDERS:
    sys.exit(f"unknown task {task_name!r}; one of {sorted(BUILDERS)}")

task = BUILDERS[task_name]()
model = task.spec.compile()
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

out_dir = Path(__file__).resolve().parent.parent / "data" / "camera-match"
out_dir.mkdir(parents=True, exist_ok=True)
for camera in ARMNETBENCH_CAMERAS:
    renderer = mujoco.Renderer(model, height=camera.height, width=camera.width)
    renderer.update_scene(data, camera=camera.camera_name)
    frame = renderer.render()
    renderer.close()
    path = out_dir / f"{task_name}-{camera.key}.png"
    Image.fromarray(frame).save(path)
    print(f"{path}  ({camera.width}x{camera.height}, max px {int(frame.max())})")
