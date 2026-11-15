#!/usr/bin/env bash
# NVIDIA's video-to-motion programs, each in its own environment, at the commits
# measured on 2026-11-15 (docs/86 §5.2; the record video-to-g1-on-a-24gb-card-2026-11-15):
#   tools/video-setup.sh            GEM-X (video -> SOMA motion) and the SOMA Retargeter
#                                   (SOMA motion -> a humanoid's joints), with their weights
# trainnr runs them as external programs; nothing here becomes a trainnr dependency.
# Downloads (SECURITY.md): both repositories from GitHub (the retargeter's SOMA and Unitree
# assets through Git LFS), nvidia/GEM-X and nvidia/SOMA-X from Hugging Face (about 7 GB),
# OpenMMLab's YOLOX person detector, and PyTorch's CUDA wheels.
# Licences: the code is Apache-2.0; GEM-X's weights are under the NVIDIA Open Model
# License, SOMA-X's assets Apache-2.0, the retargeter's sample motions NVIDIA's
# evaluation licence (not fetched).
set -euo pipefail
GEMX="${TRAINNR_GEMX:-$HOME/.cache/trainnr/GEM-X}"
SOMA="${TRAINNR_SOMA_RETARGETER:-$HOME/.cache/trainnr/soma-retargeter}"
GEMX_COMMIT="4aef1452e5df171deff4f293c08e643eb450981e"
SOMA_COMMIT="cbdb2b494a2af82137bf9b6f15122d60167281c4"
GEMX_WEIGHTS="b111f2dcf3ebcf57965fd53499cb645dfa5ba25c"  # nvidia/GEM-X on Hugging Face
SOMA_ASSETS="v0.3.3"  # nvidia/SOMA-X; GEM-X's runtime is py-soma-x 0.3.3
YOLOX_URL="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip"
YOLOX_SHA256="8e9ea96a176bd48501eaaa77216e49ee30794d2f8ba80c7b9862beca4ea972da"
# The lock files are revision 5; uv 0.5 cannot read them.
UV=(uvx --from "uv>=0.13,<0.14" uv)

checkout() {  # checkout <url> <dir> <commit> [lfs paths]
  if [ ! -d "$2/.git" ]; then GIT_LFS_SKIP_SMUDGE=1 git clone -q "$1" "$2"; fi
  git -C "$2" fetch -q origin "$3" && git -C "$2" checkout -q "$3"
  if [ -n "${4:-}" ]; then git -C "$2" lfs pull --include="$4"; fi
}

checkout https://github.com/NVlabs/GEM-X.git "$GEMX" "$GEMX_COMMIT"
cd "$GEMX"
[ -x .venv/bin/python ] || "${UV[@]}" venv -q .venv --python 3.12
export VIRTUAL_ENV="$GEMX/.venv"
"${UV[@]}" pip install -q "torch>=2.10,<3" torchvision --index-url https://download.pytorch.org/whl/cu128
"${UV[@]}" pip install -q -e '.[video]' huggingface_hub
mkdir -p weights assets/soma
.venv/bin/hf download nvidia/GEM-X body_s240000.ckpt vitpose.pth motion_native77_e774_s292175.ckpt \
  motion_stats_raw.npz body_video_native77.yaml --revision "$GEMX_WEIGHTS" --local-dir weights
.venv/bin/hf download nvidia/SOMA-X SOMA_neutral.npz SOMA_template_rig.usda \
  SOMA_procedural_transforms.json --revision "$SOMA_ASSETS" --local-dir assets/soma
if [ ! -f weights/yolox_humanart.onnx ]; then
  curl -s --fail --location "$YOLOX_URL" -o weights/yolox_humanart.zip
  .venv/bin/python -c "
import zipfile, shutil
with zipfile.ZipFile('weights/yolox_humanart.zip') as a, open('weights/yolox_humanart.onnx', 'wb') as t:
    shutil.copyfileobj(a.open(next(n for n in a.namelist() if n.endswith('/end2end.onnx'))), t)"
  rm weights/yolox_humanart.zip
fi
echo "$YOLOX_SHA256  weights/yolox_humanart.onnx" | sha256sum --check --quiet

checkout https://github.com/NVIDIA/soma-retargeter.git "$SOMA" "$SOMA_COMMIT" \
  "soma_retargeter/assets/soma/*,soma_retargeter/configs/soma/*,soma_retargeter/assets/robotics/unitree/**"
cd "$SOMA"
"${UV[@]}" sync -q
echo "GEM-X at $GEMX, the SOMA Retargeter at $SOMA"
