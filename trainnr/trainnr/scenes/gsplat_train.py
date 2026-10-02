"""The capture chain's CUDA splat trainer: gsplat (nerfstudio-project,
Apache-2.0) on a COLMAP dataset - `images/` beside `sparse/0/` - leaving
the same file Brush leaves, a 3DGS PLY in COLMAP's frame
(`splatters.SPLAT_EXPORT`). Runs in the train environment (torch), as a
subprocess of the chain; streams its training into the Studio when asked
(`--rerun`): the loss, the splat count, the cameras and the splats
themselves, on an `iterations` timeline, the entity names Brush uses.

The recipe is gsplat's own reference one (its `simple_trainer`): the
sparse points as the first gaussians, a scale from the three nearest
neighbours, Adam per parameter with the reference learning rates, the
default densification strategy, L1 with a fifth of D-SSIM, harmonics
raised a degree every thousand steps. Distortion is undone once on the
frames (OpenCV), as the reference does; a fisheye capture is refused by
name.

gsplat compiles its kernels with nvcc at first use. The CUDA tree pip
installs beside torch (`nvidia/cu<major>`) is the one this environment
carries, so it is named to the compiler here when nothing else is
(`tools/install-gsplat.py` builds once at install).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from trainnr.scenes.colmap_model import Camera, TextModel, read_text_model
from trainnr.scenes.splatters import (
    DEFAULT_STEPS,
    MAX_RESOLUTION,
    RENDERS_DIR,
    SPLAT_EXPORT,
)

if TYPE_CHECKING:
    import torch

APP_ID = "gsplat"
TIMELINE = "iterations"
SH_C0 = 0.28209479177387814  # the zeroth harmonic's constant: colour <-> sh0
KNN_NEIGHBOURS = 3
INITIAL_OPACITY = 0.1
SH_DEGREE = 3
SH_DEGREE_INTERVAL = 1000  # a degree more every this many steps
SSIM_WEIGHT = 0.2
SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
# The reference learning rates (gsplat's simple_trainer), means scaled by the scene.
LEARNING_RATES = {
    "means": 1.6e-4,
    "scales": 5e-3,
    "quats": 1e-3,
    "opacities": 5e-2,
    "sh0": 2.5e-3,
    "shN": 2.5e-3 / 20,
}
MEANS_LR_FINAL_FRACTION = 0.01  # the means' rate decays to this by the last step
REFINE_START = 500
REFINE_STOP_FRACTION = 0.5  # densify over the first half (the reference: 15k of 30k)
REFINE_EVERY = 100
RESET_EVERY = 3000
LOG_EVERY = 100
RENDER_EVERY = 1000  # the watched view re-rendered into the stream
SPLATS_LOG_FRACTION = 10  # the cloud shown this many times over a run...
SPLATS_LOG_MIN_EVERY = 500  # ...but never more often than this
RENDER_STILLS = 8  # views rendered at the end, evenly spaced through the capture
STILL_FILE = "view-{index:02d}.png"
STREAM_GAUSSIANS = 500_000  # the splat shown live, the most opaque past this
FRUSTUM_FRACTION = 0.05  # a camera's drawn image plane, as a fraction of the scene
SCENE_SCALE_MARGIN = 1.1
FISHEYE_MODELS = ("OPENCV_FISHEYE",)


def ensure_cuda_home() -> Path | None:
    """Name the pip CUDA tree to the compiler when nothing else is:
    `CUDA_HOME` to `nvidia/cu<major>`, the one tree CUDA 13's wheels lay
    out (nvcc, headers, nvvm; `tools/install-gsplat.py` fills it)."""
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return Path(os.environ.get("CUDA_HOME") or os.environ["CUDA_PATH"])
    try:
        import nvidia  # noqa: PLC0415 - the namespace package torch's wheels fill
    except ImportError:
        return None
    for base in map(Path, nvidia.__path__):
        for tree in sorted(base.glob("cu*")):
            if any((tree / "bin" / n).is_file() for n in ("nvcc", "nvcc.exe")):
                os.environ["CUDA_HOME"] = str(tree)
                return tree
    return None


def ensure_arch_list() -> None:
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return
    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"


# -- the views ------------------------------------------------------------------


@dataclass(frozen=True)
class View:
    name: str
    image: np.ndarray  # (h, w, 3) uint8, undistorted, resized
    intrinsics: np.ndarray  # (3, 3) for that image
    world_to_camera: np.ndarray  # (4, 4)


def _undistort(image: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    """The frame remapped to a pinhole and cropped to the valid region,
    with the pinhole's intrinsics (the reference's OpenCV path)."""
    import cv2  # noqa: PLC0415

    if camera.model in FISHEYE_MODELS:
        raise ValueError(
            f"camera model {camera.model}: a fisheye capture is not one this trainer "
            "undistorts; capture with the phone's main lens"
        )
    k1, k2, k3, *_ = camera.radial
    p1, p2 = camera.tangential
    dist = np.array([k1, k2, p1, p2, k3], np.float64)
    h, w = image.shape[:2]
    intrinsics = camera.K
    intrinsics_new, roi = cv2.getOptimalNewCameraMatrix(intrinsics, dist, (w, h), 0)
    map_x, map_y = cv2.initUndistortRectifyMap(
        intrinsics, dist, None, intrinsics_new, (w, h), cv2.CV_32FC1
    )
    out = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    x, y, rw, rh = roi
    out = out[y : y + rh, x : x + rw]
    intrinsics_new = intrinsics_new.copy()
    intrinsics_new[0, 2] -= x
    intrinsics_new[1, 2] -= y
    return out, intrinsics_new


def _resized(
    image: np.ndarray, intrinsics: np.ndarray, max_side: int
) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image as PilImage  # noqa: PLC0415

    h, w = image.shape[:2]
    factor = max(h, w) / max_side
    if factor <= 1.0:
        return image, intrinsics
    new_w, new_h = round(w / factor), round(h / factor)
    small = np.asarray(
        PilImage.fromarray(image).resize((new_w, new_h), PilImage.Resampling.LANCZOS)
    )
    intrinsics = intrinsics.copy()
    intrinsics[0, :] *= new_w / w
    intrinsics[1, :] *= new_h / h
    return small, intrinsics


def load_views(dataset: Path, *, max_resolution: int) -> tuple[list[View], TextModel]:
    """Every registered image as a view the rasterizer takes."""
    from PIL import Image as PilImage  # noqa: PLC0415

    model = read_text_model(dataset / "sparse" / "0")
    views: list[View] = []
    for img in model.images:
        camera = model.cameras[img.camera_id]
        pixels = np.asarray(PilImage.open(dataset / "images" / img.name).convert("RGB"))
        if camera.distorted:
            pixels, intrinsics = _undistort(pixels, camera)
        else:
            intrinsics = camera.K
        pixels, intrinsics = _resized(pixels, intrinsics, max_resolution)
        views.append(
            View(
                img.name, np.ascontiguousarray(pixels), intrinsics, img.world_to_camera
            )
        )
    if not views:
        raise ValueError(f"no registered images in {dataset}")
    return views, model


def scene_scale(views: list[View]) -> float:
    """How far the cameras spread from their centre, with a margin: the
    unit the means' learning rate and the strategy's thresholds take."""
    centres = np.stack([np.linalg.inv(v.world_to_camera)[:3, 3] for v in views])
    return (
        float(np.linalg.norm(centres - centres.mean(0), axis=1).max())
        * SCENE_SCALE_MARGIN
    )


# -- the gaussians ----------------------------------------------------------------


def initial_scales(
    points: torch.Tensor, neighbours: int = KNN_NEIGHBOURS
) -> torch.Tensor:
    """log of the mean distance to the nearest neighbours: a gaussian
    the size of the gap it sits in (the reference's initialisation)."""
    import torch  # noqa: PLC0415

    n = points.shape[0]
    chunk = 4096
    dist2 = torch.empty(n, device=points.device)
    for start in range(0, n, chunk):
        block = points[start : start + chunk]
        d = torch.cdist(block, points)  # (chunk, n)
        # the nearest is the point itself at distance 0: skip it
        nearest = d.topk(neighbours + 1, dim=1, largest=False).values[:, 1:]
        dist2[start : start + chunk] = (nearest**2).mean(1)
    return torch.log(torch.sqrt(dist2).clamp_min(1e-6)).unsqueeze(-1).repeat(1, 3)


def make_params(
    model: TextModel, *, sh_degree: int, device: str
) -> torch.nn.ParameterDict:
    import torch  # noqa: PLC0415

    points = torch.as_tensor(model.points, dtype=torch.float32, device=device)
    rgb = torch.as_tensor(model.colors, dtype=torch.float32, device=device) / 255.0
    n = points.shape[0]
    if n == 0:
        raise ValueError("the COLMAP model has no points to start the gaussians from")
    k = (sh_degree + 1) ** 2
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(points),
            "scales": torch.nn.Parameter(initial_scales(points)),
            "quats": torch.nn.Parameter(torch.rand(n, 4, device=device)),
            "opacities": torch.nn.Parameter(
                torch.logit(torch.full((n,), INITIAL_OPACITY, device=device))
            ),
            "sh0": torch.nn.Parameter(((rgb - 0.5) / SH_C0).unsqueeze(1)),
            "shN": torch.nn.Parameter(torch.zeros(n, k - 1, 3, device=device)),
        }
    )


def make_optimizers(
    params: torch.nn.ParameterDict,
    scale: float,
) -> dict[str, torch.optim.Optimizer]:
    import torch  # noqa: PLC0415

    out = {}
    for name, lr in LEARNING_RATES.items():
        rate = lr * scale if name == "means" else lr
        out[name] = torch.optim.Adam(
            [{"params": [params[name]], "lr": rate, "name": name}],
            eps=1e-15,
            betas=(0.9, 0.999),
        )
    return out


# -- the loss -----------------------------------------------------------------------


def _gaussian_window(size: int, sigma: float, device: str) -> torch.Tensor:
    import torch  # noqa: PLC0415

    x = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(3, 1, size, size).contiguous()


def ssim(a: torch.Tensor, b: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """SSIM over (1, 3, h, w) images in 0..1, the usual 11-tap window."""
    import torch.nn.functional as F  # noqa: PLC0415, N812

    pad = window.shape[-1] // 2
    mu_a = F.conv2d(a, window, padding=pad, groups=3)
    mu_b = F.conv2d(b, window, padding=pad, groups=3)
    sig_a = F.conv2d(a * a, window, padding=pad, groups=3) - mu_a**2
    sig_b = F.conv2d(b * b, window, padding=pad, groups=3) - mu_b**2
    sig_ab = F.conv2d(a * b, window, padding=pad, groups=3) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    num = (2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (sig_a + sig_b + c2)
    return (num / den).mean()


# -- the Studio ------------------------------------------------------------------------


class Narrator:
    """The run into the Studio's viewer, Brush's entity names: `losses/main`,
    `splats/num_splats`, `world/cameras`, `world/splats`. Silent without `--rerun`."""

    def __init__(self, on: bool) -> None:
        self.rr = None
        if not on:
            return
        import rerun as rr  # noqa: PLC0415

        from trainnr.viz import STUDIO_ADDRESS  # noqa: PLC0415

        rr.init(APP_ID, spawn=False)
        rr.connect_grpc(STUDIO_ADDRESS)
        self.rr = rr
        self.layout()

    def layout(self) -> None:
        """The run's own layout: the world with its cameras and splat, the
        watched view beside its truth, the loss and the count. Sent once,
        so the viewer does not guess (its guess put the images in the 3D
        view and flagged it)."""
        if self.rr is None:
            return
        import rerun.blueprint as rrb  # noqa: PLC0415

        self.rr.send_blueprint(
            rrb.Grid(
                rrb.Spatial3DView(origin="world", name="world"),
                rrb.Spatial2DView(origin="render/view", name="watched view"),
                rrb.Spatial2DView(origin="render/truth", name="its frame"),
                rrb.TimeSeriesView(origin="losses", name="loss"),
                rrb.TimeSeriesView(origin="splats", name="gaussians"),
                grid_columns=3,
            )
        )

    def cameras(self, views: list[View], scale: float) -> None:
        if self.rr is None:
            return
        self.rr.set_time(TIMELINE, sequence=0)
        for i, v in enumerate(views):
            c2w = np.linalg.inv(v.world_to_camera)
            h, w = v.image.shape[:2]
            self.rr.log(
                f"world/cameras/{i:04d}",
                self.rr.Transform3D(translation=c2w[:3, 3], mat3x3=c2w[:3, :3]),
            )
            self.rr.log(
                f"world/cameras/{i:04d}",
                self.rr.Pinhole(
                    image_from_camera=v.intrinsics,
                    width=w,
                    height=h,
                    image_plane_distance=scale * FRUSTUM_FRACTION,
                ),
            )

    def truth(self, image: np.ndarray) -> None:
        if self.rr is None:
            return
        self.rr.log("render/truth", self.rr.Image(image), static=True)

    def render(self, step: int, image: np.ndarray, name: str = "view") -> None:
        if self.rr is None:
            return
        self.rr.set_time(TIMELINE, sequence=step)
        self.rr.log(f"render/{name}", self.rr.Image(image))

    def step(self, step: int, *, loss: float, count: int, lr_means: float) -> None:
        if self.rr is None:
            return
        self.rr.set_time(TIMELINE, sequence=step)
        self.rr.log("losses/main", self.rr.Scalars(loss))
        self.rr.log("splats/num_splats", self.rr.Scalars(count))
        self.rr.log("lr/means", self.rr.Scalars(lr_means))

    def splats(self, step: int, gaussians: dict[str, np.ndarray]) -> None:
        """The splat itself, Rerun's own archetype (the viewer renders
        gaussians, not dots): the most opaque `STREAM_GAUSSIANS` of them."""
        if self.rr is None:
            return
        from trainnr.scenes.splat import Splats  # noqa: PLC0415
        from trainnr.viz import gaussians as archetype  # noqa: PLC0415

        opacities = gaussians["opacities"]
        keep = np.argsort(opacities)[::-1][:STREAM_GAUSSIANS]
        shown = Splats(
            means=gaussians["means"][keep].astype(np.float32),
            quats=gaussians["quats"][keep].astype(np.float32),
            scales=gaussians["scales"][keep].astype(np.float32),
            opacities=opacities[keep].astype(np.float32),
            colors=np.clip(gaussians["sh0"][keep] * SH_C0 + 0.5, 0, 1).astype(
                np.float32
            ),
        )
        self.rr.set_time(TIMELINE, sequence=step)
        self.rr.log("world/splats", archetype(self.rr, shown))


# -- rendering ----------------------------------------------------------------------


def activated(params: torch.nn.ParameterDict) -> dict[str, torch.Tensor]:
    """The gaussians as the rasterizer takes them: scales and opacities
    through their activations, the harmonics joined."""
    import torch  # noqa: PLC0415

    return {
        "means": params["means"],
        "quats": params["quats"],
        "scales": torch.exp(params["scales"]),
        "opacities": torch.sigmoid(params["opacities"]),
        "colors": torch.cat([params["sh0"], params["shN"]], 1),
    }


def render_image(
    gaussians: dict[str, torch.Tensor],
    view: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    sh_degree: int,
) -> np.ndarray:
    """One view of the gaussians as an 8-bit RGB image (no gradient)."""
    import torch  # noqa: PLC0415
    from gsplat import rasterization  # noqa: PLC0415

    image, intrinsics, w2c = view
    h, w = image.shape[:2]
    with torch.no_grad():
        renders, _, _ = rasterization(
            means=gaussians["means"],
            quats=gaussians["quats"],
            scales=gaussians["scales"],
            opacities=gaussians["opacities"],
            colors=gaussians["colors"],
            viewmats=w2c[None],
            Ks=intrinsics[None],
            width=w,
            height=h,
            sh_degree=sh_degree,
            packed=False,
        )
    return (renders[0, ..., :3].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()


def still_views(count: int, of: int) -> list[int]:
    """`count` view indices spread evenly through the capture."""
    if of <= count:
        return list(range(of))
    if count <= 1:
        return [of // 2]  # one still: the middle of the capture
    return [round(i * (of - 1) / (count - 1)) for i in range(count)]


def render_stills(  # noqa: PLR0913 - the stills' knobs, each named
    gaussians: dict[str, torch.Tensor],
    tensors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    out: Path,
    *,
    sh_degree: int,
    narrator: Narrator | None = None,
    step: int = 0,
) -> list[Path]:
    """`RENDER_STILLS` views of the splat as PNGs under `out`, the
    scene's own renders (the card's picture, the drawer's strip)."""
    from PIL import Image as PilImage  # noqa: PLC0415

    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, view_index in enumerate(still_views(RENDER_STILLS, len(tensors))):
        image = render_image(gaussians, tensors[view_index], sh_degree=sh_degree)
        path = out / STILL_FILE.format(index=index)
        PilImage.fromarray(image).save(path)
        written.append(path)
        if narrator is not None:
            narrator.render(step, image, name=f"still-{index:02d}")
    return written


def gaussians_from_ply(path: Path, device: str) -> tuple[dict[str, torch.Tensor], int]:
    """A trained splat back as the rasterizer's tensors, with its degree."""
    import torch  # noqa: PLC0415

    from trainnr.scenes.splat import read_ply  # noqa: PLC0415

    splats = read_ply(path)
    sh0 = ((splats.colors - 0.5) / SH_C0)[:, None, :]
    rest = splats.sh_rest if splats.sh_rest.size else np.zeros((splats.count, 0, 3))
    colors = np.concatenate([sh0, rest], 1).astype(np.float32)

    def t(a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            np.ascontiguousarray(a), dtype=torch.float32, device=device
        )

    return {
        "means": t(splats.means),
        "quats": t(splats.quats),
        "scales": t(splats.scales),
        "opacities": t(splats.opacities),
        "colors": t(colors),
    }, splats.sh_degree


def render_ply(
    dataset: Path, ply: Path, out: Path, *, max_resolution: int
) -> list[Path]:
    """The stills of an exported splat from the capture's own views (the
    `--render` mode: a scene captured before the stills existed)."""
    ensure_cuda_home()
    import torch  # noqa: PLC0415

    views, _ = load_views(dataset, max_resolution=max_resolution)
    tensors = view_tensors(views, "cuda")
    gaussians, degree = gaussians_from_ply(ply, "cuda")
    torch.cuda.synchronize()
    return render_stills(gaussians, tensors, out, sh_degree=degree)


def view_tensors(
    views: list[View], device: str
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    import torch  # noqa: PLC0415

    return [
        (
            torch.as_tensor(v.image, device=device).float() / 255.0,
            torch.as_tensor(v.intrinsics, dtype=torch.float32, device=device),
            torch.as_tensor(v.world_to_camera, dtype=torch.float32, device=device),
        )
        for v in views
    ]


# -- the run ---------------------------------------------------------------------


def train(  # noqa: PLR0913, PLR0915 - the loop, its knobs named
    dataset: Path,
    out: Path,
    *,
    steps: int,
    export_name: str,
    max_resolution: int,
    narrate: bool,
    sh_degree: int = SH_DEGREE,
    seed: int = 0,
) -> Path:
    ensure_cuda_home()
    ensure_arch_list()
    import torch  # noqa: PLC0415
    from gsplat import export_splats, rasterization  # noqa: PLC0415
    from gsplat.strategy import DefaultStrategy  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("gsplat needs a CUDA device and torch sees none")
    device = "cuda"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    began = time.time()
    views, model = load_views(dataset, max_resolution=max_resolution)
    scale = scene_scale(views)
    print(
        f"[gsplat] {len(views)} views, {model.points.shape[0]} points, "
        f"scene scale {scale:.2f}, {views[0].image.shape[1]}x{views[0].image.shape[0]}",
        flush=True,
    )
    params = make_params(model, sh_degree=sh_degree, device=device)
    optimizers = make_optimizers(params, scale)
    means_schedule = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=MEANS_LR_FINAL_FRACTION ** (1.0 / steps)
    )
    strategy = DefaultStrategy(
        refine_start_iter=REFINE_START,
        refine_stop_iter=int(steps * REFINE_STOP_FRACTION),
        refine_every=REFINE_EVERY,
        reset_every=RESET_EVERY,
    )
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scale)
    window = _gaussian_window(SSIM_WINDOW, SSIM_SIGMA, device)
    tensors = view_tensors(views, device)
    narrator = Narrator(narrate)
    narrator.cameras(views, scale)
    watched = len(views) // 2  # one view re-rendered as the run goes, its truth beside
    narrator.truth(views[watched].image)
    splats_every = max(steps // SPLATS_LOG_FRACTION, SPLATS_LOG_MIN_EVERY)
    tick = time.time()
    for step in range(steps):
        image, intrinsics, w2c = tensors[int(rng.integers(len(tensors)))]
        h, w = image.shape[:2]
        degree = min(step // SH_DEGREE_INTERVAL, sh_degree)
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        renders, _, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors,
            viewmats=w2c[None],
            Ks=intrinsics[None],
            width=w,
            height=h,
            sh_degree=degree,
            packed=False,
            absgrad=strategy.absgrad,
        )
        strategy.step_pre_backward(params, optimizers, state, step, info)
        rendered = renders[0, ..., :3]
        l1 = torch.abs(rendered - image).mean()
        structural = ssim(
            rendered.permute(2, 0, 1)[None], image.permute(2, 0, 1)[None], window
        )
        loss = l1 * (1 - SSIM_WEIGHT) + (1 - structural) * SSIM_WEIGHT
        loss.backward()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        means_schedule.step()
        if (step + 1) % LOG_EVERY == 0 or step == 0:
            now = time.time()
            rate = LOG_EVERY / max(now - tick, 1e-6) if step else 0.0
            tick = now
            count = int(params["means"].shape[0])
            print(
                f"[gsplat] step {step + 1}/{steps} loss {loss.item():.4f} "
                f"splats {count} {rate:.1f} it/s",
                flush=True,
            )
            narrator.step(
                step + 1,
                loss=loss.item(),
                count=count,
                lr_means=optimizers["means"].param_groups[0]["lr"],
            )
        if (step + 1) % RENDER_EVERY == 0 or step + 1 == steps:
            narrator.render(
                step + 1,
                render_image(activated(params), tensors[watched], sh_degree=degree),
            )
        if (step + 1) % splats_every == 0 or step + 1 == steps:
            narrator.splats(
                step + 1,
                {
                    "means": params["means"].detach().cpu().numpy(),
                    "quats": params["quats"].detach().cpu().numpy(),
                    "scales": torch.exp(params["scales"]).detach().cpu().numpy(),
                    "opacities": torch.sigmoid(params["opacities"])
                    .detach()
                    .cpu()
                    .numpy(),
                    "sh0": params["sh0"].detach()[:, 0].cpu().numpy(),
                },
            )
    out.mkdir(parents=True, exist_ok=True)
    export = out / export_name
    with torch.no_grad():
        export_splats(
            means=params["means"],
            scales=params["scales"],
            quats=params["quats"],
            opacities=params["opacities"],
            sh0=params["sh0"],
            shN=params["shN"],
            format="ply",
            save_to=str(export),
        )
    stills = render_stills(
        activated(params),
        tensors,
        out / RENDERS_DIR,
        sh_degree=sh_degree,
        narrator=narrator,
        step=steps,
    )
    minutes = (time.time() - began) / 60
    print(
        f"[gsplat] exported {params['means'].shape[0]} gaussians to {export} and "
        f"{len(stills)} stills to {out / RENDERS_DIR} in {minutes:.1f} min",
        flush=True,
    )
    return export


def link_dev_libraries(tree: Path) -> list[Path]:
    """The unversioned names the linker asks for (`-lcudart` wants
    `libcudart.so`), beside the versioned ones pip's wheels ship: each
    a link, made once at install. POSIX only; Windows links by import
    library."""
    if sys.platform.startswith("win"):
        return []
    made: list[Path] = []
    for versioned in sorted(tree.glob("lib/lib*.so.*")):
        stem = versioned.name.split(".so.", 1)[0]
        plain = versioned.with_name(f"{stem}.so")
        if not plain.exists():
            plain.symlink_to(versioned.name)
            made.append(plain)
    return made


def build_kernels() -> None:
    """Compile gsplat's CUDA kernels now (the installer's step), so the
    first capture pays nothing."""
    tree = ensure_cuda_home()
    if tree is not None:
        for plain in link_dev_libraries(tree):
            print(f"[gsplat] linked {plain.name}", flush=True)
    ensure_arch_list()
    began = time.time()
    from gsplat.cuda._backend import _C  # noqa: PLC0415

    print(f"[gsplat] kernels built in {time.time() - began:.0f} s: {_C}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dataset", type=Path, nargs="?")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--export-name", default=SPLAT_EXPORT)
    parser.add_argument("--max-resolution", type=int, default=MAX_RESOLUTION)
    parser.add_argument("--sh-degree", type=int, default=SH_DEGREE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rerun", action="store_true", help="stream into the Studio")
    parser.add_argument(
        "--build", action="store_true", help="compile the kernels and exit"
    )
    parser.add_argument(
        "--render",
        type=Path,
        default=None,
        help="render this exported splat's stills from the dataset's views, and exit",
    )
    args = parser.parse_args(argv)
    if args.build:
        build_kernels()
        return 0
    if args.render is not None:
        if args.dataset is None or args.out is None:
            parser.error("a dataset and --out are needed to render")
        stills = render_ply(
            args.dataset, args.render, args.out, max_resolution=args.max_resolution
        )
        print(f"[gsplat] rendered {len(stills)} stills to {args.out}", flush=True)
        return 0
    if args.dataset is None or args.out is None:
        parser.error("a dataset and --out are needed to train")
    train(
        args.dataset,
        args.out,
        steps=args.steps,
        export_name=args.export_name,
        max_resolution=args.max_resolution,
        narrate=args.rerun,
        sh_degree=args.sh_degree,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
