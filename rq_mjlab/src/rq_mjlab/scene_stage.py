"""A captured scene as the walk's stage (docs/78 §4 E2): the scene's
heightfield - the same grid the staged gate collides with
(`rq_pipeline.scenes.terrain`, saved beside the scene as numpy) - as an
mjlab sub-terrain at the scene's own coordinates, every world spawned
at the course's start; a head camera on the base that sees the scene's
gaussians through mujoco_warp's ray tracer across every world; and the
camera's picture as an observation term.

Two things are ours because mjlab has no seam for them yet:

- mjlab centres its terrain grid at the world origin, so a sub-terrain
  that wants the scene's coordinates has to place itself relative to
  the patch corner the generator will add (`SceneHeightfieldCfg`); the
  robot spawns at the course's start, on the surface.
- mjlab's `SensorContext` builds mujoco_warp's render context without
  the renderer's `splat_*` arguments. `render_splats` puts a thin proxy
  over the module the context calls, adding them (the patch offered
  upstream); nothing else in mujoco_warp is touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
import torch
from mjlab.managers import ObservationTermCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    TerrainGeometry,
    TerrainOutput,
)
from rq_pipeline.bundles.hashing import stamp
from rq_pipeline.scenes.cameras import VISUAL_GROUPS, splat_arguments
from rq_pipeline.scenes.heads import HEAD_CAMERA, head_mount
from rq_pipeline.scenes.record import (
    SCENE_FILE,
    SPLAT_FILE,
    floor_friction,
    load_scene_record,
)
from rq_pipeline.scenes.splat import VISIBLE_OPACITY, read_ply
from rq_pipeline.scenes.stage import course_start
from rq_pipeline.scenes.terrain import (
    GRID_FILE,
    Grid,
    add_heightfield,
    proxy_hash,
    read_grid,
)

from rq_mjlab.walks import ROBOT_ENTITY

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

SUB_TERRAIN = "scene"
TERRAIN_BODY = "terrain"  # mjlab's generator names it; the contact sensors match it
# The training picture: small, so 256 worlds render every step; the gate's
# 160x120 is the film strip a human watches (scenes.cameras).
CAMERA_WIDTH = 64
CAMERA_HEIGHT = 64
CAMERA_TERM = "head_rgb"
# The training grid's vertex spacing. mujoco_warp caps the prisms one geom
# may collide with at mujoco's MJ_MAXCONPAIR (50) and DROPS the rest with
# a warning per step (778,000 of them, 2.3 GB of log, in the first scene
# smoke): at the stage's 2 cm a calf capsule's footprint alone holds 66
# prisms; at 5 cm it holds 20 and only a trunk lying flat exceeds the cap.
TRAIN_CELL_M = 0.05
# The scene's ground in training sits in a group of its own: mjlab's height
# scans see only the groups they name, the head camera draws VISUAL_GROUPS
# (0-2), and the robot's colliders are the collision group (3). In the
# collision group the scans saw nothing: go2-scene-c1 trained blind to its
# hurdles (0/40 tracked, 2026-09-23), and without a camera mujoco_warp
# refit an empty ray structure and crashed (2026-09-25).
TERRAIN_SCAN_GROUP = 4
# Every viewer draws groups 0-2 by default (MuJoCo's, mjviser's, the Studio
# mirror's), so a scene played or recorded without its splat must turn the
# ground's group on or the robots walk on nothing (2026-09-25).
SCAN_SENSORS = ("terrain_scan", "foot_height_scan")  # the rough recipe's
NO_GRID = (
    "scene {name} has no {file}: the grid is sampled when the scene is staged "
    "(stage_deployment) or by scenes.terrain.ensure_grid, in the pipeline's "
    "environment (Open3D)"
)
STALE_GRID = (
    "scene {name}'s {file} was sampled from another proxy: re-sample it "
    "(scenes.terrain.ensure_grid) before training on this one"
)


@dataclass(kw_only=True)
class SceneHeightfieldCfg(SubTerrainCfg):
    """The scene's grid as one hfield geom, in the scene's coordinates."""

    grid: Grid
    start_xy: tuple[float, float]
    friction: list[float] | None = None

    @classmethod
    def of(
        cls, grid: Grid, start_xy: tuple[float, float], friction: list[float] | None
    ) -> SceneHeightfieldCfg:
        half_x, half_y = grid.half
        return cls(
            grid=grid,
            start_xy=start_xy,
            friction=friction,
            size=(2 * half_x, 2 * half_y),
        )

    @property
    def corner(self) -> np.ndarray:
        """Where mjlab puts this patch's corner for a one-patch grid: the
        grid centred at the world origin (`TerrainGenerator`), so local
        positions are world positions minus this."""
        return np.array([-0.5 * self.size[0], -0.5 * self.size[1], 0.0])

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del difficulty, rng  # the scene is what it is
        grid = self.grid
        # the same asset, geom, group and friction the staged gate collides
        # with (scenes.terrain.add_heightfield), placed against the corner
        # mjlab's generator will add
        field, geom = add_heightfield(
            spec,
            spec.body(TERRAIN_BODY),
            grid,
            friction=self.friction,
            shift=tuple(-self.corner),
            group=TERRAIN_SCAN_GROUP,
        )
        start = np.array(self.start_xy, dtype=np.float64)
        surface_z = float(grid.at(start[None, :])[0])
        origin = np.array([start[0], start[1], surface_z]) - self.corner
        return TerrainOutput(
            origin=origin, geometries=[TerrainGeometry(geom=geom, hfield=field)]
        )


def scene_grid(scene_dir: Path, *, cell: float = TRAIN_CELL_M) -> Grid:
    """The scene's saved grid at the training spacing (`TRAIN_CELL_M`),
    resampled from the saved one when they differ."""
    saved = read_grid(scene_dir)
    if saved is None:
        raise FileNotFoundError(
            NO_GRID.format(name=Path(scene_dir).name, file=GRID_FILE)
        )
    grid, _filled, proxy = saved
    if proxy != proxy_hash(scene_dir):  # the gate would re-sample; say so
        raise ValueError(STALE_GRID.format(name=Path(scene_dir).name, file=GRID_FILE))
    return grid if grid.cell == cell else grid.resampled(cell)


def scene_start(
    scene_dir: Path, *, cell: float = TRAIN_CELL_M
) -> tuple[float, float, float]:
    """The course's start on the scene's surface, in world coordinates
    (the patch sits in the scene's own frame): where every world spawns,
    and where a robot's default pose must stand. At the world origin the
    garden's Go2 stood inside the scene, 142 contacts before its first
    reset, and mujoco_warp refused the model (2026-09-25)."""
    record = load_scene_record(Path(scene_dir) / SCENE_FILE)
    (x, y), _heading = course_start(record)
    z = float(scene_grid(scene_dir, cell=cell).at(np.array([[x, y]]))[0])
    return float(x), float(y), z


def scene_terrain_cfg(
    scene_dir: Path, *, cell: float = TRAIN_CELL_M
) -> TerrainEntityCfg:
    """The scene as mjlab's terrain: one patch, the grid at `cell`, the
    spawn at the course's start on the surface, the scene's declared
    friction."""
    scene_dir = Path(scene_dir)
    record = load_scene_record(scene_dir / SCENE_FILE)
    start_xy, _heading = course_start(record)
    patch = SceneHeightfieldCfg.of(
        scene_grid(scene_dir, cell=cell), start_xy, floor_friction(record)
    )
    return TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=patch.size,
            num_rows=1,
            num_cols=1,
            sub_terrains={SUB_TERRAIN: patch},
            curriculum=False,
            color_scheme="none",
        ),
        # the scene's own lights and materials: none; the picture is the splat
        textures=(),
        materials=(),
    )


def scans_see_the_scene(sensors: Any) -> None:
    """Point the rough recipe's height scans at the scene's ground
    (`TERRAIN_SCAN_GROUP`), in place; refuses a recipe that lacks them."""
    found = {s.name: s for s in sensors or () if s.name in SCAN_SENSORS}
    missing = set(SCAN_SENSORS) - set(found)
    if missing:
        raise ValueError(
            f"the scene walk's height scans are missing: {sorted(missing)}"
        )
    for sensor in found.values():
        sensor.include_geom_groups = (TERRAIN_SCAN_GROUP,)


def scene_stamp(scene_dir: Path) -> str:
    """The scene's version, as a staged deployment names it."""
    scene_dir = Path(scene_dir)
    return stamp(scene_dir.name, scene_dir)


def head_camera_cfg(
    robot: str, *, width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT
) -> CameraSensorCfg:
    """The robot's head camera as the stage mounts it (`scenes.heads`, by
    robot family), as an mjlab sensor rendered for every world."""
    mount = head_mount(robot)
    if mount.body is None:
        raise ValueError(f"robot {robot!r}'s head mount names no base body")
    quat = mount.quat
    return CameraSensorCfg(
        name=HEAD_CAMERA,
        parent_body=f"{ROBOT_ENTITY}/{mount.body}",
        pos=mount.pos,
        quat=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
        fovy=mount.fovy,
        width=width,
        height=height,
        data_types=("rgb",),
        use_textures=False,
        use_shadows=False,
        enabled_geom_groups=VISUAL_GROUPS,
    )


def camera_rgb(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """The camera's picture as the policy sees it: one row per world,
    the pixels flattened, in [0, 1]."""
    rgb = env.scene.sensors[sensor_name].data.rgb
    if rgb is None:
        raise ValueError(f"camera sensor {sensor_name!r} renders no rgb")
    return rgb.reshape(rgb.shape[0], -1).to(torch.float32) / 255.0


def camera_term(sensor_name: str = HEAD_CAMERA) -> ObservationTermCfg:
    return ObservationTermCfg(func=camera_rgb, params={"sensor_name": sensor_name})


# Past the scene's footprint there is no ground: a world that walks off
# the heightfield falls forever and its observations become NaN (the
# first g3 scene run died at iteration 1017 of 1500, 2026-09-23). mjlab's
# own bounds termination judges its origin-centred grid; the scene's
# footprint is the grid's, at the scene's coordinates.
BOUNDS_MARGIN_M = 0.3


def scene_footprint(env: Any) -> tuple[float, float, float, float] | None:
    """(x_min, x_max, y_min, y_max) of the scene's heightfield; None off
    a scene."""
    terrain = env.scene.terrain
    generator = None if terrain is None else terrain.cfg.terrain_generator
    patch = None if generator is None else generator.sub_terrains.get(SUB_TERRAIN)
    if not isinstance(patch, SceneHeightfieldCfg):
        return None
    grid = patch.grid
    half_x, half_y = grid.half
    return (grid.x0, grid.x0 + 2 * half_x, grid.y0, grid.y0 + 2 * half_y)


def out_of_scene_bounds(env: Any, margin: float = BOUNDS_MARGIN_M) -> torch.Tensor:
    """Truncate a world whose base left the scene's footprint (less the
    margin); all-false off a scene."""
    footprint = scene_footprint(env)
    if footprint is None:
        return torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    x_min, x_max, y_min, y_max = footprint
    xy = env.scene["robot"].data.root_link_pos_w[:, :2]
    return (
        (xy[:, 0] < x_min + margin)
        | (xy[:, 0] > x_max - margin)
        | (xy[:, 1] < y_min + margin)
        | (xy[:, 1] > y_max - margin)
    )


class RendersSplats:
    """mujoco_warp as mjlab's sensor context sees it, with the scene's
    gaussians added to every render context it creates; everything
    else is the module's own."""

    def __init__(self, module: Any, arguments: dict[str, np.ndarray]) -> None:
        self._module = module
        self._arguments = arguments

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def create_render_context(self, *args: Any, **kwargs: Any) -> Any:
        return self._module.create_render_context(*args, **kwargs, **self._arguments)


def render_splats(scene_dir: Path, *, context_module: Any = None) -> int:
    """Put the scene's visible gaussians into every camera mjlab renders
    from now on; returns how many. `context_module` is mjlab's
    `sensor_context` module (the default), the one place that builds
    render contexts."""
    if context_module is None:
        from mjlab.sensor import sensor_context as context_module  # noqa: PLC0415

    visible = read_ply(Path(scene_dir) / SPLAT_FILE).visible(VISIBLE_OPACITY)
    inner = getattr(context_module.mjwarp, "_module", context_module.mjwarp)
    context_module.mjwarp = RendersSplats(inner, splat_arguments(visible))
    return int(visible.means.shape[0])


def unrender_splats(*, context_module: Any = None) -> None:
    """Back to the module's own contexts (tests, and a plane after a scene)."""
    if context_module is None:
        from mjlab.sensor import sensor_context as context_module  # noqa: PLC0415

    inner = getattr(context_module.mjwarp, "_module", None)
    if inner is not None:
        context_module.mjwarp = inner
