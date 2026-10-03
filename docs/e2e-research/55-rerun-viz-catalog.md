# Rerun's visualization catalog, mapped to the Studio

*Historical (2026-08-30). The split proposed in §3, native panels beside a
separately spawned `rerun` viewer, was superseded: the Studio embeds the
Rerun viewer (`rerun = 0.36.3` in `crates/trainnr-studio/Cargo.toml`) and
its pages draw the glance-level panels themselves (docs/35-the-studio.md).
The catalog in §1 and the logging inventory in §2 still hold.*

*2026-08-30, from Rerun's own reference docs (rerun.io/docs/reference/types,
fetched today) and crate metadata (docs.rs). The question: which of
Rerun's views the Studio needs, which it renders natively, and which it
delegates to the real Rerun viewer — so the app's data panels are
polished from the first pass instead of hand-rolled and replaced later.*

## 0. The one fact that settles the "native plots" question

**Rerun's own `TimeSeriesView` is built on `egui_plot` (^0.37) over
`egui` ^0.36.1** (docs.rs, `re_view_time_series`, checked 2026-08-30) —
the exact egui version `crates/trainnr-studio` already pins. A native
Studio chart drawn with `egui_plot` is not a hand-rolled imitation of
Rerun's charts; it is the same foundation Rerun's viewer itself draws
with. That makes the split below cheap on both sides.

## 1. The catalog (Rerun 0.36-era, from the reference)

**Views** (11): Spatial3D, Spatial2D, TimeSeries, BarChart, Tensor,
TextLog, TextDocument, Dataframe, Graph, Map, StateTimeline.

**Archetypes**, grouped: plotting (Scalars, SeriesLines, SeriesPoints,
BarChart, StateChange/StateConfiguration); spatial 3D (Points3D,
LineStrips3D, Boxes3D, Capsules3D, Cylinders3D, Ellipsoids3D, Arrows3D,
Mesh3D, Asset3D, InstancePoses3D, GridMap, VoxelGridMap,
**GaussianSplats3D**); spatial 2D (Points2D, LineStrips2D, Boxes2D,
Ellipses2D, Arrows2D); image (Image, EncodedImage, DepthImage,
EncodedDepthImage, SegmentationImage, Tensor); video (AssetVideo,
VideoStream, VideoFrameReference); text (TextLog, TextDocument);
transforms (Transform3D, Pinhole, CoordinateFrame, TransformAxes3D,
ViewCoordinates); graph (GraphNodes/Edges); geospatial (GeoPoints,
GeoLineStrings); **MCAP** (Channel/Message/Schema/Statistics);
bookkeeping (AnnotationContext, Clear, RecordingInfo).

Three entries earn a note against earlier research: `GaussianSplats3D`
is first-class now (the splat capture layer of the scene loop, docs/78-the-scene-loop.md,
has a native display path waiting); the MCAP archetypes land exactly
where docs/e2e-research/25-deployment-and-fleet-ops.md pointed for fleet logging; and
`StateTimeline` is purpose-built for the stage/milestone bands the
funnel already produces.

## 2. What this repo already logs, view by view

The repo is not starting from zero — three dashboards exist and name
their panels:

| Repo surface | Rerun views it exercises |
|---|---|
| `train-watch --follow` | TimeSeries (loss, l1, kld, grad norm, lr, samples/s, GPU), TextDocument (run manifest, trainer config), TextLog (stage), Image/video (eval episodes), the funnel per checkpoint |
| `rig-rerun.py` / `trainnr/viz.py` | Spatial3D (Boxes3D batch + Mesh3D twin), Spatial2D (pose trail), TimeSeries (duty, ticks, angles, errors), TextLog (stage notes) |
| `rl-watch` (`tools/rl-watch.py`) | its blueprint names verdict, reward terms, episode, losses, throughput, worlds grid |
| `show-many.py` | batched DR worlds side by side (Spatial3D grid) |

## 3. The split: native panel vs the real viewer

**Native (egui_plot in trainnr-studio)** — always-on, lightweight, lives
beside the viewport and agent panel:

- training loss / success-rate curves for a followed run (TimeSeries
  equivalent; `egui_plot` line + points)
- the eval funnel (BarChart equivalent; `egui_plot` bar chart)
- run stage / milestone band (StateTimeline equivalent; a colored strip
  is a dozen lines of egui)
- scalar readouts (tokens, steps/s, GPU util) — plain `re_ui` labels

**Delegated to the Rerun viewer** (spawn `rerun` beside the app — the
launcher pattern docs/54 recommended; the data is already logged there
by the existing tools):

- the full training dashboard (train-watch's 88-entity blueprint —
  rebuilding it natively would be rebuilding Rerun's viewer)
- episode replay with cameras, depth, segmentation
- point clouds, splats, tensor inspection, dataframe queries
- anything with a timeline scrubber

**Not built at all** (no current consumer): Map/geospatial, Graph.

The rule that falls out: **a native panel is for glanceable state while
you work; the Rerun viewer is for investigation.** The Studio never
re-implements a Rerun view that has a scrubber, a query, or a camera.

## 4. What it costs

`egui_plot` is one dependency (MIT/Apache-2.0, same family), version
matched to the egui already pinned. The Rerun-viewer path costs nothing
new — `train-watch` and `rig-rerun` already do it; the Studio's job is a
button that launches them pointed at the right run.
