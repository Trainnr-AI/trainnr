# microduck — Pollen Robotics' walking robot, exactly as they train it

The flagship's robot (docs/e2e-research/63 G1): `robot_walk.xml` from
`pollen-robotics/microduck_rl`, fetched byte-identical from branch
`develop` at commit `d424a0c899f6b33cbd3daeb279913134349c0b63`
(2026-08-27; cloned 2026-09-01), Apache-2.0 — upstream `LICENSE`
travels with it. The `assets/` here are exactly the 38 mesh files the
walk model references (their repo's assets directory also carries CAD
`.part` sources and other variants' meshes; the bundle carries what
runs). The model itself is onshape-to-robot output; the Onshape
document URL is in the XML's own header comment.

**Which variant, and why.** Their repo ships seven robot XMLs
(`walk`, `allcollisions`, `rollers`, and `_backlash` twins — the
"rival variants" catalogued in docs/e2e-research/57 §5).
`robot_walk.xml` is what `MICRODUCK_WALK_XML` names and the walk task
loads: the production walking recipe's model. The backlash twin is a
post-processor output (`add_backlash.py`) and can be regenerated; the
flagship starts from the base model their deployment story rests on.

**What the file itself declares** (the flagship's opening exhibit,
measured at wrap): the `<default>` class says `kp=50 dampratio=1`,
but every one of the 14 actuators overrides it to `kp=0.55`, and every
servo dof carries `armature=0.0018` — recognizably BAM's fitted XL330
armature (0.0018077, `xl330.m6` in our store), baked into the XML as
bare numbers with no fit id, no date, no hash. The identified values
ARE here; nothing says where they came from or which fit produced
them. In their training stack the position servos are then REPLACED at
runtime by BAM actuator classes whose parameters resolve from a laptop
path; in ours the same replacement comes from the certified `xl330.m6`
bundle with a stamp (docs/e2e-research/63 §1).

Nothing here is modified. A wrapper XML (sensors, scene) will sit
BESIDE this file when the flagship's env lands, the same rule as
`so101-nominal`: the upstream bytes stay upstream.

Onboarded before the importer audit existed (2026-09-24), so its record carries none and the robot card says "not audited"; `tools/audit-bundle.py` against robot_walk.xml finds nothing changed (an MJCF bundle is a copy of its source).
