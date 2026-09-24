# robotiq-2f85-isaac — imported from USD

Source: `Robotiq_2F_85.usda` (robotiq/isaacsim_assets @ 6d992b6), variants Fingertip=Standard, Physics=Newton_compliant, licence CC-BY-4.0 (the upstream text is `LICENSE`).

Read by Newton's USD importer and bridged to MuJoCo by its solver (newton 1.6.0, usd-core 26.3); the bundle writer renamed prim paths to leaf names, moved the inline convex hulls (64 vertices at most) into `assets/*_hull.obj` beside the USD's own visual meshes (`assets/*_visual.obj`), added jointpos and jointvel sensors, a `home` keyframe, and each position servo's joint range as its control range. docs/e2e-research/77 §6 names every rule. Contact type and affinity masks on the hulls are Newton's encoding of the USD's collision filters, kept as written.

Root: fixed. Runtime grip options declared: True.

Audited against the USD layer (`bundle.json` → `audit`, the robot card's "What the importer changed"): every mass, centre of mass, inertia tensor, joint range and joint parameter equal; 7 explained changes (the two spherical loop closures written as connect equalities, joints ordered by the kinematic tree, a hull beside each visual, the sensors and the home key added, degrees to radians), 0 unexplained.

What the USD carries that the bundle does NOT: the mimic joint's compliance (natural frequency, damping ratio) — Newton writes a rigid joint equality; PhysX-only attributes. Newton's own solver defaults were reset to MuJoCo's.
