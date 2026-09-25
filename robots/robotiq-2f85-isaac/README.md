# robotiq-2f85-isaac — imported from USD

Source: `Robotiq_2F_85.usda` (robotiq/isaacsim_assets @ 6d992b6), variants Fingertip=Standard, Physics=Newton_compliant, licence CC-BY-4.0 (the upstream text is `LICENSE`).

Read by Newton's USD importer and bridged to MuJoCo by its solver (newton 1.6.0, usd-core 26.3); the bundle writer renamed prim paths to leaf names, moved the inline convex hulls (64 vertices at most) into `assets/*_hull.obj` beside the USD's own visual meshes (`assets/*_visual.obj`), added jointpos and jointvel sensors, a `home` keyframe, and each position servo's joint range as its control range. docs/e2e-research/77 §6 names every rule. Contact type and affinity masks on the hulls are Newton's encoding of the USD's collision filters, kept as written.

Root: fixed. Runtime grip options declared: True.

A mimic's softness is carried when the USD authors it in MuJoCo's words (`mjc:solref`, `mjc:solimp`); Newton's bridge wrote MuJoCo's defaults. What the USD carries that the bundle does NOT: a PhysX mimic's natural frequency and damping ratio (no exact MuJoCo equivalent is claimed); PhysX-only attributes. Newton's own solver defaults were reset to MuJoCo's.
