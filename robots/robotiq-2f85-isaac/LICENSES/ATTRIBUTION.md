# Licences of this bundle

This bundle is converted from `Robotiq_2F_85.usda` in
[robotiq/isaacsim_assets](https://github.com/robotiq/isaacsim_assets) at
commit `6d992b6`. That repository is dual-licensed, and so is this bundle:

- **Robotiq's own content**, including the CAD visual meshes in
  `assets/*_visual.obj`: BSD 3-Clause, Copyright (c) 2026, ROBOTIQ. The full
  text, with Robotiq's own attribution section, is
  `LICENSE-robotiq-isaacsim_assets.txt` beside this file.
- **The portions Robotiq adapted from NVIDIA's Isaac Sim asset** (the
  gripper's articulation, physics and materials): Creative Commons
  Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/.
  Original author: NVIDIA Corporation. Modifications: Copyright (c) 2026,
  ROBOTIQ. The licence text is `CC-BY-4.0.txt` beside this file, and the
  bundle's own `LICENSE`.

**Changes made by trainnr** (CC BY 4.0 §3(a)(1)(B)): the USD was read by
Newton's importer (newton 1.6.0, usd-core 26.3) and written as MJCF
(`robotiq-2f85-isaac.xml`); prim paths were renamed to leaf names; the inline
convex hulls were moved into `assets/*_hull.obj`; joint position and velocity
sensors, a `home` keyframe and each servo's control range were added. The
rules are in docs/e2e-research/77-usd-import-2026-09.md §6.

"Robotiq" and "2F-85" are Robotiq's; this bundle is not made or endorsed by
Robotiq or NVIDIA.

These texts sit in `LICENSES/`, which the bundle's version stamp leaves out
(`trainnr/trainnr/bundles/hashing.py`, `BUNDLE_RECORDS`), so correcting an
attribution does not change which robot the bundle is.
