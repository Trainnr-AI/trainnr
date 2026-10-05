# The microduck's meshes are fetched, not carried

`robot_walk.xml` references 38 mesh files under `assets/`. trainnr does
not redistribute them. Pollen Robotics' repository is Apache-2.0, but its
README adds:

> 3D model files are licensed under Creative Commons BY-SA-NC.

Those terms forbid commercial use and are not an Apache-2.0 grant, so the
meshes stay with their publisher. `FETCH.json` names each one by its path
in `pollen-robotics/microduck_rl` at commit
`d424a0c899f6b33cbd3daeb279913134349c0b63` and by its git blob id.

The first time the microduck is used (a task, a training run, an
evaluation, the Studio), trainnr fetches the meshes into `assets/` from
that commit, checks each against its blob id and keeps nothing that does
not match. It says so on stderr, naming the source and the licence. The
fetched files are the bytes the bundle's stamp was taken over, so the
stamp (`microduck@ad90736153cc`) is the same once they are in place.

- Fetch ahead: `python -m trainnr.bundles.fetch robots/microduck`
- Refuse the fetch: `TRAINNR_NO_ROBOT_FETCH=1`
- Get them yourself: copy
  `src/mjlab_microduck/robot/microduck/assets/<name>.stl` from that commit
  into `assets/`, for each name `FETCH.json` lists.

Using the meshes means using them under Pollen Robotics' terms, not
under trainnr's licence.
