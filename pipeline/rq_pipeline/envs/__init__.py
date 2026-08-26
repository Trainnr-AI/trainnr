"""Our tasks behind the ecosystem's environment interface (docs/32).

`rq_pipeline.envs.robotiq` is the gymnasium env (sim extra);
`rq_pipeline.envs.lerobot_plugin` registers it with LeRobot (train extra).
`--env.discover_packages_path=rq_pipeline.envs` imports every module of
this package, which is how LeRobot finds the plugin. This init imports
neither, so the sim venv can use the env without LeRobot.
"""
