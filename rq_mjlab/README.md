# rq_mjlab

Identified actuator physics for [mjlab](https://github.com/mujocolab/mjlab),
with refusals (docs/e2e-research/58 §2). BAM's servo law — the CPU
reference's equations, the same transcription the pipeline pinned
against BAM itself — as an mjlab 1.6 actuator that builds **only** from
the certified bundle store in `robots/actuator-bundles/`:

- `BamActuatorCfg.from_bundle(path, target_names_expr=…, physics_dt=…)`
  — verifies the bundle (a tampered stamp refuses by name), carries its
  advisories, fills the passives and the identified command delay, and
  records the deployment's `kp`/`vin` register choices explicitly.
- `rq_mjlab.events.bam_expansion_event()` — the per-world
  field-expansion event, one line in an env cfg; forgetting it fails
  loudly at the actuator's `initialize`, with the fix in the message.
- `rq_mjlab.lint(events_cfg, actuator_cfgs)` — refuses DR terms that
  randomise fields the law overwrites every step (the silent no-ops of
  docs/e2e-research/57 §5).

Its own package and venv on purpose: mjlab 1.6 pins `mujoco~=3.11` with
torch/CUDA — a third instrument beside the pipeline's two. `uv sync`
here; `uv run python -m unittest discover -s tests -t .` for the
CPU-side pins. Not yet ported, refused loudly: rate-limited firmwares
(the Feetech STS line); the GPU integration test and
`entity_from_bundle`/`dr_from_bundle` are the next slices.
