# Datasheet

Every episode in this batch passed its task's own referee; the
counts below are of SUCCESSFUL episodes only.

- episodes kept: **128**
- highest attempt number recorded: 143 (keep rate ≤ 90% — attempts after the last keep are not recorded)
- expert retries across the batch: 0

## Stamps

- task: lift-study@5e95174a702a
- expert: scripted-pick@lift-study
- instrument: mujoco-3.11.0+x86_64

## Dynamics draws

Basis: a wide span that COVERS the cliff (gain 0.3-1.3): the honest alternative to guessing narrow

| parameter | low | mean | high |
|---|---|---|---|
| damping | 0.7143 | 1.001 | 1.298 |
| gain | 0.374 | 0.8328 | 1.296 |
