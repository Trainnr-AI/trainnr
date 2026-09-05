# Datasheet

Every episode in this batch passed its task's own referee; the
counts below are of SUCCESSFUL episodes only.

- episodes kept: **64**
- highest attempt number recorded: 64 (keep rate ≤ 100% — attempts after the last keep are not recorded)
- expert retries across the batch: 0

## Stamps

- task: lift-study@5e95174a702a
- expert: scripted-pick@lift-study
- instrument: mujoco-3.11.0+x86_64

## Dynamics draws

Basis: declared span ±0.30 around nominal (the folklore DR, docs/31)

| parameter | low | mean | high |
|---|---|---|---|
| damping | 0.7019 | 1.004 | 1.297 |
| gain | 0.7038 | 0.9878 | 1.299 |

## Visual draws

Basis: declared visual span: headlight 0.5-1.5, front camera ±1 cm (docs/66 §4; the so101 scenes are lit by the headlight only, docs/07 2026-09-02)

| knob | low | mean | high |
|---|---|---|---|
| front.offset_m[0] | -0.009852 | -0.0005981 | 0.009982 |
| front.offset_m[1] | -0.009895 | 0.0004177 | 0.009844 |
| front.offset_m[2] | -0.009006 | 0.0006071 | 0.009705 |
| headlight.diffuse_scale | 0.502 | 0.9246 | 1.493 |
