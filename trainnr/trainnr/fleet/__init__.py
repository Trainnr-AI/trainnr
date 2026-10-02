"""The loop's last stage: what happens after a policy is deployed.

`drift` judges fresh telemetry against the identified interval (docs/76
§9). The fleet data plane docs/30 §3.3 designed — telemetry always,
interventions on event, batches — is a service; this package is the one
record per check an agent runs when it decides to.
"""
