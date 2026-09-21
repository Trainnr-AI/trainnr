"""Captured scenes: a Gaussian splat the cameras see, a collision proxy
the solver touches, and the record that says how far apart they are
(docs/78 §3, docs/e2e-research/75).

`splat` reads and writes the splat itself; `record` is the artifact;
`gap` measures the visible surface against the proxy; `neverwhere`
imports the field's own benchmark scenes as ours.
"""
