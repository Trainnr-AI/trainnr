"""Captured scenes: a Gaussian splat the cameras see, a collision proxy
the solver touches, and the record that says how far apart they are
(docs/78 §3, docs/e2e-research/75).

`splat` reads and writes the splat itself; `record` is the artifact;
`gap` measures the visible surface against the proxy; `obj` parses and
writes meshes; `proxy` decomposes the proxy into convex parts;
`terrain` makes the proxy a terrain the solver touches; `stage` puts a
deployment on a scene and `heads` says where its head camera rides;
`cameras` renders the splat for the stage's cameras; `neverwhere`
imports the field's own benchmark scenes as ours; `capture` makes a
scene from a phone video.
"""
