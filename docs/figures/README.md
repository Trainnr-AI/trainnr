# The figures

Every figure is drawn from a record or by a tool; none is a screenshot
edited by hand.

- `<record-id>.{svg,pdf,png,csv}` at the top level: one record's figure,
  drawn by `python3 tools/finding-figure.py <record-id>` from the
  record's outcome. The CSV beside it is the plotted data. Embedded by
  `docs/68-findings.md` and the paper's figure list.
- `paper/`: the paper's designed figures (`tools/paper-figures.py`,
  `tools/paper-lift-stages.py`, `tools/paper-walk-stages.py`) and
  `paper/stills/`, the robot stills rendered from the simulator by
  `tools/paper-stills.py`. Embedded by `docs/paper/manuscript.md`.
- `attribution/`, `viewport-deploy/`: the deployment gate's attribution
  and trial pictures, cited by the Go2 records and `docs/77-the-unitree-loop.md`.
- `go2/`: the Studio after the Go2 loop closed, the README's picture.
- `go2-sysid/`: the identification runs on the public Go2 logs, cited by
  `go2-legged-fit-public-logs-2026-09-24`.
- `gripper-pick/`, `usd-import/`: the gripper imported from USD, cited by
  `docs/e2e-research/77-usd-import-2026-09.md` and the import tools.

A figure's caption lives in `docs/paper/captions.json`, keyed by the
record id; `tools/paper-figure-sheet.py` writes the contact sheet
`docs/paper/figure-sheet.html` from both.
