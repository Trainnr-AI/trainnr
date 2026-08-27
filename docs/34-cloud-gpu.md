# 34. Rented GPUs — the provider seam and the runbook

*2026-08-27. The training run at recipe scale (docs/31 §5) needs a card
for four and a half hours; the WSL box is a smoke box (docs/07
2026-08-26). This page is how a card is rented: one seam, one vendor
behind it today, one command per step.*

## 1. The seam — providers by name, like engines

`pipeline/rq_pipeline/cloud/provider.py` is the contract:

| Type | What it is |
|---|---|
| `GpuOffer` | one rentable GPU type at one provider — id, VRAM, tier, **price per card-hour**, live availability, the host CUDA versions on offer |
| `MachineSpec` | what to rent — image, GPU id, count, disk, tier (`SECURE` = the vendor's datacenter, `COMMUNITY` = a vetted host), env, whether sshd is published |
| `Machine` | what the vendor reports — status, cost/h, the actions it accepts, and two SSH doors: `direct` (its own sshd: rsync, port forwards) and `proxy` (a shell through the vendor) |
| `GpuProvider` | the Protocol: `offers`, `launch`, `machine(s)`, `act` (start/stop/restart/terminate), `logs` |

A vendor registers itself with `@provider("name")` and an entry point in
the `rq_pipeline.gpu_providers` group — the engine registry's pattern
(`physics/registry.py`), applied to compute. `resolve("runpod")` today;
a second vendor is one module and one line in `pyproject.toml`. The
package sits at layer 1 (`tests/test_layers.py`): standard library HTTP
over the plugin door, nothing from the pipeline above it.

**Credentials.** `credential(env_var)` reads the environment and refuses
by *name* when the variable is unset. The repo's `.env` (git-ignored)
holds `RUNPOD_API_KEY`; `tools/_lab.py::load_dotenv` puts it in the
environment; the key goes into one `Authorization` header and nowhere
else — never a URL, never a log, never a manifest — and `cloud-gpu push`
excludes `.env` from what it ships. A test pins the header rule.

## 2. Runpod, as measured 2026-08-27

`pipeline/rq_pipeline/cloud/runpod.py` speaks **REST API v2**
(`https://api.runpod.io/v2`; v1 retires 2026-11-15), read from its
OpenAPI schema (`/v2/openapi.json`), standard library only — the surface
we use is six calls and an SDK would pin us to its pace. What the
schema and the wire said:

- **Auth** is `Authorization: Bearer <key>`; a key carries the
  permissions it was created with, so a valid key can still get `403`.
- **Cloudflare fronts the API** and answers urllib's default agent
  string with `403 Error 1010: Access denied — browser signature`;
  `curl` and any named product token pass. Every request carries
  `User-Agent: rq-pipeline/cloud-gpu` (pinned in `tests/test_cloud.py`).
- **The catalog** (`/v2/catalog/gpus?include=AVAILABILITY&product=POD&cloud=…&minCudaVersion=13.0`)
  prices per tier and reports stock as `NONE / LOW / MEDIUM / HIGH`.
  Our train venv is torch+cu130, so the **host driver must be CUDA
  ≥ 13** — the floor the tool queries with. Tonight's stock, one card,
  community tier: RTX 3090 **$0.22/h**, RTX 4090 **$0.34/h** (CUDA 13.0
  and 13.2 hosts), RTX 5090 $0.69/h; secure tier: A40 $0.44/h, RTX 4090
  $0.74/h, L40S $0.99/h, A100 80 GB SXM $1.59/h. 47 types listed, 15
  in stock with a CUDA-13 host on community, 16 on secure.
- **A pod** is created with `name, image, gpu{id,count}, disk, cloud,
  ports, startSsh, env` (`POST /v2/pods`), transitions through
  `PROVISIONING → STARTING → RUNNING → EXITED`, accepts
  `start/stop/restart` on `/action` and `DELETE` to terminate. Its
  `ssh.direct` door appears only once `22/tcp` is published and the
  machine assigned; `ssh.proxy` is a shell only (no rsync).
- **Logs** are a `text/event-stream`; the provider reads the `data:`
  lines.
- **The account's registered SSH key is this box's `~/.ssh/id_ed25519`**
  (fingerprints compared, 2026-08-27), so `startSsh` pods accept it.
- **Image**: `runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404` (Docker
  Hub, 2026-08-27) — CUDA 13, Ubuntu 24.04, sshd. The image's torch is
  irrelevant: the bootstrap builds our own `.venv-train` from the
  lockfile, exactly the WSL box's recipe.
- The account held a **stopped B200 pod** (`$6.79/h` when running; a
  stopped pod bills its disk) when the tool first listed it. The tool
  reports; it never terminates on anyone's behalf.

## 3. The runbook — `tools/cloud-gpu.py`

Every step is a subcommand; every subcommand prints what it does.

```sh
cd pipeline
P="uv run --extra sim python ../tools/cloud-gpu.py"
$P offers --tier COMMUNITY                       # priced, in stock, CUDA >= 13 hosts
$P launch --name t5-cloud --gpu "NVIDIA GeForce RTX 4090" --tier COMMUNITY --wait
$P push <id>                                     # rsync the tree (no .env, no venvs, no runs/)
$P bootstrap <id>                                # apt + uv + .venv-train, then a torch/CUDA/mujoco check
$P run <id> -- ../tools/e2e-smoke.py --scale smoke --name t5-smoke   # prove the chain there
$P run <id> -- ../tools/e2e-smoke.py --scale cloud --name t5-cloud   # the real run (docs/31 §5)
$P pull <id> t5-cloud                            # runs/t5-cloud-* back onto this box
$P terminate <id>                                # billing stops here
```

`machines`, `status <id> [--wait]`, `ssh <id>` (both doors, with the
key), `logs <id>`, `stop`/`start`/`restart` round it out; `--provider`
selects the vendor (`runpod` is the default and the only one), `--dotenv`
and `--ssh-key` move the two files it reads. `run
--deadline-min N` wraps the remote command in `timeout` (TERM, then
KILL a minute later): a budget the machine enforces, not a clock
someone watches; checkpoints written before it survive.

**Split the chain by what each box is good at.** The demos (a render
per control tick, 95 s/episode) and the PNG → AV1 conversion (38
s/episode) are CPU-bound and free on the WSL box; training and the
evaluations are what a rented card is for. `kitting-demos.py
--first-episode K` lets N generators with disjoint ranges and seeds
fill one batch directory in parallel (six shards on the box's 24
cores); `e2e-smoke.py --until convert` runs the chain up to the
dataset; then `push` + an rsync of `runs/<name>-lerobot` and
`e2e-smoke.py --scale cloud --skip-convert --name <name>` on the
machine does only train + eval. The first run this way, 2026-08-27
night, is in docs/07.

The remote layout is `Remote` in the tool: the repo at
`/workspace/robotiq`, the venv `pipeline/.venv-train` built with
`UV_PROJECT_ENVIRONMENT=.venv-train uv sync --python 3.12.8 --extra sim
--extra viz --extra train` (the WSL box's own line), commands run as
`MUJOCO_GL=egl OMP_NUM_THREADS=1 .venv-train/bin/python …` — EGL is the
offscreen renderer on a bare Linux GPU box; none of WSL's Mesa
variables apply.

**Cost of the recipe run**, from docs/31 §5's measured rates on a
3090-class card: ~4.5 h → about **$1.50 on a community RTX 4090**, $1.00
on a 3090, $3.30 on a secure 4090. A stopped pod keeps billing its
disk; `terminate` when the results are pulled.

## 4. The Claude Code side — the vendor's plugin

Runpod also ships a Claude Code plugin (router plus six skills and a
hosted MCP server, OAuth-authenticated) — `docs.runpod.io/agent-setup`.
It is installed by the operator, not by a tool:

```
claude plugin marketplace add runpod/runpod-plugins-official
claude plugin install runpod@runpod
claude plugin list | grep -A2 'runpod@runpod'     # Status: ✔ enabled
```

then `/reload-plugins`, and `/mcp` → **runpod** → *Sign in with Runpod*.
The plugin is a convenience for the agent; the seam above is what the
pipeline depends on, and it needs only the API key.

## 5. Not built, and why

| Item | Why it waits |
|---|---|
| A second vendor | The seam is the deliverable; a vendor is added when a run needs one (price, stock, a region). Lambda, Vast, or a hyperscaler each fit the Protocol as written |
| Network volumes / a shared dataset store | The T5 batch is ~4.5 GB at 50 episodes and travels by rsync in minutes; a volume earns its keep when batches are reused across pods |
| Multi-GPU / multi-node | ACT at batch 8 fits one card; LeRobot's trainer is single-process. The `count` field is there; the distributed launcher is not |
| Spot / interruptible pricing | Runpod's v2 pods are on-demand; a checkpoint every 20k steps (docs/31 §5) already bounds what an interruption costs |
| Automatic terminate on completion | Nothing here terminates on the operator's behalf: a finished run is pulled and inspected first, then the machine is released by hand |
