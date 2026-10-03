# The rented GPU: the provider seam and the runbook

*2026-08-27. The training run at recipe scale (docs/31 §5) needs a card
for four and a half hours; the development machine is a smoke box.
This page is how a card is rented: one seam, one vendor
behind it today, one command per step.*

## 1. The seam — providers by name, like engines

`trainnr/trainnr/cloud/provider.py` is the contract:

| Type | What it is |
|---|---|
| `GpuOffer` | one rentable GPU type at one provider — id, VRAM, tier, **price per card-hour**, live availability, the host CUDA versions on offer |
| `MachineSpec` | what to rent — image, GPU id, count, disk, tier (`SECURE` = the vendor's datacenter, `COMMUNITY` = a vetted host), env, whether sshd is published |
| `Machine` | what the vendor reports — status, cost/h, the actions it accepts, and two SSH doors: `direct` (its own sshd: rsync, port forwards) and `proxy` (a shell through the vendor) |
| `GpuProvider` | the Protocol: `offers`, `launch`, `machine(s)`, `act` (start/stop/restart/terminate), `logs` |

A vendor registers itself with `@provider("name")` and an entry point in
the `trainnr.gpu_providers` group — the engine registry's pattern
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

`trainnr/trainnr/cloud/runpod.py` speaks **REST API v2**
(`https://api.runpod.io/v2`; v1 retires 2026-11-15), read from its
OpenAPI schema (`/v2/openapi.json`), standard library only — the surface
we use is six calls and an SDK would pin us to its pace. What the
schema and the wire said:

- **Auth** is `Authorization: Bearer <key>`; a key carries the
  permissions it was created with, so a valid key can still get `403`.
- **Cloudflare fronts the API** and answers urllib's default agent
  string with `403 Error 1010: Access denied — browser signature`;
  `curl` and any named product token pass. Every request carries
  `User-Agent: trainnr/cloud-gpu` (pinned in `tests/test_cloud.py`).
- **The catalog** (`/v2/catalog/gpus?include=AVAILABILITY&product=POD&cloud=…&minCudaVersion=13.0`)
  prices per tier and reports stock as `NONE / LOW / MEDIUM / HIGH`.
  Our train venv is torch+cu130, so the **host driver must be CUDA
  ≥ 13** — the floor the tool queries with. Stock on 2026-08-27, one
  card, community tier (prices as of that date): RTX 3090 **$0.22/h**, RTX 4090 **$0.34/h** (CUDA 13.0
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
- **The account's registered SSH key** is the one `--ssh-key` names
  (fingerprints compared, 2026-08-27), so `startSsh` pods accept it.
- **Image**: `runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404` (Docker
  Hub, 2026-08-27) — CUDA 13, Ubuntu 24.04, sshd. The image's torch is
  irrelevant: the bootstrap builds our own `.venv-train` from the
  lockfile, exactly the development machine's recipe.
- **The first session, on a secure-tier B200** (192 CPU cores, CUDA
  13.0, a 50 GB persistent volume at `/workspace`), 2026-08-27 night,
  measured: the pod sat in "initializing" for **52 minutes** after
  `start` before its container ran (billing started at the container);
  `bootstrap` (apt + uv + the train venv, 128 packages) took **5 min**;
  the smoke chain there: demos **45 s/episode** (the development
  machine: 95), the PNG → AV1 convert **223 s/episode** (the development
  machine: 38, slower cores),
  ACT training **11.3 steps/s with 4 dataloader workers and 11.4 with
  16, the card at 17 %, slower than a 24 GB RTX 3090 Ti's 14.8**: the training
  loop itself is CPU-bound at batch 8 (per-step Python and kernel
  launches on slower server cores), so workers are not the lever and a
  bigger batch is: **batch 64 ran at 9.26 steps/s, 58 % GPU, 17 GB** —
  81 % of the batch-8 step rate with eight times the samples per step,
  6.5× the throughput (`e2e-smoke --batch`, with `--lr` scaled by the
  square root of the batch ratio; the preset keeps the recipe's 8 for
  comparability; the decision on 2026-08-27 was 64). The pod's volume
  refuses `chown` (rsync `-a` exits 23; the tool uses `-rlptD`). Runpod's
  UI telemetry lags the pod's own `nvidia-smi` by a few minutes.
- **An evaluation of N episodes needs N distinct starts.** The env
  maps `reset(seed=k)` to trial `k % trials`; the first cloud run asked
  `lerobot-eval` for ten episodes on the four-trial kitting spec and got
  the same four starts two and a half times (LeRobot's "3/10" was one
  start three times). `--env.trials=N` rebuilds the spec with N paired
  trials — the chain does it for every evaluation it launches — and
  LeRobot's padded last batch falls into a second pass of the fold.
- A **stopped pod still bills its disk**. The tool reports what the
  account holds; it never terminates on anyone's behalf.

## 3. The runbook — `tools/cloud-gpu.py`

Every step is a subcommand; every subcommand prints what it does.

```sh
cd trainnr
P="uv run --extra sim python ../tools/cloud-gpu.py"
$P offers --tier COMMUNITY                       # priced, in stock, CUDA >= 13 hosts
$P launch --name t5-cloud --gpu "NVIDIA GeForce RTX 4090" --tier COMMUNITY --wait
$P push <id>                                     # rsync the tree (no .env, no venvs, no runs/)
$P bootstrap <id>                                # apt + uv + .venv-train, then a torch/CUDA/mujoco check
$P run <id> -- ../tools/e2e-smoke.py --scale smoke --name t5-smoke   # prove the chain there
$P run <id> -- ../tools/e2e-smoke.py --scale cloud --name t5-cloud   # the real run (docs/31 §5)
$P pull <id> t5-cloud                            # runs/t5-cloud-* back onto this machine
$P terminate <id>                                # billing stops here
```

`machines`, `status <id> [--wait]`, `ssh <id>` (both doors, with the
key), `logs <id>`, `stop`/`start`/`restart` round it out; `--provider`
selects the vendor (`runpod` is the default and the only one), `--dotenv`
and `--ssh-key` move the two files it reads. `run
--deadline-min N` wraps the remote command in `timeout` (TERM, then
KILL a minute later): a budget the machine enforces, not a clock
someone watches; checkpoints written before it survive.

**Six EGL renderers on WSL livelock.** Six parallel `kitting-demos`
shards on the development machine's card each finished five episodes and then all
parked in `futex_do_wait` at the same minute (2026-08-27, 29 of 50
kept, nothing for 30 minutes at 98 % GPU). Three shards completed the
rest without incident. Until the driver path is understood, three
shards is that machine's ceiling; on a native-EGL Linux host with 192 cores
the same split has no such ceiling in principle, but is unmeasured.

**Watch it as it trains.** The chain writes a sidecar beside the
trainer's directory from its first second — `runs/<name>-watch/` with
the run manifest (every parameter, the dataset's bundle and expert
stamps, the command line; written at the start and again at the train
stage), every stage's output teed into `chain.log` as it happens (the
demo attempts and the convert included — a dashboard that opened at the
train stage showed an empty grid for minutes, 2026-08-28), and
`nvidia-smi` samples — beside, because the trainer refuses an output
directory that already exists. `train-watch.py --follow runs/<name>-act` streams those files
into Rerun: the manifest as a text panel from step 0, the metrics
lines as `train/*`, the in-loop and final records as `eval/*` (success
rate and the milestone funnel per checkpoint, the eval videos), the
GPU samples as `machine/*`, the trainer's resolved config at every
checkpoint. For a rented card: `cloud-gpu follow <id> <name> --watch`
mirrors the light files (never the weights) every 20 s and opens the
dashboard on the mirror — same files, same panels, either box.
`--rrd` saves the stream as one portable record.

**Split the chain by what each machine is good at.** The demos (a render
per control tick, 95 s/episode) and the PNG → AV1 conversion (38
s/episode) are CPU-bound and free on the development machine; training and the
evaluations are what a rented card is for. `kitting-demos.py
--first-episode K` lets N generators with disjoint ranges and seeds
fill one batch directory in parallel (six shards on 24 cores); `e2e-smoke.py --until convert` runs the chain up to the
dataset; then `push` + an rsync of `runs/<name>-lerobot` and
`e2e-smoke.py --scale cloud --from train --name <name>` on the
machine does only train + eval. The first run this way was 2026-08-27
night.

The remote layout is `Remote` in the tool: the repo at
`/workspace/trainnr`, the venv `trainnr/.venv-train` built with
`UV_PROJECT_ENVIRONMENT=.venv-train uv sync --python 3.12.8 --extra sim
--extra viz --extra train` (the development machine's own line), commands run as
`MUJOCO_GL=egl OMP_NUM_THREADS=1 .venv-train/bin/python …` — EGL is the
offscreen renderer on a bare Linux GPU box; none of WSL's Mesa
variables apply.

**Cost of the recipe run**, from docs/31 §5's measured rates on a
3090-class card, at the 2026-08-27 prices above: ~4.5 h → about **$1.50
on a community RTX 4090**, $1.00 on a 3090, $3.30 on a secure 4090. A
stopped pod keeps billing its
disk; `terminate` when the results are pulled.

## 4. The Claude Code side — the vendor's plugin

Runpod also ships a Claude Code plugin (router plus six skills and a
hosted MCP server, OAuth-authenticated) — `docs.runpod.io/agent-setup`.
It is installed by hand, not by a tool:

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
| Several runs on one card | The way to load a B200 with a small model (ACT at batch 64 used 58 % of it): N seeds or variants at once, each its own name — what a sweep and the certificate's seed variance both want. The tool launches one run per machine today |
| Multi-GPU / multi-node | ACT at batch 8 fits one card; LeRobot's trainer is single-process. The `count` field is there; the distributed launcher is not |
| Spot / interruptible pricing | Runpod's v2 pods are on-demand; a checkpoint every 20k steps (docs/31 §5) already bounds what an interruption costs |
| Automatic terminate on completion | Nothing here terminates on anyone's behalf: a finished run is pulled and inspected first, then the machine is released by hand |
