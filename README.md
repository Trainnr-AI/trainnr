# robotiq

A Rust-first journey from zero hardware knowledge to autonomous robots.

## Vision

Build robots end-to-end in Rust — starting from a simulated robot on a laptop and
climbing, stage by stage, to real machines: simple toys, robot pets, and eventually
autonomous working machines (inspection rovers, delivery pods, autonomous service
vehicles) driven by Vision-Language-Action (VLA) models.

The end goal is **end-to-end Rust**, but the learning path deliberately touches the
whole ecosystem — C++ (the incumbent), Python (the ML/training side), Rust (the
backbone), and WebAssembly (UIs, digital twins, sandboxed robot behaviors) — because
real robotics work means interoperating with all of them.

## Operator background

Strong software engineer. New to hardware and electronics. Learning by building:
every stage must produce something that actually runs.

## The staged roadmap

| Stage | What | Where it runs | Cost | Teaches |
|-------|------|--------------|------|---------|
| 0 | 2D simulated differential-drive robot | Laptop (Rust) | $0 | Control loops, PID, kinematics, odometry, state machines |
| 1 | Perception pipeline with webcam | Laptop (Rust) | $0 | Vision models, perception→decision→action pipeline |
| 2 | First microcontroller firmware | Pico 2 / ESP32 | ~$10–30 | GPIO, PWM, I2C/SPI, motors, electronics basics |
| 3 | First real mobile robot | Chassis + MCU + laptop brain | ~$100–150 | Real-world control, drift, sensor noise, teleop→autonomy |
| 4 | Onboard AI compute | Pi 5 + Hailo, or Jetson Orin Nano | ~$110–400 | Two-tier architecture, onboard perception, SLAM |
| 5 | Arm + VLA | SO-101 leader+follower pair | ~$300–450 | Teleop data collection, VLA fine-tuning, language-conditioned action |

Each stage reuses the previous stage's code. The Stage 0 simulator remains the
permanent test bed — real robotics teams work exactly this way.

The product ladder — toy → pet → inspection rover → delivery pod — is Stages 3→5,
then the same two-tier architecture (real-time microcontroller + AI compute) scaled
up with safety, redundancy, and better actuators.

## Ecosystem map (who plays what)

- **C++** — the incumbent. ROS 2 core, Gazebo, sensor drivers, OpenCV/PCL, Autoware.
  We need reading fluency and FFI/interop skills, not authorship.
- **Python** — the glue and the ML side. PyTorch, LeRobot, and every VLA training
  pipeline. Robots run Rust; their brains are *trained* in Python.
- **Rust** — the backbone. `embassy` (microcontrollers), Zenoh (middleware, an
  official ROS 2 transport), dora-rs (Rust-native dataflow), `candle`/`ort`/`burn`
  (inference), Rerun (visualization).
- **WebAssembly** — teleop dashboards and monitoring UIs (Rust→Wasm), browser-based
  digital twins, and sandboxed hot-swappable behavior plugins on the robot itself.

Strategy: build our own stack Rust-native (dora-rs / Zenoh) as the main line, with
one deliberate detour through vanilla ROS 2 (Python/C++) around Stage 3 — it is the
industry lingua franca and every sensor assumes it.

## Key decisions (validated by web research, July 2026)

- **Stage 0 sim (revised — see [docs/06-stage0-design.md](docs/06-stage0-design.md)):**
  a pure, deterministic, tested `sim-core` crate — no game engine, no physics
  engine; we write the kinematics ourselves — visualized through **Rerun**.
  Bevy becomes an optional later frontend (wasm+WebGPU shareable demo). Adopt
  **MuJoCo** (first-class on macOS, has SO-101 models in its menagerie) as the
  "real physics" engine when manipulation starts.
- **First board:** **Raspberry Pi Pico 2 W ($7) + Pi Debug Probe ($12)** — best
  embassy/probe-rs support; E9 erratum fixed in A4 silicon. ESP32-C6 second.
- **MCU ↔ computer link:** `postcard-rpc` over USB (no Rust micro-ROS exists).
- **Rust middleware:** **dora-rs** (1.0-rc, LeRobot/VLA node ecosystem) as main
  line; **Copper** (1.0, deterministic) and **Zenoh** (Tier-1 ROS 2 middleware
  since May 2025) as the other pillars. OpenRR is dormant — don't adopt.
- **ROS 2:** learn **Lyrical Luth LTS** (May 2026) in Docker + Foxglove/
  Lichtblick; native macOS ROS 2 is Tier 3 — don't fight it.
- **VLA path:** LeRobot + SO-101 pair + **SmolVLA** (fine-tunes for $0–10 on
  Colab/rented GPU; runs on Jetson Orin Nano with TensorRT). Training stays
  Python; Rust does robot I/O, perception plumbing, safety.
- **Compute buying:** a DRAM shortage repriced everything in 2025–26 (Jetson
  Orin Nano Super $249→$399, July 2026). Buy compute only when the stage
  demands it; Pi 5 + Hailo 26-TOPS HAT (~$205) covers detection workloads at
  half the Jetson price; VLA inference can run on a PC GPU over WiFi at first.

## Knowledge base

Detailed, dated research on the current (mid-2026) state of each layer lives in
[`docs/`](docs/):

- **[`docs/17-one-page.md`](docs/17-one-page.md) — START HERE: the whole system in one
  view, with pointers into everything else**
- [`docs/00-roadmap.md`](docs/00-roadmap.md) — the staged plan in full detail
- [`docs/01-rust-robotics-stack.md`](docs/01-rust-robotics-stack.md) — dora-rs, Zenoh, Copper, ros2-rust, Rerun
- [`docs/02-embedded-rust.md`](docs/02-embedded-rust.md) — Embassy, Pico 2/RP2350, ESP32, tooling, MCU↔host link
- [`docs/03-vla-robot-learning.md`](docs/03-vla-robot-learning.md) — LeRobot, VLA models, Rust inference
- [`docs/04-hardware.md`](docs/04-hardware.md) — compute boards, sensors, budgets, shopping plan per stage
- [`docs/05-simulation-ros2-wasm.md`](docs/05-simulation-ros2-wasm.md) — simulators, ROS 2 state, Wasm in robotics
- [`docs/06-stage0-design.md`](docs/06-stage0-design.md) — Stage 0 red-team critique + revised design
- [`docs/07-progress-log.md`](docs/07-progress-log.md) — dated log of everything done and decided
- [`docs/08-hardware-sim.md`](docs/08-hardware-sim.md) — emulating a Pico before owning one
- [`docs/09-shopping-list.md`](docs/09-shopping-list.md) — what to buy, tiered
- [`docs/10-hil-protocol.md`](docs/10-hil-protocol.md) — hardware-in-the-loop wire format
- [`docs/11-perception-stack.md`](docs/11-perception-stack.md) — camera, detectors, measured limits
- [`docs/12-model-choice.md`](docs/12-model-choice.md) — fixed-class vs open-vocab vs VLM, and licences
- [`docs/13-architecture-review.md`](docs/13-architecture-review.md) — DRY/composability review + fixes
- [`docs/14-model-landscape.md`](docs/14-model-landscape.md) — detector options, **measured on this laptop**
- [`docs/15-testing-and-coverage.md`](docs/15-testing-and-coverage.md) — test suite, coverage, LOC
- [`docs/16-the-map.md`](docs/16-the-map.md) — every symbol, unit and formula, and how
  the maths, the physics and the code connect
- [`docs/17-one-page.md`](docs/17-one-page.md) — the whole system on one page
- [`recordings/README.md`](recordings/README.md) — recorded sessions, and how replay
  turns one into a regression test
- [`docs/learning/`](docs/learning/) — Rust walkthroughs of the code we write, plus exercises
  - [`math-00-symbol-decoder.md`](docs/learning/math-00-symbol-decoder.md) — what every
    maths symbol means, in plain words. Start here if formulas look alien.
  - [`hw-00-microcontroller-decoder.md`](docs/learning/hw-00-microcontroller-decoder.md) —
    the chip: pins, boot, GPIO/UART/I2C, embassy, and where the maths meets the metal.

## Repository layout

```
robotiq/
├── README.md
├── docs/                    # knowledge base (research, decisions, learning, log)
│   └── learning/            #   Rust walkthroughs + math lessons
├── Cargo.toml               # laptop workspace
├── crates/
│   ├── sim-core/            # robot math + simulator. Builds twice:
│   │                        #   std = full sim; no_std = the MCU subset
│   │                        #   exercises.rs — YOUR code goes there
│   ├── sim-run/             # the mission, and the Rerun viewer
│   ├── vision/              # camera, detectors, target lock, the chase loop
│   ├── hil-protocol/        # the host↔chip wire format, one definition
│   ├── hil-host/            # simulated body for a real chip; record/replay
│   ├── mpu6050-driver/      # IMU driver (host-tested against a mock bus)
│   └── quad-encoder/        # quadrature decoding
├── firmware/                # no_std, ARM target — outside the workspace
│   ├── build-support/       #   one copy of the linker scripts + build.rs
│   ├── pico-blink/          #   H0  async tasks
│   ├── pico-button/         #   H1  input + PWM
│   ├── pico-imu/            #   H2  I2C sensor
│   ├── pico-encoder/        #   H3  encoder decoding
│   ├── pico-odom/           #   L1  sim-core's odometry, on the chip
│   ├── pico-led/            #   the CYW43 onboard LED (a real Pico 2 W)
│   ├── pico-selftest/       #   the shared maths on real silicon, diffed
│   └── pico-robot/          #   the controller: UART (emulator) or USB (real)
├── recordings/              # committed sessions that replay as regression tests
└── tools/
    ├── verify.sh            # EVERYTHING, end to end, one command
    ├── coverage.sh          # coverage + lines-of-code report
    ├── check-docs.py        # do the docs describe code that exists?
    ├── build-robot.sh       # pico-robot for BOTH chips, together
    ├── setup-emulator.sh    # clones + patches wokwi/rp2040js (not committed)
    ├── harness/             # our emulator harnesses (virtual sensors, tests)
    ├── patches/             # our fix to the upstream emulator
    └── sim-*.sh             # build → UF2 → emulate, one command each
```

## Running it

**Stage 0 — the laptop robot** (needs [Rust](https://rustup.rs) and the
[Rerun viewer](https://rerun.io) 0.35):

```sh
tools/verify.sh             # everything: fmt, clippy, 333 tests, firmware,
                            # both replay fixtures, and the emulator HIL run
cargo run -p sim-run        # watch the robot map, plan and drive (Rerun window)
```

**Stage 2 — firmware** (needs Node ≥18 and `cargo install elf2uf2-rs`):

```sh
tools/setup-emulator.sh     # one-time: clone rp2040js, patch it, install harnesses
tools/sim-blink.sh          # H0  two async tasks blinking
tools/sim-button.sh         # H1  button + software PWM (duty measured)
tools/sim-imu.sh            # H2  MPU6050 over I2C, virtual sensor tilted
tools/sim-encoder.sh        # H3  quadrature decoding (and aliasing at speed)
tools/sim-odom.sh           # L1  sim-core's Odometry running on emulated ARM
```

No microcontroller required — `tools/rp2040js` is a local emulator, and
the firmware built here is the same UF2 that will run on a real Pico.

**Level 3 — hardware in the loop.** The chip is the brain; the laptop is
the body and the world. Same mission either way:

```sh
tools/sim-hil.sh                                     # emulated RP2040
cargo run -p hil-host -- --serial /dev/cu.usbmodem11 # a REAL Pico 2 W
```

Measured on all three, one trajectory:

| | ticks | waypoints | drift | worst compute |
|---|---:|---:|---:|---:|
| `sim-run` (Mac, hardware `f64`) | — | 1/1 | 0.052 m | — |
| emulated RP2040 (M0+, no FPU) | 1139 | 1/1 | 0.052 m | 278 µs |
| real RP2350 (M33, SP FPU) | 1139 | 1/1 | 0.052 m | 268 µs |

Both physical Pico 2 W boards command **byte-identical** duty across all
1139 ticks.

**Recording and replay.** Every session can be captured and re-run with no
hardware attached — and replay checks what the code *would now command*,
so a behaviour change fails loudly:

```sh
cargo run -p hil-host -- --serial /dev/cu.usbmodem11 --record run.wire
cargo run -p hil-host -- --replay run.wire            # ~4 s, no board

cargo run -p vision --bin chase -- --record chase.perc --video
cargo run -p vision --bin chase -- --replay chase.perc  # no camera
```

This is what found a chip carrying its previous run's pose into a new
session — from four lines of log. See [`recordings/README.md`](recordings/README.md).
