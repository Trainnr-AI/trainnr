# Recorded sessions

These recordings were made by the 2025–26 rig, whose crates (`hil-host`, the
`vision` crate, the Pico firmware) now live in the archive repository
[Trainnr-AI/rig](https://github.com/Trainnr-AI/rig); the commands below run
from that checkout. The files stay here as the evidence the identification
work cites (the drivetrain fits, the chirp and sweep studies).

Every line the host and the chip said to each other, captured from a real
run. `>` is host → chip, `<` is chip → host.

```sh
# capture
cargo run -p hil-host -- --serial /dev/cu.usbmodem11 --record recordings/mine.wire

# replay — no board, no emulator, ~4 seconds instead of ~100
cargo run -p hil-host -- --replay recordings/mine.wire
```

## What replay actually checks

It does **not** just re-read the chip's answers. Every line the host *would
now send* is compared against the recording. If today's code would command
the robot differently from the run that was captured, replay prints both
sides and exits non-zero.

So a session on real hardware becomes a permanent test. Verified: nudging
`kp_dist` from 0.8 to 0.81 — a 1% retune — is caught on the first tick:

```
REPLAY DIVERGED from the recorded session:
  - would send "G 1.3500 3.0500 4.4550", recording has "G 1.3500 3.0500 4.4000"
```

## The files

| file | what it is |
|---|---|
| `rp2350-utrap.wire` | ⚠️ **STALE since 2026-08-10 — needs a hardware re-record.** `REAL_BOT` gained its measured values, so the host commands differently and replay diverges correctly. The Stage 0 U-trap solved by a real RP2350 over USB. 1139 ticks, 1/1 waypoints, drift 0.052 m, 0 wall bumps, worst compute 311 µs of a 20 000 µs budget. |
| `chase-sweep.perc` | A **synthetic** perception session: an object enters from the right, is tracked to centre, grows as the robot closes, then vanishes. 225 frames, 20 KB. Exercises the bearing low-pass, the heading deadband, `approach_factor` and the shared PID. Synthetic on purpose — deterministic, and no footage of anyone's living room in the repo. |

```sh
# the camera path
cargo run -p vision --bin chase -- --record chase.perc [--video]
cargo run -p vision --bin chase -- --replay chase.perc

# accept a deliberate change: same perception, freshly computed commands
cargo run -p vision --bin chase -- --replay chase.perc --record chase.perc
```

A perception log records the commanded `(v, w)` per frame, so replay
checks the **output** and not just re-runs the input. Mutation-verified
against four separate pieces of glue:

| changed | frames that diverged (of 327) |
|---|---|
| bearing filter α | 233 |
| heading deadband | 143 |
| `approach_factor` curve | 153 — exactly the frames with forward motion |
| shared `heading_kp` | 213 |

Re-record it deliberately, never to make a failing replay pass. A
divergence is either a regression or a decision — and if it is a decision,
say so in the commit that re-records.

## Why this exists

A two-byte protocol change hung the rig for hours of debugging. The chip
was fast in isolation, the host was fast in isolation, and the one thing
nobody had was the conversation between them.

It earned its keep immediately. The first hardware session recorded after
it was written failed — 0/1 waypoints, 4.97 m drift, 2036 wall bumps — and
the second line of the log said why:

```
>G 1.3500 3.0500 4.4000      host: steer toward (1.35, 3.05)
<P 6.3826 3.0898 -0.6529     chip: "I am at (6.38, 3.09)"
<M 1000 -1000                chip: full-speed spin
```

(6.38, 3.09) was where the *previous* run finished. The firmware set its
pose once at boot, so a board left powered between runs began each session
believing it was wherever it last stopped. Fixed by the `I` message: the
host now tells the chip where a session starts, and the chip acknowledges.

Diagnosis took one `head -4`.

## `chase-brightness.wire` — the camera driving the wheels (2026-08-14)

Twenty-five seconds of the closed loop on the bench: 1,034 status
reports and **5 embedded camera thumbnails**, captured with
`rig_view --record` while the robot hunted the brightest patch it could
see, wheels on a stand. Replay draws the whole rig — image, belief
trail, duty, ticks — with no hardware:

```sh
cargo run -p hil-host --example rig_view -- --replay recordings/chase-brightness.wire
# -> 5 camera frames, 1034 status reports drawn
```

**A gate step since 2026-08-15**: `rig_replay` is the viewer-less reader
this note used to ask for. `tools/verify.sh` pins the exact counts —
`1034 status (0 stalled), 5 images (0 torn), 0 servo, 2 notes,
0 unparsable`.

## `track-and-stall.wire` — the arm tracking, and the stall latch (2026-08-15)

The first ~28 seconds of the camera-driven servo run: 47 `# servo us`
lines (the arm's commanded pulses), 7 thumbnails — and **1,119 of its
1,132 status reports carry `stalled:true`**, because this is the very
recording in which a marginal breadboard contact left the wheels
commanded-but-motionless and the stall guard latched in 200 ms. Kept
for exactly that: it pins the parsing of all three note formats *and*
the stalled flag against a session where the guard genuinely fired.

```sh
cargo run -p hil-host --example rig_replay -- recordings/track-and-stall.wire
# -> 1132 status (1119 stalled), 7 images (0 torn), 47 servo, 3 notes, 0 unparsable
```

⚠️ Curated by one line: the raw capture began with `1900` — the tail of
an in-flight servo note, cut at the instant DTR opened the port. A
mid-stream attach always risks a leading fragment; the fixture drops it
so `rig_replay` can stay strict instead of growing a special case.
