# Recorded sessions

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
| `rp2350-utrap.wire` | The Stage 0 U-trap solved by a real RP2350 over USB. 1139 ticks, 1/1 waypoints, drift 0.052 m, 0 wall bumps, worst compute 311 µs of a 20 000 µs budget. |

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
