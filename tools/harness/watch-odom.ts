// robotiq Level-1 proof: drive TWO virtual quadrature encoders and compare
// the pose the FIRMWARE computes (using sim-core's Odometry, on emulated
// ARM) against ground truth computed here in TypeScript.
//
// Same math, two languages, two machines. If they agree, the laptop
// robot's brain really is running on the chip.
//
// Usage (from tools/rp2040js):
//   npx tsx demo/watch-odom.ts [path-to.uf2]

import { Simulator } from '../src/simulator.js';
import { bootromB1 } from './bootrom.js';
import { loadUF2 } from './load-flash.js';

const image = process.argv[2] ?? '../../firmware/pico-odom/pico-odom.uf2';

const simulator = new Simulator();
const mcu = simulator.rp2040;
const clock = simulator.clock;
mcu.loadBootrom(bootromB1);
console.log(`Loading ${image}\n`);
loadUF2(image, mcu);

// Robot geometry — must match the firmware's constants.
const WHEEL_RADIUS = 0.03;
const TRACK_WIDTH = 0.15;
const TICKS_PER_REV = 1024;
const METERS_PER_TICK = (2 * Math.PI * WHEEL_RADIUS) / TICKS_PER_REV;

const CYCLE: [boolean, boolean][] = [
  [false, false],
  [false, true],
  [true, true],
  [true, false],
];

class VirtualEncoder {
  phase = 0;
  count = 0;
  constructor(
    readonly pinA: number,
    readonly pinB: number,
  ) {
    this.apply();
  }
  apply() {
    const [a, b] = CYCLE[((this.phase % 4) + 4) % 4];
    mcu.gpio[this.pinA].setInputValue(a);
    mcu.gpio[this.pinB].setInputValue(b);
  }
  step(dir: number) {
    this.phase += dir;
    this.count += dir;
    this.apply();
  }
}

const left = new VirtualEncoder(16, 17);
const right = new VirtualEncoder(18, 19);

// ---- ground truth pose, integrated the same way sim-core does ----
let tx = 0,
  ty = 0,
  tth = 0;
let lastL = 0,
  lastR = 0;
function integrateTruth() {
  const dl = (left.count - lastL) * METERS_PER_TICK;
  const dr = (right.count - lastR) * METERS_PER_TICK;
  lastL = left.count;
  lastR = right.count;
  const dCenter = (dr + dl) / 2;
  const dTheta = (dr - dl) / TRACK_WIDTH;
  if (Math.abs(dTheta) < 1e-9) {
    tx += dCenter * Math.cos(tth);
    ty += dCenter * Math.sin(tth);
  } else {
    const r = dCenter / dTheta;
    const next = tth + dTheta;
    tx += r * (Math.sin(next) - Math.sin(tth));
    ty -= r * (Math.cos(next) - Math.cos(tth));
    tth = next;
  }
}

let col = 0;
mcu.uart[0].onByte = (v) => {
  const ch = String.fromCharCode(v);
  if (col === 0 && ch !== '\r' && ch !== '\n') {
    process.stdout.write('  [pico] ');
    col = 1;
  }
  if (ch === '\n') col = 0;
  process.stdout.write(ch);
};

// [durationMs, left ticks/s, right ticks/s, description]
const PHASES: [number, number, number, string][] = [
  [1500, 1000, 1000, 'drive straight (both wheels equal)'],
  [1500, 400, 1600, 'arc left (right wheel faster)'],
  [1500, 1000, 1000, 'straight again'],
  [1200, -800, 800, 'spin in place (wheels opposed)'],
];

let li = 0,
  ri = 0;

function schedulePhase(i: number) {
  if (i >= PHASES.length) {
    integrateTruth();
    // sim-core wraps theta to (-pi, pi]; our truth integrator doesn't.
    // Wrap it the same way so the two are directly comparable.
    const wrapped = Math.atan2(Math.sin(tth), Math.cos(tth));
    console.log(
      `\n--- ground truth (computed here, in TypeScript) ---\n` +
        `    x=${tx.toFixed(3)} y=${ty.toFixed(3)} th=${wrapped.toFixed(3)}` +
        `   (unwrapped ${tth.toFixed(3)})\n` +
        `    ticks L=${left.count} R=${right.count}\n\n` +
        `Compare with the last [pico] line. Position should match to ~1 mm.\n` +
        `Heading will differ by whatever the wheels did in the final 100 ms\n` +
        `(the firmware reports on a 100 ms timer, so its last line is up to\n` +
        `one interval stale) — check the tick counts to confirm.`,
    );
    process.exit(0);
  }
  const [durMs, lRate, rRate, note] = PHASES[i];
  console.log(`\n[${(clock.micros / 1000).toFixed(0)} ms] >>> ${note}`);

  // Schedule this phase's encoder edges.
  const endUs = clock.micros + durMs * 1000;
  const stepL = lRate !== 0 ? 1e6 / Math.abs(lRate) : 0;
  const stepR = rRate !== 0 ? 1e6 / Math.abs(rRate) : 0;

  if (stepL > 0) {
    const a = clock.createAlarm(() => {
      if (clock.micros >= endUs) return;
      left.step(Math.sign(lRate));
      integrateTruth();
      a.schedule(stepL * 1000);
    });
    a.schedule(stepL * 1000);
  }
  if (stepR > 0) {
    const a = clock.createAlarm(() => {
      if (clock.micros >= endUs) return;
      right.step(Math.sign(rRate));
      integrateTruth();
      a.schedule(stepR * 1000);
    });
    a.schedule(stepR * 1000);
  }

  clock.createAlarm(() => schedulePhase(i + 1)).schedule(durMs * 1e6);
}

schedulePhase(0);

mcu.core.PC = 0x10000000;
simulator.execute();
