// trainnr H3 harness: drives a virtual quadrature encoder on GP16/GP17
// and compares the firmware's decoded count against ground truth.
//
// The script deliberately ramps the wheel speed past what a 10 kHz polling
// loop can follow — so you can watch the firmware's error counter wake up
// and its count fall behind the truth. Polling has a speed limit; this is
// what hitting it looks like.
//
// Usage (from tools/rp2040js):
//   npx tsx demo/watch-encoder.ts [path-to.uf2]

import { Simulator } from '../src/simulator.js';
import { bootromB1 } from './bootrom.js';
import { loadUF2 } from './load-flash.js';

const image = process.argv[2] ?? '../../firmware/pico-encoder/pico-encoder.uf2';

const simulator = new Simulator();
const mcu = simulator.rp2040;
const clock = simulator.clock;
mcu.loadBootrom(bootromB1);
console.log(`Loading ${image}\n`);
loadUF2(image, mcu);

const PIN_A = 16;
const PIN_B = 17;

// Forward quadrature cycle: (A,B) = 00 -> 01 -> 11 -> 10 -> 00
const CYCLE: [boolean, boolean][] = [
  [false, false],
  [false, true],
  [true, true],
  [true, false],
];

let phaseIndex = 0; // where we are in the 4-state cycle
let truth = 0; // ground-truth tick count

function applyState() {
  const [a, b] = CYCLE[((phaseIndex % 4) + 4) % 4];
  mcu.gpio[PIN_A].setInputValue(a);
  mcu.gpio[PIN_B].setInputValue(b);
}
applyState();

// Firmware console.
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

// [durationMs, microseconds per transition, direction, description]
const PHASES: [number, number, number, string][] = [
  [1500, 2000, +1, 'slow forward (500 transitions/s)'],
  [1500, 2000, -1, 'slow REVERSE (count should go down)'],
  [1200, 500, +1, 'faster forward (2 000 /s)'],
  [1200, 200, +1, 'fast forward (5 000 /s — near the polling limit)'],
  [1200, 50, +1, 'TOO FAST (20 000 /s — polling cannot keep up)'],
];

let phase = 0;
let stepUs = PHASES[0][1];
let dir = PHASES[0][2];

const stepAlarm = clock.createAlarm(() => {
  phaseIndex += dir;
  truth += dir;
  applyState();
  stepAlarm.schedule(stepUs * 1000);
});

function startPhase(i: number) {
  if (i >= PHASES.length) {
    console.log(`\n--- finished. ground-truth count = ${truth} ---`);
    console.log('Compare with the last [pico] line: matching count means');
    console.log('every transition was caught; a shortfall + errors means');
    console.log('the wheel outran the polling loop.');
    process.exit(0);
  }
  const [durMs, us, d, note] = PHASES[i];
  stepUs = us;
  dir = d;
  console.log(
    `\n[${(clock.micros / 1000).toFixed(0)} ms] >>> ${note}   (truth so far: ${truth})`,
  );
  clock.createAlarm(() => startPhase(i + 1)).schedule(durMs * 1e6);
}

startPhase(0);
stepAlarm.schedule(stepUs * 1000);

mcu.core.PC = 0x10000000;
simulator.execute();
