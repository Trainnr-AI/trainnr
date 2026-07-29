// robotiq H1 harness: presses a virtual button on a schedule and MEASURES
// the LED's duty cycle in each window between presses. The firmware's job
// (exercise H1) is to make the measured sequence read 0 → 25 → 50 → 100 → 0.
//
// Usage (from tools/rp2040js):
//   npx tsx demo/watch-button.ts [path-to.uf2]

import { Simulator } from '../src/simulator.js';
import { GPIOPinState } from '../src/gpio-pin.js';
import { bootromB1 } from './bootrom.js';
import { loadUF2 } from './load-flash.js';

const image = process.argv[2] ?? '../../firmware/pico-button/pico-button.uf2';

const simulator = new Simulator();
const mcu = simulator.rp2040;
const clock = simulator.clock;
mcu.loadBootrom(bootromB1);
console.log(`Loading ${image}`);
loadUF2(image, mcu);
mcu.uart[0].onByte = (v) => process.stdout.write(new Uint8Array([v]));

const BUTTON = 14;
const LED = 15;
const HEART = 25;

// Button idles HIGH (firmware uses an internal pull-up); pressed = LOW.
mcu.gpio[BUTTON].setInputValue(true);

let heartbeats = 0;
mcu.gpio[HEART].addListener((s) => {
  if (s === GPIOPinState.High) heartbeats++;
});

// Duty measurement: accumulate how long the LED spends HIGH per window.
let ledHigh = false;
let lastT = 0;
let highAccum = 0;
let windowStart = 0;
mcu.gpio[LED].addListener((s) => {
  const t = clock.micros;
  if (ledHigh) highAccum += t - lastT;
  ledHigh = s === GPIOPinState.High;
  lastT = t;
});

// Schedule presses at 810/1610/2410/3210 ms (just after each window edge,
// so every 800 ms window sees one clean duty level). 60 ms hold each.
for (const ms of [810, 1610, 2410, 3210]) {
  clock
    .createAlarm(() => {
      console.log(`[${(clock.micros / 1000).toFixed(1)} ms] >>> button PRESSED`);
      mcu.gpio[BUTTON].setInputValue(false);
    })
    .schedule(ms * 1e6);
  clock
    .createAlarm(() => {
      mcu.gpio[BUTTON].setInputValue(true);
    })
    .schedule((ms + 60) * 1e6);
}

// Report the measured duty at the end of each 800 ms window.
const expected = [0, 25, 50, 100, 0];
let windowIndex = 0;
const windowAlarm = clock.createAlarm(() => {
  const t = clock.micros;
  if (ledHigh) {
    highAccum += t - lastT;
    lastT = t;
  }
  const duty = (100 * highAccum) / (t - windowStart);
  console.log(
    `[${(t / 1000).toFixed(1)} ms] measured LED duty: ${duty.toFixed(1)}%   (expected ${expected[windowIndex]}%)`,
  );
  highAccum = 0;
  windowStart = t;
  windowIndex++;
  if (windowIndex >= expected.length) {
    console.log(`\nHeartbeats seen: ${heartbeats} (should be ~8 over 4 s)`);
    console.log('Target sequence: 0 -> 25 -> 50 -> 100 -> 0');
    process.exit(0);
  }
  windowAlarm.schedule(800 * 1e6);
});
windowAlarm.schedule(800 * 1e6);

mcu.core.PC = 0x10000000;
simulator.execute();
