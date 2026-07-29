// Debug harness: manual step loop with a stuck-detector. Reports the exact
// PC (program counter) where the firmware stops making progress, so we can
// map it to a function name with llvm-symbolizer.
import { Simulator } from '../src/simulator.js';
import { GPIOPinState } from '../src/gpio-pin.js';
import { bootromB1 } from './bootrom.js';
import { loadUF2 } from './load-flash.js';

const image = process.argv[2] ?? '../../firmware/pico-blink/pico-blink.uf2';
const LIMIT_US = Number(process.argv[3] ?? '3') * 1e6;

const simulator = new Simulator();
const mcu = simulator.rp2040;
const clock = simulator.clock;
mcu.loadBootrom(bootromB1);
loadUF2(image, mcu);
mcu.uart[0].onByte = (v) => process.stdout.write(new Uint8Array([v]));

for (const pin of [15, 25]) {
  mcu.gpio[pin].addListener((state) => {
    if (state === GPIOPinState.High || state === GPIOPinState.Low) {
      console.log(
        `[${(clock.micros / 1000).toFixed(1)} ms] GPIO${pin} ${GPIOPinState[state]}`,
      );
    }
  });
}

mcu.core.PC = 0x10000000;
const cycleNanos = 1e9 / 125_000_000;
let instructions = 0;

while (clock.micros < LIMIT_US) {
  if (mcu.core.waiting) {
    const n = clock.nanosToNextAlarm;
    if (n <= 0) {
      console.log(
        `\nSTUCK: core in WFE with NO scheduled alarm.` +
          `\n  PC = 0x${mcu.core.PC.toString(16)}  LR = 0x${mcu.core.LR.toString(16)}` +
          `\n  sim time = ${(clock.micros / 1000).toFixed(3)} ms, ${instructions} instructions executed`,
      );
      process.exit(1);
    }
    clock.tick(n);
  } else {
    const cycles = mcu.core.executeInstruction();
    clock.tick(cycles * cycleNanos);
    instructions++;
  }
}
console.log(
  `\nReached ${(LIMIT_US / 1e6).toFixed(1)} sim-seconds normally (${instructions} instructions).`,
);
