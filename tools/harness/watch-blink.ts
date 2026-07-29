// robotiq H0 harness: boots an embassy-rp UF2 in the rp2040js emulator and
// reports every LED-pin transition with SIMULATION timestamps, then prints
// a summary after N simulated seconds. Trajectories, not printouts.
//
// Usage (from tools/rp2040js):
//   npx tsx demo/watch-blink.ts [path-to.uf2] [sim-seconds]

import { Simulator } from '../src/simulator.js';
import { GPIOPinState } from '../src/gpio-pin.js';
import { bootromB1 } from './bootrom.js';
import { loadUF2 } from './load-flash.js';

const image = process.argv[2] ?? '../../firmware/pico-blink/pico-blink.uf2';
const SIM_SECONDS = Number(process.argv[3] ?? '3');

const simulator = new Simulator();
const mcu = simulator.rp2040;
mcu.loadBootrom(bootromB1);
console.log(`Loading ${image}`);
loadUF2(image, mcu);

// Anything the firmware prints on UART0 shows up here (nothing yet in H0).
mcu.uart[0].onByte = (value) => process.stdout.write(new Uint8Array([value]));

const WATCHED = [25, 16]; // onboard LED, external LED
const counts: Record<number, number> = {};
const lastHigh: Record<number, number> = {};

for (const pin of WATCHED) {
  counts[pin] = 0;
  mcu.gpio[pin].addListener((state) => {
    if (state !== GPIOPinState.High && state !== GPIOPinState.Low) return;
    const t = simulator.clock.micros / 1000; // simulated ms since boot
    counts[pin]++;
    if (state === GPIOPinState.High) {
      const period =
        lastHigh[pin] !== undefined ? `  (period ${(t - lastHigh[pin]).toFixed(1)} ms)` : '';
      console.log(`[${t.toFixed(1).padStart(8)} ms] GPIO${pin} HIGH${period}`);
      lastHigh[pin] = t;
    } else {
      console.log(`[${t.toFixed(1).padStart(8)} ms] GPIO${pin} LOW`);
    }
  });
}

// Stop after SIM_SECONDS of *simulated* time and report.
const poll = setInterval(() => {
  if (simulator.clock.micros >= SIM_SECONDS * 1e6) {
    simulator.stop();
    clearInterval(poll);
    console.log(`\n--- after ${SIM_SECONDS} simulated seconds ---`);
    for (const pin of WATCHED) {
      console.log(`GPIO${pin}: ${counts[pin]} transitions`);
    }
    process.exit(0);
  }
}, 50);

mcu.core.PC = 0x10000000;
simulator.execute();
