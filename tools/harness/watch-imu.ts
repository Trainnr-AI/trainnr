// trainnr H2b harness: boots the pico-imu firmware with a virtual MPU6050
// on I2C0, then TILTS the virtual sensor on a schedule. The firmware's
// UART output should track the tilts — proof that the whole sense path
// works: bus protocol -> driver -> physical units.
//
// Usage (from tools/rp2040js):
//   npx tsx demo/watch-imu.ts [path-to.uf2]

import { Simulator } from '../src/simulator.js';
import { bootromB1 } from './bootrom.js';
import { loadUF2 } from './load-flash.js';
import { VirtualMPU6050 } from './virtual-mpu6050.js';

const image = process.argv[2] ?? '../../firmware/pico-imu/pico-imu.uf2';

const simulator = new Simulator();
const mcu = simulator.rp2040;
const clock = simulator.clock;
mcu.loadBootrom(bootromB1);
console.log(`Loading ${image}\n`);
loadUF2(image, mcu);

// Wire the virtual IMU onto I2C0 (the peripheral the firmware uses).
const imu = new VirtualMPU6050();
imu.attach(mcu.i2c[0]);

// Firmware console output, prefixed so it's distinguishable from harness
// narration.
let col = 0;
mcu.uart[0].onByte = (v) => {
  const ch = String.fromCharCode(v);
  if (col === 0 && ch !== '\r' && ch !== '\n') {
    process.stdout.write(`  [pico] `);
    col = 1;
  }
  if (ch === '\n') col = 0;
  process.stdout.write(ch);
};

// The tilt script: (time_ms, x, y, z in g, description)
const TILTS: [number, number, number, number, string][] = [
  [1200, 0, 0, 1, 'flat on the desk (gravity straight down)'],
  [2200, 1, 0, 0, 'tipped onto its side (gravity now on +x)'],
  [3200, 0, -1, 0, 'tipped forward (gravity on -y)'],
  [4200, 0, 0, -1, 'upside down (gravity on -z)'],
  [5200, 0.5, 0.5, 0.707, 'held at an angle (split across all three)'],
];

for (const [ms, x, y, z, note] of TILTS) {
  clock
    .createAlarm(() => {
      imu.setAccelG(x, y, z);
      console.log(
        `\n[${(clock.micros / 1000).toFixed(0)} ms] >>> TILT: ${note}\n` +
          `    truth: x=${x.toFixed(3)}g y=${y.toFixed(3)}g z=${z.toFixed(3)}g`,
      );
    })
    .schedule(ms * 1e6);
}

clock
  .createAlarm(() => {
    console.log(`\n--- done. chip awake: ${imu.awake} ---`);
    process.exit(0);
  })
  .schedule(6200 * 1e6);

mcu.core.PC = 0x10000000;
simulator.execute();
