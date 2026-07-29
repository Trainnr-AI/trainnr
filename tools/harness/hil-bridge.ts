// LEVEL 3 bridge: runs the firmware on an emulated RP2040 and wires its
// UART to this process's stdin/stdout, so a Rust host can be the physics.
//
//   firmware UART TX  ->  stdout   (P and M protocol lines)
//   stdin             ->  firmware UART RX  (S protocol lines)
//
// stdout carries ONLY protocol. The emulator's own chatter goes to stderr,
// so the host never has to parse around it.
//
// Protocol: docs/10-hil-protocol.md
// Usage: npx tsx demo/hil-bridge.ts <firmware.uf2>

import { Simulator } from '../src/simulator.js';
import { bootromB1 } from './bootrom.js';
import { loadUF2 } from './load-flash.js';
import type { Logger } from '../src/utils/logging.js';

const image = process.argv[2];
if (!image) {
  process.stderr.write('usage: hil-bridge.ts <firmware.uf2>\n');
  process.exit(1);
}

/// Everything the emulator wants to say goes to stderr, never stdout.
class StderrLogger implements Logger {
  private write(level: string, component: string, message: string) {
    process.stderr.write(`[${level}] ${component}: ${message}\n`);
  }
  debug() {} // too noisy: unimplemented-peripheral reads on every boot
  info() {}
  warn(component: string, message: string) {
    // Expected and harmless: rp2040js doesn't model the clock/PLL blocks
    // or every UART register, and embassy pokes them all during init.
    if (message.startsWith('Unimplemented peripheral')) return;
    this.write('warn', component, message);
  }
  error(component: string, message: string) {
    this.write('error', component, message);
  }
}

const simulator = new Simulator();
const mcu = simulator.rp2040;
mcu.logger = new StderrLogger();
mcu.loadBootrom(bootromB1);
loadUF2(image, mcu);

// ---- firmware -> host ----
mcu.uart[0].onByte = (value) => {
  process.stdout.write(Buffer.from([value]));
};

// ---- host -> firmware ----
process.stdin.on('data', (chunk: Buffer) => {
  for (const byte of chunk) {
    mcu.uart[0].feedByte(byte);
  }
});
process.stdin.on('end', () => process.exit(0));
process.stdout.on('error', () => process.exit(0)); // host closed the pipe

process.stderr.write(`[bridge] running ${image}\n`);
mcu.core.PC = 0x10000000;
simulator.execute();
