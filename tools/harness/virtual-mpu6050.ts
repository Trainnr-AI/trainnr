// A virtual MPU6050 IMU, living on the emulator's I2C bus.
//
// This models the DEVICE side of I2C — the counterpart to the driver
// Prakhar wrote in crates/mpu6050-driver. The protocol a real chip
// implements:
//
//   write [reg, data...]      -> set register pointer, store bytes
//                                (pointer auto-increments)
//   write [reg] then read N   -> set pointer, then stream N bytes out
//                                (pointer auto-increments)
//
// Everything the driver does — who_am_i, wake, burst-read accel — is just
// those two patterns. Tilt the virtual chip with setAccelG() and the
// firmware sees it, exactly as if a hand had tilted a real breakout board.

import { I2CMode, RPI2C } from '../src/peripherals/i2c.js';

const ADDRESS = 0x68;
const REG_PWR_MGMT_1 = 0x6b;
const REG_WHO_AM_I = 0x75;
const REG_ACCEL_XOUT_H = 0x3b;
const COUNTS_PER_G = 16384;

export class VirtualMPU6050 {
  /** 128 registers, all zero at power-on except the ones set below. */
  private regs = new Uint8Array(128);
  /** Where the next read/write lands (the "register pointer"). */
  private pointer = 0;
  /** Have we been addressed in this transaction? */
  private selected = false;
  private expectingRegister = true;

  /** Set true when the firmware writes 0 to PWR_MGMT_1 (the wake command). */
  awake = false;
  /** Every transaction, for the harness to narrate. */
  onLog: (msg: string) => void = () => {};

  constructor() {
    this.regs[REG_WHO_AM_I] = 0x68; // identity, per datasheet
    this.regs[REG_PWR_MGMT_1] = 0x40; // SLEEP bit set: chips boot asleep
    this.setAccelG(0, 0, 1); // resting flat: 1 g of gravity on z
  }

  /** Tilt the virtual sensor. Values in g; writes the 6 accel registers. */
  setAccelG(x: number, y: number, z: number) {
    const axes = [x, y, z];
    for (let i = 0; i < 3; i++) {
      const counts = Math.max(-32768, Math.min(32767, Math.round(axes[i] * COUNTS_PER_G)));
      const u = counts < 0 ? counts + 65536 : counts; // two's complement
      this.regs[REG_ACCEL_XOUT_H + i * 2] = (u >> 8) & 0xff; // high byte
      this.regs[REG_ACCEL_XOUT_H + i * 2 + 1] = u & 0xff; // low byte
    }
  }

  /** Wire this device onto one of the RP2040's two I2C controllers. */
  attach(i2c: RPI2C) {
    i2c.onStart = () => i2c.completeStart();

    i2c.onConnect = (address: number, mode: I2CMode) => {
      this.selected = address === ADDRESS;
      if (this.selected && mode === I2CMode.Write) {
        // A write transaction always begins with the register address.
        this.expectingRegister = true;
      }
      // ack = "yes, that's me" — a NACK here is what a missing/miswired
      // chip does, and it's how drivers detect absent hardware.
      i2c.completeConnect(this.selected);
    };

    i2c.onWriteByte = (value: number) => {
      if (!this.selected) {
        i2c.completeWrite(false);
        return;
      }
      if (this.expectingRegister) {
        this.pointer = value & 0x7f;
        this.expectingRegister = false;
        this.onLog(`  i2c: select register 0x${value.toString(16).padStart(2, '0')}`);
      } else {
        this.regs[this.pointer] = value;
        this.onLog(
          `  i2c: write 0x${value.toString(16).padStart(2, '0')} -> reg 0x${this.pointer.toString(16)}`,
        );
        if (this.pointer === REG_PWR_MGMT_1) {
          const sleeping = (value & 0x40) !== 0;
          this.awake = !sleeping;
          this.onLog(`  >>> chip is now ${this.awake ? 'AWAKE' : 'ASLEEP'}`);
        }
        this.pointer = (this.pointer + 1) & 0x7f;
      }
      i2c.completeWrite(true);
    };

    i2c.onReadByte = () => {
      if (!this.selected) {
        i2c.completeRead(0xff);
        return;
      }
      const value = this.regs[this.pointer];
      this.pointer = (this.pointer + 1) & 0x7f;
      i2c.completeRead(value);
    };

    i2c.onStop = () => {
      this.expectingRegister = true;
      i2c.completeStop();
    };
  }
}
