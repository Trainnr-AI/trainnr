/* RP2040 memory map. The first 256 bytes of flash are "boot2" — a tiny
   second-stage bootloader that configures the external flash chip before
   any of our code can run (the RP2040 has no internal flash of its own).
   embassy-rp provides the blob (feature "boot2-w25q080"); this file
   reserves its place. */
MEMORY {
    BOOT2 : ORIGIN = 0x10000000, LENGTH = 0x100
    FLASH : ORIGIN = 0x10000100, LENGTH = 2048K - 0x100
    RAM   : ORIGIN = 0x20000000, LENGTH = 264K
}

EXTERN(BOOT2_FIRMWARE)

SECTIONS {
    /* ### Boot loader */
    .boot2 ORIGIN(BOOT2) :
    {
        KEEP(*(.boot2));
    } > BOOT2
} INSERT BEFORE .text;
