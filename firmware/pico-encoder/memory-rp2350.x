/* RP2350 (Pico 2 / Pico 2 W) memory map.

   Unlike the RP2040 there is no BOOT2 stage to reserve: the RP2350 has a
   real bootloader in ROM. Instead the ROM looks for an IMAGE_DEF block
   near the start of flash to decide whether the image is bootable and
   how to run it. embassy-rp emits that block into `.start_block` when the
   `imagedef-secure-exe` feature is on; this script places it. */
MEMORY {
    FLASH : ORIGIN = 0x10000000, LENGTH = 4096K
    RAM   : ORIGIN = 0x20000000, LENGTH = 520K
}

SECTIONS {
    .start_block : ALIGN(4)
    {
        __start_block_addr = .;
        KEEP(*(.start_block));
        KEEP(*(.boot_info));
    } > FLASH
} INSERT AFTER .vector_table;

/* Push .text past the image-def block so the ROM finds it first. */
_stext = ADDR(.start_block) + SIZEOF(.start_block);
