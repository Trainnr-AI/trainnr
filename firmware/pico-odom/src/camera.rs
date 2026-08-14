//! The OV7670 on the same board as the motors: capture, detect, report.
//!
//! ```text
//!   XCLK (PWM) ─▶ OV7670 ─▶ HREF/PCLK/D0-D7 ─▶ PIO ─▶ DMA ─▶ [frame]
//!                                                              │
//!                                       blob::find_matching ◀──┘
//!                                              │
//!                     (x, y, area) ── atomics ──▶ chase / telemetry
//! ```
//!
//! Shares nothing with the drivetrain but the report stream — which is
//! the point of running them on one chip. `PIO0` and `DMA_CH0` are the
//! radio's on a `wifi` build, so `camera,wifi` is a `compile_error!` in
//! `main.rs` rather than a corrupt frame at runtime.

use core::fmt::Write as _;
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use embassy_executor::Spawner;
use embassy_rp::bind_interrupts;
use embassy_rp::gpio::{Input, Pull};
use embassy_rp::i2c::{Async, I2c};
use embassy_rp::peripherals::{
    DMA_CH0, I2C0, PIN_0, PIN_1, PIN_13, PIN_14, PIN_15, PIN_16, PIN_17, PIN_18, PIN_19, PIN_20,
    PIN_21, PIN_22, PIO0, PWM_SLICE2,
};
use embassy_rp::pio::{
    Config as PioConfig, Direction, InterruptHandler, Pio, ShiftConfig, ShiftDirection,
};
use embassy_rp::pwm::{Config as PwmConfig, Pwm};
use embassy_rp::Peri;
use embassy_time::{Duration, Timer};

bind_interrupts!(struct CameraIrqs {
    PIO0_IRQ_0 => InterruptHandler<PIO0>;
});
bind_interrupts!(struct DmaIrqs {
    DMA_IRQ_0 => embassy_rp::dma::InterruptHandler<DMA_CH0>;
});

/// XCLK divider. A top of 5 gives 25 MHz on an RP2350 — the value every
/// OV7670 reference design uses — and 20.8 MHz on an RP2040, both inside
/// the sensor's 10–24 MHz window.
const XCLK_TOP: u16 = 5;
const XCLK_COMPARE: u16 = 3;

/// How long XCLK runs before the first transaction, and between
/// identification retries. The sensor's control interface is clocked
/// from XCLK; asking too early reads as `BUS ERROR` on a good camera —
/// measured, because the standalone bring-up firmware only worked by
/// accidentally waiting for a USB terminal first.
const SETTLE_MS: u64 = 100;
/// Identification attempts before giving up for good. One try turns
/// settling-still-in-progress into a permanent verdict.
const ATTEMPTS: usize = 5;

/// One thumbnail every this many frames.
const THUMBNAIL_EVERY: u32 = 16;

/// How often [`announce`] runs, in seconds. Also the divisor that turns a
/// frame delta into a rate, so the two cannot drift apart.
pub const ANNOUNCE_SECONDS: u32 = 10;

/// Frame geometry, and an admission.
///
/// # ⚠️ 120 wide, deliberately narrower than the sensor's line
///
/// The sensor's true line width is **not known**. An attempt to derive it
/// from the shear of a misaligned capture gave 140 and then contradicted
/// itself — the early PIO program lost a varying number of bytes per
/// line, so the shear was never a clean stride to solve for (progress
/// log, 2026-08-14).
///
/// So the width is *asserted*, not inferred: the PIO program counts out
/// exactly this many pixels per line and then blocks for the next
/// `HREF`. Any line at least this wide yields aligned rows, and the cost
/// is cropped field of view rather than a corrupt picture. ⚠️ Widening it
/// is only safe once the real line width is measured directly — counting
/// `PCLK` edges between `HREF` edges — and the PIO loop constants below
/// are retuned to match.
pub const FRAME_WIDTH: u16 = 120;
pub const FRAME_HEIGHT: u16 = 120;
const FRAME_WORDS: usize = FRAME_WIDTH as usize * FRAME_HEIGHT as usize * 2 / 4;

/// The frame buffer — 28.8 KB of the RP2350's 520 KB. `StaticCell`
/// because DMA needs a `'static` target and `unsafe_code` is forbidden
/// repo-wide, so a `static mut` is not an alternative.
static PICTURE: static_cell::StaticCell<[u32; FRAME_WORDS]> = static_cell::StaticCell::new();

/// Identity bytes, big-endian packed, so [`announce`] can restate them
/// without touching the bus again.
static CACHED: AtomicU32 = AtomicU32::new(0);
/// ⚠️ Set only on a real answer. Setting it unconditionally once made
/// `announce` restate a cached all-zeroes identity every ten seconds — a
/// failure reported in the voice of a measurement.
static SEEN: AtomicBool = AtomicBool::new(false);
/// `applied | com7 << 8 | com15 << 16` — what the sensor says its own
/// format registers hold, after being told. SCCB ACKs writes it ignores,
/// so only a read-back is evidence the table took.
static CONFIG_READBACK: AtomicU32 = AtomicU32::new(0);
/// Darkest and brightest byte of the latest frame, `low | high << 8`.
static PIXEL_RANGE: AtomicU32 = AtomicU32::new(0);
/// Frames captured since boot. A liveness signal: a count that stops
/// advancing is the camera's equivalent of a host gone silent.
static FRAMES: AtomicU32 = AtomicU32::new(0);
/// Frame count at the previous announce, for the fps delta.
static LAST_ANNOUNCED_FRAMES: AtomicU32 = AtomicU32::new(0);
/// The latest blob's offsets from frame centre, packed as two
/// millis-signed-16s: `x_milli << 16 | y_milli`. Valid only while
/// [`BLOB_AREA`] is non-zero.
static BLOB_ERROR: AtomicU32 = AtomicU32::new(0);
/// Matching pixel count. **Zero means "looked, found nothing"** — a real
/// answer, deliberately distinct from "did not look", which shows up as
/// [`FRAMES`] not advancing instead.
static BLOB_AREA: AtomicU32 = AtomicU32::new(0);

/// Every pin the camera needs, named so two cannot be swapped by
/// argument order — the same reason [`crate::MotorPins`] exists.
pub struct CameraPins {
    pub xclk_slice: Peri<'static, PWM_SLICE2>,
    pub xclk: Peri<'static, PIN_21>,
    pub vsync: Peri<'static, PIN_1>,
    pub href: Peri<'static, PIN_22>,
    pub pclk: Peri<'static, PIN_0>,
    pub pio: Peri<'static, PIO0>,
    pub dma: Peri<'static, DMA_CH0>,
    /// `D0`..`D7`. **All eight**, not just the base: every pin PIO reads
    /// must have its function switched to PIO, and claiming only `D0`
    /// left the other seven reading as nothing — a DMA timeout that
    /// looked like a wiring fault. They must also be consecutive, since
    /// `in pins` reads a contiguous group; that is the whole reason the
    /// encoders moved off GP16–GP19.
    pub data: DataPins,
}

/// The camera's eight data pins, named as distinct peripheral types so a
/// swap cannot typecheck — the tuple's whole worth, given one alias.
pub type DataPins = (
    Peri<'static, PIN_13>,
    Peri<'static, PIN_14>,
    Peri<'static, PIN_15>,
    Peri<'static, PIN_16>,
    Peri<'static, PIN_17>,
    Peri<'static, PIN_18>,
    Peri<'static, PIN_19>,
    Peri<'static, PIN_20>,
);

/// QQVGA-shaped RGB565 — the format `crates/blob` consumes.
///
/// ⚠️ **A starting point, not a tuned table.** Camera register sets are
/// order-dependent and vendor-specific. Written as data rather than code
/// so that adjusting it is editing a table, not editing logic.
const QQVGA_RGB565: &[(u8, u8)] = &[
    (0x12, 0x04), // COM7   — RGB output
    // ⚠️ Divider measured at both extremes, both wrong: /1 gave 206 fps —
    // exposure cannot exceed the frame period, so that is a ~5 ms shutter
    // and a black image indoors — and /8 put the internal clock near
    // 3 MHz, under the sensor's ~10 MHz floor, collapsing the rate ~100x
    // rather than the 8x a divider implies. /2 is in spec and explicitly
    // NOT tuned: brightness is only judgeable from a picture.
    (0x11, 0x01), // CLKRC  — internal clock / 2
    (0x0C, 0x04), // COM3   — enable downsampling (DCW)
    (0x3E, 0x1A), // COM14  — DCW on, PCLK divided by 4
    (0x40, 0xD0), // COM15  — RGB565, full 0-255 range
    // ⚠️ Bit 7 of XSC/YSC selects the sensor's internal TEST PATTERN —
    // [`TEST_PATTERN`] flips YSC's. Colour bars are generated inside the
    // sensor, after the pixel array: they travel the entire path under
    // suspicion (format, PCLK, HREF, D0-D7, PIO, DMA) while depending on
    // no lens, no light and no exposure. Bars correct = capture sound,
    // problem optical. Bars wrong = no amount of hue or exposure tuning
    // was ever going to help. This lever ended a long guessing session.
    (0x70, 0x3A), // SCALING_XSC
    (0x71, 0x35), // SCALING_YSC
    (0x72, 0x22), // SCALING_DCWCTR    — /4 horizontally and vertically
    (0x73, 0xF2), // SCALING_PCLK_DIV  — /4
    (0xA2, 0x02), // SCALING_PCLK_DELAY
    // ⚠️ Undocumented, and load-bearing. Without it the RGB565 output
    // carries a strong magenta cast — whites render lavender, seen on
    // this sensor in Rerun. 0xB0 is absent from the datasheet; 0x84 is
    // the value every working init sequence in the wild converged on,
    // and no more is known about it than that. The cast is upstream of
    // the capture: the bars decoded byte-perfect while the lens image
    // stayed pink.
    (0xB0, 0x84), // undocumented "colour mode" — kills the magenta cast
];

/// Ask the sensor for colour bars instead of the lens. A debugging
/// switch — `false` in anything that matters — kept because "compare
/// against something you know" is the move that ends guessing sessions,
/// and the next person deserves the lever without hunting for a register.
const TEST_PATTERN: bool = false;

/// ⚠️ Hunting BRIGHTNESS, not hue — the third discriminant tried, and
/// the first with a measurement behind it in both directions.
///
/// Hue failed twice on this bench (2026-08-14). Green: the post-`0xB0`
/// white balance tints the whole scene green, so the floor outscored a
/// genuinely green wire — noise ~160 pixels, signal ~0. Blue: clean
/// noise floor, but the only blue thing on the bench is a wire that must
/// be hand-held, which is a prop-dependent robot.
///
/// Brightness is immune to the cast, and the threshold is **relative**
/// (frame mean + margin): a dim room and a lit one both have a brightest
/// patch, rather than a fixed number that works in one room only. A
/// phone torch steers it.
const BRIGHT_MARGIN: u8 = 50;
/// Fewer matching pixels than this and nothing is reported.
const BRIGHT_MIN_PIXELS: u32 = 40;

/// Bring the camera up and leave it capturing in its own task.
///
/// Returns the PWM guard: **XCLK must keep running** afterwards, and
/// dropping it stops the clock — the camera works once, then never
/// again. Binding it to a name at the call site is what keeps it alive.
/// The bus is TAKEN and GIVEN BACK: GP4/GP5 carry the camera's SCCB and
/// the PCA9685 side by side, so the bus is a board resource the caller
/// owns — the first version buried its construction in here, and the
/// servo module then had no way to reach it.
#[must_use = "dropping this stops XCLK and the camera goes deaf"]
pub async fn start(
    spawner: Spawner,
    bus: I2c<'static, I2C0, Async>,
    pins: CameraPins,
) -> (Pwm<'static>, I2c<'static, I2C0, Async>) {
    // ⚠️ XCLK first, before anything touches the bus. The OV7670 has no
    // oscillator of its own; a transaction issued before the clock runs
    // fails in a way that looks exactly like bad wiring.
    let mut clock_config = PwmConfig::default();
    clock_config.top = XCLK_TOP;
    // Channel **B**, because GP21 is odd: `slice = (n/2) % 8`,
    // `channel = n % 2`. Embassy encodes the mapping in its types, so the
    // wrong choice fails to compile instead of driving the wrong pin.
    clock_config.compare_b = XCLK_COMPARE;
    let xclk = Pwm::new_output_b(pins.xclk_slice, pins.xclk, clock_config);

    Timer::after_millis(SETTLE_MS).await;
    let mut sensor = ov7670_driver::Ov7670::new(bus);

    // Spaced attempts, because this verdict is cached for the life of
    // the program.
    let mut result = sensor.identify();
    for _ in 0..ATTEMPTS - 1 {
        if result.as_ref().is_ok_and(ov7670_driver::Identity::is_ov7670) {
            break;
        }
        Timer::after_millis(SETTLE_MS).await;
        result = sensor.identify();
    }

    let mut text: heapless::String<160> = heapless::String::new();
    let identity = match result {
        Ok(identity) => {
            let _ = write!(text, "# camera {identity}");
            identity
        }
        Err(_) => {
            let _ = text
                .push_str("# camera BUS ERROR — nobody acknowledged 0x21; check SIOD/SIOC and RESET");
            ov7670_driver::Identity {
                product: 0,
                version: 0,
                manufacturer_high: 0,
                manufacturer_low: 0,
            }
        }
    };

    // ⚠️ Configure only a sensor that answered — writing a register table
    // into silence would look like configuration and be nothing of the
    // kind. Then read the format registers BACK: `let _ = apply(..)` is
    // what the first version did, and a sensor left in its power-on
    // default (VGA, YUV) while the host decodes RGB565 produces exactly
    // the full-entropy noise that cost an evening.
    if identity.product == ov7670_driver::EXPECTED_PRODUCT_ID {
        let _ = sensor.reset();
        Timer::after_millis(SETTLE_MS).await;
        let mut applied = sensor.apply(QQVGA_RGB565).is_ok();
        if TEST_PATTERN {
            // YSC bit 7 on: the eight-bar pattern.
            applied &= sensor.write_register(0x71, 0x35 | 0x80).is_ok();
        }
        Timer::after_millis(SETTLE_MS).await;
        let com7 = sensor.read_register(0x12).unwrap_or(0xEE);
        let com15 = sensor.read_register(0x40).unwrap_or(0xEE);
        CONFIG_READBACK.store(
            u32::from(applied) | (u32::from(com7) << 8) | (u32::from(com15) << 16),
            Ordering::Relaxed,
        );
    }

    crate::diag::note(&text);
    CACHED.store(
        u32::from_be_bytes([
            identity.product,
            identity.version,
            identity.manufacturer_high,
            identity.manufacturer_low,
        ]),
        Ordering::Relaxed,
    );
    SEEN.store(identity.product != 0, Ordering::Relaxed);

    let mut pio = Pio::new(pins.pio, CameraIrqs);
    // ⚠️ HREF and PCLK become PIO pins even though the program only ever
    // `wait`s on them. An earlier version probed them as `Input`s and
    // dropped them — and embassy restores the pad on `Drop`, which left
    // the input buffer disabled and `wait gpio` reading a dead line
    // forever. `make_pio_pin` makes PIO own the pad, so no diagnostic can
    // leave it in a state the capture then depends on.
    let _href = pio.common.make_pio_pin(pins.href);
    let _pclk = pio.common.make_pio_pin(pins.pclk);
    let d0 = pio.common.make_pio_pin(pins.data.0);
    let d1 = pio.common.make_pio_pin(pins.data.1);
    let d2 = pio.common.make_pio_pin(pins.data.2);
    let d3 = pio.common.make_pio_pin(pins.data.3);
    let d4 = pio.common.make_pio_pin(pins.data.4);
    let d5 = pio.common.make_pio_pin(pins.data.5);
    let d6 = pio.common.make_pio_pin(pins.data.6);
    let d7 = pio.common.make_pio_pin(pins.data.7);
    let all_data = [&d0, &d1, &d2, &d3, &d4, &d5, &d6, &d7];

    // ⚠️ A FIXED word count per line, re-armed at every HREF. Three
    // programs ran on hardware; the differences are the design:
    //
    //   1. free-running (wait HREF / wait PCLK / `in`): bar order correct
    //      within rows, rows sheared — the stride was inferred, and wrong.
    //   2. `jmp pin` until HREF fell: WORSE — the words a line yields
    //      then depend on the line's length, which varies.
    //   3. this: count out exactly 6 x 10 = 60 words (= 120 pixels), then
    //      block for the next line. Alignment is asserted, not inferred.
    //
    // The two `wait`s per byte are both needed: with only `wait 1`, a
    // machine arriving while PCLK is already high samples immediately AND
    // on the next edge — one duplicated byte per line. `mov isr, null`
    // drops any partial word at line start. And the counter is nested
    // `set` loops rather than a value pulled from the TX FIFO, because
    // `capture_frame` clears the FIFOs before every frame and
    // `clear_fifos()` clears TX too — a pushed count was deleted before
    // the program ever read it, and the board captured nothing at all.
    // (`set` takes a literal 0..=31, hence two loops for 60.)
    let program = embassy_rp::pio::program::pio_asm!(
        ".wrap_target",
        "wait 0 gpio 22", // blanking — the previous line ended
        "mov isr, null",  // drop any partial word it left
        "wait 1 gpio 22", // this line starts HERE
        "set y, 5",       // outer: 6 passes
        "outer:",
        "set x, 9", // inner: 10 words each
        "inner:",
        "wait 0 gpio 0",
        "wait 1 gpio 0",
        "in pins, 8",
        "wait 0 gpio 0",
        "wait 1 gpio 0",
        "in pins, 8",
        "wait 0 gpio 0",
        "wait 1 gpio 0",
        "in pins, 8",
        "wait 0 gpio 0",
        "wait 1 gpio 0",
        "in pins, 8",
        "jmp x-- inner",
        "jmp y-- outer",
        ".wrap",
    );

    let mut config = PioConfig::default();
    config.use_program(&pio.common.load_program(&program.program), &[]);
    config.set_in_pins(&all_data);
    // Autopush at 32 bits: four bytes per FIFO word, and [`pixel`] is the
    // one place that knows how they come back out.
    config.shift_in = ShiftConfig {
        threshold: 32,
        direction: ShiftDirection::Left,
        auto_fill: true,
    };
    pio.sm0.set_config(&config);
    pio.sm0.set_pin_dirs(Direction::In, &all_data);

    let dma = embassy_rp::dma::Channel::new(pins.dma, DmaIrqs);
    let vsync = Input::new(pins.vsync, Pull::None);
    let buffer = PICTURE.init([0u32; FRAME_WORDS]);

    // Capture gets its own task: a frame takes tens of milliseconds and
    // must never sit inside the 10 kHz encoder sampler. Same rule as
    // `diag`: nothing may block the thing being measured.
    spawner.spawn(watch(pio, dma, vsync, buffer).unwrap());
    // The watch task never touches I2C — capture is PIO and DMA — so the
    // bus leaves with the caller rather than dying in scope here.
    (xclk, sensor.free())
}

/// The pixel at `index`, out of the word-packed frame buffer.
///
/// ⚠️ **The only place the byte order is known.** Shift-left autopush
/// puts the first-sampled byte in the word's HIGH byte, `to_le_bytes`
/// hands the four back reversed, and the OV7670 sends each pixel high
/// byte first — so the pair is reassembled big-endian. Measured against
/// the sensor's own colour bars: little-endian pairing produced `6CF7`
/// where the known-good yellow is `F76C`, and the earlier capture only
/// *looked* right because an arbitrary DMA start offset re-paired every
/// pixel — two errors cancelling. Detection and thumbnail both come
/// through here, so they cannot disagree about what a pixel is.
fn pixel(buffer: &[u32], index: usize) -> u16 {
    let [a, b, c, d] = buffer[index / 2].to_le_bytes();
    if index.is_multiple_of(2) {
        u16::from_be_bytes([a, b])
    } else {
        u16::from_be_bytes([c, d])
    }
}

/// Every pixel of the frame, in raster order, through [`pixel`].
fn pixels(buffer: &[u32]) -> impl Iterator<Item = u16> + '_ {
    (0..buffer.len() * 2).map(|index| pixel(buffer, index))
}

/// Capture frames forever; hunt the brightest patch in each.
#[embassy_executor::task]
async fn watch(
    mut pio: Pio<'static, PIO0>,
    mut dma: embassy_rp::dma::Channel<'static>,
    mut vsync: Input<'static>,
    buffer: &'static mut [u32; FRAME_WORDS],
) -> ! {
    loop {
        if !capture_frame(&mut pio, &mut dma, &mut vsync, buffer).await {
            // A frame that never arrived. Say nothing rather than publish
            // a stale blob as though it were current.
            BLOB_AREA.store(0, Ordering::Relaxed);
            Timer::after_millis(100).await;
            continue;
        }
        FRAMES.fetch_add(1, Ordering::Relaxed);

        let (mut low, mut high) = (u8::MAX, u8::MIN);
        for word in buffer.iter() {
            for byte in word.to_le_bytes() {
                low = low.min(byte);
                high = high.max(byte);
            }
        }
        PIXEL_RANGE.store(u32::from(low) | (u32::from(high) << 8), Ordering::Relaxed);

        // Pass one: the frame's mean brightness. Pass two: the blob of
        // pixels well above it. Two passes over 14,400 pixels is cheap
        // next to the frame that took milliseconds to arrive, and the
        // relative threshold is what makes this work in any light.
        let mean_luma = (pixels(buffer).map(|p| u32::from(blob::luma(p))).sum::<u32>()
            / (FRAME_WIDTH as u32 * FRAME_HEIGHT as u32)) as u8;
        let floor = mean_luma.saturating_add(BRIGHT_MARGIN);
        match blob::find_matching(pixels(buffer), FRAME_WIDTH, BRIGHT_MIN_PIXELS, |p| {
            blob::luma(p) > floor
        }) {
            Some(found) => {
                // The error is computed HERE, on the real blob, and the
                // atomics carry the finished answer. The first version
                // stored the centroid and had `blob_error` fabricate a
                // `Blob` with sentinel bounds just to borrow a method —
                // a value with lying fields, waiting to be trusted.
                let (x, y) = found.error_from_centre();
                let (x_milli, y_milli) = ((x * 1000.0) as i16, (y * 1000.0) as i16);
                BLOB_ERROR.store(
                    (u32::from(x_milli as u16) << 16) | u32::from(y_milli as u16),
                    Ordering::Relaxed,
                );
                BLOB_AREA.store(found.area.max(1), Ordering::Relaxed);
            }
            None => BLOB_AREA.store(0, Ordering::Relaxed),
        }

        if FRAMES.load(Ordering::Relaxed).is_multiple_of(THUMBNAIL_EVERY) {
            send_thumbnail(buffer).await;
        }
    }
}

/// Capture one whole frame, starting at a real frame boundary.
///
/// ```text
///   wait for VSYNC high   -- frame ending, blanking begins
///   wait for VSYNC low    -- THIS is the start of a new frame
///   enable the state machine
/// ```
///
/// ⚠️ The state machine is **enabled** at the boundary rather than merely
/// unblocked: a program that waits internally still holds the previous
/// frame's tail in its FIFO, and that stale word becomes this frame's
/// first pixel — a picture shifted by a few bytes, which reads as a
/// scaling bug rather than a synchronisation one.
async fn capture_frame(
    pio: &mut Pio<'static, PIO0>,
    dma: &mut embassy_rp::dma::Channel<'static>,
    vsync: &mut Input<'static>,
    buffer: &mut [u32],
) -> bool {
    pio.sm0.set_enable(false);
    pio.sm0.clear_fifos();

    let synced = embassy_time::with_timeout(Duration::from_millis(200), async {
        vsync.wait_for_high().await;
        vsync.wait_for_low().await;
    })
    .await
    .is_ok();
    if !synced {
        return false;
    }

    pio.sm0.set_enable(true);
    // ⚠️ Bounded. A frame that never completes — HREF stuck, PCLK dead —
    // would otherwise hang this task forever, reporting nothing, which
    // reads exactly like a crash.
    let filled = embassy_time::with_timeout(
        Duration::from_millis(500),
        pio.sm0.rx().dma_pull(dma, buffer, false),
    )
    .await
    .is_ok();
    pio.sm0.set_enable(false);
    filled
}

/// Thumbnail geometry: every 4th pixel of every 4th row.
///
/// ⚠️ Small on purpose — the wire format and the reasons live in
/// `hil_protocol::thumbnail`. Nearest-neighbour rather than averaging,
/// because a thumbnail exists to be *looked at*, and sampling is honest
/// about aliasing in a way a smooth wrong picture is not.
const THUMB_STEP: usize = 4;
const THUMB_WIDTH: usize = FRAME_WIDTH as usize / THUMB_STEP;
const THUMB_HEIGHT: usize = FRAME_HEIGHT as usize / THUMB_STEP;

/// Send a downsampled copy of the frame to the host, as hex notes.
async fn send_thumbnail(buffer: &[u32]) {
    let mut line: heapless::String<192> = heapless::String::new();
    let _ = hil_protocol::thumbnail::write_header(&mut line, THUMB_WIDTH, THUMB_HEIGHT);
    crate::diag::note_blocking(&line).await;
    for row in 0..THUMB_HEIGHT {
        line.clear();
        let _ = line.push_str(hil_protocol::thumbnail::ROW_PREFIX);
        for column in 0..THUMB_WIDTH {
            let index = row * THUMB_STEP * FRAME_WIDTH as usize + column * THUMB_STEP;
            let _ = write!(line, "{:04X}", pixel(buffer, index));
        }
        crate::diag::note_blocking(&line).await;
    }
}

/// Total frames captured since boot. The chase loop and the servo
/// tracker both watch this to tell a live image from a wedged one.
#[cfg(any(feature = "chase", feature = "arm"))]
pub fn frame_total() -> u32 {
    FRAMES.load(Ordering::Relaxed)
}

/// Where the brightest patch is, as offsets from frame centre in
/// **−1..+1** (image convention: `y` positive is below centre), plus how
/// many pixels matched. `None` means the last frame had no match — a
/// real answer, distinct from a wedged camera, which [`frame_total`]
/// exposes instead.
pub fn blob_error() -> Option<(f32, f32, u32)> {
    let area = BLOB_AREA.load(Ordering::Relaxed);
    if area == 0 {
        return None;
    }
    let packed = BLOB_ERROR.load(Ordering::Relaxed);
    let x = f32::from((packed >> 16) as u16 as i16) / 1000.0;
    let y = f32::from(packed as u16 as i16) / 1000.0;
    Some((x, y, area))
}

/// Say what the camera is, again.
///
/// ⚠️ A boot-time fact gets repeated because a note said once is a note
/// nobody hears: `diag::NOTES` is drained only while a host is attached,
/// the camera is identified milliseconds after power-up, and merely
/// *opening* the port consumes the queue — `stty` probing and closing
/// ate the line twice while the camera was provably fine.
pub fn announce() {
    if !SEEN.load(Ordering::Relaxed) {
        return;
    }
    let [product, version, mid_high, mid_low] = CACHED.load(Ordering::Relaxed).to_be_bytes();
    let identity = ov7670_driver::Identity {
        product,
        version,
        manufacturer_high: mid_high,
        manufacturer_low: mid_low,
    };
    let range = PIXEL_RANGE.load(Ordering::Relaxed);
    let (low, high) = (range & 0xFF, (range >> 8) & 0xFF);
    let fps = {
        let now = FRAMES.load(Ordering::Relaxed);
        now.wrapping_sub(LAST_ANNOUNCED_FRAMES.swap(now, Ordering::Relaxed)) / ANNOUNCE_SECONDS
    };

    let mut text: heapless::String<192> = heapless::String::new();
    let _ = write!(text, "# camera {identity} | px {low}..{high} | {fps} fps");

    // What the sensor says its format actually is — see CONFIG_READBACK.
    let config = CONFIG_READBACK.load(Ordering::Relaxed);
    let (com7, com15) = ((config >> 8) & 0xFF, (config >> 16) & 0xFF);
    let _ = write!(
        text,
        " | com7=0x{com7:02X} com15=0x{com15:02X} {}",
        if config & 1 == 0 {
            "⚠️ WRITES FAILED"
        } else if com7 == 0x04 && com15 == 0xD0 {
            "format confirmed"
        } else {
            "⚠️ FORMAT NOT SET — sensor ignored the table"
        }
    );
    let _ = match blob_error() {
        Some((x, y, area)) => write!(text, " | blob x{x:+.2} y{y:+.2} area {area}"),
        None => write!(text, " | no blob"),
    };
    crate::diag::note(&text);
}
