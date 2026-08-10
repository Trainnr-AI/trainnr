//! USB CDC serial bring-up, in one place.
//!
//! Four firmware crates had a byte-identical copy of this — twenty-three
//! lines of descriptor buffers and builder plumbing whose only variables
//! were a product name and a product ID.

use embassy_rp::peripherals::USB;
use embassy_rp::usb::Driver;
use embassy_usb::class::cdc_acm::{CdcAcmClass, State};
use embassy_usb::{Builder, Config, UsbDevice};
use static_cell::StaticCell;

/// The control endpoint's max packet size, and the CDC data endpoints'.
///
/// 64 is the maximum a USB full-speed device may use, and every firmware
/// here was already using it — two by naming this constant, two by
/// writing the literal. A write longer than this must be split by the
/// caller or the transfer is rejected.
pub const MAX_PACKET: usize = 64;

/// Raspberry Pi's USB vendor ID. Product IDs differ per firmware so that
/// several boards can be plugged in at once and still be told apart.
const VENDOR_ID: u16 = 0x2e8a;

/// Concrete because there is exactly one USB peripheral on this chip and
/// exactly one driver for it. A generic version would need `StaticCell`s
/// inside a generic function, where a `static` is shared across every
/// instantiation rather than duplicated — a subtlety worth not having.
type UsbDriver = Driver<'static, USB>;

/// Builds a USB CDC serial device.
///
/// Returns the device — whose `run()` must be polled, or nothing
/// enumerates — and the class, unsplit, because callers differ: some want
/// `wait_connection()` on the whole class, others `split()` it into a
/// sender and receiver.
///
/// ⚠️ **Call this once per binary.** The buffers below are `static`, so a
/// second call panics inside `StaticCell::init`. That is the intended
/// behaviour: a board cannot have two CDC devices on one USB peripheral,
/// and a loud panic beats two devices quietly sharing one descriptor.
///
/// `driver` is built by the caller because `Driver::new` needs the
/// `bind_interrupts!` struct, and an interrupt binding has to live in the
/// binary that owns the vector table.
pub fn cdc(
    driver: UsbDriver,
    product: &'static str,
    product_id: u16,
) -> (UsbDevice<'static, UsbDriver>, CdcAcmClass<'static, UsbDriver>) {
    let mut config = Config::new(VENDOR_ID, product_id);
    config.manufacturer = Some("robotiq");
    config.product = Some(product);
    config.serial_number = Some("1");
    config.max_power = 100;
    config.max_packet_size_0 = MAX_PACKET as u8;

    static CONFIG_DESC: StaticCell<[u8; 256]> = StaticCell::new();
    static BOS_DESC: StaticCell<[u8; 256]> = StaticCell::new();
    static CONTROL_BUF: StaticCell<[u8; 64]> = StaticCell::new();
    static STATE: StaticCell<State> = StaticCell::new();

    let state = STATE.init(State::new());
    let mut builder = Builder::new(
        driver,
        config,
        CONFIG_DESC.init([0; 256]),
        BOS_DESC.init([0; 256]),
        &mut [], // no Microsoft OS descriptors
        CONTROL_BUF.init([0; 64]),
    );
    let class = CdcAcmClass::new(&mut builder, state, MAX_PACKET as u16);
    (builder.build(), class)
}
