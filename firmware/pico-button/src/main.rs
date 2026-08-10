//! H1 — inputs and PWM. A button cycles an LED's brightness.
//!
//! Three tasks:
//!   heartbeat      — onboard LED, steady 2 Hz: proof the firmware is alive
//!   button_watcher — SLEEPS until the button is pressed (async input!),
//!                    then bumps the brightness level
//!   soft_pwm       — YOUR EXERCISE: drive the LED with a duty cycle
//!
//! PWM in one sentence: switch the LED on/off far faster than the eye can
//! see, and the *fraction of time spent on* (the duty cycle) becomes
//! perceived brightness — same trick that will set motor speed in Stage 3.

#![no_std]
#![no_main]
// No unsafe anywhere in this firmware. `forbid`, not `deny`: it cannot be
// switched off locally with an `#[allow]`. Embassy's HAL already wraps the
// peripheral access that would otherwise need it — if that ever stops being
// true, the argument belongs in a commit that changes this line.
#![forbid(unsafe_code)]

use core::sync::atomic::{AtomicU8, Ordering};
use embassy_executor::Spawner;
use embassy_rp::gpio::{Input, Level, Output, Pull};
use embassy_time::Timer;
use panic_halt as _;

/// Brightness levels (percent duty) the button steps through.
const LEVELS: [u8; 4] = [0, 25, 50, 100];
/// One PWM period: 10 ms = 100 Hz. Fast enough to look steady to a human,
/// slow enough that the emulator log shows clean on/off windows.
const PERIOD_US: u64 = 10_000;

/// The current duty, shared between tasks. An atomic because two tasks
/// touch it — button_watcher writes, soft_pwm reads — and Rust will not
/// let tasks share plain mutable data (that's a data race, a compile
/// error). Atomic load/store is the simplest safe channel there is.
static DUTY_PERCENT: AtomicU8 = AtomicU8::new(0);

/// Proof of life: if this stops beating, the firmware crashed.
/// The async-input showpiece: this task consumes ZERO cpu while idle —
/// `wait_for_falling_edge` puts it to sleep until the pin's interrupt
/// fires (button pressed = pin pulled from high to low).
#[embassy_executor::task]
async fn button_watcher(mut button: Input<'static>) {
    let mut index = 0;
    loop {
        button.wait_for_falling_edge().await;
        index = (index + 1) % LEVELS.len();
        DUTY_PERCENT.store(LEVELS[index], Ordering::Relaxed);
        // Debounce: real buttons "chatter" (bounce) for a few ms on each
        // press; ignoring edges for 30 ms turns chatter into one event.
        Timer::after_millis(30).await;
    }
}

/// EXERCISE H1 — software PWM.
///
/// As shipped, this ignores the duty entirely and drives the LED full-on:
/// the harness will measure ~100% brightness no matter how often the
/// button is pressed. Your job: honor `DUTY_PERCENT`. The recipe:
///
/// 1. Each time around the loop, read the duty:
///      `let duty = DUTY_PERCENT.load(Ordering::Relaxed) as u64;`
/// 2. Compute the on-time within one period:
///      `let on_us = PERIOD_US * duty / 100;`
/// 3. If `on_us > 0`: set the LED high, then `Timer::after_micros(on_us)`.
/// 4. If `on_us < PERIOD_US`: set the LED low, then wait the remainder
///    (`PERIOD_US - on_us`).
///
/// The two `if`s matter: at 0% the LED must never flash on, at 100% it
/// must never flicker off. Success = the harness measures the duty
/// sequence 0 → 25 → 50 → 100 → 0 as the virtual button is pressed.
#[embassy_executor::task]
async fn soft_pwm(mut led: Output<'static>) {
    loop {
        let duty = DUTY_PERCENT.load(Ordering::Relaxed) as u64;
        let on_us = PERIOD_US * duty / 100;
        if on_us > 0 {
            led.set_high();
            Timer::after_micros(on_us).await;
        }
        if on_us < PERIOD_US {
            led.set_low();
            Timer::after_micros(PERIOD_US - on_us).await;
        }
    }
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_rp::init(Default::default());

    let heart = Output::new(p.PIN_25, Level::Low);
    let led = Output::new(p.PIN_15, Level::Low);
    // Pull::Up — the pin idles high through an internal resistor; the
    // button connects it to ground, so "pressed" reads LOW. (Why this
    // convention exists is a Falstad CircuitJS side quest.)
    let button = Input::new(p.PIN_14, Pull::Up);

    spawner.spawn(firmware_support::heartbeat(heart, 250).unwrap());
    spawner.spawn(button_watcher(button).unwrap());
    spawner.spawn(soft_pwm(led).unwrap());
}
