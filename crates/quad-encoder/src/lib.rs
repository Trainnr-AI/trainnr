//! Quadrature decoding — where wheel ticks actually come from.
//!
//! In Stage 0, `sim-core`'s `Encoders::advance()` handed your odometry
//! tick counts out of thin air. This is the real thing: a rotary encoder
//! produces TWO square-wave signals, A and B, offset by a quarter cycle
//! ("quadrature" = quarter). Rotate one way and A leads B; rotate the
//! other way and B leads A. That phase relationship is the ONLY way the
//! robot knows which direction a wheel turned.
//!
//! ```text
//! forward ->
//!  A  ___----____----____
//!  B  _----____----____--     B lags A by 90 degrees
//!
//! backward ->
//!  A  ___----____----____
//!  B  --____----____----_     B leads A by 90 degrees
//! ```
//!
//! Sample both lines and you get a 2-bit state that walks a fixed cycle:
//!
//! ```text
//!   state = (A << 1) | B          A B  state
//!                                 0 0    0
//!   forward:  0 -> 1 -> 3 -> 2 -> 0 1    1
//!   backward: 0 -> 2 -> 3 -> 1 -> 1 1    3
//!                                 1 0    2
//! ```
//!
//! Note only ONE bit changes per step — that's a Gray code, and it's
//! deliberate: if both bits could change at once you couldn't tell a
//! forward step from a backward one. Physical hardware encodes direction
//! into the *geometry* of the sensor. Beautiful, and free.
//!
//! Math/theory: docs/learning/math-09-quadrature.md

#![cfg_attr(not(test), no_std)]

/// Pack the two line levels into a 2-bit state (0..3).
pub fn encode_state(a: bool, b: bool) -> u8 {
    ((a as u8) << 1) | (b as u8)
}

/// EXERCISE H3 — decide which way the shaft moved.
///
/// Given the previous 2-bit state and the current one, return:
///   `+1` forward, `-1` backward, `0` no movement (or an illegal jump).
///
/// The two cycles from the module docs, as lookup tables indexed by the
/// PREVIOUS state (both are already written for you):
///
/// ```text
/// forward:  0 -> 1,  1 -> 3,  2 -> 0,  3 -> 2   =>  [1, 3, 0, 2]
/// backward: 0 -> 2,  1 -> 0,  2 -> 3,  3 -> 1   =>  [2, 0, 3, 1]
/// ```
///
/// The recipe:
/// 1. If `curr` equals `FORWARD_NEXT[prev]`, the shaft advanced: return 1.
/// 2. Else if `curr` equals `BACKWARD_NEXT[prev]`, it reversed: return -1.
/// 3. Else return 0. That covers BOTH "nothing changed" (curr == prev) and
///    "impossible jump" (both bits flipped at once — a missed sample or
///    electrical noise). Returning 0 rather than guessing is the honest
///    choice: a robot that invents motion it didn't see is worse than one
///    that misses a tick.
///
/// Rust notes: index with `prev as usize` (arrays index by usize only);
/// an `if / else if / else` chain of three arms is the whole function.
pub fn step(prev: u8, curr: u8) -> i8 {
    const FORWARD_NEXT: [u8; 4] = [1, 3, 0, 2];
    const BACKWARD_NEXT: [u8; 4] = [2, 0, 3, 1];
    if curr == FORWARD_NEXT[prev as usize] {
        1
    } else if curr == BACKWARD_NEXT[prev as usize] {
        -1
    } else {
        0
    }
}

/// Accumulates position from a stream of A/B samples.
#[derive(Debug, Clone, Copy)]
pub struct QuadratureDecoder {
    prev: u8,
    /// Net position in ticks. Signed: it counts down when reversing.
    pub count: i32,
    /// Transitions that made no sense (noise / sampling too slowly).
    /// On real hardware, a rising number here means trouble.
    pub errors: u32,
}

impl QuadratureDecoder {
    pub fn new(a: bool, b: bool) -> Self {
        QuadratureDecoder {
            prev: encode_state(a, b),
            count: 0,
            errors: 0,
        }
    }

    /// Feed one sample of both lines. Returns the delta (-1, 0, or +1).
    pub fn update(&mut self, a: bool, b: bool) -> i8 {
        let curr = encode_state(a, b);
        let delta = step(self.prev, curr);
        if delta == 0 && curr != self.prev {
            // State changed but not to a legal neighbour: we missed a step.
            self.errors += 1;
        }
        self.count += delta as i32;
        self.prev = curr;
        delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_packing() {
        assert_eq!(encode_state(false, false), 0);
        assert_eq!(encode_state(false, true), 1);
        assert_eq!(encode_state(true, true), 3);
        assert_eq!(encode_state(true, false), 2);
    }

    #[test]
    fn forward_cycle_counts_up() {
        assert_eq!(step(0, 1), 1);
        assert_eq!(step(1, 3), 1);
        assert_eq!(step(3, 2), 1);
        assert_eq!(step(2, 0), 1);
    }

    #[test]
    fn backward_cycle_counts_down() {
        assert_eq!(step(0, 2), -1);
        assert_eq!(step(2, 3), -1);
        assert_eq!(step(3, 1), -1);
        assert_eq!(step(1, 0), -1);
    }

    #[test]
    fn no_change_is_zero() {
        for s in 0..4 {
            assert_eq!(step(s, s), 0);
        }
    }

    #[test]
    fn illegal_double_flip_is_zero() {
        // Both bits changed at once — impossible on a real encoder unless
        // we sampled too slowly. Never guess a direction from this.
        assert_eq!(step(0, 3), 0);
        assert_eq!(step(3, 0), 0);
        assert_eq!(step(1, 2), 0);
        assert_eq!(step(2, 1), 0);
    }

    #[test]
    fn one_full_revolution_forward() {
        // Four states per cycle; 5 cycles = 20 ticks.
        let mut dec = QuadratureDecoder::new(false, false);
        let seq = [(false, true), (true, true), (true, false), (false, false)];
        for _ in 0..5 {
            for &(a, b) in seq.iter() {
                dec.update(a, b);
            }
        }
        assert_eq!(dec.count, 20);
        assert_eq!(dec.errors, 0);
    }

    #[test]
    fn forward_then_back_returns_to_zero() {
        let mut dec = QuadratureDecoder::new(false, false);
        let fwd = [(false, true), (true, true), (true, false), (false, false)];
        for &(a, b) in fwd.iter() {
            dec.update(a, b);
        }
        assert_eq!(dec.count, 4);
        // Same states in reverse order walks the count back down.
        for &(a, b) in fwd.iter().rev().skip(1) {
            dec.update(a, b);
        }
        dec.update(false, false);
        assert_eq!(dec.count, 0);
        assert_eq!(dec.errors, 0);
    }

    #[test]
    fn missed_samples_are_counted_as_errors() {
        let mut dec = QuadratureDecoder::new(false, false);
        dec.update(true, true); // 0 -> 3: illegal jump
        assert_eq!(dec.count, 0); // no motion invented
        assert_eq!(dec.errors, 1); // but the problem is recorded
    }
}
