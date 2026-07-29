//! A tiny deterministic random number generator.
//!
//! Why write our own instead of using the `rand` crate? Two reasons:
//! 1. sim-core stays zero-dependency (a design rule of Stage 0).
//! 2. Determinism is a *requirement* here (see docs/06-stage0-design.md):
//!    same seed → bit-identical simulation, so bugs reproduce and tests
//!    can assert exact behavior. Rolling our own makes that visible.
//!
//! The algorithm is xorshift64* — three shifts and a multiply. Not
//! cryptographic, statistically fine for simulation noise.

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // A zero state would get stuck at zero forever; nudge it.
        Rng(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }

    /// Next raw 64-bit value.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        // Take the top 53 bits — exactly an f64's precision.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Roughly normal (bell-curve) noise, mean 0, std deviation ~1.
    /// Sum of 12 uniforms minus 6 — the classic quick approximation
    /// (central limit theorem in action).
    pub fn noise(&mut self) -> f64 {
        (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.uniform(), b.uniform());
        }
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        assert_ne!(a.uniform(), b.uniform());
    }

    #[test]
    fn uniform_stays_in_range_and_covers_it() {
        let mut rng = Rng::new(7);
        let mut min = 1.0_f64;
        let mut max = 0.0_f64;
        for _ in 0..10_000 {
            let u = rng.uniform();
            assert!((0.0..1.0).contains(&u));
            min = min.min(u);
            max = max.max(u);
        }
        assert!(min < 0.05 && max > 0.95); // actually spreads out
    }

    #[test]
    fn noise_is_centered() {
        let mut rng = Rng::new(123);
        let mean: f64 = (0..10_000).map(|_| rng.noise()).sum::<f64>() / 10_000.0;
        assert!(mean.abs() < 0.05);
    }
}
