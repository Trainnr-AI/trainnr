//! Latency statistics — the arithmetic behind `bench`.
//!
//! Small enough to look obviously correct and therefore exactly the kind
//! of thing that is quietly wrong: an off-by-one in a percentile index
//! shifts every published number without ever looking suspicious. It was
//! inline in `bench`'s `main`, so it had never been checked.

/// Summary of a set of timing samples, in whatever unit they were given.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    pub mean: f64,
    pub p50: f64,
    pub p90: f64,
    pub min: f64,
    pub max: f64,
    pub count: usize,
}

impl Stats {
    /// Summarise `samples`. Returns `None` for an empty slice rather than
    /// inventing zeros — a benchmark that produced no samples must not be
    /// reported as "0.0 ms, infinitely fast".
    pub fn from_samples(samples: &[f64]) -> Option<Stats> {
        if samples.is_empty() {
            return None;
        }
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        Some(Stats {
            mean: samples.iter().sum::<f64>() / samples.len() as f64,
            p50: percentile(&sorted, 0.50),
            p90: percentile(&sorted, 0.90),
            min: sorted[0],
            max: sorted[sorted.len() - 1],
            count: samples.len(),
        })
    }

    /// Frames per second implied by the mean latency in milliseconds.
    pub fn fps(&self) -> f64 {
        if self.mean <= 0.0 {
            return f64::INFINITY;
        }
        1000.0 / self.mean
    }
}

/// Nearest-rank percentile of an **already sorted** slice.
///
/// `p` is a fraction in `[0, 1]`. Out-of-range values are clamped rather
/// than panicking — a benchmark should not abort over a bad argument.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let p = p.clamp(0.0, 1.0);
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_samples_is_none_not_zero() {
        // "0.0 ms" would read as infinitely fast in the report.
        assert!(Stats::from_samples(&[]).is_none());
    }

    #[test]
    fn a_single_sample_is_its_own_everything() {
        let s = Stats::from_samples(&[42.0]).unwrap();
        assert_eq!(s.mean, 42.0);
        assert_eq!(s.p50, 42.0);
        assert_eq!(s.p90, 42.0);
        assert_eq!(s.min, 42.0);
        assert_eq!(s.max, 42.0);
        assert_eq!(s.count, 1);
    }

    #[test]
    fn statistics_match_a_hand_computed_set() {
        let s = Stats::from_samples(&[10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
        assert!((s.mean - 30.0).abs() < 1e-12);
        assert_eq!(s.p50, 30.0);
        assert_eq!(s.min, 10.0);
        assert_eq!(s.max, 50.0);
        assert_eq!(s.count, 5);
    }

    #[test]
    fn input_order_does_not_change_the_result() {
        // from_samples must sort internally; a benchmark hands it times in
        // arrival order, not sorted order.
        let a = Stats::from_samples(&[30.0, 10.0, 50.0, 20.0, 40.0]).unwrap();
        let b = Stats::from_samples(&[10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn p90_sits_near_the_slow_tail_not_the_maximum() {
        // 1..=100: the 90th percentile should be ~90, and must NOT be the
        // max — reporting the max as p90 hides exactly the tail latency a
        // control loop cares about.
        let samples: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let s = Stats::from_samples(&samples).unwrap();
        assert!((s.p90 - 90.0).abs() <= 1.0, "p90 = {}", s.p90);
        assert!(s.p90 < s.max);
    }

    #[test]
    fn one_slow_frame_moves_the_max_but_not_the_median() {
        // The realistic pattern: a GC pause or a thermal blip. The median
        // must stay honest about typical performance.
        let mut samples = vec![50.0; 39];
        samples.push(900.0);
        let s = Stats::from_samples(&samples).unwrap();
        assert_eq!(s.p50, 50.0, "one outlier moved the median");
        assert_eq!(s.max, 900.0);
        assert!(s.mean > 50.0, "the mean should notice the outlier");
    }

    #[test]
    fn percentile_bounds_are_the_extremes() {
        let sorted = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(percentile(&sorted, 0.0), 1.0);
        assert_eq!(percentile(&sorted, 1.0), 4.0);
    }

    #[test]
    fn percentile_clamps_rather_than_panicking() {
        let sorted = [1.0, 2.0, 3.0];
        assert_eq!(percentile(&sorted, -5.0), 1.0);
        assert_eq!(percentile(&sorted, 5.0), 3.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
    }

    #[test]
    fn fps_is_the_reciprocal_of_the_mean_in_milliseconds() {
        let s = Stats::from_samples(&[50.0; 10]).unwrap();
        assert!((s.fps() - 20.0).abs() < 1e-9, "50 ms should be 20 fps");
    }

    #[test]
    fn a_zero_mean_does_not_divide_by_zero() {
        let s = Stats::from_samples(&[0.0, 0.0]).unwrap();
        assert!(s.fps().is_infinite());
    }
}
