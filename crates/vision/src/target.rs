//! The glue between "what the camera saw" and "what the robot should do".
//!
//! This logic used to live inside `chase`'s `main()`, where it could not
//! be tested — a `main` that opens a webcam and runs forever is not a unit
//! under test. It is pure arithmetic on detections, so it belongs here.
//!
//! Nothing in this module knows about cameras, models, or Rerun.

use crate::detect::Detection;

/// Pick the detection to chase.
///
/// Highest confidence wins. Returns `None` for an empty slice, which the
/// caller must treat as "target lost" — stop, and reset the controller so
/// a reappearing object does not inherit a stale integral.
///
/// # Why confidence and not size or centrality
///
/// A bigger box is nearer and a centred box is easier to track, so both
/// are tempting. Both are also wrong when the detector is unsure: chasing
/// a large low-confidence blob is exactly how a robot ends up driving at a
/// shadow. Confidence first is the conservative choice, and the
/// `MIN_CONFIDENCE` filter upstream has already discarded the noise.
///
/// # NaN is discarded, not ordered
///
/// `f32::total_cmp` sorts positive NaN **above every finite value**, so a
/// plain `max_by` hands back the NaN detection — the robot would then
/// steer at a box whose confidence is not a number. Caught by
/// `nan_confidence_does_not_panic_or_win`. Non-finite confidences are
/// filtered out first; if that leaves nothing, the target is lost, which
/// is the correct and safe answer.
pub fn pick_target(detections: &[Detection]) -> Option<&Detection> {
    detections
        .iter()
        .filter(|d| d.confidence.is_finite())
        .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
}

/// How much forward speed to allow, from apparent object size.
///
/// Box height as a fraction of frame height is a crude distance proxy:
/// a taller box means a nearer object. Returns 1.0 when far away, falling
/// to 0.0 when the box reaches `stop_at` of the frame.
///
/// **This is open-loop** and the module docs of `chase` say so: our camera
/// is bolted to a laptop, not to the robot, so driving does not change
/// what it sees. Stage 3 closes this loop.
///
/// Clamped at both ends: a box larger than `stop_at` must give 0, never a
/// negative speed that would drive the robot backwards into whatever is
/// behind it.
pub fn approach_factor(box_height: f32, frame_height: u32, stop_at: f32) -> f32 {
    if frame_height == 0 || stop_at <= 0.0 {
        return 0.0;
    }
    let height_fraction = box_height / frame_height as f32;
    (1.0 - height_fraction / stop_at).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(confidence: f32, height: f32) -> Detection {
        Detection {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height,
            confidence,
            label: "thing".into(),
            class_id: 0,
        }
    }

    // ---- pick_target ----

    #[test]
    fn no_detections_means_no_target() {
        assert!(pick_target(&[]).is_none());
    }

    #[test]
    fn the_most_confident_detection_wins() {
        let dets = [det(0.4, 10.0), det(0.9, 10.0), det(0.6, 10.0)];
        let picked = pick_target(&dets).expect("a target");
        assert!((picked.confidence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn a_single_detection_is_the_target() {
        let dets = [det(0.35, 10.0)];
        assert!((pick_target(&dets).unwrap().confidence - 0.35).abs() < 1e-6);
    }

    #[test]
    fn size_does_not_beat_confidence() {
        // The failure this guards: a huge, uncertain blob (a shadow)
        // outranking a small, certain object.
        let dets = [det(0.31, 400.0), det(0.95, 12.0)];
        let picked = pick_target(&dets).unwrap();
        assert!(
            (picked.height - 12.0).abs() < 1e-6,
            "chased the big uncertain blob"
        );
    }

    #[test]
    fn nan_confidence_does_not_panic_or_win() {
        // REGRESSION: total_cmp sorts positive NaN above every finite
        // value, so the original `max_by` returned the NaN detection and
        // the robot would have steered at it.
        let dets = [det(f32::NAN, 10.0), det(0.5, 10.0)];
        let picked = pick_target(&dets).unwrap();
        assert!(!picked.confidence.is_nan(), "NaN won the comparison");
        assert!((picked.confidence - 0.5).abs() < 1e-6);
    }

    #[test]
    fn only_nan_detections_means_target_lost() {
        // Losing the target is the safe answer: chase stops and resets.
        let dets = [det(f32::NAN, 10.0), det(f32::NAN, 20.0)];
        assert!(pick_target(&dets).is_none());
    }

    #[test]
    fn infinite_confidence_is_also_discarded() {
        let dets = [det(f32::INFINITY, 10.0), det(0.5, 10.0)];
        assert!((pick_target(&dets).unwrap().confidence - 0.5).abs() < 1e-6);
    }

    // ---- approach_factor ----

    #[test]
    fn a_distant_object_allows_full_speed() {
        // Tiny box: 1% of the frame.
        assert!((approach_factor(4.8, 480, 0.55) - 1.0).abs() < 0.02);
    }

    #[test]
    fn reaching_the_stop_size_gives_zero_speed() {
        // Box exactly 55% of a 480-pixel frame.
        let f = approach_factor(0.55 * 480.0, 480, 0.55);
        assert!(f.abs() < 1e-6, "expected 0, got {f}");
    }

    #[test]
    fn an_oversized_box_never_drives_backwards() {
        // Object closer than the stop distance: the raw formula goes
        // negative, and the clamp is what stops the robot reversing.
        for h in [0.7, 0.9, 2.0] {
            let f = approach_factor(h * 480.0, 480, 0.55);
            assert!(f >= 0.0, "height {h} gave {f}");
            assert!(f.abs() < 1e-6, "height {h} should stop, gave {f}");
        }
    }

    #[test]
    fn approach_falls_monotonically_as_the_object_nears() {
        let mut previous = f32::INFINITY;
        for step in 0..=10 {
            let height = step as f32 * 0.05 * 480.0;
            let f = approach_factor(height, 480, 0.55);
            assert!(f <= previous + 1e-6, "not monotonic at step {step}");
            previous = f;
        }
    }

    #[test]
    fn half_way_to_the_stop_size_gives_half_speed() {
        let f = approach_factor(0.275 * 480.0, 480, 0.55);
        assert!((f - 0.5).abs() < 1e-5, "expected 0.5, got {f}");
    }

    #[test]
    fn degenerate_inputs_stop_the_robot_rather_than_dividing_by_zero() {
        assert_eq!(approach_factor(10.0, 0, 0.55), 0.0, "zero frame height");
        assert_eq!(approach_factor(10.0, 480, 0.0), 0.0, "zero stop fraction");
        assert_eq!(approach_factor(10.0, 480, -1.0), 0.0, "negative stop");
    }
}
