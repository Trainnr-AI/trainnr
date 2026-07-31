//! The fast/slow handoff: name a target once, then track it cheaply.
//!
//! Measured on this laptop: Grounding DINO takes **3352 ms** per frame
//! (0.30 fps); D-FINE-N takes **52 ms** (19.1 fps). Open vocabulary cannot
//! be in a control loop, and a closed-set detector cannot understand
//! "the red mug". Neither is usable alone.
//!
//! So use each for what it is good at:
//!
//! ```text
//!   ONCE, at 0.3 fps        EVERY FRAME, at 20 fps
//!   ─────────────────       ──────────────────────
//!   "red mug"               D-FINE -> [cup, person, book, ...]
//!      |                       |
//!   Grounding DINO box   ──>  TargetLock  ──>  filter by class + hue
//!      + dominant_hue                             |
//!                                            pick_target -> steer
//! ```
//!
//! # The bridge is spatial, not lexical
//!
//! The obvious idea — take Grounding DINO's label and look it up in
//! COCO — does not work. The phrase is free-form ("mug", "coffee cup",
//! "the thing I drink from") and COCO's vocabulary is fixed and different
//! ("cup"). String matching would fail on exactly the flexible phrasing
//! that made open vocabulary worth having.
//!
//! Instead both detectors run on the **same frame** during acquisition and
//! we match boxes by overlap. Whatever COCO class the closed-set detector
//! reports *in the same place* is the class to track. That works
//! regardless of what either model calls it.
//!
//! # Colour is arithmetic, not inference
//!
//! Attributes are where CLIP-style text embeddings are weakest — see the
//! module docs in [`crate::openvocab`]. So "red" is not asked of any
//! model: [`dominant_hue`] measures it from the pixels, and the lock
//! carries a hue to match against. That is what distinguishes *the red*
//! mug from the blue one standing next to it, which no COCO class can do.
//!
//! # What this cannot do, and says so
//!
//! If nothing in COCO-80 overlaps the named object, [`TargetLock::acquire`]
//! returns `None`. "Find my keys" has no closed-set equivalent, and the
//! honest answer is to report that rather than lock onto the nearest
//! plausible box and chase a wallet.

use crate::detect::Detection;
use crate::openvocab::{dominant_hue, hue_name};
use crate::Frame;

/// Minimum box overlap for the open-vocab hit and a closed-set detection
/// to be considered the same object.
///
/// Deliberately loose. The two models were trained separately and draw
/// noticeably different boxes around the same thing — Grounding DINO tends
/// to include more context, DETR-family models hug the object. Demanding a
/// tight overlap would reject correct matches; 0.3 is enough to exclude a
/// different object elsewhere in the frame.
pub const MIN_OVERLAP: f32 = 0.30;

/// How far a hue may drift and still count as the same colour, degrees.
///
/// Generous, because hue moves with lighting: the same mug under a warm
/// bulb and by a window can differ by 20-30°. Too tight and the lock drops
/// every time the robot turns.
pub const HUE_TOLERANCE: f32 = 35.0;

/// Minimum saturation before colour is treated as meaningful.
///
/// Hue is undefined for grey. A white mug has a hue, arithmetically, and
/// it is noise. Below this the lock ignores colour entirely and matches on
/// class alone.
pub const MIN_SATURATION: f32 = 0.25;

/// Intersection over union of two boxes, in `[0, 1]`.
pub fn iou(a: &Detection, b: &Detection) -> f32 {
    let ax2 = a.x + a.width;
    let ay2 = a.y + a.height;
    let bx2 = b.x + b.width;
    let by2 = b.y + b.height;

    let iw = (ax2.min(bx2) - a.x.max(b.x)).max(0.0);
    let ih = (ay2.min(by2) - a.y.max(b.y)).max(0.0);
    let intersection = iw * ih;

    let union = a.width * a.height + b.width * b.height - intersection;
    if union <= 0.0 {
        return 0.0;
    }
    intersection / union
}

/// Shortest angular distance between two hues, degrees, in `[0, 180]`.
///
/// Hue is circular: 359° and 1° are 2° apart, not 358°. The same wrap-
/// around problem [`dominant_hue`] solves by averaging unit vectors.
pub fn hue_distance(a: f32, b: f32) -> f32 {
    let d = (a - b).abs() % 360.0;
    if d > 180.0 {
        360.0 - d
    } else {
        d
    }
}

/// A target identified once by language, expressed so a fast detector can
/// track it.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetLock {
    /// COCO class the closed-set detector should look for.
    pub class_id: u16,
    /// That class's own label, e.g. `"cup"`.
    pub class_label: String,
    /// The phrase the user actually asked for, e.g. `"red mug"`. Kept for
    /// reporting: the two rarely match, and showing both is what makes the
    /// handoff legible when it goes wrong.
    pub phrase: String,
    /// Measured hue of the named object, if it had a saturated colour.
    /// `None` means match on class alone.
    pub hue: Option<f32>,
}

impl TargetLock {
    /// Translate an open-vocabulary hit into something trackable.
    ///
    /// `hit` is the box the slow detector found for `phrase`; `candidates`
    /// are the fast detector's boxes **on the same frame**. Returns `None`
    /// when no candidate overlaps enough — the named object has no
    /// closed-set equivalent, and the caller must say so.
    pub fn acquire(
        phrase: &str,
        hit: &Detection,
        candidates: &[Detection],
        frame: &Frame,
    ) -> Option<TargetLock> {
        let best = candidates
            .iter()
            .map(|c| (c, iou(hit, c)))
            .filter(|(_, overlap)| *overlap >= MIN_OVERLAP)
            .max_by(|(_, x), (_, y)| x.total_cmp(y))
            .map(|(c, _)| c)?;

        // Colour is measured from the pixels the SLOW detector boxed: it
        // located the object we actually named.
        let hue = dominant_hue(frame, hit)
            .filter(|(_, saturation)| *saturation >= MIN_SATURATION)
            .map(|(hue, _)| hue);

        Some(TargetLock {
            class_id: best.class_id,
            class_label: best.label.clone(),
            phrase: phrase.to_string(),
            hue,
        })
    }

    /// Does this detection look like the locked target?
    pub fn matches(&self, det: &Detection, frame: &Frame) -> bool {
        if det.class_id != self.class_id {
            return false;
        }
        match self.hue {
            None => true,
            Some(wanted) => match dominant_hue(frame, det) {
                // Unsaturated now, but the target had colour: accept it.
                // Shadow and motion blur wash colour out constantly, and
                // dropping the lock every time is worse than the odd
                // false positive within an already class-matched set.
                None => true,
                Some((hue, saturation)) => {
                    saturation < MIN_SATURATION || hue_distance(hue, wanted) <= HUE_TOLERANCE
                }
            },
        }
    }

    /// The best matching detection, or `None` if the target is not in
    /// frame — which the caller must treat as "target lost": stop, and
    /// reset the controller so a reappearing object inherits no integral.
    ///
    /// The frame is only read while filtering, so its borrow deliberately
    /// does NOT tie to the returned reference — otherwise callers could
    /// not hand the frame on to the viewer afterwards.
    pub fn pick<'a>(&self, detections: &'a [Detection], frame: &Frame) -> Option<&'a Detection> {
        detections
            .iter()
            .filter(|d| d.confidence.is_finite())
            .filter(|d| self.matches(d, frame))
            .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
    }

    /// One-line description for the console.
    pub fn describe(&self) -> String {
        match self.hue {
            Some(h) => format!(
                "{:?} -> tracking COCO '{}' that is {} (hue {h:.0}°)",
                self.phrase,
                self.class_label,
                hue_name(h)
            ),
            None => format!(
                "{:?} -> tracking COCO '{}' (no reliable colour)",
                self.phrase, self.class_label
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxed(x: f32, y: f32, w: f32, h: f32, class_id: u16, label: &str) -> Detection {
        Detection {
            x,
            y,
            width: w,
            height: h,
            confidence: 0.9,
            label: label.into(),
            class_id,
        }
    }

    /// Frame filled with one solid colour, so `dominant_hue` is predictable.
    fn solid(rgb: [u8; 3]) -> Frame {
        let (w, h) = (64u32, 64u32);
        Frame {
            width: w,
            height: h,
            rgb: rgb
                .iter()
                .cycle()
                .take((w * h * 3) as usize)
                .copied()
                .collect(),
        }
    }

    // ---- iou ----

    #[test]
    fn identical_boxes_overlap_completely() {
        let a = boxed(10.0, 10.0, 20.0, 20.0, 0, "x");
        assert!((iou(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn disjoint_boxes_do_not_overlap() {
        let a = boxed(0.0, 0.0, 10.0, 10.0, 0, "x");
        let b = boxed(100.0, 100.0, 10.0, 10.0, 0, "y");
        assert_eq!(iou(&a, &b), 0.0);
    }

    #[test]
    fn touching_edges_do_not_count_as_overlap() {
        let a = boxed(0.0, 0.0, 10.0, 10.0, 0, "x");
        let b = boxed(10.0, 0.0, 10.0, 10.0, 0, "y");
        assert_eq!(iou(&a, &b), 0.0);
    }

    #[test]
    fn half_overlap_matches_the_hand_computed_value() {
        // Two 10x10 boxes sharing a 5x10 strip: inter 50, union 150.
        let a = boxed(0.0, 0.0, 10.0, 10.0, 0, "x");
        let b = boxed(5.0, 0.0, 10.0, 10.0, 0, "y");
        assert!((iou(&a, &b) - 50.0 / 150.0).abs() < 1e-6);
    }

    #[test]
    fn a_contained_box_scores_the_area_ratio() {
        let outer = boxed(0.0, 0.0, 10.0, 10.0, 0, "x");
        let inner = boxed(2.5, 2.5, 5.0, 5.0, 0, "y");
        assert!((iou(&outer, &inner) - 25.0 / 100.0).abs() < 1e-6);
    }

    #[test]
    fn zero_area_boxes_do_not_divide_by_zero() {
        let a = boxed(0.0, 0.0, 0.0, 0.0, 0, "x");
        assert_eq!(iou(&a, &a), 0.0);
    }

    #[test]
    fn iou_is_symmetric() {
        let a = boxed(0.0, 0.0, 10.0, 12.0, 0, "x");
        let b = boxed(3.0, 4.0, 9.0, 9.0, 0, "y");
        assert!((iou(&a, &b) - iou(&b, &a)).abs() < 1e-6);
    }

    // ---- hue_distance ----

    #[test]
    fn hue_distance_wraps_around_the_circle() {
        // The whole reason this function exists: 359 and 1 are 2 apart.
        assert!((hue_distance(359.0, 1.0) - 2.0).abs() < 1e-4);
        assert!((hue_distance(1.0, 359.0) - 2.0).abs() < 1e-4);
    }

    #[test]
    fn hue_distance_is_zero_for_equal_hues() {
        assert_eq!(hue_distance(120.0, 120.0), 0.0);
    }

    #[test]
    fn opposite_hues_are_180_apart_at_most() {
        assert!((hue_distance(0.0, 180.0) - 180.0).abs() < 1e-4);
        for (a, b) in [(0.0, 190.0), (350.0, 100.0), (45.0, 300.0)] {
            assert!(hue_distance(a, b) <= 180.0, "{a} vs {b}");
        }
    }

    // ---- acquire ----

    #[test]
    fn a_named_object_locks_onto_the_overlapping_coco_class() {
        let frame = solid([255, 0, 0]); // red
        let hit = boxed(10.0, 10.0, 20.0, 20.0, 0, "red mug");
        let candidates = [
            boxed(60.0, 60.0, 10.0, 10.0, 0, "person"),
            boxed(11.0, 11.0, 19.0, 19.0, 41, "cup"), // same place
        ];

        let lock = TargetLock::acquire("red mug", &hit, &candidates, &frame).expect("a lock");
        assert_eq!(lock.class_id, 41);
        assert_eq!(lock.class_label, "cup");
        assert_eq!(lock.phrase, "red mug");
        assert_eq!(hue_name(lock.hue.expect("red is saturated")), "red");
    }

    #[test]
    fn the_lexical_mismatch_does_not_matter() {
        // "mug" is not a COCO class; "cup" is. The bridge is spatial, so
        // the phrasing is irrelevant — which is the whole point.
        let frame = solid([255, 0, 0]);
        let hit = boxed(10.0, 10.0, 20.0, 20.0, 0, "the thing I drink from");
        let candidates = [boxed(10.0, 10.0, 20.0, 20.0, 41, "cup")];

        let lock = TargetLock::acquire("the thing I drink from", &hit, &candidates, &frame);
        assert_eq!(lock.expect("a lock").class_id, 41);
    }

    #[test]
    fn an_object_coco_does_not_know_yields_no_lock() {
        // "my keys" — Grounding DINO finds it, nothing in COCO-80 overlaps.
        // Reporting failure is correct; locking onto a distant box is not.
        let frame = solid([255, 0, 0]);
        let hit = boxed(10.0, 10.0, 20.0, 20.0, 0, "my keys");
        let candidates = [boxed(200.0, 200.0, 20.0, 20.0, 0, "person")];

        assert!(TargetLock::acquire("my keys", &hit, &candidates, &frame).is_none());
    }

    #[test]
    fn no_candidates_at_all_yields_no_lock() {
        let frame = solid([255, 0, 0]);
        let hit = boxed(10.0, 10.0, 20.0, 20.0, 0, "mug");
        assert!(TargetLock::acquire("mug", &hit, &[], &frame).is_none());
    }

    #[test]
    fn the_best_overlapping_candidate_wins() {
        let frame = solid([255, 0, 0]);
        let hit = boxed(10.0, 10.0, 20.0, 20.0, 0, "mug");
        let candidates = [
            boxed(16.0, 16.0, 20.0, 20.0, 1, "poor"), // overlaps a bit
            boxed(10.0, 10.0, 20.0, 20.0, 41, "cup"), // exact
        ];
        let lock = TargetLock::acquire("mug", &hit, &candidates, &frame).unwrap();
        assert_eq!(lock.class_id, 41, "picked the worse overlap");
    }

    #[test]
    fn a_grey_object_locks_on_class_without_colour() {
        let frame = solid([128, 128, 128]);
        let hit = boxed(10.0, 10.0, 20.0, 20.0, 0, "mug");
        let candidates = [boxed(10.0, 10.0, 20.0, 20.0, 41, "cup")];

        let lock = TargetLock::acquire("mug", &hit, &candidates, &frame).unwrap();
        assert_eq!(lock.hue, None, "grey must not produce a hue");
        assert!(lock.describe().contains("no reliable colour"));
    }

    // ---- matches / pick ----

    fn red_lock() -> TargetLock {
        TargetLock {
            class_id: 41,
            class_label: "cup".into(),
            phrase: "red mug".into(),
            hue: Some(0.0),
        }
    }

    #[test]
    fn a_different_class_never_matches() {
        let frame = solid([255, 0, 0]);
        let person = boxed(0.0, 0.0, 10.0, 10.0, 0, "person");
        assert!(!red_lock().matches(&person, &frame));
    }

    #[test]
    fn the_right_class_in_the_right_colour_matches() {
        let frame = solid([255, 0, 0]);
        let cup = boxed(0.0, 0.0, 10.0, 10.0, 41, "cup");
        assert!(red_lock().matches(&cup, &frame));
    }

    #[test]
    fn the_right_class_in_the_wrong_colour_does_not() {
        // THE point of the whole module: two cups, one red, one blue.
        // Class alone cannot separate them.
        let frame = solid([0, 0, 255]);
        let cup = boxed(0.0, 0.0, 10.0, 10.0, 41, "cup");
        assert!(!red_lock().matches(&cup, &frame));
    }

    #[test]
    fn a_lock_without_colour_accepts_any_hue() {
        let mut lock = red_lock();
        lock.hue = None;
        let cup = boxed(0.0, 0.0, 10.0, 10.0, 41, "cup");
        assert!(lock.matches(&cup, &solid([0, 0, 255])));
        assert!(lock.matches(&cup, &solid([255, 0, 0])));
    }

    #[test]
    fn colour_washed_out_by_shadow_keeps_the_lock() {
        // Motion blur and shadow desaturate constantly. Dropping the lock
        // every time would make tracking useless.
        let frame = solid([128, 128, 128]);
        let cup = boxed(0.0, 0.0, 10.0, 10.0, 41, "cup");
        assert!(red_lock().matches(&cup, &frame));
    }

    #[test]
    fn lighting_drift_within_tolerance_keeps_the_lock() {
        // Hue 350 vs a target of 0 is 10 degrees — inside tolerance, and
        // only correct if the wrap-around is handled.
        let lock = red_lock();
        assert!(hue_distance(350.0, lock.hue.unwrap()) <= HUE_TOLERANCE);
    }

    #[test]
    fn pick_returns_the_most_confident_match_not_the_first() {
        let frame = solid([255, 0, 0]);
        let mut weak = boxed(0.0, 0.0, 10.0, 10.0, 41, "cup");
        weak.confidence = 0.4;
        let mut strong = boxed(20.0, 20.0, 10.0, 10.0, 41, "cup");
        strong.confidence = 0.95;

        let dets = [weak, strong];
        let picked = red_lock().pick(&dets, &frame).expect("a match");
        assert!((picked.confidence - 0.95).abs() < 1e-6);
    }

    #[test]
    fn pick_ignores_more_confident_detections_of_the_wrong_class() {
        let frame = solid([255, 0, 0]);
        let mut person = boxed(0.0, 0.0, 10.0, 10.0, 0, "person");
        person.confidence = 0.99;
        let mut cup = boxed(20.0, 20.0, 10.0, 10.0, 41, "cup");
        cup.confidence = 0.42;

        let dets = [person, cup];
        let picked = red_lock().pick(&dets, &frame).expect("a match");
        assert_eq!(picked.class_id, 41, "chased the person");
    }

    #[test]
    fn nothing_in_frame_means_target_lost() {
        let frame = solid([255, 0, 0]);
        let dets = [boxed(0.0, 0.0, 10.0, 10.0, 0, "person")];
        assert!(red_lock().pick(&dets, &frame).is_none());
        assert!(red_lock().pick(&[], &frame).is_none());
    }

    #[test]
    fn nan_confidence_cannot_win_here_either() {
        // Same defect as pick_target had; this filter is separate code and
        // needs its own guard.
        let frame = solid([255, 0, 0]);
        let mut nan = boxed(0.0, 0.0, 10.0, 10.0, 41, "cup");
        nan.confidence = f32::NAN;
        let mut good = boxed(20.0, 20.0, 10.0, 10.0, 41, "cup");
        good.confidence = 0.5;

        let dets = [nan, good];
        let picked = red_lock().pick(&dets, &frame).unwrap();
        assert!(!picked.confidence.is_nan());
    }

    #[test]
    fn describe_names_both_the_phrase_and_the_coco_class() {
        // The two rarely match, and showing both is what makes a wrong
        // handoff legible instead of mysterious.
        let d = red_lock().describe();
        assert!(d.contains("red mug"), "{d}");
        assert!(d.contains("cup"), "{d}");
    }
}
