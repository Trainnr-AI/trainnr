//! Find a coloured object in a frame, on the chip.
//!
//! # Why this is enough to drive a robot
//!
//! Everything else in this project's vision path — detectors, target
//! locks, models — runs on the laptop, because a Pico has no NPU and a
//! QQVGA frame is 38 KB of its 520 KB. What a Pico *can* do is scan those
//! 19,200 pixels for a colour and report where they are.
//!
//! That turns out to be enough, because **visual servoing never asks
//! where anything is**. It asks only for the *sign of the error*:
//!
//! ```text
//!   blob left of centre   ─▶ turn left       blob small  ─▶ drive forward
//!   blob above centre     ─▶ arm up          blob large  ─▶ drive back
//! ```
//!
//! No camera height, no tilt, no hand-eye calibration, no link lengths.
//! One blob yields three independent errors, which is exactly the three
//! motions the rig has.
//!
//! # Hue, not RGB
//!
//! A red object under a desk lamp and the same object by a window are
//! very different *RGB* values and nearly the same *hue*. Matching on hue
//! with a saturation floor is what makes the tracker survive somebody
//! turning a light on.
//!
//! # What this deliberately does NOT do
//!
//! ⚠️ It finds **one** blob by averaging every matching pixel — not
//! connected components, which would need memory the chip should not
//! spend. With two matching objects in frame the centroid lands *between
//! them*, pointing at nothing.
//!
//! That failure is silent, so [`Blob::looks_like_one_object`] exists to
//! catch it: a real blob's bounding box is close to its pixel count,
//! while two distant blobs give a huge box around very few pixels.

#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

/// Circular distance between two hues, in degrees, 0–180.
///
/// ⚠️ Hue wraps: red sits at 0°, so 359° and 1° are two degrees apart,
/// not 358. A tracker that subtracted them naively would lose the target
/// every time it crossed red — which is the colour most people pick.
pub fn hue_distance(a: f32, b: f32) -> f32 {
    let d = (a - b).abs() % 360.0;
    if d > 180.0 {
        360.0 - d
    } else {
        d
    }
}

/// Hue, saturation and value from one RGB565 pixel — the format the
/// OV7670 emits.
///
/// Hue is degrees 0–360; saturation and value are 0–1. Returns hue 0 for
/// greys, where hue is undefined rather than zero — callers must reject
/// those on saturation, which is what [`Target::min_saturation`] is for.
pub fn rgb565_to_hsv(pixel: u16) -> (f32, f32, f32) {
    // 5 bits red, 6 green, 5 blue. Scaled to 0–1 by their own maxima, so
    // green is not systematically brighter than the other two.
    let red = ((pixel >> 11) & 0x1F) as f32 / 31.0;
    let green = ((pixel >> 5) & 0x3F) as f32 / 63.0;
    let blue = (pixel & 0x1F) as f32 / 31.0;

    let max = red.max(green).max(blue);
    let min = red.min(green).min(blue);
    let span = max - min;

    let hue = if span == 0.0 {
        0.0
    } else if max == red {
        60.0 * (((green - blue) / span) % 6.0)
    } else if max == green {
        60.0 * ((blue - red) / span + 2.0)
    } else {
        60.0 * ((red - green) / span + 4.0)
    };
    let hue = if hue < 0.0 { hue + 360.0 } else { hue };
    let saturation = if max == 0.0 { 0.0 } else { span / max };
    (hue, saturation, max)
}

/// The colour being hunted, and how fussy to be.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Target {
    /// Degrees on the colour wheel. Red 0, green 120, blue 240.
    pub hue_degrees: f32,
    /// How far from that hue still counts. Wide enough to survive
    /// lighting, narrow enough not to grab the carpet.
    pub hue_tolerance_degrees: f32,
    /// ⚠️ Below this, hue is meaningless — a grey pixel has no colour to
    /// be near. Without this floor a tracker locks onto shadows.
    pub min_saturation: f32,
    /// Rejects near-black pixels, whose hue is also noise.
    pub min_value: f32,
    /// Fewer matching pixels than this and nothing is reported. A noise
    /// floor, not a size preference.
    pub min_pixels: u32,
}

impl Target {
    /// A saturated colour under ordinary indoor light. Starting values —
    /// they will want tuning against the actual object and lamp, and that
    /// tuning belongs in one place rather than at each call site.
    pub const fn hue(hue_degrees: f32) -> Self {
        Target {
            hue_degrees,
            hue_tolerance_degrees: 25.0,
            min_saturation: 0.4,
            min_value: 0.2,
            min_pixels: 30,
        }
    }

    fn matches(&self, pixel: u16) -> bool {
        let (hue, saturation, value) = rgb565_to_hsv(pixel);
        saturation >= self.min_saturation
            && value >= self.min_value
            && hue_distance(hue, self.hue_degrees) <= self.hue_tolerance_degrees
    }
}

/// Where the matching pixels were.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Blob {
    /// Centre of mass, in pixels from the top-left.
    pub centroid_x: f32,
    pub centroid_y: f32,
    /// How many pixels matched. The distance proxy — closer is bigger.
    pub area: u32,
    /// Tightest box containing every match: `(min_x, min_y, max_x, max_y)`.
    pub bounds: (u16, u16, u16, u16),
    /// The frame this was found in, `(width, height)`.
    ///
    /// Carried here rather than asked for again, because
    /// [`Self::error_from_centre`] used to take them as arguments — which
    /// is the same fact in two places, and a caller passing a different
    /// size than `find` saw would get a confidently wrong steering error.
    pub frame: (u16, u16),
}

impl Blob {
    /// Offsets from frame centre, each **−1 to +1**.
    ///
    /// This is what a controller wants: `x` positive means the object is
    /// to the right, `y` positive means it is *below* centre — image
    /// convention, since that is what the camera hands us. Whoever drives
    /// an arm upward from this is responsible for the sign, once.
    pub fn error_from_centre(&self) -> (f32, f32) {
        let (width, height) = self.frame;
        let half_w = f32::from(width) / 2.0;
        let half_h = f32::from(height) / 2.0;
        (
            (self.centroid_x - half_w) / half_w,
            (self.centroid_y - half_h) / half_h,
        )
    }

    /// Is this plausibly **one** object?
    ///
    /// A single blob fills a fair share of its bounding box. Two objects
    /// on opposite sides of the frame give a huge box around few pixels,
    /// and the centroid then points at the gap between them — confidently,
    /// and at nothing.
    ///
    /// Returns the fill fraction against `at_least`; 0.25 is a permissive
    /// floor — a solid convex object fills 0.5–0.8 of its box.
    ///
    /// ⚠️ **It only catches WELL-SEPARATED blobs, and that limit is real.**
    /// Two objects side by side at the same height fill ~0.43 of the wide,
    /// short box around them and pass this check comfortably, while the
    /// centroid still points at the gap. It catches the diagonal and the
    /// far-apart cases; it does not catch everything, and a caller that
    /// treats a `true` here as proof of one object will occasionally be
    /// steered at nothing.
    pub fn looks_like_one_object(&self, at_least: f32) -> bool {
        let (min_x, min_y, max_x, max_y) = self.bounds;
        let box_area = (u32::from(max_x - min_x) + 1) * (u32::from(max_y - min_y) + 1);
        box_area > 0 && (self.area as f32 / box_area as f32) >= at_least
    }
}

/// Scan a frame for `target`.
///
/// Pixels arrive in raster order, left to right then top to bottom —
/// which is how the OV7670 emits them, so a frame can be scanned as it
/// lands rather than stored twice.
///
/// One pass, four accumulators, no allocation: it costs the same on a
/// microcontroller as on a laptop.
pub fn find(pixels: impl IntoIterator<Item = u16>, width: u16, target: &Target) -> Option<Blob> {
    // ⚠️ A zero-width frame has no pixels and would divide by zero on the
    // very first one. Refused here rather than trusted, because this crate
    // compiles into firmware and a panic is a stopped robot.
    if width == 0 {
        return None;
    }
    let (mut count, mut sum_x, mut sum_y) = (0u32, 0u64, 0u64);
    let mut total = 0u32;
    let (mut min_x, mut min_y) = (u16::MAX, u16::MAX);
    let (mut max_x, mut max_y) = (0u16, 0u16);

    for (index, pixel) in pixels.into_iter().enumerate() {
        total += 1;
        if !target.matches(pixel) {
            continue;
        }
        let x = (index % usize::from(width)) as u16;
        let y = (index / usize::from(width)) as u16;
        count += 1;
        sum_x += u64::from(x);
        sum_y += u64::from(y);
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    // ⚠️ `count > 0` is NOT implied by `count >= min_pixels`: a caller may
    // set `min_pixels: 0`, and then an empty frame produced a blob whose
    // centroid was NaN and whose bounds were still the sentinels
    // `(u16::MAX, u16::MAX, 0, 0)` — so `looks_like_one_object` panicked
    // subtracting 65535 from 0. Measured, not theorised.
    if count == 0 || count < target.min_pixels {
        return None;
    }
    // Ceiling division, so a frame shorter than one row still has height 1
    // and the centre is never a division by zero.
    let height = ((total + u32::from(width) - 1) / u32::from(width)).min(u32::from(u16::MAX));
    Some(Blob {
        centroid_x: sum_x as f32 / count as f32,
        centroid_y: sum_y as f32 / count as f32,
        area: count,
        bounds: (min_x, min_y, max_x, max_y),
        frame: (width, height as u16),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: u16 = 32;
    const H: u16 = 24;

    fn rgb565(r: u8, g: u8, b: u8) -> u16 {
        (u16::from(r >> 3) << 11) | (u16::from(g >> 2) << 5) | u16::from(b >> 3)
    }

    /// A frame of `background` with a filled rectangle of `object`.
    fn frame_with_rect(
        background: u16,
        object: u16,
        rect: (u16, u16, u16, u16),
    ) -> impl Iterator<Item = u16> {
        (0..u32::from(W) * u32::from(H)).map(move |i| {
            let (x, y) = ((i % u32::from(W)) as u16, (i / u32::from(W)) as u16);
            let (x0, y0, x1, y1) = rect;
            if x >= x0 && x <= x1 && y >= y0 && y <= y1 {
                object
            } else {
                background
            }
        })
    }

    #[test]
    fn a_red_square_is_found_at_its_own_centre() {
        // 6x6 = 36 pixels, deliberately above the 30-pixel noise floor —
        // a 5x5 square is 25 and correctly reports nothing.
        let frame = frame_with_rect(rgb565(20, 20, 20), rgb565(255, 0, 0), (10, 6, 15, 11));
        let blob = find(frame, W, &Target::hue(0.0)).expect("the square is there");
        assert!((blob.centroid_x - 12.5).abs() < 0.5, "{blob:?}");
        assert!((blob.centroid_y - 8.5).abs() < 0.5, "{blob:?}");
        assert_eq!(blob.area, 6 * 6);
    }

    /// The floor that stops a tracker chasing sensor noise.
    #[test]
    fn an_empty_frame_reports_nothing_rather_than_a_centre() {
        let frame = frame_with_rect(rgb565(20, 20, 20), rgb565(20, 20, 20), (0, 0, 0, 0));
        assert_eq!(find(frame, W, &Target::hue(0.0)), None);
    }

    #[test]
    fn a_handful_of_matching_pixels_is_below_the_noise_floor() {
        // 2x2 = 4 pixels, under the default min_pixels of 30.
        let frame = frame_with_rect(rgb565(20, 20, 20), rgb565(255, 0, 0), (4, 4, 5, 5));
        assert_eq!(find(frame, W, &Target::hue(0.0)), None);
    }

    /// ⚠️ Hue wraps, and red — the colour everyone picks first — sits
    /// exactly on the seam. A naive subtraction loses it.
    #[test]
    fn a_colour_either_side_of_the_red_seam_is_still_red() {
        assert!(hue_distance(359.0, 1.0) < 3.0);
        assert!(hue_distance(1.0, 359.0) < 3.0);
        // and a genuinely different colour is genuinely far
        assert!(hue_distance(0.0, 120.0) > 100.0);
    }

    /// Grey has no hue. Without a saturation floor it matches everything.
    #[test]
    fn grey_is_rejected_however_close_its_nominal_hue() {
        let (_, saturation, _) = rgb565_to_hsv(rgb565(128, 128, 128));
        assert!(
            saturation < 0.1,
            "grey must be unsaturated, got {saturation}"
        );
        let frame = frame_with_rect(rgb565(0, 0, 0), rgb565(128, 128, 128), (8, 4, 20, 16));
        assert_eq!(find(frame, W, &Target::hue(0.0)), None);
    }

    /// Two blobs in a frame, as a helper — the failure this crate cannot
    /// see and must therefore describe.
    fn two_blobs(a: (u16, u16, u16, u16), b: (u16, u16, u16, u16)) -> impl Iterator<Item = u16> {
        let inside =
            |x: u16, y: u16, r: (u16, u16, u16, u16)| x >= r.0 && x <= r.2 && y >= r.1 && y <= r.3;
        (0..u32::from(W) * u32::from(H)).map(move |i| {
            let (x, y) = ((i % u32::from(W)) as u16, (i / u32::from(W)) as u16);
            if inside(x, y, a) || inside(x, y, b) {
                rgb565(255, 0, 0)
            } else {
                rgb565(20, 20, 20)
            }
        })
    }

    /// The documented failure: two objects average to a point on neither.
    /// Diagonally opposite, the bounding box gives it away.
    #[test]
    fn two_distant_objects_average_to_the_gap_and_the_box_reveals_it() {
        let blob = find(
            two_blobs((0, 0, 4, 4), (27, 19, 31, 23)),
            W,
            &Target::hue(0.0),
        )
        .expect("pixels matched");
        assert!(
            (blob.centroid_x - 15.5).abs() < 2.0 && (blob.centroid_y - 11.5).abs() < 2.0,
            "the centroid lands in the empty middle: {blob:?}"
        );
        assert!(
            !blob.looks_like_one_object(0.25),
            "a box spanning the frame around 50 pixels is not one object: {blob:?}"
        );
    }

    /// ⚠️ And the blind spot, pinned so nobody trusts the check too far.
    ///
    /// Two objects side by side at the same height fill ~43% of the wide,
    /// short box around them — comfortably past the 0.25 floor — while the
    /// centroid still points at the gap between them. The check catches
    /// separated blobs, not all of them.
    #[test]
    fn side_by_side_objects_slip_past_the_one_object_check() {
        let blob = find(
            two_blobs((2, 8, 7, 14), (24, 8, 29, 14)),
            W,
            &Target::hue(0.0),
        )
        .expect("pixels matched");
        assert!(
            (blob.centroid_x - 15.5).abs() < 1.0,
            "centroid is still in the gap: {blob:?}"
        );
        assert!(
            blob.looks_like_one_object(0.25),
            "and the check does NOT catch it — this is the documented limit"
        );
    }

    #[test]
    fn one_object_passes_the_same_check_that_two_fail() {
        let frame = frame_with_rect(rgb565(20, 20, 20), rgb565(255, 0, 0), (10, 6, 16, 12));
        let blob = find(frame, W, &Target::hue(0.0)).unwrap();
        assert!(blob.looks_like_one_object(0.25));
    }

    /// ⚠️ Both found by probing rather than by reasoning, and both were
    /// real: a caller-set `min_pixels: 0` produced a blob of nothing with
    /// NaN coordinates, and `looks_like_one_object` then panicked
    /// subtracting the untouched sentinel bounds.
    #[test]
    fn no_matching_pixels_is_never_a_blob_however_low_the_floor_is_set() {
        let target = Target {
            min_pixels: 0,
            ..Target::hue(0.0)
        };
        let frame = frame_with_rect(rgb565(20, 20, 20), rgb565(20, 20, 20), (0, 0, 0, 0));
        assert_eq!(
            find(frame, W, &target),
            None,
            "a blob of zero pixels is not a blob"
        );
    }

    #[test]
    fn a_zero_width_frame_is_refused_rather_than_dividing_by_zero() {
        assert_eq!(find([0u16; 16], 0, &Target::hue(0.0)), None);
    }

    /// The frame size travels WITH the blob, so a caller cannot supply a
    /// different one than `find` saw.
    #[test]
    fn the_blob_remembers_the_frame_it_was_found_in() {
        let frame = frame_with_rect(rgb565(20, 20, 20), rgb565(255, 0, 0), (10, 6, 15, 11));
        let blob = find(frame, W, &Target::hue(0.0)).unwrap();
        assert_eq!(blob.frame, (W, H));
    }

    /// What a controller actually consumes.
    #[test]
    fn the_error_is_zero_at_the_centre_and_saturates_at_the_edges() {
        let middle = frame_with_rect(rgb565(20, 20, 20), rgb565(255, 0, 0), (13, 9, 18, 14));
        let blob = find(middle, W, &Target::hue(0.0)).unwrap();
        let (x, y) = blob.error_from_centre();
        assert!(x.abs() < 0.1 && y.abs() < 0.1, "centred: {x} {y}");

        let corner = frame_with_rect(rgb565(20, 20, 20), rgb565(255, 0, 0), (0, 0, 5, 5));
        let blob = find(corner, W, &Target::hue(0.0)).unwrap();
        let (x, y) = blob.error_from_centre();
        assert!(x < -0.7 && y < -0.7, "top-left corner: {x} {y}");
    }
}
