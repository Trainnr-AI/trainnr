//! The Studio's few own widgets, each built from Rerun's parts: a large
//! icon (Rerun's SVG icons painted into a bigger rect, the way its own
//! `large_button` does), a thumbnail card (Rerun's welcome-screen example
//! card: a cropped-to-fit picture over a description block), a tag pill
//! in its tag style, and a card frame in its card colour.

use re_ui::{DesignTokens, Icon, UiExt as _};

pub const CARD_RADIUS: f32 = 8.0;
/// Rerun's example-card thumbnail aspect (337 × 250).
pub const THUMBNAIL_ASPECT: f32 = 337.0 / 250.0;
const THUMBNAIL_RADIUS: u8 = 8;

/// An icon painted at `size` points — Rerun's icons are SVG, so this is
/// sharp at any size; its own UI paints them at 16 (small) and 22 (large).
pub fn icon_at(ui: &mut egui::Ui, icon: &Icon, size: f32, tint: egui::Color32) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::hover());
    if ui.is_rect_visible(rect) {
        icon.as_image().tint(tint).paint_at(ui, rect);
    }
    response
}

/// A card frame in Rerun's own card colour and radius.
pub fn card(ui: &egui::Ui, accent: Option<egui::Color32>) -> egui::Frame {
    let tokens = ui.tokens();
    let stroke = match accent {
        Some(color) => egui::Stroke::new(1.5, color),
        None => egui::Stroke::new(1.0, tokens.native_frame_stroke.color),
    };
    egui::Frame::new()
        .fill(tokens.example_card_background_color)
        .stroke(stroke)
        .corner_radius(CARD_RADIUS)
        .inner_margin(egui::Margin::same(14))
}

/// A small pill in the welcome screen's tag style.
pub fn tag(ui: &mut egui::Ui, text: &str) {
    let tokens = ui.tokens();
    egui::Frame::new()
        .fill(tokens.example_tag_bg_fill)
        .stroke(tokens.example_tag_stroke)
        .corner_radius(999.0)
        .inner_margin(egui::Margin::symmetric(7, 2))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(text).text_style(DesignTokens::welcome_screen_tag()));
        });
}

/// A picture cropped to fill `rect`, top corners rounded — Rerun's
/// `image_ui` on its example cards, for a file on disk.
/// The URI a preview file is registered under. Rerun's build of
/// `egui_extras` has no `file://` loader (checked: its feature list is
/// http, image, serde, svg), so the bytes are handed to egui directly
/// under a `bytes://` URI — the same route Rerun's own icons take. The
/// registration is idempotent and egui caches the decoded texture, so
/// the disk read happens once per file.
pub fn preview_uri(ui: &egui::Ui, path: &std::path::Path) -> Option<String> {
    let uri = format!("bytes://{}", path.display());
    if ui.ctx().try_load_bytes(&uri).is_err() {
        let bytes = std::fs::read(path).ok()?;
        ui.ctx().include_bytes(uri.clone(), bytes);
    }
    Some(uri)
}

pub fn thumbnail(ui: &mut egui::Ui, path: &std::path::Path, rect: egui::Rect) {
    let Some(uri) = preview_uri(ui, path) else {
        return;
    };
    let cr = egui::CornerRadius {
        nw: THUMBNAIL_RADIUS,
        ne: THUMBNAIL_RADIUS,
        sw: 0,
        se: 0,
    };
    // Cropping to fit needs the source size; egui's loader gives it once
    // the image is loaded. Until then the image is fit whole, which is a
    // frame's worth of letterboxing at most.
    let uv = ui
        .ctx()
        .try_load_image(&uri, egui::SizeHint::default())
        .ok()
        .and_then(|poll| match poll {
            egui::load::ImagePoll::Ready { image } => Some(image.size),
            egui::load::ImagePoll::Pending { .. } => None,
        })
        .map(|[w, h]| {
            let (w, h) = (w as f32, h as f32);
            let display = rect.width() / rect.height();
            let source = w / h;
            if source > display {
                let a = (w / h * rect.height() - rect.width()) / 2.0 / w;
                egui::Rect::from_min_max(egui::pos2(a, 0.0), egui::pos2(1.0 - a, 1.0))
            } else {
                let a = (h / w * rect.width() - rect.height()) / 2.0 / h;
                egui::Rect::from_min_max(egui::pos2(0.0, a), egui::pos2(1.0, 1.0 - a))
            }
        })
        .unwrap_or(egui::Rect::from_min_max(
            egui::pos2(0.0, 0.0),
            egui::pos2(1.0, 1.0),
        ));
    egui::Image::new(uri)
        .uv(uv)
        .corner_radius(cr)
        .paint_at(ui, rect);
}

/// The placeholder where a picture would go and none exists yet: the
/// kind's icon, large and dim, on the card colour. Honest about absence.
pub fn thumbnail_placeholder(ui: &mut egui::Ui, icon: &Icon, rect: egui::Rect) {
    let tokens = ui.tokens();
    let cr = egui::CornerRadius {
        nw: THUMBNAIL_RADIUS,
        ne: THUMBNAIL_RADIUS,
        sw: 0,
        se: 0,
    };
    ui.painter()
        .rect_filled(rect, cr, tokens.thumbnail_background_color);
    let size = (rect.height() * 0.38).min(56.0);
    let icon_rect = egui::Rect::from_center_size(rect.center(), egui::vec2(size, size));
    icon.as_image()
        .tint(tokens.label_button_icon_color.linear_multiply(0.7))
        .paint_at(ui, icon_rect);
}

/// The number of columns and the column width for a card grid over
/// `available` points, Rerun's welcome-screen rule (250 to 337 wide).
pub fn grid_columns(available: f32, gap: f32) -> (usize, f32) {
    const MIN: f32 = 250.0;
    const MAX: f32 = 337.0;
    let count = (((available + gap) / (MIN + gap)).floor() as usize).max(1);
    let width = ((available + gap) / count as f32 - gap).clamp(MIN, MAX);
    (count, width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_columns_follow_reruns_rule() {
        let (n, w) = grid_columns(1200.0, 20.0);
        assert_eq!(n, 4);
        assert!((250.0..=337.0).contains(&w));
        let (n, w) = grid_columns(200.0, 20.0);
        assert_eq!(n, 1);
        assert_eq!(w, 250.0);
    }
}
