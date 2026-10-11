//! The app's own palette on top of Rerun's design tokens: the surfaces and
//! the text weights the pages are built from, one set per theme. Dark is
//! the tokens' own values, so nothing moved when light arrived
//! (2026-10-02); light is designed rather than derived, because Rerun's
//! light tokens give every surface the same flat grey and the muted text
//! no contrast on it.

use re_ui::UiExt as _;

/// The surfaces and text weights of a page.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Palette {
    /// The page behind everything.
    pub canvas: egui::Color32,
    /// A card, a panel, a sheet: what sits on the canvas.
    pub surface: egui::Color32,
    /// A chip or a footer on a surface: one step deeper than the surface.
    pub surface_alt: egui::Color32,
    /// The title bar and the Running-now bar.
    pub bar: egui::Color32,
    /// The one-pixel edge of a surface; close to the surface, so it reads
    /// as an edge, never as a line.
    pub hairline: egui::Color32,
    /// Body text.
    pub text: egui::Color32,
    /// Labels, captions, secondary facts.
    pub muted: egui::Color32,
    /// A link, a reference to another artifact.
    pub link: egui::Color32,
    /// The accent as text: the next step, a running job's line. Rerun's
    /// highlight colour is a background token and measured 1.4:1 to
    /// 1.7:1 as text (design review, 2026-10-12).
    pub accent_text: egui::Color32,
    /// The soft tint behind the selected item (the rail, a chosen row).
    pub accent_soft: egui::Color32,
    /// A line that must be seen: the title bar's edge, a divider.
    pub edge: egui::Color32,
}

/// The light palette, designed 2026-10-02 after the operator's reference
/// (a white page, warm paper panels with no hard borders, soft sky-blue
/// tints, near-black text): nothing here is a grey derived from the dark
/// tokens.
pub const LIGHT: Palette = Palette {
    canvas: egui::Color32::from_rgb(255, 255, 255),
    surface: egui::Color32::from_rgb(244, 242, 237),
    surface_alt: egui::Color32::from_rgb(232, 229, 222),
    bar: egui::Color32::from_rgb(255, 255, 255),
    hairline: egui::Color32::from_rgb(236, 233, 226),
    text: egui::Color32::from_rgb(23, 23, 23),
    muted: egui::Color32::from_rgb(92, 96, 104),
    // One notch darker than the 2026-10-02 blue (0, 102, 204), which read
    // 4.4:1 on the deepest surface; this one clears 4.5:1 on all four and
    // on the selection tint (design review, 2026-10-12).
    link: egui::Color32::from_rgb(0, 96, 196),
    accent_text: egui::Color32::from_rgb(0, 96, 196),
    accent_soft: egui::Color32::from_rgb(198, 229, 250),
    edge: egui::Color32::from_rgb(214, 211, 203),
};

/// The palette of the theme in effect: dark from Rerun's tokens, light
/// from [`LIGHT`].
pub fn palette(ui: &egui::Ui) -> Palette {
    let visuals = ui.visuals();
    if visuals.dark_mode {
        let tokens = ui.tokens();
        Palette {
            canvas: visuals.panel_fill,
            surface: tokens.example_card_background_color,
            surface_alt: tokens.top_bar_color,
            bar: tokens.top_bar_color,
            hairline: tokens.native_frame_stroke.color,
            text: visuals.text_color(),
            muted: visuals.weak_text_color(),
            link: visuals.hyperlink_color,
            accent_text: tokens.info_text_color,
            accent_soft: tokens.selection_bg_fill,
            edge: tokens.native_frame_stroke.color,
        }
    } else {
        LIGHT
    }
}

/// Rerun's own grey one step below its subdued text (`Gray.500` in
/// re_ui's colour table, #827f7f): the dark theme's muted text. egui
/// otherwise fades the body text to 60 %, which measured 2.9:1 on the
/// dark page; this measures 4.9:1 on the page and the bars, 4.4:1 on a
/// card (design review, 2026-10-12). Dark keeps to Rerun's colours.
pub const DARK_MUTED: egui::Color32 = egui::Color32::from_rgb(0x82, 0x7f, 0x7f);

/// egui's visuals in both themes, aligned with the palette so what the
/// pages do not paint themselves (weak text, links, default labels)
/// matches what they do: light takes [`LIGHT`]'s designed colours, dark
/// takes [`DARK_MUTED`] for weak text. Idempotent: called every frame, it
/// writes only when something differs, because the embedded viewer may
/// re-apply its own themes. Until 2026-10-12 it checked the page fill,
/// which Rerun's light theme already sets to white, so it returned on the
/// first frame and the light theme's muted grey and link blue were never
/// applied (measured 2.0:1 for every muted line in light).
pub fn align_visuals(ctx: &egui::Context) {
    let light = ctx.style_of(egui::Theme::Light);
    let light_aligned = light.visuals.weak_text_color == Some(LIGHT.muted)
        && light.visuals.hyperlink_color == LIGHT.link
        && light.visuals.widgets.noninteractive.fg_stroke.color == LIGHT.text;
    if !light_aligned {
        ctx.style_mut_of(egui::Theme::Light, |style| {
            let v = &mut style.visuals;
            v.panel_fill = LIGHT.canvas;
            v.window_fill = LIGHT.canvas;
            v.faint_bg_color = LIGHT.surface;
            v.extreme_bg_color = LIGHT.surface_alt;
            v.weak_text_color = Some(LIGHT.muted);
            v.override_text_color = None;
            v.hyperlink_color = LIGHT.link;
            // Default labels in the designed near-black: Rerun's light
            // label grey measured 4.0:1 in the rail.
            v.widgets.noninteractive.fg_stroke.color = LIGHT.text;
        });
    }
    let dark = ctx.style_of(egui::Theme::Dark);
    if dark.visuals.weak_text_color != Some(DARK_MUTED) {
        ctx.style_mut_of(egui::Theme::Dark, |style| {
            style.visuals.weak_text_color = Some(DARK_MUTED);
        });
    }
}

/// The contrast ratio of two colours (WCAG 2), for the palette's tests.
#[cfg(test)]
fn contrast(a: egui::Color32, b: egui::Color32) -> f32 {
    fn lum(c: egui::Color32) -> f32 {
        let ch = |v: u8| {
            let v = f32::from(v) / 255.0;
            if v <= 0.039_28 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * ch(c.r()) + 0.7152 * ch(c.g()) + 0.0722 * ch(c.b())
    }
    let (x, y) = (lum(a), lum(b));
    (x.max(y) + 0.05) / (x.min(y) + 0.05)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every text colour of the light palette reads at 4.5:1 or better on
    /// every surface it sits on.
    #[test]
    fn the_light_palettes_text_reads_on_its_surfaces() {
        for surface in [LIGHT.canvas, LIGHT.surface, LIGHT.surface_alt, LIGHT.bar] {
            for (role, ink) in [
                ("text", LIGHT.text),
                ("muted", LIGHT.muted),
                ("link", LIGHT.link),
                ("accent text", LIGHT.accent_text),
            ] {
                let ratio = contrast(ink, surface);
                assert!(ratio >= 4.5, "{role} on {surface:?}: {ratio:.2}");
            }
        }
    }

    /// The dark muted grey reads at 4.5:1 on the dark page (Rerun's
    /// panel fill, #0d0d0d as measured).
    #[test]
    fn dark_muted_text_reads_on_the_page() {
        let page = egui::Color32::from_rgb(13, 13, 13);
        assert!(contrast(DARK_MUTED, page) >= 4.5);
    }

    /// The alignment applies in both themes and holds once applied.
    #[test]
    fn the_visuals_are_aligned_in_both_themes() {
        let ctx = egui::Context::default();
        re_ui::apply_style_and_install_loaders(&ctx);
        align_visuals(&ctx);
        let light = ctx.style_of(egui::Theme::Light);
        assert_eq!(light.visuals.weak_text_color, Some(LIGHT.muted));
        assert_eq!(light.visuals.hyperlink_color, LIGHT.link);
        let dark = ctx.style_of(egui::Theme::Dark);
        assert_eq!(dark.visuals.weak_text_color, Some(DARK_MUTED));
    }
}
