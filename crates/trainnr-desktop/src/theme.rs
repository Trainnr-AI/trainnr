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
    link: egui::Color32::from_rgb(0, 102, 204),
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
            accent_soft: tokens.selection_bg_fill,
            edge: tokens.native_frame_stroke.color,
        }
    } else {
        LIGHT
    }
}

/// egui's light visuals, aligned with [`LIGHT`] so what the pages do not
/// paint themselves (the page fill, weak text, links) matches what they
/// do. Idempotent: called every frame, it writes only when something
/// differs, because the embedded viewer may re-apply its own themes.
pub fn align_light_visuals(ctx: &egui::Context) {
    let aligned = ctx.style_of(egui::Theme::Light).visuals.panel_fill == LIGHT.canvas;
    if aligned {
        return;
    }
    ctx.style_mut_of(egui::Theme::Light, |style| {
        let v = &mut style.visuals;
        v.panel_fill = LIGHT.canvas;
        v.window_fill = LIGHT.canvas;
        v.faint_bg_color = LIGHT.surface;
        v.extreme_bg_color = LIGHT.surface_alt;
        v.weak_text_color = Some(LIGHT.muted);
        v.override_text_color = None;
        v.hyperlink_color = LIGHT.link;
    });
}
