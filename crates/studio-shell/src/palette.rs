//! The command palette: ⌘K (Ctrl+K elsewhere) opens a search over every
//! page and every artifact in the project; Enter opens the first hit,
//! a click opens any. The matching is a pure function, tested.

use re_ui::{DesignTokens, UiExt as _};

use crate::model::{ago_iso, split_stamp, Index};
use crate::pages::Section;

/// How many hits the palette lists.
pub const MAX_HITS: usize = 12;
const PALETTE_WIDTH: f32 = 560.0;
/// What the palette leaves free on each side in a narrow window.
const PALETTE_MARGIN: f32 = 48.0;

#[derive(Clone, Debug, PartialEq)]
pub enum Hit {
    Page(Section),
    Artifact { stamp: String, kind: String },
}

/// The palette's state while open: what was typed.
#[derive(Default, Clone)]
pub struct Palette {
    pub query: String,
    /// Focus the text box on the first frame after opening.
    pub fresh: bool,
}

impl Palette {
    pub fn open(query: Option<String>) -> Self {
        Self {
            query: query.unwrap_or_default(),
            fresh: true,
        }
    }
}

/// Pages first (by title), then artifacts (by name), each where the
/// lowercase query is a substring; an empty query lists the pages.
pub fn hits(query: &str, index: Option<&Index>) -> Vec<Hit> {
    let needle = query.trim().to_lowercase();
    let mut out: Vec<Hit> = Section::RAIL
        .iter()
        .flat_map(|(_, items)| items.iter().copied())
        .filter(|s| needle.is_empty() || s.title().to_lowercase().contains(&needle))
        .map(Hit::Page)
        .collect();
    if !needle.is_empty() {
        if let Some(index) = index {
            let mut artifacts: Vec<&crate::model::Artifact> = index
                .artifacts
                .iter()
                .filter(|a| split_stamp(&a.stamp).0.to_lowercase().contains(&needle))
                .collect();
            // The name itself first, then names that start with the query,
            // then the shortest (closest to what was typed), then the newest —
            // so "narrow" lists the policy before its forty evaluations.
            // One key per artifact, not one per comparison; newest first among equals.
            artifacts.sort_by_cached_key(|x| {
                let name = split_stamp(&x.stamp).0.to_lowercase();
                (
                    name != needle,
                    !name.starts_with(&needle),
                    name.len(),
                    std::cmp::Reverse(ordered_float(x.updated_epoch())),
                )
            });
            out.extend(artifacts.into_iter().map(|a| Hit::Artifact {
                stamp: a.stamp.clone(),
                kind: a.kind.clone(),
            }));
        }
    }
    out.truncate(MAX_HITS);
    out
}

/// An f64 as a totally ordered key (NaN and -inf sort last).
fn ordered_float(v: f64) -> i64 {
    if v.is_finite() {
        (v * 1000.0) as i64
    } else {
        i64::MIN
    }
}

/// Draw the palette; returns the chosen hit, and whether it should close.
pub fn show(
    ctx: &egui::Context,
    palette: &mut Palette,
    index: Option<&Index>,
) -> (Option<Hit>, bool) {
    let mut chosen = None;
    let mut close = false;
    let modal = egui::Modal::new(egui::Id::new("command-palette")).show(ctx, |ui| {
        ui.set_width(PALETTE_WIDTH.min(ctx.content_rect().width() - PALETTE_MARGIN));
        let edit = ui.add(
            egui::TextEdit::singleline(&mut palette.query)
                .hint_text("Jump to a page or an artifact…")
                .desired_width(f32::INFINITY)
                .font(DesignTokens::welcome_screen_body()),
        );
        if std::mem::take(&mut palette.fresh) {
            edit.request_focus();
        }
        let found = hits(&palette.query, index);
        let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
        if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            close = true;
        }
        ui.add_space(6.0);
        if found.is_empty() {
            ui.label(
                egui::RichText::new("nothing by that name in this project")
                    .color(ui.visuals().weak_text_color()),
            );
        }
        for (i, hit) in found.iter().enumerate() {
            let response = ui
                .horizontal(|ui| {
                    let (icon, title, detail) = match hit {
                        Hit::Page(section) => (
                            section.icon(),
                            section.title().to_owned(),
                            "page".to_owned(),
                        ),
                        Hit::Artifact { stamp, kind } => {
                            let when = index
                                .and_then(|x| x.artifacts.iter().find(|a| &a.stamp == stamp))
                                .map(|a| ago_iso(a.updated.as_deref()))
                                .unwrap_or_default();
                            (
                                Section::icon_for(kind),
                                split_stamp(stamp).0.to_owned(),
                                format!("{kind} · {when}"),
                            )
                        }
                    };
                    ui.small_icon(icon, Some(ui.tokens().label_button_icon_color));
                    ui.add(egui::Label::new(egui::RichText::new(title).strong()).truncate());
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(detail).color(ui.visuals().weak_text_color()),
                        )
                        .truncate(),
                    );
                    if i == 0 {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(
                                egui::RichText::new("↩").color(ui.visuals().weak_text_color()),
                            );
                        });
                    }
                })
                .response
                .interact(egui::Sense::click());
            if response.hovered() {
                ui.painter().rect_filled(
                    response.rect.expand2(egui::vec2(6.0, 2.0)),
                    4.0,
                    ui.tokens().highlight_color.linear_multiply(0.15),
                );
            }
            if response.clicked() || (enter && i == 0) {
                chosen = Some(hit.clone());
                close = true;
            }
        }
    });
    (chosen, close || modal.should_close())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index() -> Index {
        serde_json::from_value(serde_json::json!({
            "schema": "trainnr-project-index/1", "project": "p", "root": "/p",
            "indexed": "2026-09-09T00:00:00+00:00",
            "artifacts": [
                {"kind": "policy", "stamp": "narrow#2@1", "path": "policies/narrow#2",
                 "updated": "2026-09-09T10:00:00+00:00"},
                {"kind": "run", "stamp": "wide-narrow@2", "path": "runs/wide-narrow",
                 "updated": "2026-09-09T11:00:00+00:00"},
                {"kind": "robot", "stamp": "microduck@3", "path": "robots/microduck"},
                {"kind": "certificate", "stamp": "narrow#2-at-fit@4", "path": "c/x",
                 "updated": "2026-09-09T12:00:00+00:00"},
                {"kind": "policy", "stamp": "narrow@5", "path": "policies/narrow",
                 "updated": "2026-09-01T00:00:00+00:00"}
            ],
            "states": [], "next_move": null, "refused": []
        }))
        .unwrap()
    }

    #[test]
    fn an_empty_query_lists_pages_and_a_name_finds_artifacts_prefix_first() {
        let pages = hits("", Some(&index()));
        assert!(matches!(pages[0], Hit::Page(Section::Projects)));
        assert!(pages.iter().all(|h| matches!(h, Hit::Page(_))));
        let found = hits("narrow", Some(&index()));
        let names: Vec<String> = found
            .iter()
            .filter_map(|h| match h {
                Hit::Artifact { stamp, .. } => Some(split_stamp(stamp).0.to_owned()),
                Hit::Page(_) => None,
            })
            .collect();
        assert_eq!(
            names,
            vec!["narrow", "narrow#2", "narrow#2-at-fit", "wide-narrow"]
        );
        assert!(hits("eval", Some(&index()))
            .iter()
            .any(|h| matches!(h, Hit::Page(Section::Certificates))));
        assert!(hits("zzz", Some(&index())).is_empty());
    }
}
