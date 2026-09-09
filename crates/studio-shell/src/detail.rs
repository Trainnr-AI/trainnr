//! The detail view: an artifact's sections, rendered from the file the
//! Python side writes (`rq_pipeline/project/details.py`, at
//! `<project>/.index/details/<version>.json`). The Studio knows nothing
//! about MuJoCo, LeRobot or the pipeline here — a section is a titled
//! key/value list or a table with columns and rows, and this module
//! renders both in Rerun's own table style (`DesignTokens::table_*`).
//! The words on screen are the field's: the Python side chose them.

use egui_extras::{Column, TableBuilder};
use re_ui::{DesignTokens, TableStyle, UiExt as _};
use serde::Deserialize;

use crate::widgets::card;

/// A row taller than this many rendered lines in one cell is clipped;
/// long values (a datasheet, a command line) get their own block instead.
const LONG_VALUE_CHARS: usize = 160;
/// A table with more rows than this scrolls inside its card.
const SCROLL_ROWS: usize = 24;

#[derive(Deserialize, Default)]
pub struct Detail {
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub sections: Vec<Section>,
}

#[derive(Deserialize)]
pub struct Section {
    pub title: String,
    pub kind: String, // "kv" | "table"
    #[serde(default)]
    pub columns: Vec<String>,
    #[serde(default)]
    pub rows: Vec<Vec<serde_json::Value>>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub note: Option<String>,
}

/// What is shown when a fact was never recorded: the reader must not
/// mistake an empty cell for a value.
const UNRECORDED: &str = "unrecorded";

/// The field's word for a key the index writes in this repo's older
/// vocabulary (the `cites` map, whose keys are frozen by tests).
pub fn field_word(key: &str) -> &str {
    match key {
        "expert" => "scripted policy",
        "task" => "environment",
        "bundle" => "robot",
        "instrument" => "simulator build",
        "source" => "source dataset",
        "batch" => "generated dataset",
        "dynamics_basis" | "basis" => "domain randomization",
        "fit" => "system identification",
        "certificate" => "evaluation",
        other => other,
    }
}

impl Detail {
    pub fn load(path: &std::path::Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&text).ok()
    }
}

/// Every section, one card each. `expected` is the artifact's current
/// version; a detail file written for another version is flagged, not
/// trusted silently.
pub fn show(ui: &mut egui::Ui, detail: &Detail, expected: &str) {
    if !detail.version.is_empty() && detail.version != expected {
        ui.warning_label(format!(
            "detail was written for version {} — re-index to refresh",
            detail.version
        ));
    }
    for (i, section) in detail.sections.iter().enumerate() {
        ui.add_space(if i == 0 { 0.0 } else { 14.0 });
        card(ui, None).show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.label(
                egui::RichText::new(&section.title)
                    .text_style(DesignTokens::welcome_screen_example_title())
                    .strong(),
            );
            ui.add_space(6.0);
            match section.kind.as_str() {
                "table" => table(ui, section, i),
                "markdown" => markdown(ui, section),
                _ => key_values(ui, section, i),
            }
            if let Some(note) = &section.note {
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new(note)
                        .text_style(DesignTokens::welcome_screen_tag())
                        .color(ui.visuals().weak_text_color()),
                );
            }
        });
    }
}

/// A key/value list: keys weak on the left, values on the right; a long
/// value (a datasheet, a command line) drops below its key as a block.
fn key_values(ui: &mut egui::Ui, section: &Section, idx: usize) {
    egui::Grid::new(("kv", &section.title, idx))
        .num_columns(2)
        .spacing([18.0, 4.0])
        .min_col_width(160.0)
        .show(ui, |ui| {
            for row in &section.rows {
                let key = row.first().and_then(|v| v.as_str()).unwrap_or("");
                let value = row.get(1).map(render).unwrap_or_default();
                ui.label(egui::RichText::new(key).color(ui.visuals().weak_text_color()));
                if value.is_empty() {
                    ui.label(
                        egui::RichText::new(UNRECORDED)
                            .italics()
                            .color(ui.visuals().weak_text_color()),
                    );
                } else if value.len() > LONG_VALUE_CHARS || value.contains('\n') {
                    ui.add(egui::Label::new(egui::RichText::new(value).monospace().small()).wrap());
                } else if is_version(&value) {
                    ui.monospace(value);
                } else {
                    ui.label(value);
                }
                ui.end_row();
            }
        });
}

/// A markdown body (a datasheet, a README) through Rerun's own markdown
/// viewer, so headings, lists and tables read as such.
fn markdown(ui: &mut egui::Ui, section: &Section) {
    match section.text.as_deref().filter(|t| !t.trim().is_empty()) {
        Some(text) => ui.markdown_ui(text),
        None => {
            ui.label(
                egui::RichText::new(UNRECORDED)
                    .italics()
                    .color(ui.visuals().weak_text_color()),
            );
        }
    }
}

/// A table in Rerun's dense table style; scrolls past `SCROLL_ROWS`.
fn table(ui: &mut egui::Ui, section: &Section, idx: usize) {
    let tokens = ui.tokens();
    let style = TableStyle::Dense;
    let row_h = tokens.table_row_height(style) + 2.0;
    let cols = section.columns.len().max(1);
    let shown = (section.rows.len() + 1).min(SCROLL_ROWS + 1);
    let max_height = row_h * shown as f32 + 6.0;
    ui.push_id(("table", &section.title, idx), |ui| {
        egui::ScrollArea::both()
            .max_height(max_height)
            .auto_shrink([false, true])
            .show(ui, |ui| {
                let mut builder = TableBuilder::new(ui).striped(true).vscroll(false);
                for c in 0..cols {
                    let at_least = if c == 0 { 150.0 } else { 80.0 };
                    builder = builder.column(if c + 1 == cols {
                        Column::remainder().at_least(at_least)
                    } else {
                        Column::auto().at_least(at_least).resizable(true)
                    });
                }
                builder.body(|mut body| {
                    tokens.setup_table_body(&mut body, style);
                    body.row(row_h, |mut r| {
                        for name in &section.columns {
                            r.col(|ui| {
                                ui.label(
                                    egui::RichText::new(name)
                                        .strong()
                                        .color(ui.visuals().weak_text_color()),
                                );
                            });
                        }
                    });
                    for row in &section.rows {
                        body.row(row_h, |mut r| {
                            for c in 0..cols {
                                let text = row.get(c).map(render_inline).unwrap_or_default();
                                r.col(|ui| {
                                    if text.is_empty() {
                                        ui.label(
                                            egui::RichText::new("–")
                                                .color(ui.visuals().weak_text_color()),
                                        );
                                        return;
                                    }
                                    let rich = egui::RichText::new(&text);
                                    let rich = if is_version(&text) || looks_numeric(&text) {
                                        rich.monospace()
                                    } else {
                                        rich
                                    };
                                    ui.add(egui::Label::new(rich).truncate())
                                        .on_hover_text(&text);
                                });
                            }
                        });
                    }
                });
            });
    });
}

/// One line for a table cell: an object reads as `k: v, k: v`, never as
/// a clipped JSON block.
fn render_inline(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Object(map) => map
            .iter()
            .map(|(k, x)| format!("{k}: {}", render_inline(x)))
            .collect::<Vec<_>>()
            .join(", "),
        serde_json::Value::Array(items) => items
            .iter()
            .map(render_inline)
            .collect::<Vec<_>>()
            .join(", "),
        other => render(other),
    }
}

fn render(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => String::new(),
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(items) => items
            .iter()
            .map(|x| match x {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .collect::<Vec<_>>()
            .join(", "),
        serde_json::Value::Object(_) => serde_json::to_string_pretty(v).unwrap_or_default(),
        other => other.to_string(),
    }
}

fn is_version(s: &str) -> bool {
    s.contains('@') && !s.contains(' ')
}

fn looks_numeric(s: &str) -> bool {
    let t = s.trim_start_matches(['[', '-']);
    t.chars().next().is_some_and(|c| c.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_detail_file_parses_into_sections() {
        let text = r#"{"schema":"trainnr-detail/1","version":"a@000000000000","sections":[
            {"title":"Asset","kind":"kv","rows":[["version","a@000000000000"],["joints",15]],"note":null},
            {"title":"Joints","kind":"table","columns":["joint","type"],"rows":[["hip","hinge"]],"note":"n"}]}"#;
        let d: Detail = serde_json::from_str(text).expect("parses");
        assert_eq!(d.version, "a@000000000000");
        assert_eq!(d.sections.len(), 2);
        assert_eq!(d.sections[1].columns, vec!["joint", "type"]);
        assert_eq!(render(&d.sections[0].rows[1][1]), "15");
    }

    #[test]
    fn values_render_readably() {
        assert_eq!(render(&serde_json::json!(["a", "b"])), "a, b");
        assert_eq!(render(&serde_json::json!(1.5)), "1.5");
        assert_eq!(render(&serde_json::Value::Null), "");
        assert!(is_version("go2@a1b2c3d4e5f6"));
        assert!(!is_version("a robot @ home"));
        assert!(looks_numeric("[-0.43, 0.52]"));
        assert!(!looks_numeric("hinge"));
        assert_eq!(
            render_inline(&serde_json::json!({"damping": 1.1, "gain": [0.7, 0.9]})),
            "damping: 1.1, gain: 0.7, 0.9"
        );
    }

    #[test]
    fn a_markdown_section_parses_and_keys_get_the_fields_word() {
        let text = r##"{"version":"b@000000000000","sections":[
            {"title":"Datasheet","kind":"markdown","text":"# Datasheet\n\n- episodes: 2"}]}"##;
        let d: Detail = serde_json::from_str(text).expect("parses");
        assert_eq!(d.sections[0].kind, "markdown");
        assert!(d.sections[0]
            .text
            .as_deref()
            .unwrap_or("")
            .starts_with("# Datasheet"));
        assert_eq!(field_word("expert"), "scripted policy");
        assert_eq!(field_word("robot"), "robot");
    }
}
