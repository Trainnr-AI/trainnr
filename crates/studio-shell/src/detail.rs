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

/// What an empty table section says instead of a header over no rows.
pub const EMPTY_TABLE: &str = "No rows recorded.";

use crate::model::{render_value, schema_compatible, UNRECORDED};
use crate::widgets::card;

/// The detail file's schema, as `rq_pipeline/project/details.py` writes
/// it (`SCHEMA`). Same rule as the index: the family must match.
pub const DETAIL_SCHEMA: &str = "trainnr-detail/6";

/// A row taller than this many rendered lines in one cell is clipped;
/// long values (a datasheet, a command line) get their own block instead.
const LONG_VALUE_CHARS: usize = 160;
/// A table in its card shows this many rows; the rest live in the modal
/// (`table_modal`), which the table itself and its expand button open.
const PREVIEW_ROWS: usize = 8;
/// The modal's share of the window, each way.
const MODAL_SHARE: f32 = 0.86;
/// A column never grows past these in the card and in the modal.
const PREVIEW_COLUMN_CAP: f32 = 260.0;
const MODAL_COLUMN_CAP: f32 = 420.0;

#[derive(Deserialize, Default)]
pub struct Detail {
    #[serde(default)]
    pub schema: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub sections: Vec<Section>,
}

#[derive(Deserialize, Clone, Debug)]
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

/// The field's word for a key the index writes in this repo's older
/// vocabulary (the `cites` map, whose keys are frozen by tests). A key
/// that names an artifact kind takes the kind's word (`pages::kind_word`).
pub fn field_word(key: &str) -> &str {
    match key {
        "expert" => "scripted policy",
        "bundle" => "robot",
        "instrument" => "simulator build",
        "source" => "source dataset",
        "dynamics_basis" | "basis" => "domain randomization",
        "fit" => "system identification",
        other => crate::pages::kind_word(other),
    }
}

impl Detail {
    pub fn load(path: &std::path::Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&text).ok()
    }
}

/// Every section, one card each. `expected` is the artifact's current
/// version; a detail file written for another version, or for another
/// schema family, is flagged, not trusted silently. Returns the table
/// section the human asked to explore (a click on it, or its expand
/// button), if any.
pub fn show(ui: &mut egui::Ui, detail: &Detail, expected: &str) -> Option<Section> {
    if !schema_compatible(&detail.schema, DETAIL_SCHEMA) {
        ui.warning_label(format!(
            "detail was written for schema {}; this Studio reads {DETAIL_SCHEMA} — \
             re-index with a matching pipeline",
            detail.schema
        ));
    }
    if !detail.version.is_empty() && detail.version != expected {
        ui.warning_label(format!(
            "detail was written for version {} — re-index to refresh",
            detail.version
        ));
    }
    let mut explore = None;
    for (i, section) in detail.sections.iter().enumerate() {
        ui.add_space(if i == 0 { 0.0 } else { 14.0 });
        card(ui, None).show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            let is_table = section.kind == "table";
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(&section.title)
                        .text_style(DesignTokens::welcome_screen_example_title())
                        .strong(),
                );
                if is_table && !section.rows.is_empty() {
                    ui.label(
                        egui::RichText::new(crate::widgets::count_word(section.rows.len(), "row"))
                            .color(ui.visuals().weak_text_color()),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .small_icon_button(&re_ui::icons::EXPAND, "Explore the whole table")
                            .clicked()
                        {
                            explore = Some(section.clone());
                        }
                    });
                }
            });
            ui.add_space(6.0);
            match section.kind.as_str() {
                "table" if section.rows.is_empty() => {
                    // No header over nothing: the fact is the emptiness.
                    ui.label(
                        egui::RichText::new(EMPTY_TABLE).color(ui.visuals().weak_text_color()),
                    );
                }
                "table" => {
                    if table_preview(ui, section, i) {
                        explore = Some(section.clone());
                    }
                }
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
    explore
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
                let value = row.get(1).map(render_value).unwrap_or_default();
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

/// The first rows of a table in Rerun's dense style, column names on
/// top, nothing scrolling. A click anywhere on it, or on the footer,
/// opens the modal; returns true when it was asked for.
fn table_preview(ui: &mut egui::Ui, section: &Section, idx: usize) -> bool {
    let tokens = ui.tokens();
    let style = TableStyle::Dense;
    let row_h = tokens.table_row_height(style) + 2.0;
    let cols = section.columns.len().max(1);
    let shown = section.rows.len().min(PREVIEW_ROWS);
    let mut asked = false;
    let inner = ui.push_id(("table", &section.title, idx), |ui| {
        let mut builder = TableBuilder::new(ui).striped(true).vscroll(false);
        for c in 0..cols {
            let width = column_width(section, c, shown, PREVIEW_COLUMN_CAP);
            builder = builder.column(if c + 1 == cols {
                Column::remainder().at_least(width)
            } else {
                Column::exact(width).clip(true)
            });
        }
        builder
            .header(row_h, |mut header| {
                for name in &section.columns {
                    header.col(|ui| {
                        column_name(ui, name);
                    });
                }
            })
            .body(|mut body| {
                tokens.setup_table_body(&mut body, style);
                for row in &section.rows[..shown] {
                    body.row(row_h, |mut r| {
                        for c in 0..cols {
                            r.col(|ui| cell(ui, row.get(c)));
                        }
                    });
                }
            });
        let hidden = section.rows.len().saturating_sub(shown);
        ui.add_space(4.0);
        let label = if hidden > 0 {
            format!("Explore all {}", crate::widgets::count_word(section.rows.len(), "row"))
        } else {
            "Explore".to_owned()
        };
        if ui.small(label).clicked() {
            asked = true;
        }
    });
    // The table itself is the button: a click on the rows opens the modal.
    let hit = ui.interact(
        inner.response.rect,
        inner.response.id.with("open"),
        egui::Sense::click(),
    );
    if hit.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    asked || hit.clicked()
}

/// Pixels per character of the table font, for sizing a column to its
/// longest value instead of its first (a range `[-1.5708, 1.5708]` must
/// not clip behind a header that says `range`).
const PX_PER_CHAR: f32 = 7.2;
const COLUMN_PADDING: f32 = 18.0;
const COLUMN_MIN: f32 = 80.0;

/// The width a column needs to show every value whole, capped at `cap`.
fn column_width(section: &Section, col: usize, rows: usize, cap: f32) -> f32 {
    let header = section.columns.get(col).map_or(0, |n| n.chars().count());
    let longest = section
        .rows
        .iter()
        .take(rows)
        .filter_map(|r| r.get(col))
        .map(|v| render_inline(v).chars().count())
        .max()
        .unwrap_or(0);
    // The floor wins over a cap below it (`clamp` would panic there).
    (header.max(longest) as f32 * PX_PER_CHAR + COLUMN_PADDING)
        .min(cap.max(COLUMN_MIN))
        .max(COLUMN_MIN)
}

/// A column name in a preview's header (the preview does not sort).
fn column_name(ui: &mut egui::Ui, name: &str) {
    ui.label(
        egui::RichText::new(name)
            .strong()
            .color(ui.visuals().weak_text_color()),
    );
}

/// How a table is sorted: by which column, and which way. Shared by the
/// detail modal and the page listing, so a click cycles the same way
/// everywhere: ascending, descending, then not at all.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SortState(pub Option<(usize, bool)>);

impl SortState {
    /// Click a column: sort ascending, then descending, then not at all.
    pub fn toggle(&mut self, col: usize) {
        self.0 = match self.0 {
            Some((c, true)) if c == col => Some((col, false)),
            Some((c, false)) if c == col => None,
            _ => Some((col, true)),
        };
    }

    /// The column sorted by and its direction, if any.
    pub fn order(self) -> Option<(usize, bool)> {
        self.0
    }

    /// The header's text for column `col`: its name with the arrow of
    /// the direction it sorts by, when it does.
    pub fn header_text(self, col: usize, name: &str) -> String {
        match self.0 {
            Some((c, true)) if c == col => format!("{name} ▲"),
            Some((c, false)) if c == col => format!("{name} ▼"),
            _ => name.to_owned(),
        }
    }
}

/// A header cell that sorts on click. Returns true when clicked.
pub fn sortable_header(ui: &mut egui::Ui, sort: SortState, col: usize, name: &str) -> bool {
    ui.add(
        egui::Label::new(egui::RichText::new(sort.header_text(col, name)).strong())
            .sense(egui::Sense::click()),
    )
    .on_hover_text("sort by this column")
    .clicked()
}

/// One cell: the full value on hover, a dash for nothing.
fn cell(ui: &mut egui::Ui, value: Option<&serde_json::Value>) {
    let text = value.map(render_inline).unwrap_or_default();
    if text.is_empty() {
        ui.label(egui::RichText::new("–").color(ui.visuals().weak_text_color()));
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
}

// -- the table modal: the whole table, sortable, filterable -----------------

/// A table opened for exploration: which one, how it is sorted, what
/// the filter box holds.
#[derive(Clone, Debug)]
pub struct TableView {
    pub section: Section,
    pub sort: SortState,
    pub filter: String,
}

impl TableView {
    pub fn new(section: Section) -> Self {
        Self {
            section,
            sort: SortState::default(),
            filter: String::new(),
        }
    }

    /// The rows the modal lists: those matching the filter, in sort order.
    pub fn rows(&self) -> Vec<&Vec<serde_json::Value>> {
        let needle = self.filter.trim().to_lowercase();
        let mut rows: Vec<&Vec<serde_json::Value>> = self
            .section
            .rows
            .iter()
            .filter(|row| {
                needle.is_empty()
                    || row
                        .iter()
                        .any(|v| render_inline(v).to_lowercase().contains(&needle))
            })
            .collect();
        if let Some((col, ascending)) = self.sort.order() {
            rows.sort_by(|a, b| {
                let (x, y) = (
                    a.get(col).map(render_inline).unwrap_or_default(),
                    b.get(col).map(render_inline).unwrap_or_default(),
                );
                let order = match (leading_number(&x), leading_number(&y)) {
                    (Some(p), Some(q)) => p.total_cmp(&q),
                    _ => x.cmp(&y),
                };
                if ascending {
                    order
                } else {
                    order.reverse()
                }
            });
        }
        rows
    }

    /// Click a column: sort ascending, then descending, then not at all.
    pub fn toggle_sort(&mut self, col: usize) {
        self.sort.toggle(col);
    }
}

/// The modal: title, row count, a filter box, the whole table with a
/// fixed header, resizable columns and sort on click. Returns false
/// when it was closed (the × button, Escape, or a click outside).
pub fn table_modal(ctx: &egui::Context, view: &mut TableView) -> bool {
    let screen = ctx.content_rect();
    let mut close = false;
    let modal = egui::Modal::new(egui::Id::new("table-modal")).show(ctx, |ui| {
        ui.set_width(screen.width() * MODAL_SHARE);
        ui.set_max_height(screen.height() * MODAL_SHARE);
        let (matching, total) = (view.rows().len(), view.section.rows.len());
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(&view.section.title)
                    .text_style(DesignTokens::welcome_screen_h2())
                    .strong(),
            );
            ui.label(
                egui::RichText::new(if view.filter.trim().is_empty() {
                    crate::widgets::count_word(total, "row")
                } else {
                    format!("{matching} of {}", crate::widgets::count_word(total, "row"))
                })
                .color(ui.visuals().weak_text_color()),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .small_icon_button(&re_ui::icons::CLOSE, "Close")
                    .clicked()
                {
                    close = true;
                }
                ui.add(
                    egui::TextEdit::singleline(&mut view.filter)
                        .hint_text("filter rows")
                        .desired_width(220.0),
                );
            });
        });
        if let Some(note) = &view.section.note {
            ui.label(
                egui::RichText::new(note)
                    .text_style(DesignTokens::welcome_screen_tag())
                    .color(ui.visuals().weak_text_color()),
            );
        }
        ui.add_space(8.0);
        let rows = view.rows();
        let tokens = ui.tokens();
        let style = TableStyle::Dense;
        let row_h = tokens.table_row_height(style) + 2.0;
        let cols = view.section.columns.len().max(1);
        let mut clicked_col = None;
        // egui remembers resized column widths per table id: one id per
        // table, or the Bodies modal inherits the Joints widths.
        let mut builder = TableBuilder::new(ui)
            .id_salt(("table-modal", &view.section.title, view.section.rows.len()))
            .striped(true)
            .resizable(true)
            .vscroll(true)
            .max_scroll_height(screen.height() * MODAL_SHARE - 96.0);
        for c in 0..cols {
            let width = column_width(&view.section, c, usize::MAX, MODAL_COLUMN_CAP);
            builder = builder.column(if c + 1 == cols {
                Column::remainder().at_least(width)
            } else {
                Column::initial(width).at_least(COLUMN_MIN).clip(true)
            });
        }
        builder
            .header(row_h + 4.0, |mut header| {
                for (c, name) in view.section.columns.iter().enumerate() {
                    header.col(|ui| {
                        if sortable_header(ui, view.sort, c, name) {
                            clicked_col = Some(c);
                        }
                    });
                }
            })
            .body(|mut body| {
                tokens.setup_table_body(&mut body, style);
                body.rows(row_h, rows.len(), |mut r| {
                    let row = rows[r.index()];
                    for c in 0..cols {
                        r.col(|ui| cell(ui, row.get(c)));
                    }
                });
            });
        if let Some(c) = clicked_col {
            view.toggle_sort(c);
        }
    });
    !(close || modal.should_close())
}

/// The number a cell starts with (`0.053`, `[-1.57, 1.57]`, `4.2e-6`),
/// for sorting numerically where it can.
fn leading_number(text: &str) -> Option<f64> {
    let t = text.trim_start_matches(['[', ' ']);
    let end = t
        .char_indices()
        .find(|(_, ch)| !(ch.is_ascii_digit() || matches!(ch, '-' | '+' | '.' | 'e' | 'E')))
        .map_or(t.len(), |(i, _)| i);
    t[..end].parse().ok()
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
        other => render_value(other),
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
        assert_eq!(render_value(&d.sections[0].rows[1][1]), "15");
        // An older minor of the family is read; another family is not.
        assert!(schema_compatible(&d.schema, DETAIL_SCHEMA));
        assert!(!schema_compatible("trainnr-project-index/1", DETAIL_SCHEMA));
    }

    #[test]
    fn a_sort_state_cycles_and_names_its_column() {
        let mut sort = SortState::default();
        assert_eq!(sort.header_text(0, "joint"), "joint");
        sort.toggle(0);
        assert_eq!(sort.header_text(0, "joint"), "joint ▲");
        assert_eq!(sort.header_text(1, "damping"), "damping");
        sort.toggle(0);
        assert_eq!(sort.header_text(0, "joint"), "joint ▼");
        sort.toggle(0);
        assert_eq!(sort.order(), None, "third click clears");
        sort.toggle(1);
        assert_eq!(sort.order(), Some((1, true)));
    }

    #[test]
    fn values_render_readably() {
        assert_eq!(render_value(&serde_json::json!(["a", "b"])), "a, b");
        assert_eq!(render_value(&serde_json::json!(1.5)), "1.5");
        assert_eq!(render_value(&serde_json::Value::Null), "");
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
    fn a_table_view_filters_and_sorts_numerically_where_it_can() {
        let section = Section {
            title: "Joints".into(),
            kind: "table".into(),
            columns: vec!["joint".into(), "damping".into(), "range".into()],
            rows: vec![
                vec![
                    serde_json::json!("knee"),
                    serde_json::json!(0.053),
                    serde_json::json!("[-1.5, 1.5]"),
                ],
                vec![
                    serde_json::json!("ankle"),
                    serde_json::json!(0.0048),
                    serde_json::json!("[-0.5, 0.5]"),
                ],
                vec![
                    serde_json::json!("hip"),
                    serde_json::json!(0.11),
                    serde_json::json!("[-2.0, 2.0]"),
                ],
            ],
            text: None,
            note: None,
        };
        let mut view = TableView::new(section);
        assert_eq!(view.rows().len(), 3);
        view.toggle_sort(1);
        let by_damping: Vec<String> = view.rows().iter().map(|r| render_inline(&r[0])).collect();
        assert_eq!(
            by_damping,
            vec!["ankle", "knee", "hip"],
            "ascending by number"
        );
        view.toggle_sort(1);
        assert_eq!(render_inline(&view.rows()[0][0]), "hip", "descending");
        view.toggle_sort(1);
        assert!(view.sort.order().is_none(), "third click clears");
        view.toggle_sort(2);
        assert_eq!(
            render_inline(&view.rows()[0][0]),
            "hip",
            "a range sorts by its low end"
        );
        view.filter = "KNE".into();
        assert_eq!(view.rows().len(), 1);
        assert_eq!(leading_number("4.2e-6, 3.4"), Some(4.2e-6));
        assert_eq!(leading_number("hinge"), None);
        let range_col = column_width(&view.section, 2, usize::MAX, 420.0);
        let short_col = column_width(&view.section, 0, usize::MAX, 420.0);
        assert!(range_col > short_col, "a range needs more room than a name");
        assert_eq!(
            column_width(&view.section, 0, usize::MAX, 50.0),
            80.0,
            "never under the floor"
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
        assert_eq!(field_word("certificate"), "evaluation", "a kind's word");
        assert_eq!(field_word("task"), "environment");
    }
}
