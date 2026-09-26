//! The three ways a page lists its artifacts: cards (pictures, for
//! browsing), a table (sortable, filterable, for finding a number) and,
//! for evaluations, a matrix (policies by condition, for comparing).
//! The chosen view is remembered per page for the session.

use egui_extras::{Column, TableBuilder};
use re_ui::{DesignTokens, TableStyle, UiExt as _};

use crate::detail::{sortable_header, SortState};
use crate::model::{ago_iso, split_stamp, Artifact, HIDDEN_KEYS, UNRECORDED};
use crate::pages::Section;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum View {
    #[default]
    Cards,
    Table,
    Matrix,
}

impl View {
    /// Every view, in the order the switch shows them.
    pub const ALL: &'static [View] = &[View::Cards, View::Table, View::Matrix];

    pub fn label(self) -> &'static str {
        match self {
            Self::Cards => "Cards",
            Self::Table => "Table",
            Self::Matrix => "Matrix",
        }
    }

    /// A view by its name, as the agent's `open` command spells it
    /// (the label, any case).
    pub fn parse(name: &str) -> Option<Self> {
        let wanted = name.trim().to_lowercase();
        Self::ALL
            .iter()
            .copied()
            .find(|v| v.label().to_lowercase() == wanted)
    }

    /// The names `parse` accepts, for a refusal.
    pub fn names() -> Vec<String> {
        Self::ALL.iter().map(|v| v.label().to_lowercase()).collect()
    }
}

/// At most this many summary columns; the drawer holds the rest.
const MAX_SUMMARY_COLUMNS: usize = 6;
const NAME_COLUMN: f32 = 240.0;
const UPDATED_COLUMN: f32 = 110.0;
/// A listing fills the window below the heading, never less than this.
const TABLE_MIN_HEIGHT: f32 = 320.0;
/// What the heading, the switch and the margins take above a listing.
const TABLE_CHROME: f32 = 240.0;

fn listing_height(ui: &egui::Ui) -> f32 {
    (ui.ctx().content_rect().height() - TABLE_CHROME).max(TABLE_MIN_HEIGHT)
}
/// The summary key the matrix reads for its columns and its cells
/// (`rq_pipeline/project/index.py`, `_summary_certificate`).
const CONDITION_KEY: &str = "judged at";
const SUCCESS_KEY: &str = "success";
const POLICY_CITE: &str = "policy";
const MIN_MATRIX_CONDITIONS: usize = 2;
const MATRIX_CELL: f32 = 96.0;
const MATRIX_ROW_LABEL: f32 = 200.0;
const MATRIX_HEADER_LINES: f32 = 6.0;

fn view_id(section: Section) -> egui::Id {
    egui::Id::new(("trainnr.listing.view", section.slug()))
}

pub fn view_of(ctx: &egui::Context, section: Section) -> View {
    ctx.data(|d| d.get_temp(view_id(section)))
        .unwrap_or_default()
}

/// Choose a page's view (the agent's door). A matrix on a page that
/// cannot draw one falls back to cards when drawn.
pub fn set_view(ctx: &egui::Context, section: Section, view: View) {
    ctx.data_mut(|d| d.insert_temp(view_id(section), view));
}

/// The Cards / Table / Matrix switch, right-aligned on the heading row.
/// Returns the view in force after any click.
pub fn view_switch(ui: &mut egui::Ui, section: Section, matrix_available: bool) -> View {
    let mut view = view_of(ui.ctx(), section);
    if view == View::Matrix && !matrix_available {
        view = View::Cards;
    }
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        for choice in View::ALL
            .iter()
            .copied()
            .rev()
            .filter(|v| *v != View::Matrix || matrix_available)
        {
            if ui
                .selectable_label(view == choice, choice.label())
                .clicked()
            {
                view = choice;
            }
        }
    });
    set_view(ui.ctx(), section, view);
    view
}

// ---------------------------------------------------------------------
// The table

#[derive(Clone, Default)]
struct TableState {
    sort: SortState,
    filter: String,
}

fn table_state_id(section: Section) -> egui::Id {
    egui::Id::new(("trainnr.listing.table", section.slug()))
}

/// The columns a set of artifacts share: name, the summary keys in
/// order of first appearance, then when they last changed.
fn columns_of(rows: &[&Artifact]) -> Vec<String> {
    let mut keys: Vec<String> = Vec::new();
    for a in rows {
        for k in a.summary.keys() {
            if !HIDDEN_KEYS.contains(&k.as_str()) && !keys.contains(k) {
                keys.push(k.clone());
            }
        }
    }
    keys.truncate(MAX_SUMMARY_COLUMNS);
    let mut columns = vec!["name".to_owned()];
    columns.extend(keys);
    columns.push("updated".to_owned());
    columns
}

fn cell_text(a: &Artifact, column: &str) -> String {
    match column {
        "name" => split_stamp(&a.stamp).0.to_owned(),
        "updated" => ago_iso(a.updated.as_deref()),
        key => a
            .summary
            .get(key)
            .map(crate::model::render_value)
            .unwrap_or_default(),
    }
}

/// Numeric where the text starts with a number, so `19 / 40` and `0.53`
/// sort by value; text otherwise.
fn sort_key(a: &Artifact, column: &str) -> (f64, String) {
    if column == "updated" {
        return (-a.updated_epoch(), String::new());
    }
    let text = cell_text(a, column);
    let number = text
        .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e'))
        .find(|s| !s.is_empty())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(f64::INFINITY);
    (number, text.to_lowercase())
}

/// The page as a sortable, filterable table. Returns the stamp of a row
/// that was clicked.
pub fn table(
    ui: &mut egui::Ui,
    section: Section,
    rows: &[&Artifact],
    selected: Option<&str>,
) -> Option<String> {
    let id = table_state_id(section);
    let mut state: TableState = ui.ctx().data(|d| d.get_temp(id)).unwrap_or_default();
    let columns = columns_of(rows);
    let needle = state.filter.trim().to_lowercase();
    let mut shown: Vec<&Artifact> = rows
        .iter()
        .copied()
        .filter(|a| {
            needle.is_empty()
                || columns
                    .iter()
                    .any(|c| cell_text(a, c).to_lowercase().contains(&needle))
        })
        .collect();
    if let Some((col, ascending)) = state.sort.order() {
        if let Some(column) = columns.get(col) {
            // One key per row, not one per comparison.
            let mut keyed: Vec<((f64, String), &Artifact)> =
                shown.iter().map(|a| (sort_key(a, column), *a)).collect();
            keyed.sort_by(|(ka, _), (kb, _)| {
                let ord = ka.0.total_cmp(&kb.0).then_with(|| ka.1.cmp(&kb.1));
                if ascending {
                    ord
                } else {
                    ord.reverse()
                }
            });
            shown = keyed.into_iter().map(|(_, a)| a).collect();
        }
    }
    let mut clicked: Option<String> = None;
    let mut clicked_col: Option<usize> = None;
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(if needle.is_empty() {
                format!("{} rows", rows.len())
            } else {
                format!("{} of {} rows", shown.len(), rows.len())
            })
            .color(ui.visuals().weak_text_color()),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add(
                egui::TextEdit::singleline(&mut state.filter)
                    .hint_text("filter rows")
                    .desired_width(220.0),
            );
        });
    });
    ui.add_space(6.0);
    let tokens = ui.tokens();
    let style = TableStyle::Dense;
    let row_h = tokens.table_row_height(style) + 4.0;
    let height = listing_height(ui);
    // Wide tables scroll sideways inside their own box; the page never does.
    egui::ScrollArea::horizontal()
        .id_salt(("listing-scroll", section.slug()))
        .show(ui, |ui| {
            let mut builder = TableBuilder::new(ui)
                .id_salt(("listing", section.slug(), columns.len()))
                .striped(true)
                .resizable(true)
                .vscroll(true)
                .min_scrolled_height(height)
                .max_scroll_height(height)
                .sense(egui::Sense::click());
            for (c, column) in columns.iter().enumerate() {
                builder = builder.column(match column.as_str() {
                    "name" => Column::initial(NAME_COLUMN).at_least(120.0).clip(true),
                    "updated" => Column::exact(UPDATED_COLUMN),
                    _ if c + 2 == columns.len() => Column::remainder().at_least(120.0).clip(true),
                    _ => Column::initial(160.0).at_least(80.0).clip(true),
                });
            }
            builder
                .header(row_h + 2.0, |mut header| {
                    for (c, name) in columns.iter().enumerate() {
                        header.col(|ui| {
                            if sortable_header(ui, state.sort, c, name) {
                                clicked_col = Some(c);
                            }
                        });
                    }
                })
                .body(|mut body| {
                    tokens.setup_table_body(&mut body, style);
                    body.rows(row_h, shown.len(), |mut r| {
                        let a = shown[r.index()];
                        r.set_selected(selected == Some(a.stamp.as_str()));
                        for column in &columns {
                            r.col(|ui| {
                                let text = cell_text(a, column);
                                if column == "name" {
                                    ui.add(
                                        egui::Label::new(egui::RichText::new(text).strong())
                                            .truncate(),
                                    );
                                } else {
                                    ui.add(egui::Label::new(text).truncate());
                                }
                            });
                        }
                        if r.response().clicked() {
                            clicked = Some(a.stamp.clone());
                        }
                    });
                });
        });
    if let Some(c) = clicked_col {
        state.sort.toggle(c);
    }
    ui.ctx().data_mut(|d| d.insert_temp(id, state));
    clicked
}

// ---------------------------------------------------------------------
// The matrix: policies by the condition they were judged under

pub fn condition_of(a: &Artifact) -> Option<String> {
    a.summary
        .get(CONDITION_KEY)
        .and_then(|v| v.as_str())
        .map(str::to_owned)
}

pub fn policy_of(a: &Artifact) -> String {
    a.cites
        .get(POLICY_CITE)
        .and_then(|v| v.as_str())
        .map(|s| split_stamp(s).0.to_owned())
        .unwrap_or_else(|| UNRECORDED.to_owned())
}

/// `19 / 40` → (19, 40).
pub fn successes_of(a: &Artifact) -> Option<(u32, u32)> {
    let text = a.summary.get(SUCCESS_KEY)?.as_str()?;
    let (k, n) = text.split_once('/')?;
    Some((k.trim().parse().ok()?, n.trim().parse().ok()?))
}

/// A matrix reads when the rows are evaluations judged under at least
/// two named conditions.
pub fn matrix_available(rows: &[&Artifact]) -> bool {
    let mut conditions: Vec<String> = Vec::new();
    for a in rows {
        if let Some(c) = condition_of(a) {
            if !conditions.contains(&c) {
                conditions.push(c);
            }
        }
    }
    conditions.len() >= MIN_MATRIX_CONDITIONS
}

/// Rows the policies, columns the conditions, each cell the newest
/// evaluation of that policy under that condition: its rate as text on
/// a fill whose strength is the rate (one hue, light to strong).
/// Returns the stamp of a clicked cell.
pub fn matrix(ui: &mut egui::Ui, rows: &[&Artifact], selected: Option<&str>) -> Option<String> {
    let mut policies: Vec<String> = Vec::new();
    let mut conditions: Vec<String> = Vec::new();
    let mut cells: std::collections::HashMap<(String, String), &Artifact> =
        std::collections::HashMap::new();
    for a in rows {
        let Some(condition) = condition_of(a) else {
            continue;
        };
        let policy = policy_of(a);
        if !policies.contains(&policy) {
            policies.push(policy.clone());
        }
        if !conditions.contains(&condition) {
            conditions.push(condition.clone());
        }
        let key = (policy, condition);
        let newer = cells
            .get(&key)
            .is_none_or(|old| a.updated_epoch() > old.updated_epoch());
        if newer {
            cells.insert(key, a);
        }
    }
    policies.sort();
    let mut clicked = None;
    ui.label(
        egui::RichText::new(format!(
            "{} policies × {} conditions · a cell is the newest evaluation of that policy under that condition; the fill is its success rate",
            policies.len(),
            conditions.len()
        ))
        .color(ui.visuals().weak_text_color()),
    );
    ui.add_space(6.0);
    let tokens = ui.tokens();
    let style = TableStyle::Dense;
    let row_h = tokens.table_row_height(style) + 10.0;
    let small_h = ui.text_style_height(&egui::TextStyle::Small);
    let hue = tokens.highlight_color;
    let height = listing_height(ui);
    // Wide matrices scroll sideways inside their own box; the page never does.
    egui::ScrollArea::horizontal()
        .id_salt("matrix-scroll")
        .show(ui, |ui| {
            let mut builder = TableBuilder::new(ui)
                .id_salt(("matrix", conditions.len(), policies.len()))
                .striped(false)
                .resizable(true)
                .vscroll(true)
                .min_scrolled_height(height)
                .max_scroll_height(height)
                .column(Column::initial(MATRIX_ROW_LABEL).at_least(120.0).clip(true));
            for _ in &conditions {
                builder = builder.column(Column::initial(MATRIX_CELL).at_least(72.0).clip(true));
            }
            builder
                // A condition's name is a sentence ("CROSS-evaluation: trained
                // in the fit, judged in declared; law DR …"): it wraps over
                // MATRIX_HEADER_LINES small lines rather than truncating to
                // its first word; the whole of it is on hover.
                .header(small_h * MATRIX_HEADER_LINES + 8.0, |mut header| {
                    header.col(|ui| {
                        ui.label(egui::RichText::new("policy").strong());
                    });
                    for condition in &conditions {
                        header.col(|ui| {
                            ui.add(
                                egui::Label::new(egui::RichText::new(condition).strong().small())
                                    .wrap(),
                            )
                            .on_hover_text(condition);
                        });
                    }
                })
                .body(|mut body| {
                    tokens.setup_table_body(&mut body, style);
                    body.rows(row_h, policies.len(), |mut r| {
                        let policy = &policies[r.index()];
                        r.col(|ui| {
                            ui.add(
                                egui::Label::new(egui::RichText::new(policy).strong()).truncate(),
                            );
                        });
                        for condition in &conditions {
                            r.col(|ui| {
                                let Some(a) = cells.get(&(policy.clone(), condition.clone()))
                                else {
                                    ui.label(
                                        egui::RichText::new("·")
                                            .color(ui.visuals().weak_text_color()),
                                    );
                                    return;
                                };
                                let rect = ui.available_rect_before_wrap();
                                let (text, rate) = match successes_of(a) {
                                    Some((k, n)) if n > 0 => {
                                        (format!("{k} / {n}"), k as f32 / n as f32)
                                    }
                                    _ => (UNRECORDED.to_owned(), 0.0),
                                };
                                let response =
                                    ui.interact(rect, ui.id().with(&a.stamp), egui::Sense::click());
                                let fill = hue.linear_multiply(0.12 + 0.6 * rate);
                                ui.painter().rect_filled(rect.shrink(1.0), 4.0, fill);
                                if selected == Some(a.stamp.as_str()) {
                                    ui.painter().rect_stroke(
                                        rect.shrink(1.0),
                                        4.0,
                                        egui::Stroke::new(1.5, hue),
                                        egui::StrokeKind::Inside,
                                    );
                                } else if response.hovered() {
                                    ui.painter().rect_stroke(
                                        rect.shrink(1.0),
                                        4.0,
                                        egui::Stroke::new(1.0, hue.linear_multiply(0.7)),
                                        egui::StrokeKind::Inside,
                                    );
                                }
                                ui.put(
                                    rect,
                                    egui::Label::new(
                                        egui::RichText::new(text)
                                            .text_style(DesignTokens::welcome_screen_body())
                                            .strong(),
                                    )
                                    .selectable(false),
                                );
                                let hover = format!(
                                    "{}\n{} under {}",
                                    split_stamp(&a.stamp).0,
                                    policy,
                                    condition
                                );
                                if response.on_hover_text(hover).clicked() {
                                    clicked = Some(a.stamp.clone());
                                }
                            });
                        }
                    });
                });
        });
    clicked
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(stamp: &str, summary: serde_json::Value, cites: serde_json::Value) -> Artifact {
        serde_json::from_value(serde_json::json!({
            "kind": "certificate", "stamp": stamp, "path": stamp,
            "summary": summary, "cites": cites
        }))
        .unwrap()
    }

    #[test]
    fn a_matrix_needs_two_conditions_and_reads_its_cells() {
        let a = artifact(
            "p1-fit@1",
            serde_json::json!({"success": "19 / 40", "judged at": "at the fit"}),
            serde_json::json!({"policy": "p1@a"}),
        );
        let b = artifact(
            "p1-x0.7@2",
            serde_json::json!({"success": "0 / 40", "judged at": "fit x 0.7"}),
            serde_json::json!({"policy": "p1@a"}),
        );
        assert!(!matrix_available(&[&a]));
        assert!(matrix_available(&[&a, &b]));
        assert_eq!(successes_of(&a), Some((19, 40)));
        assert_eq!(policy_of(&a), "p1");
        assert_eq!(condition_of(&b).as_deref(), Some("fit x 0.7"));
    }

    #[test]
    fn table_columns_follow_the_summaries_and_sort_by_value() {
        let a = artifact(
            "x@1",
            serde_json::json!({"success": "19 / 40", "files": 3}),
            serde_json::json!({}),
        );
        let b = artifact(
            "y@2",
            serde_json::json!({"success": "3 / 40"}),
            serde_json::json!({}),
        );
        assert_eq!(columns_of(&[&a, &b]), vec!["name", "success", "updated"]);
        assert!(sort_key(&b, "success").0 < sort_key(&a, "success").0);
        assert_eq!(cell_text(&a, "name"), "x");
    }

    #[test]
    fn views_parse_from_their_names() {
        assert_eq!(View::parse("cards"), Some(View::Cards));
        assert_eq!(View::parse(" Matrix "), Some(View::Matrix));
        assert_eq!(View::parse("grid"), None);
        assert_eq!(View::names(), vec!["cards", "table", "matrix"]);
        assert_eq!(View::ALL.len(), View::names().len());
    }
}
