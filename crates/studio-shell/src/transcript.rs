//! Rendering for the agent transcript — the "first class" half of the
//! panel rework (`agent.rs` holds the model). The interaction ideas are
//! Zed's and Cursor's, reimplemented on egui/re_ui from scratch (Zed is
//! GPL — ideas only, never code, the standing rule):
//!
//! - the agent's replies are RENDERED markdown (`egui_commonmark`, the
//!   same crate re_ui already pulls in — zero new dependency weight),
//!   flowing unlabeled: the user's messages are the visually distinct
//!   ones (filled cards), so labeling every bubble "you"/"claude" was
//!   chat-log noise, not information;
//! - a tool call is ONE compact row — chevron, live spinner → ✓/✗, kind,
//!   title — and the whole row is the click target that expands the
//!   detail (full command, output, diffs), Zed's shape exactly;
//! - file edits inside tool calls are real ± diffs, computed here with
//!   `similar` (also already in the tree, via Rerun);
//! - thinking is collapsed and dimmed; the plan is a checklist card.

use agent_client_protocol::schema::v1::{PlanEntry, PlanEntryStatus, ToolCallStatus, ToolKind};
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};
use re_ui::UiExt as _;
use similar::{ChangeTag, TextDiff};

use crate::agent::{ToolCard, ToolDetail, TranscriptItem};

/// Past this many rendered diff lines the rest is elided — a whole-file
/// rewrite must not turn the transcript into the file.
const DIFF_MAX_LINES: usize = 120;

/// Breathing room between conversation turns; tool rows inside a turn
/// sit tighter (they read as one burst of activity, not new turns).
const TURN_GAP: f32 = 8.0;
const ROW_GAP: f32 = 2.0;

/// The one in-progress spinner size, shared with the composer's
/// turn-active spinner in `main.rs` — it was 12.0 in three places once.
pub const SPINNER_SIZE: f32 = 12.0;

/// The user card's shape — filled, strokeless (a stroked version read
/// as an editable text field on screen).
const USER_CARD_CORNER_RADIUS: f32 = 6.0;
const USER_CARD_MARGIN: f32 = 8.0;

/// Draws transcript items. Owns the markdown cache (`egui_commonmark`
/// keys its layout state by content, one cache for the whole panel).
pub struct TranscriptView {
    cache: CommonMarkCache,
}

impl TranscriptView {
    pub fn new() -> Self {
        Self {
            cache: CommonMarkCache::default(),
        }
    }

    /// Draws one item. `index` is the item's position in the transcript —
    /// the id salt keeping each collapsible row's open/closed state its
    /// own.
    pub fn show(&mut self, ui: &mut egui::Ui, index: usize, item: &TranscriptItem) {
        match item {
            TranscriptItem::User { text, specialist } => {
                ui.add_space(TURN_GAP);
                // A FILLED card, deliberately strokeless — the earlier
                // stroked version read as an editable text field, not a
                // sent message (seen live).
                egui::Frame::new()
                    .fill(ui.visuals().code_bg_color)
                    .corner_radius(USER_CARD_CORNER_RADIUS)
                    .inner_margin(USER_CARD_MARGIN)
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        if let Some(name) = specialist {
                            ui.weak(egui::RichText::new(format!("→ {name}")).small());
                        }
                        ui.label(text);
                    });
                ui.add_space(TURN_GAP);
            }
            TranscriptItem::Agent(text) => {
                CommonMarkViewer::new().show(ui, &mut self.cache, text);
                ui.add_space(TURN_GAP);
            }
            TranscriptItem::Thought(text) => {
                collapsing_row(ui, ("thought", index), |ui| {
                    // No spinner here, deliberately: whether a thought is
                    // still streaming isn't tracked, and an animated
                    // spinner is both a guess AND a permanent repaint
                    // loop (egui spinners repaint every frame, forever).
                    ui.weak(egui::RichText::new("thinking…").italics());
                })
                .body_unindented(|ui| {
                    indented(ui, |ui| {
                        ui.weak(text);
                    });
                });
                ui.add_space(ROW_GAP);
            }
            TranscriptItem::Tool(tool) => {
                self.tool_row(ui, index, tool);
                ui.add_space(ROW_GAP);
            }
            TranscriptItem::Plan(entries) => {
                plan_card(ui, entries);
                ui.add_space(ROW_GAP);
            }
            TranscriptItem::Status(text) => {
                ui.weak(egui::RichText::new(text).small());
                ui.add_space(ROW_GAP);
            }
            TranscriptItem::Error(text) => {
                ui.error_label(text);
                ui.add_space(ROW_GAP);
            }
        }
    }

    /// One compact row per tool call; the row IS the disclosure control.
    fn tool_row(&mut self, ui: &mut egui::Ui, index: usize, tool: &ToolCard) {
        collapsing_row(ui, ("tool", index), |ui| {
            status_glyph(ui, tool.status);
            ui.weak(egui::RichText::new(kind_label(tool.kind)).small());
            // One line however long the command is — the full title
            // lives in the expanded body.
            ui.add(
                egui::Label::new(
                    egui::RichText::new(first_line(&tool.title))
                        .monospace()
                        .small(),
                )
                .truncate(),
            );
        })
        .body_unindented(|ui| {
            indented(ui, |ui| {
                if first_line(&tool.title) != tool.title {
                    ui.weak(egui::RichText::new(&tool.title).monospace().small());
                }
                for detail in &tool.detail {
                    match detail {
                        ToolDetail::Text(text) => {
                            ui.weak(egui::RichText::new(text).monospace().small());
                        }
                        ToolDetail::Diff { path, old, new } => diff_view(ui, path, old, new),
                    }
                }
                if tool.detail.is_empty() && first_line(&tool.title) == tool.title {
                    ui.weak(egui::RichText::new("no output yet").small());
                }
            });
        });
    }
}

/// A Zed-style disclosure row: small chevron + caller's widgets, the
/// whole row clickable. Thin wrapper over egui's own `CollapsingState`
/// so every collapsible in the transcript shares one look.
fn collapsing_row<'a>(
    ui: &'a mut egui::Ui,
    id_salt: impl std::hash::Hash + std::fmt::Debug,
    header: impl FnOnce(&mut egui::Ui),
) -> egui::collapsing_header::HeaderResponse<'a, ()> {
    let id = ui.make_persistent_id(id_salt);
    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
        .show_header(ui, |ui| {
            header(ui);
        })
}

/// The expanded body of a disclosure row, inset under its header.
fn indented(ui: &mut egui::Ui, body: impl FnOnce(&mut egui::Ui)) {
    ui.indent("row_body", body);
}

/// The plan card's frame (user messages carry their own filled,
/// strokeless frame — see `show`; this one keeps the group stroke).
fn card_frame(ui: &egui::Ui) -> egui::Frame {
    egui::Frame::group(ui.style()).inner_margin(6.0)
}

/// Spinner while the call runs, ✓/✗ once it's done — the whole point of
/// mutate-in-place cards: the SAME row changes state instead of a new
/// "tool update" row appearing.
fn status_glyph(ui: &mut egui::Ui, status: ToolCallStatus) {
    match status {
        ToolCallStatus::Completed => {
            let color = ui.tokens().alert_success.icon;
            ui.label(egui::RichText::new("✓").color(color));
        }
        ToolCallStatus::Failed => {
            let color = ui.tokens().alert_error.icon;
            ui.label(egui::RichText::new("✗").color(color));
        }
        // Pending, InProgress, and whatever the protocol adds later:
        // still running as far as this panel knows.
        _ => {
            ui.add(egui::Spinner::new().size(SPINNER_SIZE));
        }
    }
}

fn kind_label(kind: ToolKind) -> &'static str {
    match kind {
        ToolKind::Read => "read",
        ToolKind::Edit => "edit",
        ToolKind::Delete => "delete",
        ToolKind::Move => "move",
        ToolKind::Search => "search",
        ToolKind::Execute => "run",
        ToolKind::Think => "think",
        ToolKind::Fetch => "fetch",
        _ => "tool",
    }
}

fn first_line(text: &str) -> &str {
    text.lines().next().unwrap_or_default()
}

/// A real ± diff, line-diffed here with `similar` from the texts the
/// protocol carries verbatim. Green/red come from the theme's own
/// success/error tokens — identity by sign AND color, readable either way.
fn diff_view(ui: &mut egui::Ui, path: &str, old: &str, new: &str) {
    ui.monospace(egui::RichText::new(path).small());
    let diff = TextDiff::from_lines(old, new);
    for (shown, change) in diff.iter_all_changes().enumerate() {
        if shown == DIFF_MAX_LINES {
            ui.weak("… diff truncated");
            break;
        }
        let (sign, color) = match change.tag() {
            ChangeTag::Delete => ("-", ui.tokens().alert_error.icon),
            ChangeTag::Insert => ("+", ui.tokens().alert_success.icon),
            ChangeTag::Equal => (" ", ui.visuals().weak_text_color()),
        };
        let line = change.value().trim_end_matches('\n');
        ui.label(
            egui::RichText::new(format!("{sign} {line}"))
                .monospace()
                .small()
                .color(color),
        );
    }
}

/// The agent's plan as a checklist — one card, ticking over in place as
/// the plan updates (see `apply`'s replace-in-place rule).
fn plan_card(ui: &mut egui::Ui, entries: &[PlanEntry]) {
    card_frame(ui).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.weak(egui::RichText::new("plan").small());
        for entry in entries {
            ui.horizontal(|ui| {
                match entry.status {
                    PlanEntryStatus::Completed => {
                        let color = ui.tokens().alert_success.icon;
                        ui.label(egui::RichText::new("✓").color(color));
                        ui.weak(egui::RichText::new(&entry.content).strikethrough());
                    }
                    PlanEntryStatus::InProgress => {
                        ui.label("▸");
                        ui.label(&entry.content);
                    }
                    // Pending and future variants: not started.
                    _ => {
                        ui.weak("○");
                        ui.weak(&entry.content);
                    }
                }
            });
        }
    });
}
