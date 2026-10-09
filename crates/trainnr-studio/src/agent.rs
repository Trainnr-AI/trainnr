//! Whether the developer's agent is connected, as the window shows it.
//!
//! trainnr is driven from an agent in the developer's own tool, and the
//! Studio is what that agent opens (docs/35 §10.2). Until 2026-10-10 the
//! window never said whether one was there: a user whose MCP
//! configuration had not taken saw exactly what a correctly wired user
//! saw. `trainnr mcp` now keeps `<user home>/agent.json` while it serves
//! (`trainnr/trainnr/mcp_link.py`); this reads it
//! (`model::read_agent_link`) and draws it in two places.
//!
//! The **status bar** carries a dot on every page, at every stage of a
//! project: an agent can drop at any time, and the one screen that is
//! always on show is where an ambient signal belongs. The **Welcome page
//! and an empty project's Start here card** carry the fuller line, with
//! the command to run when nothing is connected — those two screens
//! promise that trainnr "runs through your coding agent", and a new user
//! reading that promise is the one who needs the fix spelled out.
//!
//! This is not a chat panel and must not grow into one (docs/35 §8).

use std::time::Duration;

use re_ui::{DesignTokens, UiExt as _};

use crate::model::{read_agent_link, AgentLink};

/// How often the connection file is re-read while a page shows it: it is
/// a few hundred bytes, but a page repaints far more often than an agent
/// comes or goes.
const POLL: Duration = Duration::from_secs(1);
/// What a user runs to connect their agent, as the README's "Connect your
/// agent" gives it.
pub const CONNECT_COMMAND: &str =
    "claude mcp add trainnr -- uvx --from \"trainnr[sim,mcp]\" trainnr mcp";
/// The connection dot's radius on a page.
const DOT_RADIUS: f32 = 4.0;
/// The same dot in the status bar, where everything is smaller.
const SMALL_DOT_RADIUS: f32 = 3.0;

/// The connection, re-read at most once a second.
pub fn link(ui: &egui::Ui) -> AgentLink {
    let id = egui::Id::new("agent-link");
    let now = ui.input(|i| i.time);
    let cached = ui.ctx().data(|d| d.get_temp::<(f64, AgentLink)>(id));
    if let Some((at, link)) = cached {
        if now - at < POLL.as_secs_f64() {
            return link;
        }
    }
    let link = read_agent_link(crate::model::user_home().as_deref());
    ui.ctx()
        .data_mut(|d| d.insert_temp(id, (now, link.clone())));
    // The page is otherwise still: ask for the repaint that notices the
    // agent arriving, rather than waiting for the user to move the mouse.
    ui.ctx().request_repaint_after(POLL);
    link
}

/// The dot: the success colour when connected, muted when not.
fn dot(ui: &mut egui::Ui, connected: bool, radius: f32) {
    let colour = if connected {
        ui.tokens().alert_success.icon
    } else {
        crate::theme::palette(ui).muted
    };
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(radius * 2.0, radius * 2.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), radius, colour);
}

/// What the hover says in the status bar, where the line itself is two
/// words.
fn tooltip(link: &AgentLink) -> String {
    if !link.connected {
        return format!(
            "No agent connected. trainnr is driven from your coding agent; \
             connect one and restart it:\n\n{CONNECT_COMMAND}"
        );
    }
    match (&link.last_tool, link.calls) {
        (Some(tool), calls) => {
            format!("Agent connected: {calls} tool call(s), last {tool}")
        }
        (None, _) => "Agent connected; it has called nothing yet".to_owned(),
    }
}

/// The status bar's dot and two words, on every page at every stage: an
/// agent can drop at any time, not only while a project is empty.
pub fn status_bar_chip(ui: &mut egui::Ui, link: &AgentLink) {
    let palette = crate::theme::palette(ui);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 5.0;
        dot(ui, link.connected, SMALL_DOT_RADIUS);
        let (text, colour) = if link.connected {
            ("agent", palette.muted)
        } else {
            ("no agent", ui.visuals().warn_fg_color)
        };
        ui.label(egui::RichText::new(text).small().color(colour));
    })
    .response
    .on_hover_text(tooltip(link));
}

/// The fuller line, for the two screens that promise trainnr "runs
/// through your coding agent": the dot, what it means, and — when
/// nothing is connected — the command that fixes it.
pub fn status(ui: &mut egui::Ui) {
    let link = link(ui);
    let palette = crate::theme::palette(ui);
    ui.horizontal(|ui| {
        dot(ui, link.connected, DOT_RADIUS);
        ui.add_space(8.0);
        let headline = if link.connected {
            "Agent connected"
        } else {
            "No agent connected"
        };
        ui.label(
            egui::RichText::new(headline)
                .text_style(DesignTokens::welcome_screen_body())
                .strong()
                .color(palette.text),
        );
        if let Some(tool) = link.last_tool.filter(|_| link.connected) {
            ui.label(
                egui::RichText::new(format!("· last call {tool}"))
                    .text_style(DesignTokens::welcome_screen_body())
                    .color(palette.muted),
            );
        }
    });
    if link.connected {
        return;
    }
    ui.add_space(6.0);
    ui.label(
        egui::RichText::new(
            "Nothing here moves until one is. Connect your agent, restart it, and \
             this turns green:",
        )
        .text_style(DesignTokens::welcome_screen_body())
        .color(palette.text),
    );
    ui.add_space(6.0);
    crate::onboarding::command_chip(ui, CONNECT_COMMAND);
}
