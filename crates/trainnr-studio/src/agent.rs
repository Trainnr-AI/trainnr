//! Whether the developer's agent is connected, as the window shows it.
//!
//! trainnr is driven from an agent in the developer's own tool, and the
//! Studio is what that agent opens (docs/35 §10.2). Until 2026-10-10 the
//! window never said whether one was there: a user whose MCP
//! configuration had not taken saw exactly what a correctly wired user
//! saw. `trainnr mcp` now keeps a file while it serves
//! (`trainnr/trainnr/mcp_link.py`); this reads the folder of them
//! (`model::read_agent_link`) and draws the answer in two places.
//!
//! The **status bar** carries a dot on every page, at every stage of a
//! project: an agent can drop at any time, and the one screen that is
//! always on show is where an ambient signal belongs. The **Welcome page
//! and an empty project's Start here card** carry the fuller line, with
//! how to connect one when none is. Those two screens promise that
//! trainnr "runs through your coding agent", and a new user reading that
//! promise is the one who needs the fix spelled out.
//!
//! This is not a chat panel and must not grow into one (docs/35 §8).

use std::time::Duration;

use re_ui::{DesignTokens, UiExt as _};

use crate::model::{read_agent_link, AgentLink};

/// How often the connection folder is re-read while a page shows it: the
/// files are a few hundred bytes, but a page repaints far more often than
/// an agent comes or goes.
const POLL: Duration = Duration::from_secs(1);
/// How to connect one, the README's own first path ("Install the
/// plugin"). Pinned to the README by `tests/test_studio_mirrors.py`: a
/// window that teaches an install the project no longer documents is
/// worse than one that says nothing.
pub const PLUGIN_MARKETPLACE: &str = "claude plugin marketplace add Trainnr-AI/trainnr";
pub const PLUGIN_INSTALL: &str = "claude plugin install trainnr@trainnr";
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

/// What the line says: one agent is named by what it last did, several
/// are only counted.
///
/// Several are counted as **sessions**, not agents, because that is what
/// the folder holds. One agent window configured with two trainnr servers
/// — the plugin's and a checkout's `.mcp.json`, which Claude Code loads
/// side by side as `plugin:trainnr:trainnr` and `trainnr` — serves twice
/// and would otherwise be announced as two agents (found by installing
/// the plugin over a checkout, 2026-10-12).
fn headline(link: &AgentLink) -> String {
    match link.count {
        0 => "No agent connected".to_owned(),
        1 => "Agent connected".to_owned(),
        n => format!("{n} agent sessions connected"),
    }
}

/// What the hover says in the status bar, where the line itself is two
/// words.
fn tooltip(link: &AgentLink) -> String {
    match (link.count, &link.last_tool) {
        (0, _) => format!(
            "No agent connected. trainnr is driven from your coding agent; \
             install the plugin and restart it:\n\n{PLUGIN_MARKETPLACE}\n{PLUGIN_INSTALL}"
        ),
        (1, Some(tool)) => format!("Agent connected: {} tool call(s), last {tool}", link.calls),
        (1, None) => "Agent connected; it has called nothing yet".to_owned(),
        (n, _) => format!(
            "{n} trainnr sessions are serving: {n} agent windows, or fewer \
             windows with more than one trainnr server between them."
        ),
    }
}

/// The status bar's dot and two words, on every page at every stage: an
/// agent can drop at any time, not only while a project is empty.
pub fn status_bar_chip(ui: &mut egui::Ui, link: &AgentLink) {
    let palette = crate::theme::palette(ui);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 5.0;
        dot(ui, link.connected(), SMALL_DOT_RADIUS);
        let (text, colour) = match link.count {
            0 => ("no agent".to_owned(), ui.visuals().warn_fg_color),
            1 => ("agent".to_owned(), palette.muted),
            // Sessions, not agents, for the reason on `headline`.
            n => (format!("{n} sessions"), palette.muted),
        };
        ui.label(egui::RichText::new(text).small().color(colour));
    })
    .response
    .on_hover_text(tooltip(link));
}

/// The fuller line, for the two screens that promise trainnr "runs
/// through your coding agent": the dot, what it means, and — when
/// nothing is connected — how to connect one.
///
/// The block keeps the tallest height it has drawn, so the cards beneath
/// it do not jump by the height of the instructions when an agent
/// arrives or drops (review, 2026-10-11).
pub fn status(ui: &mut egui::Ui) {
    let link = link(ui);
    let palette = crate::theme::palette(ui);
    let id = egui::Id::new("agent-status-height");
    let top = ui.cursor().top();

    ui.horizontal(|ui| {
        dot(ui, link.connected(), DOT_RADIUS);
        ui.add_space(8.0);
        ui.label(
            egui::RichText::new(headline(&link))
                .text_style(DesignTokens::welcome_screen_body())
                .strong()
                .color(palette.text),
        );
        if let Some(tool) = link.last_tool.as_deref() {
            ui.label(
                egui::RichText::new(format!("· last call {tool}"))
                    .text_style(DesignTokens::welcome_screen_body())
                    .color(palette.muted),
            );
        }
    });
    if !link.connected() {
        ui.add_space(6.0);
        ui.label(
            // The Welcome page's "Open the Go2 walk" works with no agent
            // at all, so this cannot say that nothing moves without one.
            egui::RichText::new(
                "trainnr's tools run through your coding agent. Install the plugin and \
                 restart it, and this turns green:",
            )
            .text_style(DesignTokens::welcome_screen_body())
            .color(palette.text),
        );
        ui.add_space(6.0);
        crate::onboarding::command_chip(ui, PLUGIN_MARKETPLACE);
        ui.add_space(4.0);
        crate::onboarding::command_chip(ui, PLUGIN_INSTALL);
        ui.add_space(6.0);
        ui.label(
            egui::RichText::new(
                "Cursor, Codex or a checkout: see \"Any other MCP client\" in the README.",
            )
            .text_style(DesignTokens::welcome_screen_body())
            .color(palette.muted),
        );
    }

    // Hold the tallest height drawn this session, so the page below is
    // still when the state flips.
    let used = ui.cursor().top() - top;
    let tallest = ui
        .ctx()
        .data(|d| d.get_temp::<f32>(id))
        .unwrap_or(used)
        .max(used);
    ui.ctx().data_mut(|d| d.insert_temp(id, tallest));
    if tallest > used {
        ui.add_space(tallest - used);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn link(count: usize, calls: u64, last_tool: Option<&str>) -> AgentLink {
        AgentLink {
            count,
            calls,
            last_tool: last_tool.map(str::to_owned),
        }
    }

    #[test]
    fn nothing_connected_says_so_and_offers_the_plugin() {
        let none = link(0, 0, None);
        assert_eq!(headline(&none), "No agent connected");
        let hover = tooltip(&none);
        assert!(hover.contains(PLUGIN_MARKETPLACE), "{hover}");
        assert!(hover.contains(PLUGIN_INSTALL), "{hover}");
    }

    #[test]
    fn one_agent_is_named_by_what_it_last_did() {
        let one = link(1, 5, Some("onboard_robot"));
        assert_eq!(headline(&one), "Agent connected");
        let hover = tooltip(&one);
        assert!(hover.contains('5'), "the call count: {hover}");
        assert!(hover.contains("onboard_robot"), "the last tool: {hover}");
    }

    #[test]
    fn an_agent_that_has_called_nothing_is_still_connected() {
        let idle = link(1, 0, None);
        assert_eq!(headline(&idle), "Agent connected");
        assert_eq!(tooltip(&idle), "Agent connected; it has called nothing yet");
    }

    #[test]
    fn several_are_counted_as_sessions_and_none_is_named() {
        // One agent window with two trainnr servers configured - the
        // plugin's and a checkout's - serves twice, so the plural cannot
        // promise two agents (2026-10-12).
        let many = link(2, 9, Some("list_robots"));
        assert_eq!(headline(&many), "2 agent sessions connected");
        let hover = tooltip(&many);
        assert!(!hover.contains("list_robots"), "whose call is it: {hover}");
        assert!(hover.contains("2 trainnr sessions"), "{hover}");
    }

    #[test]
    fn connected_is_whether_anything_is_serving() {
        assert!(!link(0, 0, None).connected());
        assert!(link(1, 0, None).connected());
        assert!(link(3, 0, None).connected());
    }
}
