//! Whether the developer's agent is connected, as the window shows it.
//!
//! The Studio is an app used on its own, and an agent can drive it over
//! MCP (docs/85). Until 2026-10-10 the window never said whether one was
//! there: a user whose MCP configuration had not taken saw exactly what a
//! correctly wired user saw. `trainnr mcp` keeps a file while it serves
//! (`trainnr/trainnr/mcp_link.py`); this reads the folder of them
//! (`model::read_agent_link`) and draws the answer (#32, by a first-run
//! contributor).
//!
//! In the app it lives where connecting lives (2026-10-12): the status
//! bar's Connect agent button says whether one is connected and opens the
//! Connect panel; the panel and the Welcome page carry the fuller line.
//! No agent is an ordinary state for the app, so it is said in the muted
//! colour, never as a warning, and how to connect one is the panel's job.
//!
//! This is not a chat panel and must not grow into one (docs/35 §8).

use std::time::Duration;

use re_ui::{DesignTokens, UiExt as _};

use crate::model::{read_agent_link, AgentLink};

/// How often the connection folder is re-read while a page shows it: the
/// files are a few hundred bytes, but a page repaints far more often than
/// an agent comes or goes.
const POLL: Duration = Duration::from_secs(1);
/// The connection dot's radius on a page.
const DOT_RADIUS: f32 = 4.0;
/// The status bar's dot, a glyph in the button's own words.
const BAR_DOT: &str = "● ";

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

/// The dot's colour: the success colour when connected, muted when not.
fn dot_colour(ui: &egui::Ui, connected: bool) -> egui::Color32 {
    if connected {
        ui.tokens().alert_success.icon
    } else {
        crate::theme::palette(ui).muted
    }
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

/// The status bar button's words: what to do when nothing is connected,
/// what is there when something is.
fn bar_words(link: &AgentLink) -> String {
    match link.count {
        0 => "Connect agent".to_owned(),
        1 => "Agent connected".to_owned(),
        n => format!("{n} agent sessions"),
    }
}

/// What the hover says.
fn tooltip(link: &AgentLink) -> String {
    match (link.count, &link.last_tool) {
        (0, _) => "No agent connected. Connect Claude Code, Codex, Cursor or another \
                   MCP client"
            .to_owned(),
        (1, Some(tool)) => format!("Agent connected: {} tool call(s), last {tool}", link.calls),
        (1, None) => "Agent connected; it has called nothing yet".to_owned(),
        (n, _) => format!(
            "{n} trainnr sessions are serving: {n} agent windows, or fewer \
             windows with more than one trainnr server between them."
        ),
    }
}

/// The status bar's Connect agent button, which says whether an agent is
/// connected: a dot and its words, on every page at every stage, since an
/// agent can drop at any time. A click opens the Connect panel.
pub fn status_bar_button(ui: &mut egui::Ui, link: &AgentLink) -> egui::Response {
    let palette = crate::theme::palette(ui);
    let font = egui::TextStyle::Small.resolve(ui.style());
    let mut words = egui::text::LayoutJob::default();
    words.append(
        BAR_DOT,
        0.0,
        egui::TextFormat::simple(font.clone(), dot_colour(ui, link.connected())),
    );
    words.append(
        &bar_words(link),
        0.0,
        egui::TextFormat::simple(font, palette.muted),
    );
    ui.add(egui::Button::new(words).frame_when_inactive(false))
        .on_hover_text(tooltip(link))
}

/// The fuller line, for the Connect panel and the Welcome page: the dot,
/// what it means, and the last tool one agent called.
pub fn status_line(ui: &mut egui::Ui, link: &AgentLink) {
    let palette = crate::theme::palette(ui);
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(DOT_RADIUS * 2.0, DOT_RADIUS * 2.0),
            egui::Sense::hover(),
        );
        ui.painter()
            .circle_filled(rect.center(), DOT_RADIUS, dot_colour(ui, link.connected()));
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new(headline(link))
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
    })
    .response
    .on_hover_text(tooltip(link));
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
    fn nothing_connected_offers_to_connect_without_alarm() {
        let none = link(0, 0, None);
        assert_eq!(headline(&none), "No agent connected");
        assert_eq!(bar_words(&none), "Connect agent");
        assert!(tooltip(&none).contains("Connect"), "{}", tooltip(&none));
    }

    #[test]
    fn one_agent_is_named_by_what_it_last_did() {
        let one = link(1, 5, Some("onboard_robot"));
        assert_eq!(headline(&one), "Agent connected");
        assert_eq!(bar_words(&one), "Agent connected");
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
        assert_eq!(bar_words(&many), "2 agent sessions");
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

    /// Both drawings render headless in either state.
    #[test]
    fn the_button_and_the_line_draw() {
        let ctx = egui::Context::default();
        re_ui::apply_style_and_install_loaders(&ctx);
        for state in [
            link(0, 0, None),
            link(1, 2, Some("list_robots")),
            link(2, 0, None),
        ] {
            let mut output = ctx.run_ui(Default::default(), |ui| {
                let _ = status_bar_button(ui, &state);
                status_line(ui, &state);
            });
            output.textures_delta.clear(); // no renderer here to apply them
        }
    }
}
