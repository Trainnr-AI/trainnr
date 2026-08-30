//! The Studio's shell (docs/35-the-studio.md, phase 1 groundwork).
//!
//! One window, Rerun's own theme (`re_ui`), a real MuJoCo-rendered
//! viewport (via `viewport::ViewportFeed` — MuJoCo itself never runs
//! in-process, see that module) and a real agent-client-protocol session
//! (via `agent::AgentSession` — talking to Claude Code's own ACP adapter,
//! see that module) in the side panel.

mod agent;
mod viewport;

use agent::{AgentLine, AgentSession};
use re_ui::UiExt as _;
use viewport::ViewportFeed;

const DEFAULT_TASK: &str = "block_stack";

/// Agent panel layout, sized by eye against this window's default size,
/// not measured. Named rather than inlined so a future pass tuning the
/// panel doesn't have to first figure out which bare number means what.
const AGENT_PANEL_DEFAULT_WIDTH: f32 = 360.0;
const INPUT_ROW_RESERVED_HEIGHT: f32 = 40.0;
const PERMISSION_OPTION_ROW_HEIGHT: f32 = 30.0;
const PERMISSION_BLOCK_PADDING: f32 = 30.0;

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "robotiq studio",
        native_options,
        Box::new(|creation_context| {
            re_ui::apply_style_and_install_loaders(&creation_context.egui_ctx);
            let viewport = ViewportFeed::spawn(&creation_context.egui_ctx, DEFAULT_TASK);
            let agent = AgentSession::spawn(&creation_context.egui_ctx);
            Ok(Box::new(StudioShell {
                viewport,
                agent,
                transcript: Vec::new(),
                draft: String::new(),
            }))
        }),
    )
}

struct StudioShell {
    viewport: ViewportFeed,
    agent: AgentSession,
    transcript: Vec<AgentLine>,
    draft: String,
}

/// `crates/studio-shell` is always two directories under the repo root —
/// true regardless of the shell's own current working directory, unlike
/// relying on `std::env::current_dir()`. One home; both the viewport's
/// render subprocess and the agent's MCP server config build on it.
fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/studio-shell is two directories under the repo root")
        .to_path_buf()
}

impl eframe::App for StudioShell {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.agent.drain_into(&mut self.transcript);

        let pending = self.agent.pending_permission();

        egui::Panel::right("agent_panel")
            .default_size(AGENT_PANEL_DEFAULT_WIDTH)
            .show(ui, |ui| {
                ui.heading("Agent");

                // Reserve room below the scroll area for the input row,
                // plus the permission block when one is waiting.
                let reserved = INPUT_ROW_RESERVED_HEIGHT
                    + pending.as_ref().map_or(0.0, |(_, opts)| {
                        PERMISSION_OPTION_ROW_HEIGHT * opts.len() as f32 + PERMISSION_BLOCK_PADDING
                    });
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .max_height((ui.available_height() - reserved).max(0.0))
                    .show(ui, |ui| {
                        for line in &self.transcript {
                            match line {
                                AgentLine::User(text) => {
                                    ui.strong("you");
                                    ui.label(text);
                                }
                                AgentLine::AgentText(text) => {
                                    ui.strong("claude");
                                    ui.label(text);
                                }
                                AgentLine::Status(text) => {
                                    ui.info_label(text);
                                }
                                AgentLine::Error(text) => {
                                    ui.error_label(text);
                                }
                                AgentLine::Other(text) => {
                                    ui.colored_label(egui::Color32::DARK_GRAY, text);
                                }
                            }
                        }
                    });

                if let Some((tool_title, options)) = pending {
                    ui.separator();
                    re_ui::alert::Alert::warning().show(ui, |ui| {
                        ui.vertical(|ui| {
                            ui.label(format!("Allow: {tool_title}?"));
                            ui.horizontal_wrapped(|ui| {
                                for (option_id, label) in options {
                                    if ui.button(label).clicked() {
                                        self.agent.resolve_permission(option_id);
                                    }
                                }
                            });
                        });
                    });
                }

                ui.separator();
                ui.horizontal(|ui| {
                    let response = ui
                        .add(egui::TextEdit::singleline(&mut self.draft).hint_text("Ask Claude…"));
                    let sent = (response.lost_focus()
                        && ui.input(|i| i.key_pressed(egui::Key::Enter)))
                        || ui.button("Send").clicked();
                    if sent && !self.draft.trim().is_empty() {
                        self.agent.send(std::mem::take(&mut self.draft));
                    }
                });
            });

        egui::CentralPanel::default().show(ui, |ui| {
            self.viewport.show(ui);
        });
    }
}
