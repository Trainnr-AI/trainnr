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
use viewport::ViewportFeed;

const DEFAULT_TASK: &str = "block_stack";

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

impl eframe::App for StudioShell {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.agent.drain_into(&mut self.transcript);

        egui::Panel::right("agent_panel")
            .default_size(360.0)
            .show(ui, |ui| {
                ui.heading("Agent");

                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .max_height(ui.available_height() - 40.0)
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
                                    ui.colored_label(egui::Color32::GRAY, text);
                                }
                                AgentLine::Other(text) => {
                                    ui.colored_label(egui::Color32::DARK_GRAY, text);
                                }
                            }
                        }
                    });

                ui.separator();
                ui.horizontal(|ui| {
                    let response =
                        ui.add(egui::TextEdit::singleline(&mut self.draft).hint_text("Ask Claude…"));
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
