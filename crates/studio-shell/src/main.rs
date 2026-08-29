//! The Studio's shell (docs/35-the-studio.md, phase 1 groundwork).
//!
//! One window, Rerun's own theme (`re_ui`), a real MuJoCo-rendered
//! viewport (via `viewport::ViewportFeed` — MuJoCo itself never runs
//! in-process, see that module) and a placeholder agent panel. The agent
//! panel stays a placeholder until agent-client-protocol is wired in.

mod viewport;

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
            Ok(Box::new(StudioShell { viewport }))
        }),
    )
}

struct StudioShell {
    viewport: ViewportFeed,
}

impl eframe::App for StudioShell {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::right("agent_panel")
            .default_size(320.0)
            .show(ui, |ui| {
                ui.heading("Agent");
                ui.label("An agent-client-protocol session lands here.");
            });

        egui::CentralPanel::default().show(ui, |ui| {
            self.viewport.show(ui);
        });
    }
}
