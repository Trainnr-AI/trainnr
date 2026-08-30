//! The Studio's shell (the-studio doc, phase 1).
//!
//! One window, three real things: a MuJoCo-rendered viewport (via
//! `viewport::ViewportFeed` — MuJoCo never runs in-process, see that
//! module), a live agent-client-protocol session against Claude Code's
//! own adapter (`agent::AgentSession`), and — filling the rest — **the
//! actual Rerun viewer, embedded**, not an imitation of it. The embed
//! follows Rerun's own `extend_viewer_ui` example (Apache-2.0, 0.36.3)
//! line for line where it matters: a gRPC server on the standard :9876
//! feeds it, so every tool this repo already has that speaks the Rerun
//! SDK (`train-watch --follow`, `rig-rerun`, `replay-errand`, a
//! five-line script) streams straight into this window. A bespoke
//! egui_plot data panel lived here for one evening; the operator's
//! verdict — "use exactly what Rerun does, don't reinvent the wheel" —
//! replaced it with the wheel.

mod agent;
mod viewport;

use agent::{AgentLine, AgentSession};
use re_ui::UiExt as _;
use rerun::external::{re_crash_handler, re_grpc_server, re_log, re_memory, re_viewer};
use viewport::ViewportFeed;

// Rerun's own allocator setup, verbatim: the accounting wrapper is what
// lets the viewer measure and prune its store; mimalloc is just faster.
#[global_allocator]
static GLOBAL: re_memory::AccountingAllocator<mimalloc::MiMalloc> =
    re_memory::AccountingAllocator::new(mimalloc::MiMalloc);

const DEFAULT_TASK: &str = "block_stack";

/// Agent panel layout, sized by eye against this window's default size.
const AGENT_PANEL_DEFAULT_WIDTH: f32 = 360.0;
/// A permission prompt's tool title can be a full multi-line shell
/// command; past this height it scrolls instead of pushing the layout.
const PERMISSION_TITLE_MAX_HEIGHT: f32 = 120.0;
/// The MuJoCo viewport's starting height above the Rerun viewer, and the
/// floor it can be dragged down to — a panel that can collapse to an
/// invisible sliver looks like a missing feature, not a closed panel
/// (seen live on the embed's first launch).
const VIEWPORT_DEFAULT_HEIGHT: f32 = 420.0;
const VIEWPORT_MIN_HEIGHT: f32 = 160.0;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let main_thread_token = re_viewer::MainThreadToken::i_promise_i_am_on_the_main_thread();
    re_log::setup_logging();
    re_crash_handler::install_crash_handlers(re_viewer::build_info());

    // The standard SDK port: anything calling `rr.connect_grpc()` lands
    // in this window. Fails loudly if another viewer already holds it —
    // close the standalone viewer rather than silently split streams.
    let (rx, _grpc_server_handle) = re_grpc_server::spawn_with_recv(
        "0.0.0.0:9876".parse()?,
        Default::default(),
        re_grpc_server::shutdown::never(),
    );

    let mut native_options = re_viewer::native::eframe_options(None);
    native_options.viewport = native_options.viewport.with_app_id("robotiq_studio");

    eframe::run_native(
        "robotiq studio",
        native_options,
        Box::new(move |cc| {
            re_viewer::customize_eframe_and_setup_renderer(cc)?;

            let mut rerun_app = re_viewer::App::new(
                main_thread_token,
                re_viewer::build_info(),
                re_viewer::AppEnvironment::Custom("robotiq studio".to_owned()),
                re_viewer::StartupOptions::default(),
                cc,
                None,
                re_viewer::AsyncRuntimeHandle::from_current_tokio_runtime_or_wasmbindgen()?,
            );
            rerun_app.add_log_receiver(rx);

            let viewport = ViewportFeed::spawn(&cc.egui_ctx, DEFAULT_TASK);
            let agent = AgentSession::spawn(&cc.egui_ctx);
            Ok(Box::new(StudioShell {
                rerun_app,
                viewport,
                agent,
                transcript: Vec::new(),
                draft: String::new(),
            }))
        }),
    )?;
    Ok(())
}

struct StudioShell {
    rerun_app: re_viewer::App,
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
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        // The viewer persists its blueprint/state exactly as standalone
        // Rerun would.
        self.rerun_app.save(storage);
    }

    fn logic(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.rerun_app.logic(ctx, frame);
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        self.agent.drain_into(&mut self.transcript);
        let pending = self.agent.pending_permission();

        egui::Panel::right("agent_panel")
            .default_size(AGENT_PANEL_DEFAULT_WIDTH)
            .show(ui, |ui| {
                ui.heading("Agent");

                egui::Panel::bottom("agent_panel_input").show(ui, |ui| {
                    if let Some((tool_title, options)) = pending {
                        re_ui::alert::Alert::warning().show(ui, |ui| {
                            ui.vertical(|ui| {
                                ui.label("Allow this tool call?");
                                // Buttons BEFORE the command text: a tool
                                // title can be arbitrarily long, and the
                                // answer must never scroll out of reach.
                                ui.horizontal_wrapped(|ui| {
                                    for (option_id, label) in options {
                                        if ui.button(label).clicked() {
                                            self.agent.resolve_permission(option_id);
                                        }
                                    }
                                });
                                egui::ScrollArea::vertical()
                                    .id_salt("permission_title")
                                    .max_height(PERMISSION_TITLE_MAX_HEIGHT)
                                    .show(ui, |ui| {
                                        ui.label(tool_title);
                                    });
                            });
                        });
                    }
                    ui.horizontal(|ui| {
                        let response = ui.add(
                            egui::TextEdit::singleline(&mut self.draft).hint_text("Ask Claude…"),
                        );
                        let sent = (response.lost_focus()
                            && ui.input(|i| i.key_pressed(egui::Key::Enter)))
                            || ui.button("Send").clicked();
                        if sent && !self.draft.trim().is_empty() {
                            self.agent.send(std::mem::take(&mut self.draft));
                        }
                    });
                });

                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
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
            });

        // The MuJoCo sim viewport on top; the whole rest of the window IS
        // the Rerun viewer — blueprint panel, timeline, views, exactly as
        // the standalone app renders them.
        egui::Panel::top("sim_viewport")
            .resizable(true)
            .default_size(VIEWPORT_DEFAULT_HEIGHT)
            .min_size(VIEWPORT_MIN_HEIGHT)
            .show(ui, |ui| {
                self.viewport.show(ui);
            });
        self.rerun_app.ui(ui, frame);
    }
}
