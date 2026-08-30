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
mod code;
mod transcript;
mod viewport;

use agent::{AgentSession, TranscriptItem};
use code::CodePanel;
use re_ui::UiExt as _;
use rerun::external::{re_crash_handler, re_grpc_server, re_log, re_memory, re_viewer};
use transcript::TranscriptView;
use viewport::ViewportFeed;

// Rerun's own allocator setup, verbatim: the accounting wrapper is what
// lets the viewer measure and prune its store; mimalloc is just faster.
#[global_allocator]
static GLOBAL: re_memory::AccountingAllocator<mimalloc::MiMalloc> =
    re_memory::AccountingAllocator::new(mimalloc::MiMalloc);

// The task whose ACCEPTED scripted expert drives the viewport for real —
// a genuine dual-arm pick-and-place cycling the protocol's own paired
// trial starts, not placeholder motion (tools/studio-render-stream.py).
const DEFAULT_TASK: &str = "kitting";

/// The robot-development pipeline as the panel's first-class entry
/// points — (chip label, subagent name in `.claude/agents/`), in
/// pipeline order. Picking one routes the prompt to that specialist via
/// Claude Code's own subagent dispatch; the specialists' definitions
/// carry the Studio streaming contract (Rerun on :9876 + the MuJoCo
/// pipeline tools), so their evidence lands in THIS window live.
const SPECIALISTS: &[(&str, &str)] = &[
    ("onboard", "robot-onboarding"),
    ("identify", "system-identification"),
    ("tasks", "task-designer"),
    ("data", "data-generator"),
    ("train", "policy-trainer"),
    ("eval", "evaluator"),
    ("deploy", "deploy-engineer"),
];

/// Agent panel layout, sized by eye against this window's default size.
const AGENT_PANEL_DEFAULT_WIDTH: f32 = 360.0;
/// A permission prompt's tool title can be a full multi-line shell
/// command; past this height it scrolls instead of pushing the layout.
const PERMISSION_TITLE_MAX_HEIGHT: f32 = 120.0;
/// Width of the macOS traffic-light cluster the brand bar must clear
/// (the window uses a fullsize content view).
#[cfg(target_os = "macos")]
const TRAFFIC_LIGHTS_INSET: f32 = 72.0;
/// The MuJoCo viewport's starting height above the Rerun viewer, and the
/// floor it can be dragged down to — a panel that can collapse to an
/// invisible sliver looks like a missing feature, not a closed panel
/// (seen live on the embed's first launch).
const VIEWPORT_DEFAULT_HEIGHT: f32 = 420.0;
const VIEWPORT_MIN_HEIGHT: f32 = 160.0;
/// Transcript prose size — re_ui's inspector-density default reads as a
/// small grey wall in a conversation (seen live).
const TRANSCRIPT_BODY_SIZE: f32 = 14.0;
/// The code editor's starting width when toggled on (file tree + a
/// readable column of code).
const CODE_PANEL_DEFAULT_WIDTH: f32 = 640.0;

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
    // Our own dock/window icon in place of the Rerun logo the viewer's
    // eframe options install. Raw RGBA committed beside a generator with
    // provenance (tools/gen-app-icon.py) — no PNG decoder in the tree.
    let icon = egui::IconData {
        rgba: include_bytes!("../assets/icon-256.rgba").to_vec(),
        width: 256,
        height: 256,
    };
    native_options.viewport = native_options
        .viewport
        .with_app_id("robotiq_studio")
        .with_icon(std::sync::Arc::new(icon));

    eframe::run_native(
        "robotiq studio",
        native_options,
        Box::new(move |cc| {
            re_viewer::customize_eframe_and_setup_renderer(cc)?;

            // The viewer's own top bar is hidden via its sanctioned
            // override: painting our wordmark OVER its logo was tried
            // and looked exactly like the patch it was (the wordmark
            // peeked out beneath the overlay). One header, ours — the
            // brand bar below carries the panel toggles the viewer's bar
            // would have offered.
            let mut startup_options = re_viewer::StartupOptions::default();
            startup_options.panel_state_overrides.top =
                Some(rerun::external::re_sdk_types::blueprint::components::PanelState::Hidden);

            let mut rerun_app = re_viewer::App::new(
                main_thread_token,
                re_viewer::build_info(),
                re_viewer::AppEnvironment::Custom("robotiq studio".to_owned()),
                startup_options,
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
                transcript_view: TranscriptView::new(),
                draft: String::new(),
                specialist: None,
                code: CodePanel::new(repo_root()),
                show_code: false,
            }))
        }),
    )?;
    Ok(())
}

struct StudioShell {
    rerun_app: re_viewer::App,
    viewport: ViewportFeed,
    agent: AgentSession,
    transcript: Vec<TranscriptItem>,
    transcript_view: TranscriptView,
    draft: String,
    /// Index into `SPECIALISTS` — which pipeline stage the next prompt
    /// is addressed to. `None` is the plain generalist session.
    specialist: Option<usize>,
    code: CodePanel,
    /// Header toggle: when on, the left side is the code editor.
    show_code: bool,
}

/// "17.1k" / "1.0M" instead of "17102" / "1000000" — the footer states a
/// magnitude, not a ledger ("1000.0k" shipped once; a unit that never
/// rolls over isn't a unit).
fn compact_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
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

        // The window's one header: wordmark left, the viewer's panel
        // toggles right (its own top bar is hidden — see the startup
        // override). Same icons, same commands as the native bar.
        egui::Panel::top("brand_bar").show(ui, |ui| {
            ui.horizontal(|ui| {
                #[cfg(target_os = "macos")]
                ui.add_space(TRAFFIC_LIGHTS_INSET);
                ui.add_space(4.0);
                let accent = ui.tokens().alert_info.icon;
                ui.label(egui::RichText::new("●").color(accent));
                ui.label(egui::RichText::new("robotiq studio").strong().size(15.0));
                ui.weak("· measure, don't guess");

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    use re_ui::{UICommand, UICommandSender as _};
                    let sender = &self.rerun_app.command_sender;
                    if ui
                        .small_icon_button(&re_ui::icons::RIGHT_PANEL_TOGGLE, "Selection panel")
                        .clicked()
                    {
                        sender.send_ui(UICommand::ToggleSelectionPanel);
                    }
                    // No ToggleTimePanel command exists — the time panel
                    // carries its own collapse control at its left edge.
                    if ui
                        .small_icon_button(&re_ui::icons::LEFT_PANEL_TOGGLE, "Blueprint panel")
                        .clicked()
                    {
                        sender.send_ui(UICommand::ToggleBlueprintPanel);
                    }
                    // The code editor toggle — the left side becomes a
                    // syntax-highlighted view of the repo (code.rs; NOT
                    // an embedded Lapce, see that module's header).
                    if ui
                        .selectable_label(self.show_code, egui::RichText::new("code").small())
                        .on_hover_text("Code editor")
                        .clicked()
                    {
                        self.show_code = !self.show_code;
                    }
                });
            });
        });

        egui::Panel::right("agent_panel")
            .default_size(AGENT_PANEL_DEFAULT_WIDTH)
            .show(ui, |ui| {
                // Compact header: who this is, and one line of session
                // state (starting…/connected/the session's own title) —
                // which used to open the transcript as debris rows.
                ui.horizontal(|ui| {
                    ui.strong("claude");
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(self.agent.status_line()).weak().small(),
                        )
                        .truncate(),
                    );
                });
                ui.separator();

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
                    // The robot-development pipeline as chips: pick a
                    // stage and the prompt goes to that specialist (the
                    // agents in .claude/agents/, dispatched by Claude
                    // Code itself). This panel is an agentic workbench
                    // for building robots, not raw chat access.
                    ui.horizontal_wrapped(|ui| {
                        for (index, (label, agent_name)) in SPECIALISTS.iter().enumerate() {
                            let selected = self.specialist == Some(index);
                            let response = ui
                                .selectable_label(selected, *label)
                                .on_hover_text(*agent_name);
                            if response.clicked() {
                                // Click again to drop back to generalist.
                                self.specialist = if selected { None } else { Some(index) };
                            }
                        }
                    });

                    // Enter sends, Shift+Enter breaks the line — consumed
                    // BEFORE the editor sees the key, or the editor would
                    // insert the newline first (the standard egui order
                    // for stealing a key from a focused TextEdit).
                    let draft_id = egui::Id::new("agent_draft");
                    let enter_sends = ui.memory(|m| m.has_focus(draft_id))
                        && ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Enter));
                    let hint = match self.specialist {
                        Some(index) => format!("Ask the {} agent…", SPECIALISTS[index].1),
                        None => "Ask Claude…  (Enter sends, Shift+Enter for a new line)".to_owned(),
                    };
                    egui::Frame::group(ui.style())
                        .fill(ui.visuals().extreme_bg_color)
                        .show(ui, |ui| {
                            ui.add(
                                egui::TextEdit::multiline(&mut self.draft)
                                    .id(draft_id)
                                    // The composer's outer group frame is
                                    // the chrome; the editor itself draws
                                    // none (an empty Frame — 0.36's way to
                                    // say "no frame").
                                    .frame(egui::Frame::new())
                                    .desired_rows(2)
                                    .desired_width(f32::INFINITY)
                                    .hint_text(hint),
                            );
                            // Bottom row of the composer, actions on the
                            // right where every chat app puts them; the
                            // context-window usage sits quietly on the
                            // left (Zed keeps it near the editor too —
                            // the stream of per-chunk token counts was
                            // pure noise as transcript rows).
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    let turn_active = self.agent.turn_active();
                                    // Mid-turn a prompt QUEUES behind the
                                    // running one (it shows in the
                                    // transcript immediately) — the label
                                    // says so instead of pretending to
                                    // interrupt.
                                    let send_label = if turn_active { "Queue" } else { "Send" };
                                    if (ui.button(send_label).clicked() || enter_sends)
                                        && !self.draft.trim().is_empty()
                                    {
                                        let text = std::mem::take(&mut self.draft);
                                        let specialist =
                                            self.specialist.map(|index| SPECIALISTS[index].1);
                                        self.agent.send(text, specialist);
                                    }
                                    if turn_active {
                                        if ui.button("Stop").clicked() {
                                            self.agent.cancel();
                                        }
                                        ui.add(egui::Spinner::new().size(12.0));
                                    }
                                    ui.with_layout(
                                        egui::Layout::left_to_right(egui::Align::Center),
                                        |ui| {
                                            if let Some((used, size)) = self.agent.usage() {
                                                ui.weak(
                                                    egui::RichText::new(format!(
                                                        "{} / {} tokens",
                                                        compact_count(used),
                                                        compact_count(size)
                                                    ))
                                                    .small(),
                                                );
                                            }
                                        },
                                    );
                                },
                            );
                        });
                });

                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        // Conversation type: re_ui's default body text is
                        // sized for dense inspector panels; a transcript
                        // is read as prose and needs the step up (the
                        // small grey wall was the first live complaint).
                        if let Some(body) =
                            ui.style_mut().text_styles.get_mut(&egui::TextStyle::Body)
                        {
                            body.size = TRANSCRIPT_BODY_SIZE;
                        }
                        ui.spacing_mut().item_spacing.y = 6.0;
                        // Breathing room at the panel edges.
                        egui::Frame::new()
                            .inner_margin(egui::Margin::symmetric(8, 4))
                            .show(ui, |ui| {
                                for (index, item) in self.transcript.iter().enumerate() {
                                    self.transcript_view.show(ui, index, item);
                                }
                            });
                    });
            });

        // The code editor claims the left side when toggled on — laid
        // out before the viewport so it runs full height and the
        // sim/viewer split shares what remains.
        if self.show_code {
            egui::Panel::left("code_panel")
                .resizable(true)
                .default_size(CODE_PANEL_DEFAULT_WIDTH)
                .show(ui, |ui| self.code.show(ui));
        }

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
