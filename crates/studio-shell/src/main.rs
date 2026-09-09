//! The Studio's shell (the-studio doc, phase 1).
//!
//! One window with a shell (`shell.rs`: top bar, a rail in the field's
//! vocabulary — Assets, Data, Training, Evaluation, Deployment — and a
//! page router) over three real things: the project's pages (`pages.rs`,
//! read from the index and job table the Python side writes —
//! `model.rs`), a MuJoCo-rendered viewport (via
//! `viewport::ViewportFeed` — MuJoCo never runs in-process, see that
//! module) and — filling the rest — **the actual Rerun viewer,
//! embedded**, not an imitation of it. No chat panel (docs/64,
//! 2026-09-02) and no code panel (2026-09-09), deliberately: the agent
//! lives in the developer's own tool and drives this window through the
//! MCP surface; code stays on the developer's laptop or GitHub; the
//! Studio is the window where the magic shows, not another place to
//! talk or type. The embed
//! follows Rerun's own `extend_viewer_ui` example (Apache-2.0, 0.36.3)
//! line for line where it matters: a gRPC server on the standard :9876
//! feeds it, so every tool this repo already has that speaks the Rerun
//! SDK (`train-watch --follow`, `rig-rerun`, `replay-errand`, a
//! five-line script) streams straight into this window. A bespoke
//! egui_plot data panel lived here for one evening; the operator's
//! verdict — "use exactly what Rerun does, don't reinvent the wheel" —
//! replaced it with the wheel.

mod control;
mod detail;
mod model;
mod pages;
mod shell;
mod viewport;
mod widgets;

use model::Model;
use pages::Section;
use re_ui::UiExt as _;
use rerun::external::{re_crash_handler, re_grpc_server, re_log, re_memory, re_viewer};
use shell::Shell;
use viewport::ViewportFeed;

// Rerun's own allocator setup, verbatim: the accounting wrapper is what
// lets the viewer measure and prune its store; mimalloc is just faster.
#[global_allocator]
static GLOBAL: re_memory::AccountingAllocator<mimalloc::MiMalloc> =
    re_memory::AccountingAllocator::new(mimalloc::MiMalloc);

// The task whose ACCEPTED scripted expert drives the viewport for real —
// a genuine dual-arm pick-and-place cycling the protocol's own paired
// trial starts, not placeholder motion (tools/studio-render-stream.py).

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

    // On Linux, re_ui's chrome probe defaults to client-drawn
    // decorations (always, when WAYLAND_DISPLAY is unset) — but the
    // client here is US, and our header is a brand bar, not a drag
    // region: the window came up borderless and could neither move nor
    // resize (seen live on WSLg, 2026-08-31; the only grabbable thing
    // near the top was the viewport panel's resize handle). Ask for the
    // native frame instead; WSLg draws a movable, resizable one.
    #[cfg(target_os = "linux")]
    {
        native_options.viewport = native_options
            .viewport
            .with_decorations(true)
            .with_title_shown(true)
            .with_titlebar_shown(true)
            .with_transparent(false);
    }

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
            // Rerun's welcome screen (its marketing splash, example
            // recordings, Hub promotion) is not this app's front page.
            // The project home (project.rs) is the empty state instead;
            // the viewer's panes appear the moment a recording exists.
            startup_options.hide_welcome_screen = true;

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

            // The builder's `.with_decorations(true)` above is not enough:
            // the embedded viewer re-asserts its OWN chrome preference every
            // frame (`sync_native_window_decorations`, driven by this
            // AppOptions flag, whose Linux default is client-drawn) and
            // strips the frame right back off. Tell the app itself to want
            // native decorations.
            #[cfg(target_os = "linux")]
            {
                rerun_app.app_options_mut().custom_window_decorations = false;
            }

            let viewport = ViewportFeed::idle();
            let shell = Shell::new(Model::open(&repo_root()), repo_root());
            let control = control::Control::new(shell.model.project_root.clone());
            Ok(Box::new(StudioShell {
                rerun_app,
                viewport,
                shell,
                control,
                before: (Section::Overview, None),
                seen_recording: false,
            }))
        }),
    )?;
    Ok(())
}

struct StudioShell {
    rerun_app: re_viewer::App,
    viewport: ViewportFeed,
    shell: Shell,
    /// The agent's control surface: commands in, state and events out
    /// (control.rs), all files under the project's `.index/`.
    control: control::Control,
    /// The page and selection before this frame's page ran, so a change
    /// the human made (not a command) becomes an event.
    before: (Section, Option<String>),
    /// Whether a recording was loaded last frame — a fresh arrival
    /// switches the page to Live once, without trapping the user there.
    seen_recording: bool,
}

/// `crates/studio-shell` is always two directories under the repo root —
/// true regardless of the shell's own current working directory, unlike
/// relying on `std::env::current_dir()`. One home; the viewport's render
/// subprocess and the default project both build on it.
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
        self.apply_commands(ui);
        self.before = (self.shell.section, self.shell.selected.clone());
        // One header, ours: the viewer's own top bar is hidden (startup
        // override above); its panel toggles ride on ours, and matter on
        // the Live view.
        let sender = self.rerun_app.command_sender.clone();
        self.shell.top_bar(ui, |ui| {
            use re_ui::{UICommand, UICommandSender as _};
            if ui
                .small_icon_button(&re_ui::icons::RIGHT_PANEL_TOGGLE, "Selection panel")
                .clicked()
            {
                sender.send_ui(UICommand::ToggleSelectionPanel);
            }
            if ui
                .small_icon_button(&re_ui::icons::LEFT_PANEL_TOGGLE, "Blueprint panel")
                .clicked()
            {
                sender.send_ui(UICommand::ToggleBlueprintPanel);
            }
        });
        self.shell.rail(ui);

        // A recording arriving while another page is up switches to Live:
        // a run streaming in is the thing to look at.
        let has_recording = self.rerun_app.recording_db().is_some();
        if has_recording && (!self.seen_recording || self.shell.show_requested) {
            if self.shell.section != Section::Live {
                self.control.event(
                    "open",
                    serde_json::json!({"section": "live", "by": "studio"}),
                );
                self.before.0 = Section::Live; // not the human's doing
            }
            self.shell.section = Section::Live;
            self.shell.show_requested = false;
        }
        self.seen_recording = has_recording;

        if self.shell.section == Section::Live {
            // The MuJoCo sim viewport on top; the rest IS the Rerun
            // viewer — blueprint panel, timeline, views, exactly as the
            // standalone app renders them. Two panel ids on purpose:
            // egui remembers a panel's size by id, and the idle strip
            // must not inherit a 420 px preview height.
            let active = self.viewport.is_active();
            egui::Panel::top(if active {
                "sim_viewport"
            } else {
                "sim_viewport_idle"
            })
            .resizable(active)
            .default_size(if active {
                VIEWPORT_DEFAULT_HEIGHT
            } else {
                36.0
            })
            .min_size(if active { VIEWPORT_MIN_HEIGHT } else { 36.0 })
            .show(ui, |ui| {
                self.viewport.show(ui);
            });
            if has_recording {
                self.rerun_app.ui(ui, frame);
            } else {
                egui::CentralPanel::default().show(ui, |ui| {
                    pages::page(ui, |ui| {
                        pages::heading(ui, "Live view");
                        ui.add_space(8.0);
                        ui.label(
                            egui::RichText::new(
                                "Nothing is streaming. Anything speaking the Rerun SDK lands \
                                 here on :9876 — a press, a training run, a certificate's \
                                 trials — and the viewer's panes appear the moment it does. \
                                 Preview a scene from the strip above meanwhile.",
                            )
                            .text_style(re_ui::DesignTokens::welcome_screen_body())
                            .color(ui.visuals().weak_text_color()),
                        );
                    });
                });
            }
        } else {
            egui::CentralPanel::default().show(ui, |ui| self.shell.page(ui));
        }
        self.report(ui);
    }
}

// -- the control surface: commands applied, state and events written --------

impl StudioShell {
    /// Apply every command the agent wrote since the last poll (20 Hz),
    /// answering each on disk. A command that cannot be applied is
    /// refused with the reason, never dropped.
    fn apply_commands(&mut self, ui: &mut egui::Ui) {
        // The next frame must come soon even when the human is idle: a
        // command waits at most one poll interval.
        ui.ctx().request_repaint_after(control::POLL_EVERY);
        self.control.set_root(self.shell.model.project_root.clone());
        for pending in self.control.poll() {
            let outcome = match pending.command {
                Ok(command) => self.apply(ui, command),
                Err(why) => Err(why),
            };
            match outcome {
                Ok(()) => self.control.ack(&pending.id, "done", None),
                Err(why) => self.control.ack(&pending.id, "refused", Some(&why)),
            }
        }
    }

    fn apply(&mut self, ui: &mut egui::Ui, command: control::Command) -> Result<(), String> {
        use control::Command;
        match command {
            Command::Open {
                project,
                section,
                artifact,
            } => {
                if let Some(root) = project {
                    let root = std::path::PathBuf::from(root);
                    if !root.join("project.json").is_file() {
                        return Err(format!(
                            "{} is not a project (no project.json)",
                            root.display()
                        ));
                    }
                    self.shell.switch_project(root.clone());
                    self.control
                        .event("open", serde_json::json!({"project": root, "by": "agent"}));
                }
                if let Some(name) = section {
                    let page =
                        Section::parse(&name).ok_or_else(|| format!("no page named {name:?}"))?;
                    self.shell.section = page;
                    self.shell.selected = None;
                    self.control.event(
                        "open",
                        serde_json::json!({"section": page.slug(), "by": "agent"}),
                    );
                }
                if let Some(stamp) = artifact {
                    let kind = self
                        .shell
                        .model
                        .artifact(&stamp)
                        .map(|a| a.kind.clone())
                        .ok_or_else(|| format!("no artifact {stamp:?} in this project"))?;
                    let page = Section::for_kind(&kind)
                        .ok_or_else(|| format!("no page lists a {kind}"))?;
                    self.shell.section = page;
                    self.shell.selected = Some(stamp.clone());
                    self.control.event(
                        "select",
                        serde_json::json!({"artifact": stamp, "by": "agent"}),
                    );
                }
                Ok(())
            }
            Command::Show { artifact } => {
                if self.shell.model.artifact(&artifact).is_none() {
                    return Err(format!("no artifact {artifact:?} in this project"));
                }
                self.shell.show(&artifact);
                Ok(())
            }
            Command::Compare { a, b } => {
                for stamp in [&a, &b] {
                    if self.shell.model.artifact(stamp).is_none() {
                        return Err(format!("no artifact {stamp:?} in this project"));
                    }
                }
                self.shell.compare(&a, &b);
                Ok(())
            }
            Command::Time {
                timeline,
                seconds,
                sequence,
                play,
                speed,
                start,
                end,
                follow,
                step,
            } => {
                use re_viewer::external::re_log_types::{
                    AbsoluteTimeRange, TimeInt, TimeReal, TimeType, TimelineName,
                };
                use re_viewer::external::re_sdk_types::blueprint::components::PlayState;
                use re_viewer::external::re_viewer_context::{
                    SystemCommand, SystemCommandSender as _, TimeControlCommand,
                };
                let store_id = self
                    .rerun_app
                    .active_recording_id()
                    .cloned()
                    .ok_or_else(|| "nothing is streaming in the viewer".to_owned())?;
                let mut time_commands = Vec::new();
                let name = match timeline {
                    Some(n) => {
                        let name = TimelineName::try_new(&n)
                            .map_err(|e| format!("timeline {n:?}: {e}"))?;
                        time_commands.push(TimeControlCommand::SetActiveTimeline(name));
                        Some(name)
                    }
                    None => self.rerun_app.current_query().and_then(|q| q.timeline()),
                };
                let typ = name.and_then(|n| {
                    self.rerun_app
                        .recording_db()
                        .and_then(|db| db.timelines().get(&n).map(|t| t.typ()))
                });
                let temporal = !matches!(typ, Some(TimeType::Sequence));
                if let Some(secs) = seconds {
                    if !temporal {
                        return Err(
                            "this timeline counts steps: pass `sequence`, not `seconds`".into()
                        );
                    }
                    time_commands.push(TimeControlCommand::SetTime(TimeReal::from_secs(secs)));
                }
                if let Some(seq) = sequence {
                    if temporal && typ.is_some() {
                        return Err(
                            "this timeline is in seconds: pass `seconds`, not `sequence`".into(),
                        );
                    }
                    time_commands.push(TimeControlCommand::SetTime(TimeReal::from(seq)));
                }
                if let (Some(lo), Some(hi)) = (start, end) {
                    let to_int = |v: f64| {
                        if temporal {
                            TimeInt::from_secs(v)
                        } else {
                            TimeInt::new_temporal(v as i64)
                        }
                    };
                    time_commands.push(TimeControlCommand::SetTimeSelection(
                        AbsoluteTimeRange::new(to_int(lo), to_int(hi)),
                    ));
                } else if start.is_some() || end.is_some() {
                    return Err("a time selection needs both `start` and `end`".into());
                }
                if let Some(s) = speed {
                    time_commands.push(TimeControlCommand::SetSpeed(s));
                }
                if let Some(n) = step {
                    let cmd = if n < 0 {
                        TimeControlCommand::StepTimeBack
                    } else {
                        TimeControlCommand::StepTimeForward
                    };
                    for _ in 0..n.unsigned_abs().min(1000) {
                        time_commands.push(cmd_clone(&cmd));
                    }
                }
                if follow == Some(true) {
                    time_commands.push(TimeControlCommand::MoveEndAndFollow);
                }
                match play {
                    Some(true) => {
                        time_commands.push(TimeControlCommand::SetPlayState(PlayState::Playing))
                    }
                    Some(false) => time_commands.push(TimeControlCommand::Pause),
                    None => {}
                }
                if time_commands.is_empty() {
                    return Err("nothing to do: pass a timeline, a cursor, play, speed, a selection, follow or step".into());
                }
                self.rerun_app
                    .command_sender
                    .send_system(SystemCommand::TimeControlCommands {
                        store_id,
                        time_commands,
                    });
                self.control.note_commanded_time();
                Ok(())
            }
            Command::Panels {
                blueprint,
                selection,
                time,
            } => {
                use re_ui::{UICommand, UICommandSender as _};
                if time.is_some() {
                    return Err("the time panel has no command in this viewer build".into());
                }
                let sender = &self.rerun_app.command_sender;
                for (panel, action) in [("blueprint", blueprint), ("selection", selection)] {
                    let Some(action) = action else { continue };
                    let cmd = match (panel, action.as_str()) {
                        ("blueprint", "expand") => UICommand::ExpandBlueprintPanel,
                        ("blueprint", "toggle") => UICommand::ToggleBlueprintPanel,
                        ("selection", "expand") => UICommand::ExpandSelectionPanel,
                        ("selection", "toggle") => UICommand::ToggleSelectionPanel,
                        _ => {
                            return Err(format!("{panel}: {action:?} is not `expand` or `toggle`"))
                        }
                    };
                    sender.send_ui(cmd);
                }
                Ok(())
            }
            Command::Quit => {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                Ok(())
            }
        }
    }

    /// After the page ran: what the human changed becomes an event, and
    /// the state file says what the window shows.
    fn report(&mut self, _ui: &mut egui::Ui) {
        let (section_before, selected_before) = self.before.clone();
        if self.shell.section != section_before {
            self.control.event(
                "open",
                serde_json::json!({"section": self.shell.section.slug(), "by": "user"}),
            );
        }
        if self.shell.selected != selected_before {
            match &self.shell.selected {
                Some(stamp) => self.control.event(
                    "select",
                    serde_json::json!({"artifact": stamp, "by": "user"}),
                ),
                None => self
                    .control
                    .event("deselect", serde_json::json!({"by": "user"})),
            }
        }
        if let Some(shown) = self.shell.shown.take() {
            self.control
                .event("show", serde_json::json!({"artifact": shown}));
        }
        let live = self.live();
        if let Some(rested) = self.control.watch_live(&live) {
            self.control.event(
                "time",
                serde_json::json!({
                    "timeline": rested.timeline, "seconds": rested.seconds,
                    "sequence": rested.sequence, "by": "user"
                }),
            );
        }
        let state = control::StudioState {
            schema: control::STATE_SCHEMA,
            pid: std::process::id(),
            heartbeat: 0.0,
            project: self.shell.model.project_root.display().to_string(),
            project_name: self.shell.model.name(),
            section: self.shell.section.slug(),
            selected: self.shell.selected.clone(),
            live,
            presenter_running: self.shell.presenter_running(),
            jobs_running: self.shell.model.running_jobs(),
        };
        self.control.record_state(state);
    }

    /// The viewer's recording and cursor, as the state file reports them.
    fn live(&self) -> control::Live {
        use re_viewer::external::re_log_types::TimeType;
        let Some(db) = self.rerun_app.recording_db() else {
            return control::Live::default();
        };
        let query = self.rerun_app.current_query();
        let name = query.as_ref().and_then(|q| q.timeline());
        let typ = name.and_then(|n| db.timelines().get(&n).map(|t| t.typ()));
        let at = query.as_ref().map(|q| q.at().as_i64());
        control::Live {
            recording: Some(db.application_id().to_string()),
            timeline: name.map(|n| n.to_string()),
            seconds: match (typ, at) {
                (Some(TimeType::DurationNs | TimeType::TimestampNs), Some(v)) => {
                    Some(v as f64 / 1e9)
                }
                _ => None,
            },
            sequence: match (typ, at) {
                (Some(TimeType::Sequence), Some(v)) => Some(v),
                _ => None,
            },
        }
    }
}

/// `TimeControlCommand` is not `Clone`; the two step variants are unit-like.
fn cmd_clone(
    cmd: &re_viewer::external::re_viewer_context::TimeControlCommand,
) -> re_viewer::external::re_viewer_context::TimeControlCommand {
    use re_viewer::external::re_viewer_context::TimeControlCommand;
    match cmd {
        TimeControlCommand::StepTimeBack => TimeControlCommand::StepTimeBack,
        _ => TimeControlCommand::StepTimeForward,
    }
}
