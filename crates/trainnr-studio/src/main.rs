//! The Studio's shell (the-studio doc, phase 1).
//!
//! One window with a shell (`shell.rs`: top bar, a rail in the field's
//! vocabulary — Assets, Data, Training, Evaluation, Deployment — and a
//! page router) over three real things: the project's pages (`pages.rs`,
//! read from the index and job table the Python side writes —
//! `model.rs`), a MuJoCo-rendered viewport (via
//! `viewport::ViewportFeed` — MuJoCo never runs in-process, see that
//! module) and — filling the rest — **the actual Rerun viewer,
//! embedded**, not an imitation of it. No chat panel (docs/80,
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

mod chrome;
mod control;
mod detail;
mod keys;
mod listing;
mod model;
mod pages;
mod palette;
mod pictures;
mod running;
mod shell;
mod simulator;
mod spawn;
mod theme;
mod viewport;
mod widgets;

use control::{Event, BY_AGENT, BY_STUDIO, BY_USER};
use model::Model;
use pages::Section;
use re_ui::UiExt as _;
use rerun::external::{re_crash_handler, re_grpc_server, re_log, re_memory, re_viewer};
use shell::Shell;
use spawn::{on_wsl, repo_root};
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
const VIEWPORT_DEFAULT_HEIGHT: f32 = 520.0;
const VIEWPORT_MIN_HEIGHT: f32 = 240.0;
/// A frame slower than this halves the simulation's picture; faster than
/// `FAST_FRAME_MS` restores it (viewport.rs::set_render_scale).
const SLOW_FRAME_MS: f32 = 150.0;
const FAST_FRAME_MS: f32 = 60.0;
/// The picker row alone, when no scene runs above a streaming viewer.
const VIEWPORT_PICKER_HEIGHT: f32 = 34.0;
/// The standard Rerun SDK port, on this machine only: anything here
/// calling `rr.connect_grpc()` lands in this window, and nothing on the
/// network can stream into it (it listened on every interface until
/// 2026-10-03; every client, the cloud feed included, connects from this
/// machine). `$TRAINNR_VIEWER_BIND` opens it deliberately, e.g.
/// `0.0.0.0:9876` to stream from another machine on a trusted network.
const GRPC_BIND: &str = "127.0.0.1:9876";
const GRPC_BIND_ENV: &str = "TRAINNR_VIEWER_BIND";
/// Bounds on what one agent command may ask of the viewer and the
/// simulator: time steps in the viewer, physics steps in the stream.
const MAX_TIME_STEPS: u64 = 1000;
const MAX_SIM_STEPS: u32 = 100_000;
/// How long a capture waits for the window to produce a frame.
const CAPTURE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

/// Under WSLg a native Wayland window did not show on Windows on
/// 2026-09-11, so the app took the X11 path; on 2026-10-02 the Wayland
/// window shows, and the X11 path presents a frame in 385 to 570 ms
/// against 50 to 64 ms on Wayland (the heartbeat's frame_ms). So winit's
/// own choice (Wayland when `WAYLAND_DISPLAY` is set) stands by default;
/// `TRAINNR_X11=1` takes the X11 path again, with its window fixes.
/// Comfortably inside a 1080-line monitor with the manager's frame.
const WSL_WINDOW_SIZE: [f32; 2] = [1600.0, 900.0];

/// Whether the X11 path was asked for under WSLg.
fn x11_under_wslg() -> bool {
    on_wsl() && std::env::var_os("TRAINNR_X11").is_some_and(|v| v == "1")
}

fn prefer_x11_under_wslg(options: &mut eframe::NativeOptions) {
    if !x11_under_wslg() {
        return;
    }
    re_log::info!("WSLg: taking the X11 path so the window shows");
    // A remembered size that nearly fills the monitor comes back
    // MAXIMIZED under WSLg's window manager, with the frame 32 px above
    // the screen: the window cannot be dragged and every pointer hit
    // lands 32 px low (2026-09-12, twice; the X11 state read it). So on
    // WSL the window opens at a fixed size, never maximized, and its
    // size is not remembered between runs.
    options.persist_window = false;
    options.viewport = options
        .viewport
        .clone()
        .with_inner_size(WSL_WINDOW_SIZE)
        .with_maximized(false);
    options.event_loop_builder = Some(Box::new(|builder| {
        #[cfg(all(target_os = "linux", not(target_arch = "wasm32")))]
        {
            use winit::platform::x11::EventLoopBuilderExtX11 as _;
            builder.with_x11();
        }
        #[cfg(not(all(target_os = "linux", not(target_arch = "wasm32"))))]
        {
            let _ = builder; // WSL is Linux; this arm never runs a window
        }
    }));
}

/// A simulator command that needs the model before the stream described it.
const NOT_DESCRIBED: &str = "the model is not described yet";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let main_thread_token = re_viewer::MainThreadToken::i_promise_i_am_on_the_main_thread();
    re_log::setup_logging();
    re_crash_handler::install_crash_handlers(re_viewer::build_info());

    // Fails loudly if another viewer already holds the port — close the
    // standalone viewer rather than silently split streams.
    let bind = std::env::var(GRPC_BIND_ENV).unwrap_or_else(|_| GRPC_BIND.to_owned());
    let (rx, _grpc_server_handle) = re_grpc_server::spawn_with_recv(
        bind.parse()?,
        Default::default(),
        re_grpc_server::shutdown::never(),
    );

    let mut native_options = re_viewer::native::eframe_options(None);
    prefer_x11_under_wslg(&mut native_options);
    no_vsync_under_wslg(&mut native_options);
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
        .with_app_id("ai.trainnr.studio")
        .with_icon(std::sync::Arc::new(icon));

    // The chrome is ours where a client may draw it (chrome.rs): our top
    // bar is the title bar, with drag, double-click and the caption
    // buttons; the Mac keeps its traffic lights over a full-size content
    // view. Rerun's helper sets the per-platform flags.
    native_options.viewport =
        re_ui::viewport_with_window_chrome(native_options.viewport, chrome::custom_chrome());

    eframe::run_native(
        "trainnr Studio",
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
                re_viewer::AppEnvironment::Custom("trainnr Studio".to_owned()),
                startup_options,
                cc,
                None,
                re_viewer::AsyncRuntimeHandle::from_current_tokio_runtime_or_wasmbindgen()?,
            );
            rerun_app.add_log_receiver(rx);

            // The embedded viewer re-asserts its own chrome preference every
            // frame (`sync_native_window_decorations`); it must agree with
            // the window we built.
            rerun_app.app_options_mut().custom_window_decorations = chrome::custom_chrome();

            // The theme the person chose last time, before the first frame;
            // light when nothing was chosen (the default since 2026-10-03:
            // the designed light palette is the product's face, theme.rs).
            let chosen = cc
                .storage
                .and_then(|s| s.get_string(THEME_KEY))
                .map(|pref| theme_preference(&pref))
                .unwrap_or(DEFAULT_THEME);
            cc.egui_ctx.set_theme(chosen);

            let viewport = ViewportFeed::idle();
            let mut shell = Shell::new(Model::open(repo_root()));
            shell.open_project();
            let mut control = control::Control::new(shell.model.project_root.clone());
            control.watch(cc.egui_ctx.clone());
            Ok(Box::new(StudioShell {
                rerun_app,
                viewport,
                shell,
                control,
                before: (Section::Overview, None, None),
                shot: None,
                last_navigation: None,
                seen_recording: false,
                activated_shown: None,
                viewport_full: false,
                closing: false,
                theme_preference: chosen,
                frames: FrameClock::default(),
            }))
        }),
    )?;
    Ok(())
}

struct StudioShell {
    /// The theme preference as of this frame (system, dark or light),
    /// persisted by `save`.
    theme_preference: egui::ThemePreference,
    /// Frame times, for the heartbeat.
    frames: FrameClock,
    rerun_app: re_viewer::App,
    viewport: ViewportFeed,
    shell: Shell,
    /// The agent's control surface: commands in, state and events out
    /// (control.rs), all files under the project's `.index/`.
    control: control::Control,
    /// The page and selection before this frame's page ran, so a change
    /// the human made (not a command) becomes an event.
    before: (Section, Option<String>, Option<String>),
    /// A screenshot command waiting for its frame: the command id, the
    /// width asked, and when it was asked (a capture that never arrives
    /// is answered `failed`, not left hanging).
    shot: Option<Shot>,
    /// When the agent last moved the window (a page, an artifact, a
    /// table, a time): a capture waits for the frame to settle after it.
    last_navigation: Option<std::time::Instant>,
    /// Whether a recording was loaded last frame — a fresh arrival
    /// switches the page to Live once, without trapping the user there.
    seen_recording: bool,
    /// The presenter's last `shown` we brought to the front.
    activated_shown: Option<String>,
    /// The viewport alone on the Live page: no rail, no viewer panels,
    /// the picture and its transport bar (the `f` key; Escape leaves).
    viewport_full: bool,
    /// A close that waits one frame for full screen to end.
    closing: bool,
}

impl eframe::App for StudioShell {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        storage.set_string(THEME_KEY, preference_name(self.theme_preference).to_owned());
        // The viewer persists its blueprint/state exactly as standalone
        // Rerun would.
        self.rerun_app.save(storage);
    }

    fn logic(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.rerun_app.logic(ctx, frame);
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        if chrome::custom_chrome() {
            // The window is transparent under our own chrome (rounded
            // corners need it); paint the app's ground under everything
            // first, or the gaps between panels show the desktop
            // (a see-through strip under the header, 2026-09-12).
            chrome::paint_ground(ui);
            chrome::resize_handles(ui);
            // A frameless window has no manager to keep it on the screen.
            chrome::keep_on_screen(ui.ctx(), x11_under_wslg() && spawn::on_wsl());
        }
        self.fullscreen_keys(ui.ctx());
        self.leave_fullscreen_before_close(ui.ctx());
        self.shell.tick(ui.ctx());
        self.answer_screenshot(ui);
        self.apply_commands(ui);
        self.before = (
            self.shell.section,
            self.shell.selected.clone(),
            self.shell.table.as_ref().map(|t| t.section.title.clone()),
        );
        // One header, ours: the viewer's own top bar is hidden (startup
        // override above); its panel toggles ride on ours, and matter on
        // the Live view.
        let sender = self.rerun_app.command_sender.clone();
        self.shell.top_bar(ui, chrome::custom_chrome());
        // The status bar along the bottom (Zed's shape): what runs, the
        // presenter, the viewer's panel toggles, the theme, the frame time.
        let frame_ms = self.frames.mean_ms();
        egui::Panel::bottom("status_bar")
            .resizable(false)
            .frame(egui::Frame::NONE)
            .show(ui, |ui| {
                self.shell.status_bar(ui, frame_ms, |ui| {
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
            });
        // A presenter failure is a panel under the header, before the page;
        // so is Running now, when its indicator was clicked.
        self.shell.presenter_failure(ui);
        self.shell.running_panel(ui);
        // A run's viewer file, asked for from Running now: loaded into
        // the viewer as a file, and the page turns to it.
        if let Some(path) = self.shell.viewer_request.take() {
            use re_viewer::external::re_log_types::FileSource;
            use re_viewer::external::re_viewer_context::{SystemCommand, SystemCommandSender as _};
            self.rerun_app
                .command_sender
                .send_system(SystemCommand::LoadDataSource(
                    re_data_source::LogDataSource::File {
                        file_source: FileSource::Cli,
                        path,
                    },
                ));
            self.shell.show_requested = true;
        }
        let full = self.viewport_full && self.shell.section == Section::Live;
        if !full {
            self.shell.rail(ui);
        }

        // A recording arriving while another page is up switches to Live:
        // a run streaming in is the thing to look at.
        let has_recording = self.rerun_app.recording_db().is_some();
        if has_recording && (!self.seen_recording || self.shell.show_requested) {
            if self.shell.section != Section::Live {
                self.control
                    .event(Event::open(BY_STUDIO).in_section(Section::Live.slug()));
                self.before.0 = Section::Live; // not the human's doing
            }
            self.shell.section = Section::Live;
            self.shell.show_requested = false;
        }
        self.seen_recording = has_recording;
        // The presenter's freshly landed recording comes to the front: the
        // viewer keeps the active one otherwise, and a second Show (a
        // batch after a dataset) streamed unseen behind it (2026-09-27).
        let landed = self
            .shell
            .model
            .present_status
            .as_ref()
            .and_then(|status| status.shown.clone());
        if landed.is_some() && landed != self.activated_shown {
            use re_viewer::external::re_log_types::ApplicationId;
            use re_viewer::external::re_viewer_context::{SystemCommand, SystemCommandSender as _};
            if let Some(app) = landed
                .as_deref()
                .and_then(|id| ApplicationId::try_new(control::entry_name(id)).ok())
            {
                self.rerun_app
                    .command_sender
                    .send_system(SystemCommand::ActivateApp(app));
            }
            self.activated_shown = landed;
        }
        // A deployment played or replayed from its drawer: the viewport
        // runs it and the page turns to it.
        if let Some(scene) = self.shell.scene_request.take() {
            self.viewport = ViewportFeed::spawn(ui.ctx(), &scene, &self.shell.model.project_root);
            if self.shell.section != Section::Live {
                self.control
                    .event(Event::open(BY_USER).in_section(Section::Live.slug()));
                self.before.0 = Section::Live; // said once, not again by the diff
            }
            self.shell.section = Section::Live;
        }

        if full {
            // Full screen: the picture and its transport bar, nothing else.
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ui, |ui| self.viewport_body(ui));
        } else if self.shell.section == Section::Live {
            // The simulator's controls (simulate's own sections: Simulation,
            // Physics, Joint, Control, Visualization, Rendering) on the
            // right; the viewport on top of the rest; the Rerun viewer
            // below it — blueprint panel, timeline, views, exactly as the
            // standalone app renders them. Two panel ids on purpose:
            // egui remembers a panel's size by id, and the idle strip
            // must not inherit a 420 px preview height.
            let active = self.viewport.is_active();
            if !active && !has_recording {
                // Nothing runs and nothing streams: one card says so and
                // holds the Scene picker; no idle band, no transport bar,
                // no second sentence (2026-10-02, "too many border lines").
                egui::CentralPanel::default().show(ui, |ui| {
                    pages::page(ui, |ui| {
                        pages::heading(ui, "Simulator");
                        ui.add_space(8.0);
                        ui.label(
                            egui::RichText::new(
                                "No scene is running. Pick one here, or ask your agent to \
                                 simulate an environment; a data generation, a training \
                                 run or an evaluation streams into the viewer the moment \
                                 it starts.",
                            )
                            .text_style(re_ui::DesignTokens::welcome_screen_body())
                            .color(ui.visuals().weak_text_color()),
                        );
                        ui.add_space(12.0);
                        let deployments = self.shell.model.deploy_scenes();
                        if let Some(simulator::Action::Spawn(task)) =
                            simulator::transport(ui, &mut self.viewport, &deployments)
                        {
                            self.viewport = ViewportFeed::spawn(
                                ui.ctx(),
                                &task,
                                &self.shell.model.project_root,
                            );
                        }
                    });
                });
                self.shell.overlays(ui.ctx());
                if chrome::custom_chrome() {
                    chrome::window_border(ui.ctx(), theme::palette(ui).edge);
                }
                self.report(ui);
                return;
            }
            // Two panels on purpose: egui remembers a panel's size by id
            // (and persists it), so the picker row never inherits the
            // picture's height, nor a band an older build left behind
            // (2026-10-03, "why do we have this space").
            let panel = if active {
                egui::Panel::top("sim_viewport")
                    .resizable(true)
                    .default_size(VIEWPORT_DEFAULT_HEIGHT)
                    .min_size(VIEWPORT_MIN_HEIGHT)
            } else {
                egui::Panel::top("sim_scene_picker")
                    .resizable(false)
                    .exact_size(VIEWPORT_PICKER_HEIGHT)
            };
            panel.show(ui, |ui| self.viewport_body(ui));
            if has_recording {
                self.viewer_ui(ui, frame);
            } else {
                // A scene runs and nothing streams yet: the viewport above is
                // the page; the viewer takes this space when a recording
                // arrives.
                egui::CentralPanel::default().show(ui, |ui| {
                    ui.add_space(12.0);
                    ui.label(
                        egui::RichText::new(
                            "The viewer appears here the moment a run, an evaluation or \
                             this scene's twin streams.",
                        )
                        .color(ui.visuals().weak_text_color()),
                    );
                });
            }
        } else {
            egui::CentralPanel::default().show(ui, |ui| self.shell.page(ui));
        }
        self.shell.overlays(ui.ctx());
        if chrome::custom_chrome() {
            chrome::window_border(ui.ctx(), theme::palette(ui).edge);
        }
        self.report(ui);
    }
}

impl StudioShell {
    /// The picture, its transport bar under it, the overlays and the
    /// Inspect drawer floating over it (simulator.rs) - the same body in
    /// the Live page's top panel and alone in full screen.
    fn viewport_body(&mut self, ui: &mut egui::Ui) {
        let mut action = None;
        if !self.viewport.is_active() {
            // No scene: the row is the picker and nothing else; no idle
            // placeholder above it (2026-10-02, "too many border lines").
            // Drawn in a foreground area over the panel's own rect: the
            // viewer below paints its hidden top bar into this region
            // (`viewer_ui`), and the row must win both paint and input.
            let rect = ui.max_rect();
            let palette = crate::theme::palette(ui);
            let deployments = self.shell.model.deploy_scenes();
            let viewport = &mut self.viewport;
            let inner = egui::Area::new(ui.id().with("scene-picker-row"))
                .order(egui::Order::Foreground)
                .fixed_pos(rect.min)
                .show(ui.ctx(), |ui| {
                    egui::Frame::new()
                        .fill(palette.canvas)
                        .inner_margin(egui::Margin::symmetric(8, 4))
                        .show(ui, |ui| {
                            ui.set_width(rect.width() - 16.0);
                            simulator::transport(ui, viewport, &deployments)
                        })
                        .inner
                });
            action = inner.inner;
            self.spawn_or_stop(ui, action);
            return;
        }
        egui::Panel::bottom("simulator_transport")
            .resizable(false)
            .show(ui, |ui| {
                let deployments = self.shell.model.deploy_scenes();
                action = simulator::transport(ui, &mut self.viewport, &deployments);
            });
        if let Some(ms) = self.frames.mean_ms() {
            let scale = self.viewport.render_scale();
            if ms > SLOW_FRAME_MS && scale > 0.5 {
                self.viewport.set_render_scale(0.5);
            } else if ms < FAST_FRAME_MS && scale < 1.0 {
                self.viewport.set_render_scale(1.0);
            }
        }
        let picture = self.viewport.show(ui);
        if let Some(picture) = picture {
            simulator::overlays(ui.ctx(), picture, &mut self.viewport);
            simulator::drawer(ui.ctx(), picture, &mut self.viewport);
        }
        if let Some(press) = simulator::shortcuts(ui.ctx(), &mut self.viewport) {
            self.control.event(Event::key(BY_USER, &press));
        }
        simulator::apply_follow(ui.ctx(), &mut self.viewport);
        self.spawn_or_stop(ui, action);
    }

    /// What the transport bar asked for.
    fn spawn_or_stop(&mut self, ui: &egui::Ui, action: Option<simulator::Action>) {
        match action {
            Some(simulator::Action::Spawn(task)) => {
                self.viewport =
                    ViewportFeed::spawn(ui.ctx(), &task, &self.shell.model.project_root);
            }
            Some(simulator::Action::Stop) => self.stop_scene(),
            Some(simulator::Action::ToggleFullscreen) => {
                self.viewport_full = !self.viewport_full;
            }
            None => {}
        }
    }

    /// The embedded viewer, with its own top bar out of sight. The viewer
    /// keeps that bar's full height even when its content is hidden
    /// (`exact_size` on its panel; the content is what the override
    /// hides), which left a blank band above every recording (2026-10-03,
    /// "why do we have this space"). So the viewer draws into a child
    /// area that begins one bar height above the visible region, clipped
    /// to it: the bar lands out of sight, the rest fills the space.
    fn viewer_ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        use re_ui::ContextExt as _;
        let hidden = ui.ctx().top_bar_style(frame, false).height;
        let rect = ui.available_rect_before_wrap();
        let above = egui::Rect::from_min_max(egui::pos2(rect.min.x, rect.min.y - hidden), rect.max);
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(above));
        child.set_clip_rect(rect);
        eframe::App::ui(&mut self.rerun_app, &mut child, frame);
    }

    /// No scene: the viewport idle and the page back to its rails — the
    /// one way a scene ends, from the button and from the door alike.
    /// Close the previous viewport twin's recordings before a new one
    /// streams under the same application id: restarted twice, the viewer
    /// held two twins of one scene and showed the older (2026-09-26).
    fn close_twin(&self, scene: &str) {
        use re_viewer::external::re_log_types::ApplicationId;
        use re_viewer::external::re_viewer_context::{SystemCommand, SystemCommandSender as _};
        if let Ok(app) = ApplicationId::try_new(format!("{}{scene}", viewport::TWIN_APP_PREFIX)) {
            self.rerun_app
                .command_sender
                .send_system(SystemCommand::CloseApp(app));
        }
    }

    fn stop_scene(&mut self) {
        self.viewport = ViewportFeed::idle();
        self.viewport_full = false;
    }

    /// `f` puts the viewport alone on the page and back (Live, a scene
    /// running); Escape leaves it, and leaves the window's own full
    /// screen too; F11 toggles the window itself. First presses only:
    /// a held F toggled at repeat rate (2026-09-13).
    fn fullscreen_keys(&mut self, ctx: &egui::Context) {
        let typing = keys::typing(ctx);
        let window_full = ctx.input(|i| i.viewport().fullscreen.unwrap_or(false));
        let f = !typing && keys::first_press(ctx, egui::Key::F);
        let f11 = keys::first_press(ctx, egui::Key::F11);
        let escape = !typing
            && (self.viewport_full || window_full)
            && keys::first_press(ctx, egui::Key::Escape);
        ctx.input_mut(|i| {
            if !typing {
                i.consume_key(egui::Modifiers::NONE, egui::Key::F);
            }
            i.consume_key(egui::Modifiers::NONE, egui::Key::F11);
            if !typing && (self.viewport_full || window_full) {
                i.consume_key(egui::Modifiers::NONE, egui::Key::Escape);
            }
        });
        if f && self.shell.section == Section::Live && self.viewport.is_active() {
            self.viewport_full = !self.viewport_full;
        }
        if escape {
            self.viewport_full = false;
            if window_full {
                ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
            }
        }
        if f11 {
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(!window_full));
        }
    }

    /// A window that closes full screen would reopen full screen: eframe
    /// remembers the window as it was at close. So a close request while
    /// full screen leaves full screen first and closes on the next frame.
    fn leave_fullscreen_before_close(&mut self, ctx: &egui::Context) {
        let (close_requested, window_full) = ctx.input(|i| {
            let v = i.viewport();
            (v.close_requested(), v.fullscreen.unwrap_or(false))
        });
        if self.closing {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        if close_requested && window_full {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
            self.closing = true;
            ctx.request_repaint();
        }
    }
}

// -- the control surface: commands applied, state and events written --------

/// How long a capture waits after the agent's last move, so the frame
/// it takes is settled (egui's own fades run ~100 ms).
const SETTLE_AFTER_NAVIGATION: std::time::Duration = std::time::Duration::from_millis(300);

/// A screenshot in flight: asked for, sent to the viewport once the
/// frame has settled, answered when the frame arrives.
#[derive(Clone)]
struct Shot {
    id: String,
    width: u32,
    asked: std::time::Instant,
    ready_at: std::time::Instant,
    sent: bool,
}

impl StudioShell {
    /// Apply every command the agent wrote since the last poll (20 Hz),
    /// answering each on disk. A command that cannot be applied is
    /// refused with the reason, never dropped.
    fn apply_commands(&mut self, ui: &mut egui::Ui) {
        // The watcher thread (`Control::watch`) wakes the window when a
        // command file appears; the frame loop itself never polls.
        self.control.set_root(self.shell.model.project_root.clone());
        for pending in self.control.poll() {
            let outcome = match pending.command {
                Ok(control::Command::Screenshot { width }) => {
                    // Answered when the frame arrives, not now.
                    match self.request_screenshot(ui, &pending.id, width) {
                        Ok(()) => continue,
                        Err(why) => Err(why),
                    }
                }
                Ok(command) => {
                    if command.moves_the_window() {
                        self.last_navigation = Some(std::time::Instant::now());
                    }
                    self.apply(ui, command)
                }
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
                table,
                view,
                search,
            } => {
                // A move by the agent closes the palette; a search opens it.
                self.shell.palette = search.map(|q| crate::palette::Palette::open(Some(q)));
                let switching = project.is_some();
                if let Some(root) = project {
                    let root = std::path::PathBuf::from(root);
                    if !root.join(model::MANIFEST_FILE).is_file() {
                        return Err(format!(
                            "{} is not a project (no {})",
                            root.display(),
                            model::MANIFEST_FILE
                        ));
                    }
                    self.shell.switch_project(root.clone());
                    self.control
                        .event(Event::open(BY_AGENT).in_project(root.display().to_string()));
                }
                if let Some(name) = section {
                    let page =
                        Section::parse(&name).ok_or_else(|| format!("no page named {name:?}"))?;
                    if switching {
                        self.shell.switch_section = Some(page); // after the switch lands
                    }
                    self.shell.section = page;
                    self.shell.selected = None;
                    self.shell.entered = true;
                    pages::scroll_to_top(ui.ctx());
                    self.control
                        .event(Event::open(BY_AGENT).in_section(page.slug()));
                }
                if let Some(stamp) = artifact {
                    let kind = self
                        .shell
                        .model
                        .artifact(&stamp)
                        .map(|a| a.kind.clone())
                        .ok_or_else(|| format!("no artifact {stamp:?} in this project"))?;
                    let page = Section::for_kind(&kind)
                        .ok_or_else(|| format!("no page lists a {}", pages::kind_word(&kind)))?;
                    self.shell.section = page;
                    self.shell.selected = Some(stamp.clone());
                    self.shell.scroll_to_detail = true;
                    self.control
                        .event(Event::select(BY_AGENT).of_artifact(stamp));
                }
                if let Some(name) = view {
                    let view = crate::listing::View::parse(&name).ok_or_else(|| {
                        format!(
                            "no view {name:?}; one of {}",
                            crate::listing::View::names().join(", ")
                        )
                    })?;
                    crate::listing::set_view(ui.ctx(), self.shell.section, view);
                }
                if let Some(title) = table {
                    let wanted = (!title.trim().is_empty()).then_some(title.as_str());
                    self.shell.open_table(wanted)?;
                    self.control
                        .event(Event::table(BY_AGENT).on_table(wanted.map(str::to_owned)));
                }
                Ok(())
            }
            Command::Show { artifact } => {
                if self.shell.model.artifact(&artifact).is_none() {
                    return Err(format!("no artifact {artifact:?} in this project"));
                }
                self.shell.show(&artifact);
                if let Some(shown) = self.shell.shown.take() {
                    self.control.event(Event::show(BY_AGENT).of_artifact(shown));
                }
                Ok(())
            }
            Command::Focus { recording } => {
                use re_viewer::external::re_log_types::ApplicationId;
                use re_viewer::external::re_viewer_context::{
                    SystemCommand, SystemCommandSender as _,
                };
                // The same migration the SDK applied to the tool's id, so the
                // raw name resolves to the entry name the viewer holds.
                let app = ApplicationId::try_new(control::entry_name(&recording))
                    .map_err(|e| format!("recording {recording:?}: {e}"))?;
                self.rerun_app
                    .command_sender
                    .send_system(SystemCommand::ActivateApp(app));
                Ok(())
            }
            Command::Compare { a, b } => {
                for stamp in [&a, &b] {
                    if self.shell.model.artifact(stamp).is_none() {
                        return Err(format!("no artifact {stamp:?} in this project"));
                    }
                }
                self.shell.compare(&a, &b);
                if let Some(shown) = self.shell.shown.take() {
                    self.control.event(Event::show(BY_AGENT).of_artifact(shown));
                }
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
                if let (Some(n), None) = (name, typ) {
                    // A timeline the recording has not got: named, not acked
                    // as done and left on the old one (2026-09-27).
                    let known: Vec<String> = self
                        .rerun_app
                        .recording_db()
                        .map(|db| db.timelines().keys().map(|k| k.to_string()).collect())
                        .unwrap_or_default();
                    return Err(format!(
                        "no timeline {n:?} in the recording; it has {}",
                        if known.is_empty() {
                            "none".to_owned()
                        } else {
                            known.join(", ")
                        }
                    ));
                }
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
                    for _ in 0..n.unsigned_abs().min(MAX_TIME_STEPS) {
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
            Command::Theme { theme } => {
                if !matches!(theme.as_str(), "system" | "dark" | "light") {
                    return Err(format!("theme: {theme:?} is not system, dark or light"));
                }
                ui.ctx().set_theme(theme_preference(&theme));
                Ok(())
            }
            Command::Simulate { task } => {
                match task {
                    Some(name) => {
                        if !viewport::is_known_scene(&name) {
                            return Err(format!(
                                "no preview scene {name:?}; one of {:?}, {:?}, or a \
                                 deployment as {}<name>",
                                viewport::PREVIEW_TASKS,
                                viewport::WALK_TASK,
                                viewport::DEPLOY_PREFIX
                            ));
                        }
                        self.close_twin(&name);
                        self.viewport =
                            ViewportFeed::spawn(ui.ctx(), &name, &self.shell.model.project_root);
                        if self.shell.section != Section::Live {
                            self.control
                                .event(Event::open(BY_AGENT).in_section(Section::Live.slug()));
                        }
                        self.shell.section = Section::Live;
                    }
                    None => self.stop_scene(),
                }
                Ok(())
            }
            Command::Simulator {
                run,
                step,
                reset,
                keyframe,
                speed,
                manual,
                actuator,
                joint,
                value,
                flag,
                on,
                group,
                kind,
                command,
                inspect,
                view,
                follow,
                fullscreen,
            } => {
                if !self.viewport.is_active() {
                    return Err("no scene runs in the simulator; simulate a task first".into());
                }
                let (_, model) = self.viewport.report();
                let described = || model.as_deref().ok_or(NOT_DESCRIBED);
                // Every field is checked before any is applied: a `run`
                // sent and then a bad `keyframe` refused left the scene
                // running with the agent told only "refused" (2026-09-23).
                let key = match keyframe.as_ref() {
                    Some(name) => {
                        let m = described()?;
                        Some(m.keyframes.iter().position(|k| k == name).ok_or_else(|| {
                            format!("no keyframe {name:?}; one of {:?}", m.keyframes)
                        })? as u32)
                    }
                    None => None,
                };
                if let Some(n) = step {
                    if !(1..=MAX_SIM_STEPS).contains(&n) {
                        return Err(format!("step must be within 1..={MAX_SIM_STEPS}"));
                    }
                }
                // A field that only means something with another is refused
                // with it, not silently ignored (the flash named nothing).
                if kind.is_some() && group.is_none() {
                    return Err("kind needs group".into());
                }
                if on.is_some() && flag.is_none() && group.is_none() {
                    return Err("on needs flag or group".into());
                }
                if value.is_some() && actuator.is_none() && joint.is_none() && command.is_none() {
                    return Err("value needs actuator, joint or command".into());
                }
                if let Some(factor) = speed {
                    if !viewport::SPEED_RANGE.contains(&factor) {
                        return Err(format!(
                            "speed must be within {}..={}",
                            viewport::SPEED_RANGE.start(),
                            viewport::SPEED_RANGE.end()
                        ));
                    }
                }
                let mut sliders: Vec<(viewport::SliderKind, usize, f32)> = Vec::new();
                if actuator.is_some() || joint.is_some() {
                    let value = value.ok_or("an actuator or joint needs a `value`")?;
                    let m = described()?;
                    if let Some(name) = actuator.as_ref() {
                        let index = m
                            .actuators
                            .iter()
                            .position(|a| a.name == *name)
                            .ok_or_else(|| format!("no actuator {name:?}"))?;
                        sliders.push((viewport::SliderKind::Actuator, index, value));
                    }
                    if let Some(name) = joint.as_ref() {
                        let j = m.joints.iter().find(|j| j.name == *name).ok_or_else(|| {
                            format!(
                                "no scalar joint {name:?} (hinge or slide; free and ball \
                                 joints have no scalar)"
                            )
                        })?;
                        sliders.push((viewport::SliderKind::Joint, j.qpos, value));
                    }
                }
                let inspect_tab = match inspect.as_ref() {
                    Some(what) => Some(simulator::inspect_tab(what)?),
                    None => None,
                };
                let follow_choice = match follow.as_ref() {
                    Some(rule) => {
                        let nworld = described()?.nworld;
                        if nworld <= 1 {
                            return Err("follow needs a many-worlds scene (walk); this scene \
                                        has one world"
                                .into());
                        }
                        Some(match rule.trim() {
                            "none" | "" => simulator::Follow::None,
                            "worst" => simulator::Follow::Worst,
                            "failing" => simulator::Follow::Failing,
                            "cycle" => simulator::Follow::Cycle,
                            n => {
                                let world: u32 =
                                    n.trim_start_matches('w').parse().map_err(|_| {
                                        format!(
                                            "follow {rule:?}: none, worst, failing, cycle or a \
                                             world index"
                                        )
                                    })?;
                                if world >= nworld {
                                    return Err(format!(
                                        "follow {rule:?}: this scene has worlds 0 to {}",
                                        nworld - 1
                                    ));
                                }
                                simulator::Follow::World(world)
                            }
                        })
                    }
                    None => None,
                };
                let view_preset = match view.as_ref() {
                    Some(name) => Some(viewport::view_preset(name).ok_or_else(|| {
                        format!(
                            "view {name:?}: one of {}",
                            viewport::view_preset_names().join(", ")
                        )
                    })?),
                    None => None,
                };
                // (visualization?, index, on): a flag lives in one of two tables
                let flag_send = match flag.as_ref() {
                    Some(name) => {
                        let on = on.ok_or("a flag needs `on`")?;
                        let m = described()?;
                        if let Some(i) = m.vis_flags.iter().position(|f| f == name) {
                            Some((true, i as u32, on))
                        } else if let Some(i) = m.rnd_flags.iter().position(|f| f == name) {
                            Some((false, i as u32, on))
                        } else {
                            return Err(format!(
                                "no flag {name:?}; visualization {:?}, rendering {:?}",
                                m.vis_flags, m.rnd_flags
                            ));
                        }
                    }
                    None => None,
                };
                let twist = match command.as_ref() {
                    Some(axis) => Some(simulator::twist_from_door(
                        ui.ctx(),
                        &self.viewport,
                        axis,
                        value,
                    )?),
                    None => None,
                };
                let group_send = match group {
                    Some(group) => {
                        let on = on.ok_or("a group needs `on`")?;
                        let m = described()?;
                        if m.groups.is_empty() || m.ngroup == 0 {
                            return Err("this scene reports no group kinds".into());
                        }
                        // No kind named: the stream's first (geom, as it orders them).
                        let index = match kind.as_ref() {
                            Some(kind) => {
                                m.groups.iter().position(|k| k == kind).ok_or_else(|| {
                                    format!("no group kind {kind:?}; one of {:?}", m.groups)
                                })?
                            }
                            None => 0,
                        };
                        let last = m.ngroup.saturating_sub(1);
                        if group > last {
                            return Err(format!("group {group}: 0 to {last}"));
                        }
                        let kind_byte = u8::try_from(index)
                            .map_err(|_| format!("group kind {index} is past the wire's byte"))?;
                        let group_byte = u8::try_from(group)
                            .map_err(|_| format!("group {group} is past the wire's byte"))?;
                        Some((kind_byte, group_byte, on))
                    }
                    None => None,
                };

                // Everything passed: apply, in the order the fields are named.
                if let Some(full) = fullscreen {
                    self.viewport_full = full;
                    self.shell.section = Section::Live;
                }
                let pressed = [
                    run.map(|r| if r { "run" } else { "pause" }),
                    step.map(|_| "step"),
                    keyframe.as_ref().map(|_| "keyframe"),
                    reset.filter(|r| *r).map(|_| "reset"),
                    speed.map(|_| "speed"),
                    manual.map(|m| if m { "drive" } else { "hand back" }),
                    actuator.as_ref().map(|_| "actuator"),
                    joint.as_ref().map(|_| "joint"),
                    flag.as_ref().map(|_| "overlay"),
                    group.map(|_| "group"),
                    command.as_ref().map(|_| "command"),
                    inspect.as_ref().map(|_| "inspect"),
                    view.as_ref().map(|_| "view"),
                    follow.as_ref().map(|_| "follow"),
                    fullscreen.map(|f| if f { "full screen" } else { "page" }),
                ]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join(", ");
                if !pressed.is_empty() {
                    self.viewport.flash(&pressed);
                }
                if let Some(run) = run {
                    self.viewport.send_run(run);
                }
                if let Some(n) = step {
                    self.viewport.send_step(n);
                }
                if let Some(key) = key {
                    self.viewport.send_reset(Some(key));
                } else if reset == Some(true) {
                    self.viewport.send_reset(None);
                }
                if let Some(factor) = speed {
                    self.viewport.send_speed(factor);
                }
                if let Some(on) = manual {
                    self.viewport.send_manual(on);
                }
                for (slider, index, value) in sliders {
                    match slider {
                        viewport::SliderKind::Actuator => {
                            self.viewport.send_ctrl(index as u32, value)
                        }
                        viewport::SliderKind::Joint => self.viewport.send_qpos(index as u32, value),
                        viewport::SliderKind::Twist => {} // never queued here: `command` sends twists
                    }
                    self.viewport.stop_editing(slider, index);
                }
                if let Some(tab) = inspect_tab {
                    simulator::apply_inspect(ui.ctx(), tab);
                }
                if let Some(choice) = follow_choice {
                    simulator::set_follow(ui.ctx(), choice);
                }
                if let Some(preset) = view_preset {
                    self.viewport.send_view(preset);
                }
                if let Some((visualization, i, on)) = flag_send {
                    if visualization {
                        self.viewport.send_vis(i, on);
                    } else {
                        self.viewport.send_rnd(i, on);
                    }
                }
                if let Some(twist) = twist {
                    simulator::apply_twist_from_door(ui.ctx(), &mut self.viewport, twist);
                }
                if let Some((kind_byte, group_byte, on)) = group_send {
                    self.viewport.send_group(kind_byte, group_byte, on);
                }
                Ok(())
            }
            Command::Screenshot { .. } => {
                unreachable!("screenshots are taken in apply_commands, after the page")
            }
            Command::Quit => {
                // Never close full screen: the window would reopen so.
                ui.ctx()
                    .send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                Ok(())
            }
        }
    }

    /// Ask egui for the frame; the answer comes through `Event::Screenshot`
    /// on a later frame and is written then (`answer_screenshot`).
    fn request_screenshot(
        &mut self,
        ui: &egui::Ui,
        id: &str,
        width: Option<u32>,
    ) -> Result<(), String> {
        if let Some(shot) = &self.shot {
            return Err(format!("a screenshot ({}) is still being taken", shot.id));
        }
        let width = width.unwrap_or(control::SCREENSHOT_WIDTH);
        if width == 0 || width > control::SCREENSHOT_WIDTH_MAX {
            return Err(format!(
                "width must be 1..={}",
                control::SCREENSHOT_WIDTH_MAX
            ));
        }
        // A capture right after a move would show a fade half done (a
        // modal opening, a drawer sliding); it waits for the frame to settle.
        let ready_at = self
            .last_navigation
            .map(|t| t + SETTLE_AFTER_NAVIGATION)
            .unwrap_or_else(std::time::Instant::now);
        self.shot = Some(Shot {
            id: id.to_owned(),
            width,
            asked: std::time::Instant::now(),
            ready_at,
            sent: false,
        });
        ui.ctx().request_repaint_after(SETTLE_AFTER_NAVIGATION);
        Ok(())
    }

    /// The captured frame, if one arrived: write it and answer the command.
    fn answer_screenshot(&mut self, ui: &egui::Ui) {
        let Some(shot) = self.shot.clone() else {
            return;
        };
        if !shot.sent {
            if std::time::Instant::now() < shot.ready_at {
                ui.ctx().request_repaint_after(
                    shot.ready_at
                        .saturating_duration_since(std::time::Instant::now()),
                );
                return;
            }
            ui.ctx()
                .send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::new(
                    shot.id.clone(),
                )));
            if let Some(s) = self.shot.as_mut() {
                s.sent = true;
            }
            return;
        }
        let (id, width, asked) = (shot.id, shot.width, shot.asked);
        let captured = ui.ctx().input(|input| {
            input.events.iter().find_map(|event| match event {
                egui::Event::Screenshot {
                    image, user_data, ..
                } if user_data
                    .data
                    .as_ref()
                    .and_then(|d| d.downcast_ref::<String>())
                    .is_some_and(|got| *got == id) =>
                {
                    Some(image.clone())
                }
                _ => None,
            })
        });
        match captured {
            Some(frame) => {
                self.shot = None;
                match self.control.save_screenshot(&id, &frame, width) {
                    Ok((path, w, h)) => {
                        let mut extra = serde_json::Map::new();
                        extra.insert("path".into(), serde_json::json!(path.display().to_string()));
                        extra.insert("width".into(), serde_json::json!(w));
                        extra.insert("height".into(), serde_json::json!(h));
                        self.control.ack_with(&id, "done", None, extra);
                    }
                    Err(why) => self.control.ack(&id, "failed", Some(&why)),
                }
            }
            None if asked.elapsed() > CAPTURE_TIMEOUT => {
                self.shot = None;
                self.control.ack(
                    &id,
                    "failed",
                    Some(&format!(
                        "the window produced no frame within {} s (a hidden or minimized \
                         window cannot be captured)",
                        CAPTURE_TIMEOUT.as_secs()
                    )),
                );
            }
            None => {}
        }
    }

    /// After the page ran: what the human changed becomes an event, and
    /// the state file says what the window shows.
    fn report(&mut self, ui: &mut egui::Ui) {
        let (section_before, selected_before, table_before) = self.before.clone();
        let table_now = self.shell.table.as_ref().map(|t| t.section.title.clone());
        if table_now != table_before {
            self.control
                .event(Event::table(BY_USER).on_table(table_now.clone()));
        }
        if self.shell.section != section_before {
            self.control
                .event(Event::open(BY_USER).in_section(self.shell.section.slug()));
        }
        if self.shell.selected != selected_before {
            let by = if std::mem::take(&mut self.shell.auto_selected) {
                BY_STUDIO
            } else {
                BY_USER
            };
            match &self.shell.selected {
                Some(stamp) => self
                    .control
                    .event(Event::select(by).of_artifact(stamp.clone())),
                None => self.control.event(Event::deselect(by)),
            }
        } else {
            self.shell.auto_selected = false;
        }
        if let Some(shown) = self.shell.shown.take() {
            self.control.event(Event::show(BY_USER).of_artifact(shown));
        }
        let live = self.live();
        if let Some(rested) = self.control.watch_live(&live) {
            self.control.event(Event::time(BY_USER).at(&rested));
        }
        self.frames.tick();
        self.theme_preference = ui.ctx().options(|o| o.theme_preference);
        theme::align_light_visuals(ui.ctx());
        let state = control::StudioState {
            schema: control::STATE_SCHEMA,
            pid: std::process::id(),
            heartbeat: 0.0,
            project: self.shell.model.project_root.display().to_string(),
            project_name: self.shell.model.name(),
            section: self.shell.section.slug(),
            selected: self.shell.selected.clone(),
            table: table_now,
            live,
            presenter_running: self.shell.presenter_running(),
            jobs_running: self.shell.model.running_jobs(),
            theme: theme_name(ui.ctx().theme()).to_owned(),
            frame_ms: self.frames.mean_ms(),
            frames_per_s: self.frames.per_second(),
            repaint_causes: ui
                .ctx()
                .repaint_causes()
                .iter()
                .map(|c| format!("{}:{}", c.file, c.line))
                .collect(),
            viewport_task: self.viewport.task().map(str::to_owned),
            viewport_fps: self.viewport.fps(),
            window: Some(control::WindowState {
                width: ui.ctx().content_rect().width(),
                height: ui.ctx().content_rect().height(),
                pixels_per_point: ui.ctx().pixels_per_point(),
            }),
            simulator: self.viewport.report().0.map(|s| control::SimulatorState {
                time: s.time,
                rtf: s.rtf,
                paused: s.paused,
                manual: s.manual,
                speed: s.speed,
                render_ms: s.render_ms,
                camera: s.camera.clone(),
            }),
            moved_to: None,
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
        let tip = name
            .and_then(|n| db.time_range_for(&n))
            .map(|r| r.max().as_i64());
        control::Live {
            at_tip: at.is_some() && at == tip,
            tip,
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

/// The storage key of the chosen theme preference.
const THEME_KEY: &str = "trainnr.theme";
/// The theme a first launch opens in, before anyone picks one.
const DEFAULT_THEME: egui::ThemePreference = egui::ThemePreference::Light;

/// The heartbeat's word for the theme in effect.
fn theme_name(theme: egui::Theme) -> &'static str {
    match theme {
        egui::Theme::Dark => "dark",
        egui::Theme::Light => "light",
    }
}

/// The stored word for a preference, and back.
fn preference_name(pref: egui::ThemePreference) -> &'static str {
    match pref {
        egui::ThemePreference::Dark => "dark",
        egui::ThemePreference::Light => "light",
        egui::ThemePreference::System => "system",
    }
}

fn theme_preference(name: &str) -> egui::ThemePreference {
    match name {
        "dark" => egui::ThemePreference::Dark,
        "light" => egui::ThemePreference::Light,
        _ => egui::ThemePreference::System,
    }
}

/// The last second of frames: when each began. Mean frame time and
/// frames per second come from it; both are None until two frames exist.
#[derive(Default)]
struct FrameClock {
    starts: std::collections::VecDeque<std::time::Instant>,
}

impl FrameClock {
    fn tick(&mut self) {
        let now = std::time::Instant::now();
        self.starts.push_back(now);
        while self
            .starts
            .front()
            .is_some_and(|t| now.duration_since(*t) > std::time::Duration::from_secs(1))
        {
            self.starts.pop_front();
        }
    }

    fn mean_ms(&self) -> Option<f32> {
        let (first, last) = (self.starts.front()?, self.starts.back()?);
        let n = self.starts.len();
        (n > 1).then(|| last.duration_since(*first).as_secs_f32() * 1000.0 / (n - 1) as f32)
    }

    fn per_second(&self) -> Option<f32> {
        let n = self.starts.len();
        (n > 1).then_some(n as f32)
    }
}

/// Under WSLg the Vulkan layer (dzn over Direct3D) stalls about half a
/// second on every vsynced present: 2 frames a second on a 24-core box,
/// the app's own threads idle (measured 2026-10-02 through the
/// heartbeat's frame_ms; the same build on software Vulkan drew a frame
/// in 13 ms). egui repaints only when asked, so no vsync costs nothing
/// at rest; `TRAINNR_VSYNC=1` restores it.
fn no_vsync_under_wslg(options: &mut eframe::NativeOptions) {
    if !on_wsl() || std::env::var_os("TRAINNR_VSYNC").is_some_and(|v| v == "1") {
        return;
    }
    re_log::info!("WSLg: presenting without vsync (TRAINNR_VSYNC=1 restores it)");
    options.wgpu_options.surface.present_mode = eframe::wgpu::PresentMode::AutoNoVsync;
}
