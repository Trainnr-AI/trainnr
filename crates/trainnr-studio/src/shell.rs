//! The application shell: top bar, left rail, page router.
//!
//! The rail groups sections in the field's vocabulary (Assets, Data,
//! Training, Evaluation, Deployment) with live counts; the top bar
//! carries the wordmark, the project name and a running-jobs indicator;
//! the page area routes to Overview, a section table, or the Live view
//! (the MuJoCo viewport and the embedded Rerun viewer). The Live view is
//! the only page that shows the viewer; the other pages are the Studio's
//! own, painted with Rerun's tokens so the two halves read as one app.

use re_ui::{icons, DesignTokens, UiExt as _};

use crate::model::Model;
use crate::onboarding::{ActionState, Pick};
use crate::pages::{self, Section, RAIL_ICON};
use crate::spawn::{end_tree, pipeline_command, trainnr_command, PRESENTER_SCRIPT};
use crate::widgets::icon_at;

/// The rail's width: wide enough for "Environments" plus a count.
const RAIL_WIDTH: f32 = 196.0;
/// The files a button's `trainnr` command writes, in the projects home's
/// `.index`: its errors, and what it prints. A file, never a pipe: a
/// pipe read only after the exit fills at about 64 KB and the command
/// then waits forever on its write (review, 2026-10-11).
const ACTION_LOG: &str = "studio-action.log";
const ACTION_OUT: &str = "studio-action.out";
/// `trainnr call`'s exit code for a tool that ran and refused
/// (`trainnr/cli.py::REFUSED_EXIT`).
const REFUSED_EXIT: i32 = 3;
/// Width of the macOS traffic-light cluster the top bar must clear.
#[cfg(target_os = "macos")]
const TRAFFIC_LIGHTS_INSET: f32 = 72.0;

pub struct Shell {
    pub model: Model,
    pub section: Section,
    /// The selected artifact stamp on a section page.
    pub selected: Option<String>,
    /// A project the user picked; applied at the top of the next frame.
    switch_to: Option<std::path::PathBuf>,
    /// The page to land on after the switch, when the command named one
    /// (`open_in_studio(project=…, section=…)` landed on Overview until
    /// 2026-09-28).
    pub switch_section: Option<Section>,
    /// The presenter process for the open project (`tools/studio-present.py`),
    /// spawned on first Show and killed with the shell or on a project switch.
    presenter: Option<std::process::Child>,
    /// Set when the user asked to show an artifact; the frame loop
    /// switches to Live once the viewer has a recording.
    pub show_requested: bool,
    /// A viewport scene the drawer asked for (a deployment played or
    /// replayed), taken by the frame loop that owns the viewport.
    pub scene_request: Option<String>,
    /// This frame's selection was the page's own (a lone artifact on
    /// arrival), not a click: `report` logs it by the Studio.
    pub auto_selected: bool,
    /// What was last sent to the viewer (a version, or `a vs b`), taken
    /// by the frame loop to log the event.
    pub shown: Option<String>,
    /// The last artifact asked for, kept after `shown` is taken by the
    /// frame loop, so a late presenter answer still finds its click.
    pub last_shown: Option<String>,
    /// A table opened for exploration (the modal over the page).
    pub table: Option<crate::detail::TableView>,
    /// Bring the drawer into view on the next frame (an artifact opened
    /// by the agent, or a card far down the grid).
    pub scroll_to_detail: bool,
    /// Where the user came from, for the back button: a lineage link
    /// pushes the page and selection it left.
    history: Vec<(Section, Option<String>)>,
    /// True on the first frame of a freshly opened page: a page with a
    /// single artifact opens its drawer then, and only then, so the
    /// user can still close it.
    pub entered: bool,
    /// The command palette, while open.
    pub palette: Option<crate::palette::Palette>,
    /// A presenter failure the user has dismissed (its stamp and reason).
    dismissed_failure: Option<(String, String)>,
    /// The Running now panel, opened from the top bar's indicator.
    pub running_open: bool,
    /// The run whose stop was clicked once (the second click stops it).
    stop_armed: Option<String>,
    /// The run whose log is open in the panel.
    log_open: Option<String>,
    /// The run whose chip was clicked: its full detail shows under the
    /// chips (by job id, so it survives the once-a-second refresh).
    pub run_selected: Option<String>,
    /// The "+N more" chip was clicked: every run, as a list.
    runs_listed: bool,
    /// What the last stop answered, shown in the panel until the next.
    stop_note: Option<String>,
    /// A run's viewer file to load into the viewer, taken by the frame
    /// loop that owns the viewer.
    pub viewer_request: Option<std::path::PathBuf>,
    /// A button's `trainnr` command while it runs (a sample opening, a
    /// new project): its key, the process, and the log its errors go to.
    action_job: Option<ActionJob>,
    /// What the running command prints (`ACTION_OUT`), read when it ends.
    action_out: std::path::PathBuf,
    /// What the buttons show: the command running, the last failure.
    pub actions: ActionState,
    /// The New project dialog's name, while the dialog is open.
    new_project_name: Option<String>,
    /// The forms over the agent's tools (`trainnr forms`), loaded once.
    pub forms: crate::forms::Catalog,
    /// A form being filled in.
    open_form: Option<crate::forms::Open>,
    /// The engine's setup, started by the first action that needs it.
    engine: crate::engine::Installer,
    /// The action that waits on the engine's setup, run when it is ready.
    pending: Option<(String, Vec<String>, After)>,
    /// The engine's setup finished in this session: the project's presenter
    /// and forms have been started once for it.
    engine_settled: bool,
    /// A sample being opened by the Studio itself: its name, and where the
    /// answer arrives.
    sample_wait: Option<(
        String,
        std::sync::mpsc::Receiver<Result<std::path::PathBuf, String>>,
    )>,
}

/// A button's `trainnr` command while it runs.
struct ActionJob {
    key: String,
    child: std::process::Child,
    log: std::path::PathBuf,
    after: After,
}

/// What a button's command leads to when it ends well.
enum After {
    /// Open the project it made or opened.
    OpenProject,
    /// Stay on the page: a job opens Running now, anything else says done.
    Stay,
}

impl Shell {
    pub fn new(model: Model) -> Self {
        Self {
            model,
            section: Section::Overview,
            selected: None,
            switch_to: None,
            switch_section: None,
            presenter: None,
            show_requested: false,
            scene_request: None,
            auto_selected: false,
            shown: None,
            last_shown: None,
            table: None,
            scroll_to_detail: false,
            history: Vec::new(),
            entered: true,
            palette: None,
            dismissed_failure: None,
            running_open: false,
            stop_armed: None,
            log_open: None,
            run_selected: None,
            runs_listed: false,
            stop_note: None,
            viewer_request: None,
            action_job: None,
            action_out: std::path::PathBuf::new(),
            actions: ActionState::default(),
            new_project_name: None,
            forms: crate::forms::Catalog::default(),
            open_form: None,
            engine: crate::engine::Installer::default(),
            pending: None,
            engine_settled: false,
            sample_wait: None,
        }
    }

    /// Show an artifact in the viewer: make sure the presenter runs for
    /// this project, then write the intent it watches for.
    /// The presenter runs from the moment a project is open, not from the
    /// first Show: it is also the live loop that turns a training run's
    /// console log into its record and re-indexes (the Go2 run was
    /// invisible until a Show, 2026-09-10).
    pub fn open_project(&mut self) {
        if !self.model.welcome && crate::spawn::engine_ready() {
            self.ensure_presenter();
            self.load_forms();
        }
    }

    /// The forms over this build's tools, once the engine can list them;
    /// until then the pages show the ones built in (`forms::built_in`).
    fn load_forms(&mut self) {
        if let Some(home) = crate::model::user_projects_home() {
            self.forms.load(&home.join(crate::model::INDEX_DIR));
        }
    }

    /// Whether the engine can run a tool now; when it cannot, its setup
    /// starts (the first action that needs it), and false is returned.
    pub fn ensure_engine(&mut self) -> bool {
        if crate::spawn::engine_ready() {
            return true;
        }
        self.engine.start();
        false
    }

    /// The engine's setup, each frame while it runs: its step shown on the
    /// waiting button; when it is done, the waiting action runs (or says
    /// why it cannot), and the project's presenter and forms start.
    fn follow_engine(&mut self) {
        use crate::engine::Setup;
        match self.engine.state() {
            Setup::Idle => {}
            Setup::Working(step) => self.actions.progress = Some(step),
            Setup::Ready => {
                self.actions.progress = None;
                if let Some((key, args, after)) = self.pending.take() {
                    self.actions.running = None;
                    let args: Vec<&str> = args.iter().map(String::as_str).collect();
                    self.run_action(&key, &args, after);
                }
                if !self.engine_settled {
                    self.engine_settled = true;
                    self.open_project();
                }
            }
            Setup::Failed(why) => {
                self.actions.progress = None;
                if let Some((key, _, _)) = self.pending.take() {
                    self.actions.running = None;
                    self.actions.failed = Some((key, format!("the engine did not set up: {why}")));
                }
            }
        }
    }

    /// Open a sample with no engine: the Studio downloads, checks and
    /// unpacks it itself (`sample.rs`), and `tick` switches to it.
    pub fn open_sample(&mut self, name: &'static str) {
        if self.action_job.is_some() || self.pending.is_some() || self.sample_wait.is_some() {
            return;
        }
        let Some(sample) = crate::onboarding::SAMPLES.iter().find(|s| s.name == name) else {
            return;
        };
        let Some(home) = crate::model::user_projects_home() else {
            self.actions.failed = Some((name.to_owned(), "no projects home".to_owned()));
            return;
        };
        self.actions.failed = None;
        self.actions.done = None;
        self.actions.running = Some(name.to_owned());
        self.sample_wait = Some((
            name.to_owned(),
            crate::sample::open_in_background(sample, home),
        ));
    }

    /// The sample's opening, when it has ended: switch to it, or say why not.
    fn follow_sample(&mut self) {
        let Some((key, receive)) = &self.sample_wait else {
            return;
        };
        let Ok(outcome) = receive.try_recv() else {
            return;
        };
        let key = key.clone();
        self.sample_wait = None;
        self.actions.running = None;
        match outcome {
            Ok(root) => self.switch_to = Some(root),
            Err(why) => self.actions.failed = Some((key, why)),
        }
    }

    /// Make a project without an agent: the same `create_project` tool an
    /// agent calls (`trainnr call`), which checks the name and refuses by
    /// name; `tick` switches to the project it made.
    pub fn create_project(&mut self, name: &str) {
        // The name alone: create_project makes it a folder in the projects
        // home, and refuses a name that is not one.
        let arguments = serde_json::json!({ "name": name }).to_string();
        self.run_action(
            crate::onboarding::NEW_PROJECT,
            &["call", "create_project", "--args", &arguments],
            After::OpenProject,
        );
    }

    /// Run a form: its tool on the open project, through `trainnr call`.
    pub fn run_form(&mut self, tool: &str, arguments: &serde_json::Value) {
        let arguments = arguments.to_string();
        self.run_action(tool, &["call", tool, "--args", &arguments], After::Stay);
    }

    /// Run `trainnr <args>` for a button, one at a time, on the open
    /// project: `$TRAINNR_PROJECT` names it, because the tools otherwise
    /// act on the projects home's `.current`, which a switch in this
    /// window does not move. Not killed with the window: a sample's
    /// download cut short leaves its temporary folder in the home.
    fn run_action(&mut self, key: &str, args: &[&str], after: After) {
        if self.action_job.is_some() || self.pending.is_some() {
            return;
        }
        if !self.ensure_engine() {
            // The first action: it runs once the engine is set up.
            self.actions.failed = None;
            self.actions.done = None;
            self.actions.running = Some(key.to_owned());
            let args = args.iter().map(|a| (*a).to_owned()).collect();
            self.pending = Some((key.to_owned(), args, after));
            return;
        }
        let dir = crate::model::user_projects_home()
            .unwrap_or_else(std::env::temp_dir)
            .join(crate::model::INDEX_DIR);
        let log = dir.join(ACTION_LOG);
        let out = dir.join(ACTION_OUT);
        let (printed, errors) = match crate::forms::output_files(&dir, &out, &log) {
            Ok(files) => files,
            Err(why) => {
                let why = format!("cannot write in {}: {why}", dir.display());
                self.actions.failed = Some((key.to_owned(), why));
                return;
            }
        };
        let mut command = trainnr_command();
        command
            .args(args)
            .stdin(std::process::Stdio::null())
            .stdout(printed)
            .stderr(errors);
        if self.model.welcome {
            command.env_remove(crate::model::PROJECT_ENV);
        } else {
            command.env(crate::model::PROJECT_ENV, &self.model.project_root);
        }
        self.action_out = out;
        self.actions.failed = None;
        self.actions.done = None;
        match command.spawn() {
            Ok(child) => {
                self.actions.running = Some(key.to_owned());
                self.action_job = Some(ActionJob {
                    key: key.to_owned(),
                    child,
                    log,
                    after,
                });
            }
            Err(why) => {
                self.actions.failed = Some((key.to_owned(), format!("uv did not start: {why}")));
            }
        }
    }

    /// A button's command, when it ended: open the project it made, or
    /// open Running now on the job it started, or say it is done; or say
    /// why not (a refusal's reason, else the log's last line).
    fn reap_action(&mut self) {
        let Some(job) = self.action_job.as_mut() else {
            return;
        };
        let status = match job.child.try_wait() {
            Ok(None) => return,
            Ok(Some(status)) => Some(status),
            Err(_) => None,
        };
        let Some(job) = self.action_job.take() else {
            return;
        };
        self.actions.running = None;
        let printed = std::fs::read_to_string(&self.action_out).unwrap_or_default();
        let code = status.and_then(|s| s.code());
        let failed = match job.after {
            After::OpenProject => match action_outcome(code, &printed) {
                Ok(root) => {
                    self.switch_to = Some(root);
                    None
                }
                Err(why) => Some(why),
            },
            After::Stay => match stay_outcome(code, &printed) {
                Ok(Some(job_id)) => {
                    // A job: Running now opens on it.
                    self.running_open = true;
                    self.runs_listed = false;
                    self.run_selected = Some(job_id);
                    let note = "Started; it runs in Running now.".to_owned();
                    self.actions.done = Some((job.key.clone(), note));
                    None
                }
                Ok(None) => {
                    self.actions.done = Some((job.key.clone(), "Done.".to_owned()));
                    None
                }
                Err(why) => Some(why),
            },
        };
        if let Some(why) = failed {
            let why = why.unwrap_or_else(|| crate::forms::last_line(&job.log));
            self.actions.failed = Some((job.key, why));
        }
    }

    pub fn show(&mut self, stamp: &str) {
        self.ensure_presenter();
        if self.model.request_show(stamp).is_ok() {
            self.show_requested = true;
            self.shown = Some(stamp.to_owned());
            self.last_shown = Some(stamp.to_owned());
            self.dismissed_failure = None;
        }
    }

    /// Two artifacts side by side in the viewer.
    pub fn compare(&mut self, a: &str, b: &str) {
        self.ensure_presenter();
        if self.model.request_compare(a, b).is_ok() {
            self.show_requested = true;
            self.shown = Some(format!("{a} vs {b}"));
            self.last_shown = Some(format!("{a} vs {b}"));
            self.dismissed_failure = None;
        }
    }

    /// Switch projects at the top of the next frame (what the switcher does).
    pub fn switch_project(&mut self, root: std::path::PathBuf) {
        self.switch_to = Some(root);
    }

    /// Whether the presenter process is alive right now.
    pub fn presenter_running(&mut self) -> bool {
        match self.presenter.as_mut() {
            Some(child) => matches!(child.try_wait(), Ok(None)),
            None => false,
        }
    }

    fn ensure_presenter(&mut self) {
        if !self.ensure_engine() {
            return; // started with the project once the engine is ready
        }
        if let Some(child) = self.presenter.as_mut() {
            if matches!(child.try_wait(), Ok(None)) {
                return;
            }
        }
        let mut command = pipeline_command(PRESENTER_SCRIPT);
        command
            .arg("--project")
            .arg(&self.model.project_root)
            // The presenter exits on its own when this window is gone —
            // six of them outlived their Studios on 2026-09-09, because
            // killing the `uv` wrapper leaves the python grandchild.
            .arg("--parent-pid")
            .arg(std::process::id().to_string())
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::inherit());
        self.presenter = command.spawn().ok();
    }

    fn kill_presenter(&mut self) {
        if let Some(child) = self.presenter.take() {
            end_tree(child);
        }
    }

    /// The top bar: wordmark, project, jobs, and the viewer's panel
    /// toggles (meaningful on the Live view; harmless elsewhere).
    /// The title bar, after Zed's: the brand and a crumb (the open project,
    /// the page) on the left, the window's caption buttons on the right,
    /// nothing else; what runs and the switches live in the status bar
    /// (2026-10-03, "headers and footer like zed editor").
    pub fn top_bar(&mut self, ui: &mut egui::Ui, custom_chrome: bool) {
        let tokens = ui.tokens();
        let palette = crate::theme::palette(ui);
        let bar = egui::Frame::new()
            .fill(palette.bar)
            .inner_margin(egui::Margin::symmetric(8, 6))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                if custom_chrome {
                    // This bar is the title bar: drag and double-click,
                    // registered first so the widgets on it win input.
                    let rect = crate::chrome::title_bar_rect(ui.available_rect_before_wrap());
                    crate::chrome::title_bar_interaction(ui, rect, ui.id().with("titlebar"));
                }
                ui.horizontal(|ui| {
                    #[cfg(target_os = "macos")]
                    ui.add_space(TRAFFIC_LIGHTS_INSET);
                    ui.label(egui::RichText::new("●").color(tokens.highlight_color));
                    ui.label(egui::RichText::new("trainnr").strong().size(15.0));
                    ui.add_space(10.0);
                    // The crumb: where the person is, read-only; the rail
                    // and its switcher are where it changes.
                    let crumb = ui.visuals().weak_text_color();
                    ui.label(egui::RichText::new(self.model.name()).color(crumb));
                    if !self.model.welcome {
                        ui.label(egui::RichText::new("›").color(crumb));
                        ui.label(egui::RichText::new(self.section.title()).color(crumb));
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if custom_chrome {
                            // re_ui's caption buttons: close, maximize,
                            // minimize, in the platform's own order.
                            ui.native_window_buttons_ui();
                        }
                    });
                });
            });
        // The bar's edge: the one line between the chrome and the page
        // (2026-10-03, "the top header is all mixed up in the body").
        let rect = bar.response.rect;
        ui.painter().hline(
            rect.x_range(),
            rect.bottom() - 0.5,
            egui::Stroke::new(1.0, palette.edge),
        );
    }

    /// The status bar along the bottom, after Zed's: on the left what
    /// runs (a click opens Running now) and the presenter's state; on the
    /// right the viewer's panel toggles, the theme switch and the frame
    /// time the heartbeat reports.
    pub fn status_bar(
        &mut self,
        ui: &mut egui::Ui,
        frame_ms: Option<f32>,
        viewer_buttons: impl FnOnce(&mut egui::Ui),
    ) {
        let tokens = ui.tokens();
        let palette = crate::theme::palette(ui);
        let rect = ui.available_rect_before_wrap();
        ui.painter().hline(
            rect.x_range(),
            rect.top() + 0.5,
            egui::Stroke::new(1.0, palette.edge),
        );
        egui::Frame::new()
            .fill(palette.bar)
            .inner_margin(egui::Margin::symmetric(10, 4))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.horizontal(|ui| {
                    // What runs, whoever started it: a click opens the
                    // Running now panel (2026-09-25). The line rotates
                    // through the running runs, so a short gate never
                    // hides a long training.
                    let t = ui.input(|i| i.time);
                    let (text, color) = match crate::running::indicator_line(&self.model.jobs, t) {
                        Some(line) => {
                            ui.ctx().request_repaint_after(crate::running::ROTATE_EVERY);
                            (line, tokens.highlight_color)
                        }
                        None => ("idle".to_owned(), ui.visuals().weak_text_color()),
                    };
                    if self.model.running_jobs() > 0 {
                        ui.small_icon(&icons::PLAY, Some(tokens.highlight_color));
                    }
                    let clicked = ui
                        .add(
                            egui::Button::selectable(
                                self.running_open,
                                egui::RichText::new(text).small().color(color),
                            )
                            .frame_when_inactive(false),
                        )
                        .on_hover_text("Running now: every run in the job table")
                        .clicked();
                    if clicked {
                        self.running_open = !self.running_open;
                        if !self.running_open {
                            self.close_log();
                        }
                    }
                    if let Some(why) = self.forms.problem() {
                        ui.add_space(8.0);
                        ui.label(
                            egui::RichText::new("buttons unavailable")
                                .small()
                                .color(ui.visuals().warn_fg_color),
                        )
                        .on_hover_text(format!(
                            "The pages' buttons could not load (trainnr forms): {why}. \
                             Your agent can still run every tool."
                        ));
                    }
                    let presenting = self.presenter_running() || !crate::spawn::engine_ready();
                    if !self.model.welcome && !presenting {
                        ui.add_space(8.0);
                        ui.label(
                            egui::RichText::new("presenter stopped")
                                .small()
                                .color(ui.visuals().warn_fg_color),
                        )
                        .on_hover_text("The process that draws the cards and shows recordings is not running; launch_studio restarts it");
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if let Some(ms) = frame_ms {
                            ui.label(
                                egui::RichText::new(format!("{ms:.0} ms"))
                                    .small()
                                    .color(ui.visuals().weak_text_color()),
                            )
                            .on_hover_text("One frame, drawn to shown, on this machine");
                            ui.add_space(6.0);
                        }
                        // Dark, light or the system's: egui's own switch,
                        // so the embedded viewer and this chrome change
                        // together.
                        egui::global_theme_preference_switch(ui);
                        ui.add_space(6.0);
                        viewer_buttons(ui);
                    });
                });
            });
    }

    /// The left rail: grouped sections with counts, the current one lit.
    pub fn rail(&mut self, ui: &mut egui::Ui) {
        egui::Panel::left("rail")
            .resizable(false)
            .exact_size(RAIL_WIDTH)
            .show(ui, |ui| {
                ui.add_space(8.0);
                self.project_switcher(ui);
                for (group, items) in Section::RAIL {
                    if !group.is_empty() {
                        ui.add_space(10.0);
                        ui.label(
                            egui::RichText::new(*group)
                                .text_style(DesignTokens::welcome_screen_tag())
                                .strong()
                                .color(ui.visuals().weak_text_color()),
                        );
                        ui.add_space(2.0);
                    } else {
                        ui.add_space(4.0);
                    }
                    for item in items.iter().copied() {
                        self.rail_item(ui, item);
                    }
                }
            });
    }

    /// The project switcher at the head of the rail: the open project's
    /// name in a field with a chevron; click for the list, or the
    /// Projects page. It lived in the title bar until 2026-10-02, where a
    /// bare name beside the brand read as a stray label.
    fn project_switcher(&mut self, ui: &mut egui::Ui) {
        let palette = crate::theme::palette(ui);
        let name = self.model.name();
        let response = egui::Frame::new()
            .fill(palette.surface)
            .stroke(egui::Stroke::new(1.0, palette.hairline))
            .corner_radius(6.0)
            .inner_margin(egui::Margin::symmetric(10, 6))
            .show(ui, |ui| {
                ui.set_min_width(RAIL_WIDTH - 36.0);
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Project")
                            .small()
                            .color(ui.visuals().weak_text_color()),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.small_icon(
                            &icons::ARROW_DOWN,
                            Some(ui.tokens().label_button_icon_color),
                        );
                        ui.label(egui::RichText::new(name).strong());
                    });
                });
            })
            .response
            .interact(egui::Sense::click())
            .on_hover_text("Switch project");
        if response.clicked() {
            self.model.refresh_projects();
        }
        let projects = self.model.projects();
        egui::Popup::menu(&response).show(|ui| {
            ui.set_min_width(RAIL_WIDTH - 36.0);
            for project in projects {
                let current = project.root == self.model.project_root;
                let label = if current {
                    egui::RichText::new(&project.name).strong()
                } else {
                    egui::RichText::new(&project.name)
                };
                if ui.button(label).clicked() && !current {
                    self.switch_to = Some(project.root.clone());
                }
            }
            ui.separator();
            if ui.button("All projects…").clicked() {
                self.section = Section::Projects;
            }
        });
        ui.add_space(6.0);
    }

    fn rail_item(&mut self, ui: &mut egui::Ui, item: Section) {
        let tokens = ui.tokens();
        let selected = self.section == item;
        let count = self
            .model
            .index()
            .map(|index| item.kinds().map(|k| index.count(k)).sum::<usize>())
            .unwrap_or(0);
        let palette = crate::theme::palette(ui);
        let fill = if selected {
            palette.accent_soft
        } else {
            egui::Color32::TRANSPARENT
        };
        let response = egui::Frame::new()
            .fill(fill)
            .corner_radius(6.0)
            .inner_margin(egui::Margin::symmetric(10, 6))
            .show(ui, |ui| {
                ui.set_min_width(RAIL_WIDTH - 36.0);
                ui.horizontal(|ui| {
                    let tint = if selected && ui.visuals().dark_mode {
                        tokens.strong_fg_color
                    } else if selected {
                        palette.text
                    } else {
                        tokens.label_button_icon_color
                    };
                    icon_at(ui, item.icon(), RAIL_ICON, tint);
                    let text = egui::RichText::new(item.title());
                    ui.label(if selected { text.strong() } else { text });
                    if item.lists_artifacts() {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(
                                egui::RichText::new(count.to_string())
                                    .small()
                                    .color(ui.visuals().text_color()),
                            );
                        });
                    }
                });
            })
            .response
            .interact(egui::Sense::click());
        if response.hovered() && !selected {
            ui.painter().rect_filled(
                response.rect,
                6.0,
                tokens.form_selectable_bg_color.linear_multiply(0.6),
            );
        }
        if response.clicked() {
            self.section = item;
            self.selected = None;
            self.history.clear();
            self.entered = true;
            crate::pages::scroll_to_top(ui.ctx());
        }
    }

    /// The page for the current section, except Live, which the caller
    /// renders (it owns the viewer and the viewport).
    pub fn page(&mut self, ui: &mut egui::Ui) {
        if self.model.welcome {
            self.model.refresh_projects();
            let home = crate::model::user_projects_home();
            let picked = crate::onboarding::welcome(
                ui,
                self.model.projects(),
                &self.actions,
                home.as_deref(),
            );
            self.pick(picked);
            return;
        }
        if let Some(problem) = self.model.problem.clone() {
            if self.section != Section::Projects {
                pages::problem_page(ui, &problem);
                return;
            }
        }
        match self.section {
            Section::Projects => {
                let picked = pages::projects(ui, &mut self.model, &self.actions);
                self.pick(picked);
            }
            Section::Overview => {
                let mut go_to = None;
                let mut open_run = None;
                pages::overview(ui, &self.model, &mut go_to, &mut open_run);
                if let Some(id) = open_run {
                    // A chip on the Compute card: the panel opens on it.
                    self.running_open = true;
                    self.runs_listed = false;
                    self.run_selected = Some(id);
                }
                if let Some((section, stamp)) = go_to {
                    self.section = section;
                    self.scroll_to_detail = stamp.is_some();
                    self.selected = stamp;
                    self.entered = true;
                    pages::scroll_to_top(ui.ctx());
                }
            }
            Section::Live => {}
            section => {
                let mut show: Option<String> = None;
                let mut explore = None;
                let scroll = std::mem::take(&mut self.scroll_to_detail);
                let mut nav = pages::Nav {
                    can_back: !self.history.is_empty(),
                    scroll_to_detail: scroll,
                    entered: std::mem::take(&mut self.entered),
                    ..Default::default()
                };
                let forms = self.forms.for_page(section);
                pages::section(
                    ui,
                    &self.model,
                    section,
                    &mut self.selected,
                    &mut show,
                    &mut explore,
                    &mut nav,
                    (&forms, &self.actions),
                );
                if let Some(tool) = nav.form.take() {
                    if let Some(form) = self.forms.form(&tool) {
                        self.open_form = Some(crate::forms::Open::new(form.clone()));
                    }
                }
                if let Some(stamp) = show {
                    self.show(&stamp);
                }
                if let Some(table) = explore {
                    self.table = Some(crate::detail::TableView::new(table));
                }
                if let Some(stamp) = nav.open {
                    self.navigate(&stamp);
                }
                if let Some(scene) = nav.scene.take() {
                    self.scene_request = Some(scene);
                }
                if nav.auto_selected {
                    self.auto_selected = true;
                }
                if nav.back {
                    if let Some((section, selected)) = self.history.pop() {
                        self.section = section;
                        self.scroll_to_detail = selected.is_some();
                        self.selected = selected;
                        pages::scroll_to_top(ui.ctx());
                    }
                }
            }
        }
        if let Some(view) = self.table.as_mut() {
            if !crate::detail::table_modal(ui.ctx(), view) {
                self.table = None;
            }
        }
    }

    /// Once per frame, before anything is drawn: the project switch the
    /// user asked for, and the model's watch on its files (the index,
    /// the jobs, the events, the presenter's answer) — on every page,
    /// the Simulator included, not only where a page is drawn.
    pub fn tick(&mut self, ctx: &egui::Context) {
        self.reap_action();
        self.follow_sample();
        self.follow_engine();
        self.forms.poll();
        if self.switch_to.is_none() {
            // On the Welcome page: the first project the agent makes or
            // opens is the one to show.
            self.switch_to = self.model.followed_project();
        }
        if let Some(root) = self.switch_to.take() {
            self.kill_presenter();
            self.model.switch(root);
            self.selected = None;
            self.section = self.switch_section.take().unwrap_or(Section::Overview);
            self.entered = true;
            self.open_project();
        }
        self.model.refresh();
        ctx.request_repaint_after(crate::model::RELOAD_EVERY);
    }

    /// A click on the Welcome or Projects page: a project to switch to,
    /// or a sample to open.
    fn pick(&mut self, picked: Option<Pick>) {
        match picked {
            Some(Pick::Project(root)) => self.switch_to = Some(root),
            Some(Pick::Sample(name)) => self.open_sample(name),
            Some(Pick::NewProject) => self.new_project_name = Some(String::new()),
            None => {}
        }
    }

    /// What floats over every page, the Simulator included: the keys and
    /// the command palette. Called once per frame after the page (or the
    /// viewer) has been drawn; the failure banner is a panel and goes
    /// before the page (`presenter_failure`).
    pub fn overlays(&mut self, ctx: &egui::Context) {
        if let Some(name) = self.new_project_name.as_mut() {
            match crate::onboarding::new_project_dialog(ctx, name) {
                crate::onboarding::Dialog::Open => {}
                crate::onboarding::Dialog::Cancel => self.new_project_name = None,
                crate::onboarding::Dialog::Create(name) => {
                    self.new_project_name = None;
                    self.create_project(&name);
                }
            }
            // The dialog has the keys: no palette, no page moves under it.
            return;
        }
        if let Some(open) = self.open_form.as_mut() {
            let artifacts = |kind: &str| -> Vec<String> {
                self.model
                    .index()
                    .map(|index| {
                        index
                            .by_kind(kind)
                            .iter()
                            .map(|a| a.stamp.clone())
                            .collect()
                    })
                    .unwrap_or_default()
            };
            match crate::forms::dialog(ctx, open, &artifacts) {
                crate::forms::Answer::Open => {}
                crate::forms::Answer::Cancel => self.open_form = None,
                crate::forms::Answer::Submit(arguments) => {
                    let tool = open.form.tool.clone();
                    self.open_form = None;
                    self.run_form(&tool, &arguments);
                }
            }
            return;
        }
        self.keys(ctx);
        if let Some(palette) = self.palette.as_mut() {
            let (chosen, close) = crate::palette::show(ctx, palette, self.model.index());
            if close {
                self.palette = None;
            }
            match chosen {
                Some(crate::palette::Hit::Page(section)) => {
                    self.section = section;
                    self.selected = None;
                    self.history.clear();
                    self.entered = true;
                    pages::scroll_to_top(ctx);
                }
                Some(crate::palette::Hit::Artifact { stamp, .. }) => self.navigate(&stamp),
                None => {}
            }
        }
    }

    /// The keys that move the window: ⌘K / Ctrl+K opens the palette, Esc
    /// closes what is open (the palette, the table, then the drawer), and
    /// ←/→ walk the page's artifacts in the order they are listed. Text
    /// boxes keep their keys.
    fn keys(&mut self, ctx: &egui::Context) {
        let typing = crate::keys::typing(ctx);
        // First presses (a held Escape closed everything at repeat rate);
        // the keys are consumed either way so no widget below sees them.
        let escape = !typing && crate::keys::first_press(ctx, egui::Key::Escape);
        let left = !typing && crate::keys::first_press(ctx, egui::Key::ArrowLeft);
        let right = !typing && crate::keys::first_press(ctx, egui::Key::ArrowRight);
        let palette_key = ctx.input_mut(|i| {
            if !typing {
                i.consume_key(egui::Modifiers::NONE, egui::Key::Escape);
                i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowLeft);
                i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight);
            }
            i.consume_key(egui::Modifiers::COMMAND, egui::Key::K)
        });
        if palette_key {
            self.palette = Some(crate::palette::Palette::open(None));
        }
        if escape {
            if self.palette.is_some() {
                self.palette = None;
            } else if self.table.is_some() {
                self.table = None;
            } else {
                self.selected = None;
            }
        }
        if (left || right) && self.palette.is_none() && self.table.is_none() {
            let stamps = pages::ordered(&self.model, self.section);
            if !stamps.is_empty() {
                let at = self
                    .selected
                    .as_ref()
                    .and_then(|s| stamps.iter().position(|x| x == s));
                let next = match (at, right) {
                    (None, _) => 0,
                    (Some(i), true) => (i + 1).min(stamps.len() - 1),
                    (Some(i), false) => i.saturating_sub(1),
                };
                self.selected = Some(stamps[next].clone());
                self.scroll_to_detail = true;
            }
        }
    }

    /// When "Show in viewer" could not be honoured, say so where the
    /// click happened, with the presenter's reason, until dismissed.
    pub fn presenter_failure(&mut self, ui: &mut egui::Ui) {
        let Some(status) = self.model.present_status.clone() else {
            return;
        };
        let (Some(stamp), Some(error)) = (status.stamp, status.error) else {
            return;
        };
        if self.last_shown.as_deref() != Some(stamp.as_str()) {
            return;
        }
        if self.dismissed_failure.as_ref() == Some(&(stamp.clone(), error.clone())) {
            return;
        }
        egui::Panel::top("presenter-failure")
            .frame(egui::Frame::new().inner_margin(egui::Margin::symmetric(12, 8)))
            .show(ui, |ui| {
                ui.painter().rect_filled(
                    ui.max_rect().expand(8.0),
                    0.0,
                    ui.visuals().warn_fg_color.linear_multiply(0.12),
                );
                ui.horizontal_wrapped(|ui| {
                    ui.small_icon(&icons::WARNING, Some(ui.visuals().warn_fg_color));
                    ui.label(
                        egui::RichText::new(format!(
                            "The viewer could not show {}: {error}",
                            crate::model::split_stamp(&stamp).0
                        ))
                        .color(ui.visuals().warn_fg_color),
                    );
                    if ui.small_button("dismiss").clicked() {
                        self.dismissed_failure = Some((stamp.clone(), error.clone()));
                    }
                });
            });
    }

    /// The Running now panel, under the header on every page: each run
    /// in the job table (door, tool or agent) as a chip, the running
    /// first, then what ended within `running::FINISHED_KEPT` with its
    /// exit code; a chip's click opens that run's full detail below, the
    /// "+N more" chip the full list (2026-09-25).
    pub fn running_panel(&mut self, ui: &mut egui::Ui) {
        use crate::running::{
            chip_strip, kept_selection, panel_rows, row, RowAction, StripClick, FINISHED_KEPT,
        };
        if !self.running_open {
            return;
        }
        let now = crate::model::now_epoch();
        let jobs = self.model.jobs.clone();
        let rows = panel_rows(&jobs, now, FINISHED_KEPT);
        // A selected run that dropped off the table clears the selection.
        self.run_selected = kept_selection(self.run_selected.as_deref(), &rows);
        let mut chosen: Option<(crate::running::Job, RowAction)> = None;
        let mut strip_click: Option<StripClick> = None;
        egui::Panel::top("running-now")
            .frame(
                egui::Frame::new()
                    .fill(crate::theme::palette(ui).bar)
                    .inner_margin(egui::Margin::symmetric(12, 8)),
            )
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Running now").strong());
                    ui.label(
                        egui::RichText::new(format!(
                            "the job table, read once a second · ended runs kept {} min",
                            FINISHED_KEPT.as_secs() / 60
                        ))
                        .small()
                        .color(ui.visuals().weak_text_color()),
                    );
                    if ui.small_button("close").clicked() {
                        self.running_open = false;
                    }
                });
                if let Some(note) = &self.stop_note {
                    ui.label(
                        egui::RichText::new(note)
                            .small()
                            .color(ui.visuals().warn_fg_color),
                    );
                }
                if rows.is_empty() {
                    ui.label(
                        egui::RichText::new(
                            "Nothing is running, and nothing ended in the last few minutes.",
                        )
                        .color(ui.visuals().weak_text_color()),
                    );
                }
                ui.add_space(4.0);
                strip_click = chip_strip(
                    ui,
                    &rows,
                    now,
                    self.run_selected.as_deref(),
                    self.runs_listed,
                );
                // Below the chips: the selected run in full, or every run
                // as a list when "+N more" is open.
                let detail: Vec<&crate::running::Job> = if self.runs_listed {
                    rows.clone()
                } else {
                    rows.iter()
                        .copied()
                        .filter(|j| self.run_selected.as_deref() == Some(j.id.as_str()))
                        .collect()
                };
                if !detail.is_empty() {
                    egui::ScrollArea::vertical()
                        .max_height(ui.ctx().content_rect().height() * 0.5)
                        .show(ui, |ui| {
                            for job in &detail {
                                ui.separator();
                                let log_open = self.log_open.as_deref() == Some(job.id.as_str());
                                let armed = self.stop_armed.as_deref() == Some(job.id.as_str());
                                if let Some(action) = row(ui, job, now, log_open, armed) {
                                    chosen = Some(((*job).clone(), action));
                                }
                                if log_open {
                                    match &self.model.log_tail {
                                        Some((id, lines)) if id == &job.id => {
                                            crate::running::log_tail(ui, lines)
                                        }
                                        _ => {
                                            ui.label(
                                                egui::RichText::new("reading the log…").small(),
                                            );
                                        }
                                    }
                                }
                            }
                        });
                }
            });
        match strip_click {
            Some(StripClick::Chip(id)) => {
                self.runs_listed = false;
                self.run_selected = if self.run_selected.as_deref() == Some(id.as_str()) {
                    None
                } else {
                    Some(id)
                };
            }
            Some(StripClick::More) => {
                self.runs_listed = !self.runs_listed;
                self.run_selected = None;
            }
            None => {}
        }
        if !self.running_open {
            self.close_log();
        }
        if let Some((job, action)) = chosen {
            self.act_on_run(&job, action);
        }
    }

    fn close_log(&mut self) {
        self.log_open = None;
        self.model.watch_log(None);
    }

    fn act_on_run(&mut self, job: &crate::running::Job, action: crate::running::RowAction) {
        use crate::running::RowAction;
        match action {
            RowAction::Log => {
                if self.log_open.as_deref() == Some(job.id.as_str()) {
                    self.close_log();
                } else {
                    self.log_open = Some(job.id.clone());
                    self.model.watch_log(Some(&job.id));
                }
            }
            RowAction::Viewer => {
                // A record names the stream file; only one inside the
                // project is opened (review, 2026-10-05).
                let path = std::path::PathBuf::from(&job.viewer);
                if crate::control::inside(&self.model.project_root, &path) {
                    self.viewer_request = Some(path);
                } else {
                    eprintln!(
                        "studio: {} is outside the project; not opened",
                        path.display()
                    );
                }
            }
            RowAction::Viewport => self.scene_request = Some(job.viewport.clone()),
            RowAction::ArmStop => self.stop_armed = Some(job.id.clone()),
            RowAction::Disarm => self.stop_armed = None,
            RowAction::Stop => {
                self.stop_armed = None;
                self.stop_note = Some(match crate::running::stop(job) {
                    Ok(()) => format!("asked {} (pid {}) to stop", job.tool, job.pid),
                    Err(err) => format!("could not stop {} (pid {}): {err}", job.tool, job.pid),
                });
            }
        }
    }

    /// Go to an artifact by its stamp, from a lineage link: the page it
    /// belongs to, its drawer open, the way back remembered.
    pub fn navigate(&mut self, stamp: &str) {
        let Some(kind) = self.model.artifact(stamp).map(|a| a.kind.clone()) else {
            return;
        };
        let Some(section) = Section::for_kind(&kind) else {
            return;
        };
        self.history.push((self.section, self.selected.clone()));
        self.section = section;
        self.selected = Some(stamp.to_owned());
        self.scroll_to_detail = true;
    }

    /// Open one of the selected artifact's tables by title (the agent's
    /// door); `None` closes whatever is open.
    pub fn open_table(&mut self, title: Option<&str>) -> Result<(), String> {
        let Some(title) = title else {
            self.table = None;
            return Ok(());
        };
        let stamp = self
            .selected
            .clone()
            .ok_or_else(|| "no artifact is selected; open one first".to_owned())?;
        let artifact = self
            .model
            .artifact(&stamp)
            .ok_or_else(|| format!("{stamp} is not in the index"))?;
        let detail = self
            .model
            .detail(artifact)
            .ok_or_else(|| format!("{stamp} has no detail file"))?;
        let wanted = title.trim().to_lowercase();
        let section = detail
            .sections
            .iter()
            .find(|s| s.kind == "table" && s.title.to_lowercase() == wanted)
            .cloned()
            .ok_or_else(|| format!("{stamp} has no table named {title:?}"))?;
        self.table = Some(crate::detail::TableView::new(section));
        Ok(())
    }
}

impl Drop for Shell {
    fn drop(&mut self) {
        self.kill_presenter();
    }
}

/// What a button's `trainnr` command came to, from its exit code and what
/// it printed: the project folder to switch to (`sample open` prints the
/// path; `call` prints the tool's JSON, whose `root` names it); a refusal
/// with its reason; or `Err(None)`, a failure whose reason is in the log.
fn action_outcome(code: Option<i32>, printed: &str) -> Result<std::path::PathBuf, Option<String>> {
    let reply: Option<serde_json::Value> = serde_json::from_str(printed).ok();
    if code == Some(REFUSED_EXIT) {
        return Err(Some(refusal_reason(reply.as_ref())));
    }
    if code != Some(0) {
        return Err(None);
    }
    let root = match reply.as_ref().and_then(|r| r.get("root")) {
        Some(root) => root.as_str().map(std::path::PathBuf::from),
        None => Some(std::path::PathBuf::from(printed.trim())),
    };
    root.filter(|r| r.join(crate::model::MANIFEST_FILE).is_file())
        .ok_or_else(|| Some("it finished without naming a project".to_owned()))
}

/// What a form's `trainnr call` came to: the job it started (its
/// `job_id`), or `None` for a tool that finished in place; a refusal
/// with its reason; or `Err(None)`, a failure whose reason is in the log.
fn stay_outcome(code: Option<i32>, printed: &str) -> Result<Option<String>, Option<String>> {
    let reply: Option<serde_json::Value> = serde_json::from_str(printed).ok();
    if code == Some(REFUSED_EXIT) {
        return Err(Some(refusal_reason(reply.as_ref())));
    }
    if code != Some(0) {
        return Err(None);
    }
    Ok(reply
        .as_ref()
        .and_then(|r| r.get("job_id"))
        .and_then(|j| j.as_str())
        .map(str::to_owned))
}

/// A refusal's reason, from the tool's reply.
fn refusal_reason(reply: Option<&serde_json::Value>) -> String {
    reply
        .and_then(|r| r.get("reason"))
        .and_then(|r| r.as_str())
        .map_or_else(|| "the tool refused".to_owned(), str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A shell on a machine with no checkout and no engine, its projects
    /// home under `TRAINNR_TEST_FRESH_HOME` (an empty folder).
    fn fresh_shell() -> Shell {
        let home = std::path::PathBuf::from(
            std::env::var("TRAINNR_TEST_FRESH_HOME").expect("an empty home"),
        );
        std::env::set_var("TRAINNR_HOME", &home);
        std::env::remove_var("TRAINNR_PROJECTS");
        std::env::remove_var(crate::spawn::REPO_ENV);
        crate::spawn::hide_checkout();
        let projects = home.join("projects");
        std::fs::create_dir_all(&projects).unwrap();
        let shell = Shell::new(Model::at(projects.clone(), vec![projects]));
        assert!(shell.model.welcome);
        assert!(
            !crate::spawn::engine_ready(),
            "a fresh machine has no engine"
        );
        shell
    }

    /// Frames until the running button is done, its steps printed.
    fn run_frames(shell: &mut Shell) {
        let ctx = egui::Context::default();
        let started = std::time::Instant::now();
        let mut said = String::new();
        while started.elapsed() < std::time::Duration::from_secs(900) {
            shell.tick(&ctx);
            if let Some(step) = shell.actions.progress.clone() {
                if step != said {
                    eprintln!("{:>6.1}s  {step}", started.elapsed().as_secs_f32());
                    said = step;
                }
            }
            if shell.actions.running.is_none() {
                shell.tick(&ctx); // the switch lands on the next frame
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
        eprintln!("{:>6.1}s  done", started.elapsed().as_secs_f32());
    }

    /// Open sample on a machine with nothing: the Studio downloads, checks
    /// and opens it, and no engine is set up. By hand, alone (network):
    /// `TRAINNR_TEST_FRESH_HOME=<empty dir> cargo test --release --
    /// --ignored a_sample_opens --nocapture`.
    #[test]
    #[ignore = "downloads the sample; run by hand"]
    fn a_sample_opens_with_no_engine() {
        let mut shell = fresh_shell();
        shell.open_sample("go2-walk");
        run_frames(&mut shell);
        assert_eq!(shell.actions.failed, None);
        assert!(shell.model.project_root.ends_with("go2-walk-sample"));
        assert!(shell.model.index().is_some_and(|i| !i.artifacts.is_empty()));
        assert!(!crate::spawn::engine_ready(), "no engine was needed");
    }

    /// The first tool on a machine with nothing: New project sets the engine
    /// up (its steps shown on the button), then creates the project. By
    /// hand, alone: `TRAINNR_TEST_FRESH_HOME=<empty dir>
    /// TRAINNR_RUNTIME_URL=<archive> cargo test --release -- --ignored
    /// a_first_tool --nocapture`.
    #[test]
    #[ignore = "sets an engine up; run by hand"]
    fn a_first_tool_sets_the_engine_up_then_runs() {
        let mut shell = fresh_shell();
        shell.create_project("lab-arm");
        run_frames(&mut shell);
        assert_eq!(shell.actions.failed, None);
        assert!(crate::spawn::engine_ready());
        assert!(shell.model.project_root.ends_with("lab-arm"));
    }

    #[test]
    fn a_forms_command_opens_its_job_or_says_it_is_done() {
        let started = serde_json::json!({ "job_id": "j1", "status": "started" }).to_string();
        assert_eq!(stay_outcome(Some(0), &started), Ok(Some("j1".to_owned())));
        let done = serde_json::json!({ "status": "done" }).to_string();
        assert_eq!(stay_outcome(Some(0), &done), Ok(None));
        let refused = serde_json::json!({ "status": "refused", "reason": "no robot" });
        assert_eq!(
            stay_outcome(Some(REFUSED_EXIT), &refused.to_string()),
            Err(Some("no robot".to_owned()))
        );
        assert_eq!(stay_outcome(Some(1), ""), Err(None));
    }

    #[test]
    fn a_buttons_command_switches_only_to_a_project_it_names() {
        let dir = std::env::temp_dir().join(format!("studio-action-{}", std::process::id()));
        let project = dir.join("mine");
        std::fs::create_dir_all(&project).unwrap();
        std::fs::write(project.join(crate::model::MANIFEST_FILE), "{}").unwrap();
        // Built by serde, so a Windows path's backslashes are escaped.
        let called = serde_json::json!({ "status": "done", "root": project }).to_string();
        assert_eq!(
            action_outcome(Some(0), &called),
            Ok(project.clone()),
            "call's JSON"
        );
        let printed = format!("{}\n", project.display());
        assert_eq!(
            action_outcome(Some(0), &printed),
            Ok(project.clone()),
            "a path"
        );
        let refused = r#"{"status": "refused", "reason": "name taken"}"#;
        assert_eq!(
            action_outcome(Some(REFUSED_EXIT), refused),
            Err(Some("name taken".to_owned()))
        );
        assert_eq!(action_outcome(Some(1), ""), Err(None), "the log says why");
        assert_eq!(action_outcome(None, ""), Err(None), "killed");
        let elsewhere = r#"{"status": "done", "root": "/nowhere/at/all"}"#;
        assert!(matches!(action_outcome(Some(0), elsewhere), Err(Some(_))));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
