//! What a page shows before it holds anything, and what an empty project's
//! Overview shows instead of a row of zeros: what the page is for, and the
//! words to say to the agent that fills it, each one a chip that copies
//! itself. A fresh-install review (2026-10-09) found every empty page a
//! single faint sentence, and the first project's Overview nine grey chips
//! and four zeros with nothing to do next.
//!
//! And the Welcome page, the window before any project exists: start your
//! own project through the agent, or open a finished sample, or one of
//! your projects. A new user's first screen was a finished Go2 walk, or
//! the checkout's empty sample, until 2026-10-09.

use std::path::{Path, PathBuf};
use std::time::Duration;

use re_ui::{icons, DesignTokens, Icon};

use crate::model::ProjectSummary;
use crate::pages::Section;
use crate::widgets::{card, icon_at, tag};

// Sentences use the text colour, never the muted one: muted measured
// 2.7:1 on a dark card, under the 4.5:1 body text needs (2026-10-09).
/// How long a copied chip says so.
const COPIED_FOR: Duration = Duration::from_millis(1500);
/// The icon above an empty page's title.
const EMPTY_ICON_SIZE: f32 = 40.0;
/// The prompt column's width cap: a chip reads as one line on a wide page.
const PROMPT_MAX_WIDTH: f32 = 560.0;

/// An empty page's guide: its title, what the page holds, and what to say.
pub struct Guide {
    pub title: &'static str,
    pub body: &'static str,
    pub prompts: &'static [&'static str],
}

impl Section {
    /// The guide this page shows while it has nothing.
    pub fn guide(self) -> Guide {
        match self {
            Self::Projects => Guide {
                title: "No projects yet",
                body: "A project holds one robot effort: its model, its data, and every \
                       result made from them.",
                prompts: &["Create a project called my-robot"],
            },
            Self::Robots => Guide {
                title: "No robots yet",
                body: "A robot enters as its model (MJCF, URDF or USD). Its page shows the \
                       model's parts and every fit of its dynamics.",
                prompts: &["Onboard my robot from <path to its MJCF, URDF or USD>"],
            },
            Self::Environments => Guide {
                title: "No environments yet",
                body: "An environment is a task declared for a robot, a walk or a \
                       manipulation scene, and checked to be learnable before any \
                       training.",
                prompts: &[
                    "Define a task for my robot and check that it is learnable",
                    "Which task families can my robot use?",
                ],
            },
            Self::Recordings => Guide {
                title: "No recordings yet",
                body: "A recording is telemetry from a robot: a ROS 2 bag, a LeRobot dataset, \
                       a .wire log or motion capture. System identification fits the \
                       robot's dynamics to it.",
                prompts: &[
                    "Ingest my robot's telemetry from <path>",
                    "Which public robot logs can I use?",
                ],
            },
            Self::Datasets => Guide {
                title: "No datasets yet",
                body: "Demonstrations generated in simulation, for imitation learning. A walk \
                       trained by reinforcement needs none.",
                prompts: &["Generate demonstrations for my task"],
            },
            Self::Experiments => Guide {
                title: "No experiments yet",
                body: "A training run: its reward curves, its settings, and the checkpoints \
                       it saves.",
                prompts: &["Train a walk for my robot, a smoke run first"],
            },
            Self::Policies => Guide {
                title: "No policies yet",
                body: "The checkpoints a training run saves, ready to evaluate and to \
                       export.",
                prompts: &["Train a walk for my robot, a smoke run first"],
            },
            Self::Certificates => Guide {
                title: "No evaluations yet",
                body: "A policy's paired trials: its success rate with an exact 95 % \
                       interval, and every trial on the record.",
                prompts: &["Evaluate my latest policy with 40 trials"],
            },
            Self::Deployments => Guide {
                title: "No deployments yet",
                body: "An exported policy, its ONNX and manifest, gated against the robot's \
                       own runtime in simulation before it meets the real robot.",
                prompts: &["Export my best policy and run the sim-to-sim gate"],
            },
            Self::Monitoring => Guide {
                title: "Nothing monitored yet",
                body: "Fresh telemetry checked against the fitted model: which parameters \
                       drifted, and whether to identify again.",
                prompts: &["Check my robot's newest recording for drift"],
            },
            Self::Findings => Guide {
                title: "No findings yet",
                body: "A finding is a measured result kept with the commit, the command and \
                       the simulator build that produced it.",
                prompts: &[],
            },
            Self::Overview | Self::Live => Guide {
                title: "",
                body: "",
                prompts: &[],
            },
        }
    }
}

/// An empty page: its icon, its title, what it holds, and the prompts.
pub fn empty_state(ui: &mut egui::Ui, icon: &Icon, guide: &Guide) {
    let palette = crate::theme::palette(ui);
    card(ui, None).show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        ui.vertical_centered(|ui| {
            ui.add_space(24.0);
            icon_at(ui, icon, EMPTY_ICON_SIZE, palette.muted);
            ui.add_space(12.0);
            ui.label(
                egui::RichText::new(guide.title)
                    .text_style(DesignTokens::welcome_screen_h2())
                    .strong()
                    .color(ui.visuals().strong_text_color()),
            );
            ui.add_space(6.0);
            ui.set_max_width(PROMPT_MAX_WIDTH);
            ui.label(
                egui::RichText::new(guide.body)
                    .text_style(DesignTokens::welcome_screen_body())
                    .color(palette.text),
            );
            if !guide.prompts.is_empty() {
                ui.add_space(18.0);
                ask_your_agent(ui, guide.prompts);
            }
            ui.add_space(24.0);
        });
    });
}

/// "Ask your agent:" and a chip per prompt.
pub fn ask_your_agent(ui: &mut egui::Ui, prompts: &[&str]) {
    let palette = crate::theme::palette(ui);
    ui.label(
        egui::RichText::new("Ask your agent:")
            .text_style(DesignTokens::welcome_screen_body())
            .color(palette.muted),
    );
    ui.add_space(6.0);
    for prompt in prompts {
        prompt_chip(ui, prompt);
        ui.add_space(6.0);
    }
}

/// A prompt in quotes with a copy icon, sized to its words so a centred
/// column centres it; a click anywhere on it copies it, and it says
/// "copied" for a moment.
pub fn prompt_chip(ui: &mut egui::Ui, prompt: &str) {
    chip(
        ui,
        &format!("\u{201c}{prompt}\u{201d}"),
        prompt,
        "Copy, then paste it to your agent",
    );
}

/// A command, shown as it is typed rather than quoted as speech, and
/// copied as it is.
pub fn command_chip(ui: &mut egui::Ui, command: &str) {
    chip(
        ui,
        command,
        command,
        "Copy, then use it in your agent's MCP settings",
    );
}

/// The chip both of those wear: `shown` on its face, `copy` to the
/// clipboard, `hover` as its tooltip.
fn chip(ui: &mut egui::Ui, shown: &str, copy: &str, hover: &str) {
    let palette = crate::theme::palette(ui);
    let id = egui::Id::new(("prompt-chip", copy));
    let now = ui.input(|i| i.time);
    let copied = ui
        .ctx()
        .data(|d| d.get_temp::<f64>(id))
        .is_some_and(|t| now - t < COPIED_FOR.as_secs_f64());
    let font = DesignTokens::welcome_screen_body().resolve(ui.style());
    let words = ui
        .painter()
        .layout_no_wrap(shown.to_owned(), font.clone(), palette.text);
    let mark = ui
        .painter()
        .layout_no_wrap("copied".to_owned(), font, palette.link);
    let trailing = if copied { mark.size().x } else { CHIP_ICON };
    let size = egui::vec2(
        CHIP_PAD.x * 2.0 + words.size().x + COPY_ICON_GAP + trailing,
        CHIP_PAD.y * 2.0 + words.size().y.max(CHIP_ICON),
    );
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    if ui.is_rect_visible(rect) {
        let edge = if response.hovered() {
            palette.link
        } else {
            palette.edge
        };
        ui.painter().rect(
            rect,
            8.0,
            palette.canvas,
            egui::Stroke::new(1.0, edge),
            egui::StrokeKind::Inside,
        );
        let text_top = rect.center().y - words.size().y / 2.0;
        ui.painter().galley(
            egui::pos2(rect.left() + CHIP_PAD.x, text_top),
            words.clone(),
            palette.text,
        );
        let right = rect.right() - CHIP_PAD.x;
        if copied {
            ui.painter().galley(
                egui::pos2(right - mark.size().x, rect.center().y - mark.size().y / 2.0),
                mark,
                palette.link,
            );
        } else {
            let icon = egui::Rect::from_center_size(
                egui::pos2(right - CHIP_ICON / 2.0, rect.center().y),
                egui::vec2(CHIP_ICON, CHIP_ICON),
            );
            icons::COPY
                .as_image()
                .tint(palette.muted)
                .paint_at(ui, icon);
        }
    }
    // Painted, so it names itself to a screen reader: a copy button with
    // its words (it was unlabelled until the 2026-10-12 design review).
    response.widget_info(|| {
        egui::WidgetInfo::labeled(egui::WidgetType::Button, true, format!("Copy {shown}"))
    });
    let response = response
        .on_hover_cursor(egui::CursorIcon::PointingHand)
        .on_hover_text(hover);
    if response.clicked() {
        ui.ctx().copy_text(copy.to_owned());
        ui.ctx().data_mut(|d| d.insert_temp(id, now));
    }
    if copied {
        // Repaint once more when the "copied" word should go; never a
        // continuous redraw.
        ui.ctx().request_repaint_after(COPIED_FOR);
    }
}

/// A copy chip's inner margin, the gap before its icon, and the icon's
/// size (named apart from `running::CHIP_GAP`, the gap between run chips).
const CHIP_PAD: egui::Vec2 = egui::vec2(12.0, 8.0);
const COPY_ICON_GAP: f32 = 10.0;
const CHIP_ICON: f32 = 14.0;

/// A first step of a project: what it is, the buttons that do it (the
/// tools whose forms they open, the first one filled), and the words to
/// say to an agent instead.
struct FirstStep {
    title: &'static str,
    body: &'static str,
    actions: &'static [(&'static str, &'static str)],
    prompt: &'static str,
}

/// What an empty project's Overview shows instead of zeros. Each step has
/// its button since the app works without an agent (design review,
/// 2026-10-12); the prompt is the other way in.
const FIRST_STEPS: [FirstStep; 3] = [
    FirstStep {
        title: "Bring in your robot",
        body: "Its model enters as an asset: the parts, the joints, the actuators.",
        actions: &[("Add robot", "onboard_robot")],
        prompt: "Onboard my robot from <path to its MJCF, URDF or USD>",
    },
    FirstStep {
        title: "Give it data",
        body: "Telemetry from the real robot, or a public log of the same robot.",
        actions: &[
            ("Add recording", "ingest_recording"),
            ("Add public log", "ingest_public_log"),
        ],
        prompt: "Ingest my robot's telemetry from <path>",
    },
    FirstStep {
        title: "Fit its dynamics",
        body: "System identification turns the data into a simulation that behaves like \
               the robot, with an interval on every parameter.",
        actions: &[("Identify dynamics", "identify_system")],
        prompt: "Identify my robot's dynamics from that recording",
    },
];

/// An empty project's start: three steps, each with its button and the
/// words for an agent. Returns the tool whose form was asked for. A
/// button whose step needs something the project does not hold yet is
/// off, and says what (`forms::missing_input`).
pub fn start_here(
    ui: &mut egui::Ui,
    forms: &crate::forms::Catalog,
    count: &dyn Fn(&str) -> usize,
) -> Option<String> {
    let palette = crate::theme::palette(ui);
    let mut asked = None;
    card(ui, None).show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        ui.label(
            egui::RichText::new("Start here")
                .text_style(DesignTokens::welcome_screen_h2())
                .strong()
                .color(ui.visuals().strong_text_color()),
        );
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new(
                "Each step is a button here, or say it to your coding agent in your own \
                 words. The stages below light up as each result lands.",
            )
            .text_style(DesignTokens::welcome_screen_body())
            .color(palette.text),
        );
        ui.add_space(16.0);
        for (number, step) in FIRST_STEPS.iter().enumerate() {
            ui.horizontal_top(|ui| {
                step_number(ui, number + 1);
                ui.add_space(10.0);
                ui.vertical(|ui| {
                    ui.label(
                        egui::RichText::new(step.title)
                            .text_style(DesignTokens::welcome_screen_body())
                            .strong()
                            .color(palette.text),
                    );
                    ui.label(
                        egui::RichText::new(step.body)
                            .text_style(DesignTokens::welcome_screen_body())
                            .color(palette.text),
                    );
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        for (i, (label, tool)) in step.actions.iter().enumerate() {
                            let Some(form) = forms.form(tool) else {
                                continue;
                            };
                            let missing = crate::forms::missing_input(form, count);
                            let enabled = missing.is_none();
                            let button = if i == 0 {
                                primary_button(ui, label, enabled)
                            } else {
                                secondary_button(ui, label, enabled)
                            };
                            let hover = missing.unwrap_or_else(|| form.summary.clone());
                            if button
                                .on_hover_text(&hover)
                                .on_disabled_hover_text(&hover)
                                .clicked()
                            {
                                asked = Some((*tool).to_owned());
                            }
                        }
                    });
                    ui.add_space(6.0);
                    ui.horizontal_wrapped(|ui| {
                        ui.label(
                            egui::RichText::new("or ask your agent:")
                                .text_style(DesignTokens::welcome_screen_body())
                                .color(palette.muted),
                        );
                        prompt_chip(ui, step.prompt);
                    });
                });
            });
            ui.add_space(16.0);
        }
        ui.label(
            egui::RichText::new(
                "Then define an environment, train, evaluate and deploy: each page has its \
                 buttons, and your agent knows every step.",
            )
            .text_style(DesignTokens::welcome_screen_body())
            .color(palette.text),
        );
    });
    asked
}

/// A step number's disc, across, and the gap before its words.
const STEP_DISC: f32 = 24.0;
const STEP_TEXT_GAP: f32 = 6.0;

/// A step's number in a soft disc.
fn step_number(ui: &mut egui::Ui, n: usize) {
    let palette = crate::theme::palette(ui);
    let size = egui::vec2(STEP_DISC, STEP_DISC);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::hover());
    response.widget_info(|| {
        egui::WidgetInfo::labeled(egui::WidgetType::Label, true, format!("Step {n}"))
    });
    ui.painter()
        .circle_filled(rect.center(), size.x / 2.0, palette.accent_soft);
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        n.to_string(),
        egui::FontId::proportional(13.0),
        // The strongest text: the body grey on the dark selection blue
        // read 1.6:1 (design review, 2026-10-12).
        ui.visuals().strong_text_color(),
    );
}

/// A sample project, as the Welcome and Projects pages offer it: mirrored
/// from `trainnr/samples.py::SAMPLES` (pinned by
/// `tests/test_studio_mirrors.py`); the download and its checks are the
/// Python side's (`trainnr sample open`).
pub struct SampleCard {
    pub name: &'static str,
    pub title: &'static str,
    pub summary: &'static str,
    /// The folder it unpacks to in the projects home.
    pub project: &'static str,
    pub bytes: u64,
    /// The archive's SHA-256, and where it is (`trainnr/samples.py`'s
    /// dataset at its pinned commit): the Studio opens it itself.
    pub sha256: &'static str,
    pub url: &'static str,
}

pub const SAMPLES: &[SampleCard] = &[SampleCard {
    name: "go2-walk",
    title: "Go2 walk",
    summary: "A Unitree Go2 through the whole loop: identified from a public log, a walk declared and accepted, trained by reinforcement, evaluated in seven conditions, exported and gated against Unitree's own runtime in simulation, and checked for drift.",
    project: "go2-walk-sample",
    bytes: 12_555_361,
    sha256: "8b0b3fb5a029c41ad674984c6b152df9ae0d637abc210c57d6a3dca8ad182cc5",
    url: "https://huggingface.co/datasets/trainnr/trainnr-artifacts/resolve/e5879a2709763f23967fff911e3fbb39d9e6a074/samples/go2-walk-sample.tar.gz",
}];

/// Where a button's `trainnr` command stands (a sample opening, a new
/// project): the one running, or the last failure (its key and why). One
/// at a time: the buttons are disabled while one runs.
#[derive(Default)]
pub struct ActionState {
    pub running: Option<String>,
    pub failed: Option<(String, String)>,
    /// The last command that finished well: its key and a word on it.
    pub done: Option<(String, String)>,
    /// What the engine's setup is doing while the running command waits
    /// on it (`engine.rs`): shown in place of the button's own word.
    pub progress: Option<String>,
    /// The running download's progress and its cancel: the sample's, or
    /// the engine's while a command waits on its setup.
    pub transfer: Option<std::sync::Arc<crate::engine::Transfer>>,
}

/// What a running button shows in its place: the download's bar and
/// Cancel while one runs, else a spinner and its word.
pub fn busy_line(ui: &mut egui::Ui, actions: &ActionState, own: &str) {
    match &actions.transfer {
        Some(transfer) => transfer_row(ui, actions.busy_word(own), transfer),
        None => {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(
                    egui::RichText::new(actions.busy_word(own))
                        .color(crate::theme::palette(ui).text),
                );
            });
        }
    }
}

impl ActionState {
    /// Whether the command under `key` is the one running.
    pub fn is_running(&self, key: &str) -> bool {
        self.running.as_deref() == Some(key)
    }

    /// What a running button says: the engine's step while it sets up,
    /// else its own word.
    pub fn busy_word<'a>(&'a self, own: &'a str) -> &'a str {
        self.progress.as_deref().unwrap_or(own)
    }
}

/// The key a new project's `trainnr call create_project` runs under.
pub const NEW_PROJECT: &str = "new-project";

/// What a click on the Welcome or Projects page asks for.
pub enum Pick {
    Project(PathBuf),
    Sample(&'static str),
    /// The New project dialog, to name it.
    NewProject,
    /// The Connect your agent panel.
    ConnectAgent,
}

/// What the New project dialog answered this frame.
pub enum Dialog {
    Open,
    Create(String),
    Cancel,
}

/// The New project dialog: a name, Create and Cancel; Enter creates and
/// Escape cancels. The name is checked by `create_project` itself, which
/// refuses by name, so the rule lives in one place.
pub fn new_project_dialog(ctx: &egui::Context, name: &mut String) -> Dialog {
    let mut answer = Dialog::Open;
    let modal = egui::Modal::new(egui::Id::new("new-project")).show(ctx, |ui| {
        let palette = crate::theme::palette(ui);
        ui.set_width(NEW_PROJECT_DIALOG_WIDTH);
        ui.label(
            egui::RichText::new("New project")
                .text_style(DesignTokens::welcome_screen_h2())
                .strong()
                .color(ui.visuals().strong_text_color()),
        );
        ui.add_space(6.0);
        ui.label(
            egui::RichText::new("One project per robot effort. Name it for the robot or the task.")
                .color(palette.text),
        );
        ui.add_space(12.0);
        let field = ui.add(
            egui::TextEdit::singleline(name)
                .hint_text("my-robot")
                .desired_width(f32::INFINITY),
        );
        // Focus once, when the dialog opens: asking every frame took the
        // focus back the frame Enter released it, and Enter never created.
        if ui.memory(|m| m.focused().is_none()) {
            field.request_focus();
        }
        // Enter in the field creates; Enter on Cancel cancels. egui 0.36
        // keeps the field focused the frame Enter is pressed (measured in
        // the test below), so the field's focus, kept or just given up,
        // tells the two apart.
        let in_field = field.has_focus() || field.lost_focus();
        let entered = in_field && ui.input(|i| i.key_pressed(egui::Key::Enter));
        ui.add_space(14.0);
        egui::Sides::new().height(BUTTON_HEIGHT).show(
            ui,
            |_ui| {},
            |ui| {
                let ready = !name.trim().is_empty();
                if primary_button(ui, "Create", ready).clicked() || (entered && ready) {
                    answer = Dialog::Create(name.trim().to_owned());
                }
                if secondary_button(ui, "Cancel", true).clicked() {
                    answer = Dialog::Cancel;
                }
            },
        );
    });
    if crate::widgets::escape_closes(&modal) && matches!(answer, Dialog::Open) {
        answer = Dialog::Cancel;
    }
    answer
}

/// A download under way: its step in words, a bar when its size is known
/// (a spinner when it is not), the megabytes so far, and Cancel. The work
/// stops at its next look at the flag; until then Cancel says so.
pub fn transfer_row(ui: &mut egui::Ui, step: &str, transfer: &crate::engine::Transfer) {
    let palette = crate::theme::palette(ui);
    ui.label(egui::RichText::new(step).color(palette.text));
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        match transfer.fraction() {
            Some(done) => {
                ui.add(
                    egui::ProgressBar::new(done.min(1.0))
                        .desired_width(TRANSFER_BAR_WIDTH)
                        .fill(palette.link),
                );
            }
            None => {
                ui.spinner();
            }
        }
        if let Some(amount) = transfer.amount() {
            ui.label(egui::RichText::new(amount).color(palette.muted));
        }
        let stopping = transfer.is_cancelled();
        let label = if stopping { "Cancelling…" } else { "Cancel" };
        if secondary_button(ui, label, !stopping).clicked() {
            transfer.cancel();
        }
    });
    ui.ctx().request_repaint_after(TRANSFER_REDRAW);
}

/// A download bar's width: room for the words beside it on a card.
const TRANSFER_BAR_WIDTH: f32 = 220.0;
/// How often a download's bar is redrawn.
const TRANSFER_REDRAW: std::time::Duration = std::time::Duration::from_millis(100);

/// The filled button a page's one main action wears.
pub fn primary_button(ui: &mut egui::Ui, label: &str, enabled: bool) -> egui::Response {
    let palette = crate::theme::palette(ui);
    ui.spacing_mut().button_padding = BUTTON_PADDING;
    let button = egui::Button::new(
        egui::RichText::new(label)
            .text_style(DesignTokens::welcome_screen_body())
            .strong()
            .color(ui.visuals().strong_text_color()),
    )
    .fill(palette.accent_soft)
    .stroke(egui::Stroke::new(1.0, palette.link))
    .corner_radius(8.0)
    .min_size(egui::vec2(0.0, BUTTON_HEIGHT));
    ui.add_enabled(enabled, button)
}

/// The outlined button a page's other actions wear, the primary's size.
pub fn secondary_button(ui: &mut egui::Ui, label: &str, enabled: bool) -> egui::Response {
    let palette = crate::theme::palette(ui);
    ui.spacing_mut().button_padding = BUTTON_PADDING;
    let button = egui::Button::new(
        egui::RichText::new(label)
            .text_style(DesignTokens::welcome_screen_body())
            .color(palette.text),
    )
    .fill(palette.surface)
    .stroke(egui::Stroke::new(1.0, palette.edge))
    .corner_radius(8.0)
    .min_size(egui::vec2(0.0, BUTTON_HEIGHT));
    ui.add_enabled(enabled, button)
}

/// A line under a button when the last run under `key` went well.
pub fn done_line(ui: &mut egui::Ui, state: &ActionState, key: &str) {
    if let Some((done, what)) = &state.done {
        if done == key {
            ui.add_space(6.0);
            ui.label(egui::RichText::new(what).color(crate::theme::palette(ui).text));
        }
    }
}

/// A failure line under a button, when the last run under `key` failed.
pub fn failure_line(ui: &mut egui::Ui, state: &ActionState, key: &str, what: &str) {
    if let Some((failed, why)) = &state.failed {
        if failed == key {
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new(format!("{what}: {why}")).color(crate::theme::palette(ui).warn),
            );
        }
    }
}

/// The Welcome page's widest pair of cards side by side; narrower, one
/// above the other.
const WELCOME_TWO_UP: f32 = 760.0;
/// The height of a primary or secondary button, and of the row of them
/// a dialog ends with.
pub const BUTTON_HEIGHT: f32 = 36.0;
/// Room around an Open button's words.
const BUTTON_PADDING: egui::Vec2 = egui::vec2(16.0, 8.0);
/// The two cards' height, so they read as a pair.
const WELCOME_CARD_HEIGHT: f32 = 270.0;
/// The loop's stages in order, by the index's names
/// (`trainnr/project/index.py::STATES`, pinned by
/// `tests/test_studio_mirrors.py`), with the page each one fills.
pub const LOOP_STATES: [(&str, Section); 9] = [
    ("telemetry recorded", Section::Recordings),
    ("asset onboarded", Section::Robots),
    ("system identified", Section::Robots),
    ("environment defined", Section::Environments),
    ("data generated", Section::Datasets),
    ("policy trained", Section::Policies),
    ("policy evaluated", Section::Certificates),
    ("deployment exported", Section::Deployments),
    ("drift monitored", Section::Monitoring),
];
/// What a user says to start their own project.
const START_PROMPT: &str = "Create a project called my-robot";
/// The New project dialog's width.
const NEW_PROJECT_DIALOG_WIDTH: f32 = 380.0;

/// The window before any project exists: what trainnr is, start your own
/// project, or explore a sample; then the projects already there.
pub fn welcome(
    ui: &mut egui::Ui,
    projects: &[ProjectSummary],
    samples: &ActionState,
    home: Option<&Path>,
) -> Option<Pick> {
    let palette = crate::theme::palette(ui);
    let mut picked = None;
    crate::pages::page(ui, |ui| {
        ui.add_space(16.0);
        ui.label(
            egui::RichText::new("Welcome to trainnr")
                .text_style(DesignTokens::welcome_screen_h1())
                .strong()
                .color(ui.visuals().strong_text_color()),
        );
        ui.add_space(6.0);
        ui.label(
            egui::RichText::new(
                "The robotics platform on your desktop: real-to-sim, train, \
                 sim-to-real, and back. Work here, or let your coding agent drive it.",
            )
            .text_style(DesignTokens::welcome_screen_body())
            .color(palette.text),
        );
        ui.add_space(12.0);
        ui.horizontal(|ui| {
            if secondary_button(ui, "Connect your coding agent", true)
                .on_hover_text("Claude Code, Codex, Cursor or any MCP client")
                .clicked()
            {
                picked = Some(Pick::ConnectAgent);
            }
            // Beside the button, one row high either way, so nothing below
            // moves when an agent arrives or leaves.
            let link = crate::agent::link(ui);
            if link.connected() {
                ui.add_space(12.0);
                crate::agent::status_line(ui, &link);
            }
        });
        ui.add_space(22.0);
        let mut create = false;
        let mut open = false;
        let mut start = |ui: &mut egui::Ui| create = start_your_own(ui, samples, home);
        let mut explore = |ui: &mut egui::Ui| open = explore_a_sample(ui, samples, projects);
        if ui.available_width() >= WELCOME_TWO_UP {
            ui.columns(2, |cols| {
                start(&mut cols[0]);
                explore(&mut cols[1]);
            });
        } else {
            start(ui);
            ui.add_space(16.0);
            explore(ui);
        }
        if create {
            picked = Some(Pick::NewProject);
        } else if open {
            picked = Some(Pick::Sample(SAMPLES[0].name));
        }
        ui.add_space(16.0);
        the_loop(ui);
        if !projects.is_empty() {
            ui.add_space(32.0);
            crate::pages::subheading(ui, "Your projects");
            ui.add_space(12.0);
            if let Some(root) = crate::pages::project_grid(ui, projects, Path::new("")) {
                picked = Some(Pick::Project(root));
            }
        }
    });
    picked
}

/// What every project goes through: the loop's stages, numbered, each
/// saying on hover what its page holds.
fn the_loop(ui: &mut egui::Ui) {
    let palette = crate::theme::palette(ui);
    card(ui, None).show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        crate::pages::subheading(ui, "What a project goes through");
        ui.add_space(2.0);
        ui.label(
            egui::RichText::new(
                "Each stage lights up on the project's Overview as its result lands. \
                 A walk trained by reinforcement needs no dataset.",
            )
            .text_style(DesignTokens::welcome_screen_body())
            .color(palette.text),
        );
        ui.add_space(12.0);
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(6.0, 10.0);
            // One row height, the disc's, so every item centres on the same
            // line: the nested rows grew it step by step and the row sloped
            // (design review, 2026-10-12).
            ui.set_row_height(STEP_DISC);
            for (number, (state, section)) in LOOP_STATES.iter().enumerate() {
                if number > 0 {
                    ui.label(
                        egui::RichText::new("›")
                            .text_style(DesignTokens::welcome_screen_body())
                            .color(palette.muted),
                    );
                }
                // A block of the disc's height, its width measured: a nested
                // row grew the wrapped row a little each time.
                let words = crate::pages::stage_label(state);
                let font = DesignTokens::welcome_screen_body().resolve(ui.style());
                let text_w = ui
                    .painter()
                    .layout_no_wrap(words.to_owned(), font, palette.text)
                    .size()
                    .x;
                let size = egui::vec2(STEP_DISC + STEP_TEXT_GAP + text_w, STEP_DISC);
                ui.allocate_ui_with_layout(
                    size,
                    egui::Layout::left_to_right(egui::Align::Center),
                    |ui| {
                        ui.spacing_mut().item_spacing.x = STEP_TEXT_GAP;
                        step_number(ui, number + 1);
                        ui.label(
                            egui::RichText::new(words)
                                .text_style(DesignTokens::welcome_screen_body())
                                .color(palette.text),
                        );
                    },
                )
                .response
                .on_hover_text(section.guide().body);
            }
        });
    });
}

/// The left card: a project of one's own, made here or by the agent.
/// Whether New project was clicked.
fn start_your_own(ui: &mut egui::Ui, actions: &ActionState, home: Option<&Path>) -> bool {
    let palette = crate::theme::palette(ui);
    let mut clicked = false;
    welcome_card(ui, Section::Robots.icon(), "Start your own", |ui| {
        ui.label(
            egui::RichText::new(
                "A project holds one robot effort: its model, its data, and \
                 everything trained from them.",
            )
            .text_style(DesignTokens::welcome_screen_body())
            .color(palette.text),
        );
        ui.add_space(16.0);
        if actions.is_running(NEW_PROJECT) {
            busy_line(ui, actions, "Creating…");
        } else {
            clicked = primary_button(ui, "New project", actions.running.is_none()).clicked();
        }
        failure_line(ui, actions, NEW_PROJECT, "It was not created");
        ui.add_space(16.0);
        ask_your_agent(ui, &[START_PROMPT]);
        if let Some(home) = home {
            ui.add_space(6.0);
            ui.label(
                egui::RichText::new(format!(
                    "Projects live in {}.",
                    crate::widgets::home_relative(&home.display().to_string())
                ))
                .color(palette.text),
            );
        }
    });
    clicked
}

/// The right card: the first sample, with its Open button.
fn explore_a_sample(ui: &mut egui::Ui, samples: &ActionState, projects: &[ProjectSummary]) -> bool {
    let mut clicked = false;
    welcome_card(ui, &icons::PLAY, "Explore a sample", |ui| {
        clicked = sample_body(ui, &SAMPLES[0], samples, projects);
    });
    clicked
}

/// A Welcome card: an icon, a title, and what goes under them, at the
/// pair's height.
fn welcome_card(ui: &mut egui::Ui, icon: &Icon, title: &str, add: impl FnOnce(&mut egui::Ui)) {
    let palette = crate::theme::palette(ui);
    card(ui, None).show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        ui.set_min_height(WELCOME_CARD_HEIGHT);
        // Left-aligned and unjustified: `ui.columns` justifies, which
        // spread a wrapped line's words and stretched the button.
        ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
            ui.horizontal(|ui| {
                icon_at(ui, icon, 22.0, palette.muted);
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(title)
                        .text_style(DesignTokens::welcome_screen_h2())
                        .strong()
                        .color(ui.visuals().strong_text_color()),
                );
            });
            ui.add_space(10.0);
            add(ui);
        });
    });
}

/// A sample on the Projects page: a card with its title, what it shows
/// and its Open button, or a word that it is the project open now.
/// Whether Open was clicked.
pub fn sample_card(
    ui: &mut egui::Ui,
    sample: &SampleCard,
    samples: &ActionState,
    projects: &[ProjectSummary],
    open: &Path,
) -> bool {
    let mut clicked = false;
    card(ui, None).show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        if open.file_name().is_some_and(|n| n == sample.project) {
            sample_text(ui, sample);
            ui.add_space(12.0);
            ui.label(
                egui::RichText::new("This is the project open now.")
                    .color(crate::theme::palette(ui).text),
            );
        } else {
            clicked = sample_body(ui, sample, samples, projects);
        }
    });
    clicked
}

/// A sample's title, its tag, and what it shows.
fn sample_text(ui: &mut egui::Ui, sample: &SampleCard) {
    let palette = crate::theme::palette(ui);
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(sample.title)
                .text_style(DesignTokens::welcome_screen_example_title())
                .strong()
                .color(palette.text),
        );
        tag(ui, "sample");
    });
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(sample.summary)
            .text_style(DesignTokens::welcome_screen_body())
            .color(palette.text),
    );
}

/// A sample's title, summary, and Open (or where its opening stands).
fn sample_body(
    ui: &mut egui::Ui,
    sample: &SampleCard,
    samples: &ActionState,
    projects: &[ProjectSummary],
) -> bool {
    let palette = crate::theme::palette(ui);
    let here = projects
        .iter()
        .any(|p| p.root.file_name().is_some_and(|n| n == sample.project));
    sample_text(ui, sample);
    ui.add_space(16.0);
    let mut clicked = false;
    if samples.is_running(sample.name) {
        let own = if here {
            "Opening…"
        } else {
            "Downloading and checking…"
        };
        busy_line(ui, samples, own);
    } else {
        let label = if here {
            format!("Open the {}", sample.title)
        } else {
            format!(
                "Open the {} · {:.0} MB",
                sample.title,
                sample.bytes as f64 / 1e6
            )
        };
        clicked = primary_button(ui, &label, samples.running.is_none())
            .on_hover_text(if here {
                "Open it: it is in your projects"
            } else {
                "Download it, check its SHA-256, and open it as a project"
            })
            .clicked();
    }
    failure_line(ui, samples, sample.name, "It did not open");
    ui.add_space(8.0);
    ui.label(
        egui::RichText::new(
            "It opens as a project of your own: look through every page, or ask \
             your agent to evaluate or retrain it.",
        )
        .color(palette.text),
    );
    clicked
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The New project dialog stays open while typed into, creates the
    /// trimmed name on Enter, and cancels on Escape; drawn headless.
    #[test]
    fn the_new_project_dialog_creates_on_enter_and_cancels_on_escape() {
        let press = |key| egui::RawInput {
            events: vec![egui::Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..Default::default()
        };
        let frame = |ctx: &egui::Context, input, name: &mut String| {
            let mut answer = Dialog::Open;
            let mut output = ctx.run_ui(input, |ui| answer = new_project_dialog(ui.ctx(), name));
            output.textures_delta.clear(); // no renderer here to apply them
            answer
        };
        let ctx = egui::Context::default();
        re_ui::apply_style_and_install_loaders(&ctx);
        let mut name = "  lab-arm ".to_owned();
        assert!(matches!(
            frame(&ctx, Default::default(), &mut name),
            Dialog::Open
        ));
        assert!(matches!(
            frame(&ctx, Default::default(), &mut name),
            Dialog::Open
        ));
        match frame(&ctx, press(egui::Key::Enter), &mut name) {
            Dialog::Create(made) => assert_eq!(made, "lab-arm"),
            _ => panic!("Enter did not create"),
        }
        let ctx = egui::Context::default();
        re_ui::apply_style_and_install_loaders(&ctx);
        let mut empty = String::new();
        let _ = frame(&ctx, Default::default(), &mut empty);
        assert!(
            matches!(
                frame(&ctx, press(egui::Key::Enter), &mut empty),
                Dialog::Open
            ),
            "an empty name creates nothing"
        );
        assert!(matches!(
            frame(&ctx, press(egui::Key::Escape), &mut empty),
            Dialog::Cancel
        ));
    }

    /// Every page that can be empty says what it is for, and a prompt is
    /// words to paste, never a tool call.
    #[test]
    fn every_page_that_can_be_empty_has_a_guide() {
        // Every page the rail opens: the rail is the one list of them.
        for &section in Section::RAIL.iter().flat_map(|(_, items)| items.iter()) {
            if matches!(section, Section::Overview | Section::Live) {
                continue;
            }
            let guide = section.guide();
            assert!(!guide.title.is_empty(), "{section:?} has no title");
            assert!(
                guide.body.ends_with('.'),
                "{section:?}'s body is not a sentence"
            );
            for prompt in guide.prompts {
                assert!(
                    !prompt.contains('('),
                    "{section:?}: {prompt} reads as a call"
                );
            }
        }
    }
}
