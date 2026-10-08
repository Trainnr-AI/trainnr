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
    let palette = crate::theme::palette(ui);
    let id = egui::Id::new(("prompt-chip", prompt));
    let now = ui.input(|i| i.time);
    let copied = ui
        .ctx()
        .data(|d| d.get_temp::<f64>(id))
        .is_some_and(|t| now - t < COPIED_FOR.as_secs_f64());
    let font = DesignTokens::welcome_screen_body().resolve(ui.style());
    let words = ui.painter().layout_no_wrap(
        format!("\u{201c}{prompt}\u{201d}"),
        font.clone(),
        palette.text,
    );
    let mark = ui
        .painter()
        .layout_no_wrap("copied".to_owned(), font, palette.link);
    let trailing = if copied { mark.size().x } else { CHIP_ICON };
    let size = egui::vec2(
        CHIP_PAD.x * 2.0 + words.size().x + CHIP_GAP + trailing,
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
    let response = response
        .on_hover_cursor(egui::CursorIcon::PointingHand)
        .on_hover_text("Copy, then paste it to your agent");
    if response.clicked() {
        ui.ctx().copy_text(prompt.to_owned());
        ui.ctx().data_mut(|d| d.insert_temp(id, now));
    }
    if copied {
        // Repaint once more when the "copied" word should go; never a
        // continuous redraw.
        ui.ctx().request_repaint_after(COPIED_FOR);
    }
}

/// A chip's inner margin, the gap before its icon, and the icon's size.
const CHIP_PAD: egui::Vec2 = egui::vec2(12.0, 8.0);
const CHIP_GAP: f32 = 10.0;
const CHIP_ICON: f32 = 14.0;

/// The first steps of a project, each with its words: what an empty
/// project's Overview shows instead of zeros.
const FIRST_STEPS: [(&str, &str, &str); 3] = [
    (
        "Bring in your robot",
        "Its model enters as an asset: the parts, the joints, the actuators.",
        "Onboard my robot from <path to its MJCF, URDF or USD>",
    ),
    (
        "Give it data",
        "Telemetry from the real robot, or a public log of the same robot.",
        "Ingest my robot's telemetry from <path>",
    ),
    (
        "Fit its dynamics",
        "System identification turns the data into a simulation that behaves like \
         the robot, with an interval on every parameter.",
        "Identify my robot's dynamics from that recording",
    ),
];

/// An empty project's start: what trainnr does with it, three steps, and
/// where the rest of the loop goes.
pub fn start_here(ui: &mut egui::Ui) {
    let palette = crate::theme::palette(ui);
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
                "trainnr runs through your coding agent: say what you want in your own \
                 words, and each stage above lights up as its result lands.",
            )
            .text_style(DesignTokens::welcome_screen_body())
            .color(palette.text),
        );
        ui.add_space(16.0);
        for (number, (title, body, prompt)) in FIRST_STEPS.iter().enumerate() {
            ui.horizontal_top(|ui| {
                step_number(ui, number + 1);
                ui.add_space(10.0);
                ui.vertical(|ui| {
                    ui.label(
                        egui::RichText::new(*title)
                            .text_style(DesignTokens::welcome_screen_body())
                            .strong()
                            .color(palette.text),
                    );
                    ui.label(
                        egui::RichText::new(*body)
                            .text_style(DesignTokens::welcome_screen_body())
                            .color(palette.text),
                    );
                    ui.add_space(6.0);
                    prompt_chip(ui, prompt);
                });
            });
            ui.add_space(14.0);
        }
        ui.label(
            egui::RichText::new(
                "Then define an environment, train, evaluate and deploy; the agent \
                 knows each step. The README's quickstart walks the same path with \
                 Unitree's Go2 and a public log.",
            )
            .text_style(DesignTokens::welcome_screen_body())
            .color(palette.text),
        );
    });
}

/// A step's number in a soft disc.
fn step_number(ui: &mut egui::Ui, n: usize) {
    let palette = crate::theme::palette(ui);
    let size = egui::vec2(24.0, 24.0);
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    ui.painter()
        .circle_filled(rect.center(), size.x / 2.0, palette.accent_soft);
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        n.to_string(),
        egui::FontId::proportional(13.0),
        palette.text,
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
}

pub const SAMPLES: &[SampleCard] = &[SampleCard {
    name: "go2-walk",
    title: "Go2 walk",
    summary: "A Unitree Go2 through the whole loop: identified from a public log, a walk declared and accepted, trained by reinforcement, evaluated in seven conditions, exported and gated against Unitree's own runtime in simulation, and checked for drift.",
    project: "go2-walk-sample",
    bytes: 11_973_917,
}];

/// Where opening a sample stands: the one being opened, or the last
/// failure (the sample's name and why).
#[derive(Default)]
pub struct SampleState {
    pub opening: Option<&'static str>,
    pub failed: Option<(&'static str, String)>,
}

/// What a click on the Welcome or Projects page asks for.
pub enum Pick {
    Project(PathBuf),
    Sample(&'static str),
}

/// The Welcome page's widest pair of cards side by side; narrower, one
/// above the other.
const WELCOME_TWO_UP: f32 = 760.0;
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

/// The window before any project exists: what trainnr is, start your own
/// project, or explore a sample; then the projects already there.
pub fn welcome(
    ui: &mut egui::Ui,
    projects: &[ProjectSummary],
    samples: &SampleState,
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
                "The end-to-end robotics platform, run from your coding agent: \
                 real-to-sim, train, sim-to-real, and back.",
            )
            .text_style(DesignTokens::welcome_screen_body())
            .color(palette.text),
        );
        ui.add_space(28.0);
        let start = |ui: &mut egui::Ui| start_your_own(ui, home);
        let mut explore = |ui: &mut egui::Ui| {
            if explore_a_sample(ui, samples, projects) {
                picked = Some(Pick::Sample(SAMPLES[0].name));
            }
        };
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
            for (number, (state, section)) in LOOP_STATES.iter().enumerate() {
                if number > 0 {
                    ui.label(egui::RichText::new("›").color(palette.muted));
                }
                ui.horizontal(|ui| {
                    step_number(ui, number + 1);
                    ui.label(
                        egui::RichText::new(crate::pages::stage_label(state))
                            .text_style(DesignTokens::welcome_screen_body())
                            .color(palette.text),
                    );
                })
                .response
                .on_hover_text(section.guide().body);
            }
        });
    });
}

/// The left card: a project of one's own, through the agent.
fn start_your_own(ui: &mut egui::Ui, home: Option<&Path>) {
    let palette = crate::theme::palette(ui);
    welcome_card(ui, Section::Robots.icon(), "Start your own", |ui| {
        ui.label(
            egui::RichText::new(
                "A project holds one robot effort: its model, its data, and \
                 everything trained from them. Your agent makes it, and this \
                 window opens it the moment it exists.",
            )
            .text_style(DesignTokens::welcome_screen_body())
            .color(palette.text),
        );
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
}

/// The right card: the first sample, with its Open button.
fn explore_a_sample(ui: &mut egui::Ui, samples: &SampleState, projects: &[ProjectSummary]) -> bool {
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
    samples: &SampleState,
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
    samples: &SampleState,
    projects: &[ProjectSummary],
) -> bool {
    let palette = crate::theme::palette(ui);
    let here = projects
        .iter()
        .any(|p| p.root.file_name().is_some_and(|n| n == sample.project));
    sample_text(ui, sample);
    ui.add_space(16.0);
    let mut clicked = false;
    if samples.opening == Some(sample.name) {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(
                egui::RichText::new(if here {
                    "Opening…"
                } else {
                    "Downloading and checking…"
                })
                .color(palette.text),
            );
        });
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
        .min_size(egui::vec2(0.0, 36.0));
        clicked = ui
            .add_enabled(samples.opening.is_none(), button)
            .on_hover_text(if here {
                "Open it: it is in your projects"
            } else {
                "Download it, check its SHA-256, and open it as a project"
            })
            .clicked();
    }
    if let Some((name, why)) = &samples.failed {
        if *name == sample.name {
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new(format!("It did not open: {why}"))
                    .color(ui.visuals().warn_fg_color),
            );
        }
    }
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
