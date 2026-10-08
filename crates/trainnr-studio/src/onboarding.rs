//! What a page shows before it holds anything, and what an empty project's
//! Overview shows instead of a row of zeros: what the page is for, and the
//! words to say to the agent that fills it, each one a chip that copies
//! itself. A fresh-install review (2026-10-09) found every empty page a
//! single faint sentence, and the first project's Overview nine grey chips
//! and four zeros with nothing to do next.

use std::time::Duration;

use re_ui::{icons, DesignTokens, Icon};

use crate::pages::Section;
use crate::widgets::{card, icon_at};

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
