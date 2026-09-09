//! The pages behind the rail: Projects, Overview, one card grid per
//! section, and the empty/problem states. Built from Rerun's parts
//! (`widgets.rs`) and painted with its tokens and text styles, so the
//! Studio reads as one app.
//!
//! Vocabulary (decision 2026-09-09): the field's, not ours — Assets,
//! Environments, Recordings, Datasets, Experiments, Policies,
//! Certificates, Deployments, Monitoring. Our internal kinds map onto
//! those names in `Section`; the one word that stays ours is
//! "certificate", because nobody else has one.
//!
//! Every artifact that has a picture shows it (decision 2026-09-09: a
//! platform shows pictures, not hashes): a robot's rendered model, a
//! batch's first frame, a run's loss curve — written by the indexer into
//! `.index/previews/` and painted here the way Rerun paints its example
//! cards. An artifact without one shows its kind's icon, large and dim.

use re_ui::{icons, DesignTokens, UiExt as _};

use crate::model::{
    clock, elapsed, render_value, short_time, split_stamp, summary_line, Artifact, Index, Job,
    Model,
};
use crate::widgets::{
    card, grid_columns, icon_at, tag, thumbnail, thumbnail_placeholder, CARD_RADIUS,
    THUMBNAIL_ASPECT,
};

/// The page's content column, capped so a wide window does not stretch
/// cards into strips; Rerun's welcome screen does the same.
const MAX_CONTENT_WIDTH: f32 = 1280.0;
const PAGE_MARGIN: f32 = 32.0;
const GRID_GAP: f32 = 20.0;
/// A pipeline stage chip.
const STAGE_MIN_WIDTH: f32 = 116.0;
/// How many activity rows the Overview shows.
const ACTIVITY_ROWS: usize = 8;
/// Icon sizes: the rail and cards; Rerun's own are 16 and 22.
pub const RAIL_ICON: f32 = 18.0;
const KPI_ICON: f32 = 28.0;
/// One height for the four KPI cards, so the row aligns.
const KPI_CARD_HEIGHT: f32 = 118.0;
/// The description block under a picture card.
const CARD_DESCRIPTION_HEIGHT: f32 = 78.0;
/// The detail drawer's picture width.
const DETAIL_PICTURE_WIDTH: f32 = 320.0;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Section {
    Projects,
    Overview,
    Robots,
    Environments,
    Recordings,
    Datasets,
    Experiments,
    Policies,
    Certificates,
    Deployments,
    Monitoring,
    Findings,
    Live,
}

impl Section {
    pub const RAIL: &'static [(&'static str, &'static [Section])] = &[
        ("", &[Section::Projects, Section::Overview]),
        ("ASSETS", &[Section::Robots, Section::Environments]),
        ("DATA", &[Section::Recordings, Section::Datasets]),
        ("TRAINING", &[Section::Experiments, Section::Policies]),
        ("EVALUATION", &[Section::Certificates, Section::Findings]),
        ("DEPLOYMENT", &[Section::Deployments, Section::Monitoring]),
        ("", &[Section::Live]),
    ];

    pub fn title(self) -> &'static str {
        match self {
            Self::Projects => "Projects",
            Self::Overview => "Overview",
            Self::Robots => "Robots",
            Self::Environments => "Environments",
            Self::Recordings => "Recordings",
            Self::Datasets => "Datasets",
            Self::Experiments => "Experiments",
            Self::Policies => "Policies",
            Self::Certificates => "Evaluations",
            Self::Deployments => "Deployments",
            Self::Monitoring => "Monitoring",
            Self::Findings => "Findings",
            Self::Live => "Live view",
        }
    }

    /// The internal artifact kinds this section lists. Environments hold
    /// tasks (a task is a scene plus a referee); Datasets hold pressed
    /// batches and exported datasets alike; Experiments are training runs.
    pub fn kinds(self) -> &'static [&'static str] {
        match self {
            Self::Robots => &["robot"],
            Self::Environments => &["task"],
            Self::Recordings => &["recording"],
            Self::Datasets => &["batch", "dataset"],
            Self::Experiments => &["run"],
            Self::Policies => &["policy"],
            Self::Certificates => &["certificate"],
            Self::Deployments => &["deploy"],
            Self::Monitoring => &["drift"],
            Self::Findings => &["finding"],
            Self::Projects | Self::Overview | Self::Live => &[],
        }
    }

    pub fn icon(self) -> &'static re_ui::Icon {
        match self {
            Self::Projects => &icons::APPLICATION,
            Self::Overview => &icons::HOME,
            Self::Robots => &icons::ENTITY,
            Self::Environments => &icons::VIEW_3D,
            Self::Recordings => &icons::RECORDING,
            Self::Datasets => &icons::DATASET,
            Self::Experiments => &icons::VIEW_TIMESERIES,
            Self::Policies => &icons::COMPONENT_STATIC,
            Self::Certificates => &icons::SUCCESS,
            Self::Deployments => &icons::EXTERNAL_LINK,
            Self::Monitoring => &icons::LOOP,
            Self::Findings => &icons::INFO,
            Self::Live => &icons::PLAY,
        }
    }

    /// The icon for one of this section's kinds (Datasets holds two).
    pub fn icon_for(kind: &str) -> &'static re_ui::Icon {
        match kind {
            "robot" => &icons::ENTITY,
            "task" => &icons::VIEW_3D,
            "recording" => &icons::RECORDING,
            "batch" => &icons::DATA_SOURCE,
            "dataset" => &icons::DATASET,
            "run" => &icons::VIEW_TIMESERIES,
            "policy" => &icons::COMPONENT_STATIC,
            "certificate" => &icons::SUCCESS,
            "deploy" => &icons::EXTERNAL_LINK,
            "drift" => &icons::LOOP,
            "finding" => &icons::INFO,
            _ => &icons::ENTITY_EMPTY,
        }
    }

    /// The sentence a section shows when it has nothing yet — what the
    /// agent would do to fill it.
    pub fn empty_hint(self) -> &'static str {
        match self {
            Self::Robots => "No assets yet. Ask your agent to onboard a robot model (MJCF or URDF) or record its telemetry.",
            Self::Environments => "No environments yet. Ask your agent to define a task and run acceptance.",
            Self::Recordings => "No recordings yet. Record telemetry from a robot, a ROS 2 bag, a LeRobot dataset, or motion capture.",
            Self::Datasets => "No datasets yet. Ask your agent to generate demonstrations.",
            Self::Experiments => "No experiments yet. Ask your agent to train a policy on a dataset.",
            Self::Policies => "No policies yet. A training experiment leaves checkpoints here.",
            Self::Certificates => "No evaluations yet. Ask your agent to evaluate a policy with paired trials.",
            Self::Deployments => "No deployments yet. Export a deployment manifest and pass the sim-to-sim check.",
            Self::Monitoring => "Nothing monitored yet. Record fresh telemetry and check for parameter drift.",
            Self::Findings => "No findings yet. A finding is a recorded result with its commit, command line and simulator build.",
            Self::Projects => "No projects yet. Ask your agent to run `create_project`.",
            Self::Overview | Self::Live => "",
        }
    }
}

/// The eight loop states, in the field's words, keyed by the index's
/// state names (`rq_pipeline/project/index.py::STATES`).
fn stage_label(name: &str) -> &str {
    match name {
        "telemetry recorded" => "Telemetry",
        "asset onboarded" => "Asset",
        "system identified" => "Sys ID",
        "environment defined" => "Environment",
        "data generated" => "Dataset",
        "policy trained" => "Policy",
        "policy evaluated" => "Evaluation",
        "deployment exported" => "Deployment",
        "drift monitored" => "Monitoring",
        other => other,
    }
}

fn section_of(kind: &str) -> Option<Section> {
    Section::RAIL
        .iter()
        .flat_map(|(_, items)| items.iter().copied())
        .find(|s| s.kinds().contains(&kind))
}

// ---------------------------------------------------------------------
// Page frame and type

/// A centred, scrolling content column; every page draws inside one.
pub fn page<R>(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let available = ui.available_width();
            let width = available.min(MAX_CONTENT_WIDTH);
            let inset = ((available - width) / 2.0).max(PAGE_MARGIN * 0.5);
            ui.horizontal(|ui| {
                ui.add_space(inset);
                ui.vertical(|ui| {
                    ui.set_max_width(width - PAGE_MARGIN);
                    ui.add_space(PAGE_MARGIN * 0.75);
                    let r = add(ui);
                    ui.add_space(PAGE_MARGIN);
                    r
                })
                .inner
            })
            .inner
        })
        .inner
}

pub fn heading(ui: &mut egui::Ui, title: &str) {
    ui.label(
        egui::RichText::new(title)
            .text_style(DesignTokens::welcome_screen_h2())
            .strong(),
    );
}

fn subheading(ui: &mut egui::Ui, title: &str) {
    ui.label(
        egui::RichText::new(title)
            .text_style(DesignTokens::welcome_screen_example_title())
            .strong(),
    );
    ui.add_space(8.0);
}

fn weak_body(ui: &mut egui::Ui, text: impl Into<String>) {
    ui.label(
        egui::RichText::new(text.into())
            .text_style(DesignTokens::welcome_screen_body())
            .color(ui.visuals().weak_text_color()),
    );
}

/// What a picture card says under its picture.
struct CardText<'a> {
    title: &'a str,
    facts: &'a str,
    footer: Option<String>,
}

/// A thumbnail card in Rerun's example-card shape: picture (or the kind's
/// icon, large and dim) over a description block. Clickable.
fn picture_card(
    ui: &mut egui::Ui,
    width: f32,
    picture: Option<&std::path::Path>,
    icon: &re_ui::Icon,
    text: CardText<'_>,
    selected: bool,
) -> egui::Response {
    let tokens = ui.tokens();
    let thumb_h = width / THUMBNAIL_ASPECT;
    let rect = egui::Rect::from_min_size(
        ui.cursor().min,
        egui::vec2(width, thumb_h + CARD_DESCRIPTION_HEIGHT),
    );
    let response = ui.interact(
        rect,
        ui.id().with(text.title).with(width as u32),
        egui::Sense::click(),
    );
    ui.painter()
        .rect_filled(rect, CARD_RADIUS, tokens.example_card_background_color);
    let thumb_rect = egui::Rect::from_min_size(rect.min, egui::vec2(width, thumb_h));
    match picture {
        Some(path) => thumbnail(ui, path, thumb_rect),
        None => thumbnail_placeholder(ui, icon, thumb_rect),
    }
    let stroke = if selected {
        egui::Stroke::new(1.5, tokens.highlight_color)
    } else if response.hovered() {
        egui::Stroke::new(1.0, tokens.highlight_color.linear_multiply(0.6))
    } else {
        egui::Stroke::new(1.0, tokens.native_frame_stroke.color)
    };
    ui.painter()
        .rect_stroke(rect, CARD_RADIUS, stroke, egui::StrokeKind::Inside);
    let desc = egui::Rect::from_min_size(
        egui::pos2(rect.min.x + 14.0, thumb_rect.max.y + 10.0),
        egui::vec2(width - 28.0, CARD_DESCRIPTION_HEIGHT - 20.0),
    );
    let mut child = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(desc)
            .layout(egui::Layout::top_down(egui::Align::Min)),
    );
    child.style_mut().interaction.selectable_labels = false;
    child.spacing_mut().item_spacing.y = 2.0;
    child.horizontal(|ui| {
        icon_at(ui, icon, 16.0, tokens.label_button_icon_color);
        ui.add(
            egui::Label::new(
                egui::RichText::new(text.title)
                    .text_style(DesignTokens::welcome_screen_example_title())
                    .strong(),
            )
            .truncate(),
        );
    });
    child.add(
        egui::Label::new(
            egui::RichText::new(text.facts)
                .text_style(DesignTokens::welcome_screen_body())
                .color(ui.visuals().weak_text_color()),
        )
        .truncate(),
    );
    if let Some(footer) = text.footer {
        child.add(
            egui::Label::new(
                egui::RichText::new(footer)
                    .text_style(DesignTokens::welcome_screen_tag())
                    .color(ui.visuals().weak_text_color()),
            )
            .truncate(),
        );
    }
    ui.advance_cursor_after_rect(rect);
    response
}

// ---------------------------------------------------------------------
// The problem page (no project / no index)

pub fn problem_page(ui: &mut egui::Ui, problem: &str) {
    page(ui, |ui| {
        ui.label(
            egui::RichText::new("No project open")
                .text_style(DesignTokens::welcome_screen_h1())
                .strong(),
        );
        ui.add_space(12.0);
        for line in problem.lines() {
            weak_body(ui, line);
        }
    });
}

// ---------------------------------------------------------------------
// Projects

/// Every project under `projects/` as a picture card: its first artifact's
/// picture, its name, its stage progress and artifact count. Returns a
/// project root to switch to when one is clicked.
pub fn projects(ui: &mut egui::Ui, model: &Model) -> Option<std::path::PathBuf> {
    let list = model.projects();
    let mut switch_to = None;
    page(ui, |ui| {
        ui.horizontal(|ui| {
            heading(ui, "Projects");
            ui.add_space(8.0);
            tag(ui, &list.len().to_string());
        });
        ui.add_space(4.0);
        weak_body(
            ui,
            "One directory per effort under projects/. Your agent makes one with `create_project` \
             and points every tool at it with TRAINNR_PROJECT.",
        );
        ui.add_space(20.0);
        if list.is_empty() {
            card(ui, None).show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                weak_body(ui, Section::Projects.empty_hint());
            });
            return;
        }
        let (columns, width) = grid_columns(ui.available_width(), GRID_GAP);
        egui::Grid::new("projects_grid")
            .spacing(egui::vec2(GRID_GAP, GRID_GAP))
            .min_col_width(width)
            .max_col_width(width)
            .show(ui, |ui| {
                for (i, project) in list.iter().enumerate() {
                    let current = project.root == model.project_root;
                    let facts = if project.stages > 0 {
                        format!(
                            "{} of {} stages · {} artifacts",
                            project.stages_proved, project.stages, project.artifacts
                        )
                    } else {
                        "not indexed yet".to_owned()
                    };
                    let footer = (!project.indexed.is_empty())
                        .then(|| format!("indexed {}", short_time(&project.indexed)));
                    let response = picture_card(
                        ui,
                        width,
                        project.preview.as_deref(),
                        Section::Projects.icon(),
                        CardText {
                            title: &project.name,
                            facts: &facts,
                            footer,
                        },
                        current,
                    );
                    if response.clicked() && !current {
                        switch_to = Some(project.root.clone());
                    }
                    if (i + 1) % columns == 0 {
                        ui.end_row();
                    }
                }
            });
    });
    switch_to
}

// ---------------------------------------------------------------------
// Overview

pub fn overview(ui: &mut egui::Ui, model: &Model, go_to: &mut Option<Section>) {
    let Some(index) = model.index() else {
        return;
    };
    page(ui, |ui| {
        ui.label(
            egui::RichText::new(&index.project)
                .text_style(DesignTokens::welcome_screen_h1())
                .strong(),
        );
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            weak_body(ui, &index.root);
            weak_body(ui, format!("· indexed {}", short_time(&index.indexed)));
        });
        ui.add_space(24.0);

        pipeline_strip(ui, index);
        ui.add_space(24.0);

        kpi_row(ui, index, go_to);
        ui.add_space(24.0);

        latest_pictures(ui, model, index, go_to);

        ui.columns(2, |cols| {
            activity(&mut cols[0], &model.jobs);
            compute(&mut cols[1], model);
        });

        if !index.refused.is_empty() {
            ui.add_space(20.0);
            for r in &index.refused {
                ui.warning_label(format!("{} — {}", r.path, r.reason));
            }
        }
    });
}

/// The newest pictures in the project: what it looks like, one row.
fn latest_pictures(ui: &mut egui::Ui, model: &Model, index: &Index, go_to: &mut Option<Section>) {
    let with_pictures: Vec<&Artifact> = index
        .artifacts
        .iter()
        .filter(|a| a.preview.is_some())
        .collect();
    if with_pictures.is_empty() {
        return;
    }
    subheading(ui, "Latest");
    let (columns, width) = grid_columns(ui.available_width(), GRID_GAP);
    egui::Grid::new("latest_grid")
        .spacing(egui::vec2(GRID_GAP, GRID_GAP))
        .min_col_width(width)
        .max_col_width(width)
        .show(ui, |ui| {
            for artifact in with_pictures.iter().take(columns) {
                let (name, hash) = split_stamp(&artifact.stamp);
                let facts =
                    summary_line(&artifact.summary).unwrap_or_else(|| artifact.kind.clone());
                let response = picture_card(
                    ui,
                    width,
                    model.preview_path(artifact).as_deref(),
                    Section::icon_for(&artifact.kind),
                    CardText {
                        title: name,
                        facts: &facts,
                        footer: Some(format!("{} · @{hash}", artifact.kind)),
                    },
                    false,
                );
                if response.clicked() {
                    *go_to = section_of(&artifact.kind);
                }
            }
        });
    ui.add_space(24.0);
}

/// The sim-to-real pipeline as one card: eight stage chips in a row,
/// filled where an artifact proves the stage, the first missing one
/// accented; the next move spelled out beneath.
fn pipeline_strip(ui: &mut egui::Ui, index: &Index) {
    let tokens = ui.tokens();
    let first_missing = index.states.iter().position(|s| !s.present);
    let proved = index.states.iter().filter(|s| s.present).count();
    card(ui, None).show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        ui.horizontal(|ui| {
            subheading(ui, "Sim-to-real pipeline");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                tag(ui, &format!("{proved} of {} stages", index.states.len()));
            });
        });
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(6.0, 8.0);
            for (i, state) in index.states.iter().enumerate() {
                let is_next = first_missing == Some(i);
                let (fill, stroke, text_color) = if state.present {
                    (
                        tokens.alert_success.fill,
                        tokens.alert_success.icon,
                        tokens.strong_fg_color,
                    )
                } else if is_next {
                    (
                        egui::Color32::TRANSPARENT,
                        tokens.highlight_color,
                        tokens.highlight_color,
                    )
                } else {
                    (
                        egui::Color32::TRANSPARENT,
                        tokens.native_frame_stroke.color,
                        ui.visuals().weak_text_color(),
                    )
                };
                let response = egui::Frame::new()
                    .fill(fill)
                    .stroke(egui::Stroke::new(1.0, stroke))
                    .corner_radius(6.0)
                    .inner_margin(egui::Margin::symmetric(10, 6))
                    .show(ui, |ui| {
                        ui.set_min_width(STAGE_MIN_WIDTH);
                        ui.horizontal(|ui| {
                            let icon = if state.present {
                                &icons::SUCCESS
                            } else {
                                &icons::FLAG_UNTOGGLED
                            };
                            icon_at(ui, icon, 16.0, text_color);
                            ui.label(
                                egui::RichText::new(stage_label(&state.name))
                                    .text_style(DesignTokens::welcome_screen_body())
                                    .strong()
                                    .color(text_color),
                            );
                        });
                    })
                    .response;
                let mut hover = state.name.clone();
                if !state.proved_by.is_empty() {
                    hover.push_str("\nproved by\n");
                    hover.push_str(&state.proved_by.join("\n"));
                }
                response.on_hover_text(hover);
                if i + 1 < index.states.len() {
                    ui.label(egui::RichText::new("›").color(ui.visuals().weak_text_color()));
                }
            }
        });
        ui.add_space(10.0);
        match &index.next_move {
            Some(next) => {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Next")
                            .text_style(DesignTokens::welcome_screen_body())
                            .strong()
                            .color(tokens.highlight_color),
                    );
                    ui.label(
                        egui::RichText::new(next).text_style(DesignTokens::welcome_screen_body()),
                    );
                });
            }
            None => {
                ui.success_label("Every stage proved — the loop is closed.");
            }
        }
    });
}

/// Four KPI cards: what a platform leads with. Each is a button into
/// its section, with a large icon.
fn kpi_row(ui: &mut egui::Ui, index: &Index, go_to: &mut Option<Section>) {
    let episodes: u64 = index
        .by_kind("batch")
        .iter()
        .filter_map(|a| a.summary.get("episodes").and_then(|v| v.as_u64()))
        .sum();
    let cards: [(Section, String, String); 4] = [
        (
            Section::Robots,
            index.count("robot").to_string(),
            first_name(index, "robot").unwrap_or_else(|| "no asset".into()),
        ),
        (
            Section::Datasets,
            (index.count("batch") + index.count("dataset")).to_string(),
            format!("{episodes} episodes"),
        ),
        (
            Section::Experiments,
            index.count("run").to_string(),
            first_name(index, "run").unwrap_or_else(|| "no experiment".into()),
        ),
        (
            Section::Certificates,
            index.count("certificate").to_string(),
            first_name(index, "certificate").unwrap_or_else(|| "none yet".into()),
        ),
    ];
    let gap = GRID_GAP;
    let width = (ui.available_width() - gap * 3.0) / 4.0;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(gap, gap);
        for (section, value, caption) in cards {
            let response = card(ui, None)
                .show(ui, |ui| {
                    ui.set_min_width(width - 28.0);
                    ui.set_max_width(width - 28.0);
                    ui.set_min_height(KPI_CARD_HEIGHT);
                    ui.set_max_height(KPI_CARD_HEIGHT);
                    ui.horizontal_top(|ui| {
                        ui.vertical(|ui| {
                            weak_body(ui, section.title());
                            ui.label(
                                egui::RichText::new(value)
                                    .text_style(DesignTokens::welcome_screen_h1())
                                    .strong(),
                            );
                            ui.label(
                                egui::RichText::new(caption)
                                    .text_style(DesignTokens::welcome_screen_tag())
                                    .color(ui.visuals().weak_text_color()),
                            );
                        });
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
                            let tint = ui.tokens().label_button_icon_color;
                            icon_at(ui, section.icon(), KPI_ICON, tint);
                        });
                    });
                })
                .response
                .interact(egui::Sense::click());
            if response.clicked() {
                *go_to = Some(section);
            }
            if response.hovered() {
                ui.painter().rect_stroke(
                    response.rect,
                    CARD_RADIUS,
                    egui::Stroke::new(1.0, ui.tokens().highlight_color),
                    egui::StrokeKind::Inside,
                );
            }
        }
    });
}

fn first_name(index: &Index, kind: &str) -> Option<String> {
    index
        .by_kind(kind)
        .first()
        .map(|a| split_stamp(&a.stamp).0.to_owned())
}

/// The job table as an activity feed: newest first, running jobs marked.
fn activity(ui: &mut egui::Ui, jobs: &[Job]) {
    subheading(ui, "Recent activity");
    card(ui, None).show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        if jobs.is_empty() {
            weak_body(ui, "No jobs yet. Every tool your agent runs appears here.");
            return;
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        for job in jobs.iter().take(ACTIVITY_ROWS) {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(clock(job.started))
                        .monospace()
                        .small()
                        .color(ui.visuals().weak_text_color()),
                );
                let (icon, tint) = if job.running() {
                    (&icons::PLAY, ui.tokens().highlight_color)
                } else if job.exit == Some(0) {
                    (&icons::SUCCESS, ui.tokens().success_text_color)
                } else {
                    (&icons::ERROR, ui.visuals().error_fg_color)
                };
                icon_at(ui, icon, 16.0, tint);
                ui.label(egui::RichText::new(&job.tool).strong())
                    .on_hover_text(format!("{}\n{}", job.id, job.argv.join(" ")));
                let status = if job.running() {
                    format!("running · {}", elapsed(now - job.started))
                } else if let Some(code) = job.exit {
                    if code == 0 {
                        "done".to_owned()
                    } else {
                        format!("failed (exit {code})")
                    }
                } else {
                    String::new()
                };
                ui.label(
                    egui::RichText::new(status)
                        .text_style(DesignTokens::welcome_screen_tag())
                        .color(ui.visuals().weak_text_color()),
                );
            });
        }
    });
}

/// Where work runs: local by default; a rented machine when the cloud
/// tools are configured. Honest about what it does not know.
fn compute(ui: &mut egui::Ui, model: &Model) {
    subheading(ui, "Compute");
    card(ui, None).show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        let running = model.running_jobs();
        ui.horizontal(|ui| {
            icon_at(
                ui,
                &icons::SETTINGS,
                16.0,
                ui.tokens().label_button_icon_color,
            );
            ui.label(egui::RichText::new("local").strong());
            tag(ui, std::env::consts::ARCH);
            tag(ui, std::env::consts::OS);
        });
        ui.add_space(4.0);
        weak_body(
            ui,
            if running == 0 {
                "idle · no jobs running".to_owned()
            } else {
                format!("{running} job(s) running")
            },
        );
        ui.add_space(6.0);
        weak_body(
            ui,
            "Cloud: offline. Set TRAINNR_ENDPOINT to reach a control plane.",
        );
    });
}

// ---------------------------------------------------------------------
// Section pages: a card grid, then the selected artifact's detail

/// One section as a grid of picture cards (Rerun's example-card shape):
/// the artifact's preview or its kind's icon, the name, its facts, the
/// version hash. Click to select; the detail drawer opens beneath.
pub fn section(
    ui: &mut egui::Ui,
    model: &Model,
    section: Section,
    selected: &mut Option<String>,
    show: &mut Option<String>,
) {
    let Some(index) = model.index() else {
        return;
    };
    let rows: Vec<&Artifact> = section
        .kinds()
        .iter()
        .flat_map(|k| index.by_kind(k))
        .collect();
    page(ui, |ui| {
        ui.horizontal(|ui| {
            heading(ui, section.title());
            ui.add_space(8.0);
            tag(ui, &rows.len().to_string());
        });
        ui.add_space(16.0);
        if rows.is_empty() {
            card(ui, None).show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                weak_body(ui, section.empty_hint());
            });
            return;
        }
        let (columns, width) = grid_columns(ui.available_width(), GRID_GAP);
        let mut clicked: Option<String> = None;
        egui::Grid::new(("section_grid", section.title()))
            .spacing(egui::vec2(GRID_GAP, GRID_GAP))
            .min_col_width(width)
            .max_col_width(width)
            .show(ui, |ui| {
                for (i, artifact) in rows.iter().enumerate() {
                    let (name, hash) = split_stamp(&artifact.stamp);
                    let facts =
                        summary_line(&artifact.summary).unwrap_or_else(|| artifact.path.clone());
                    let is_selected = selected.as_deref() == Some(artifact.stamp.as_str());
                    let response = picture_card(
                        ui,
                        width,
                        model.preview_path(artifact).as_deref(),
                        Section::icon_for(&artifact.kind),
                        CardText {
                            title: name,
                            facts: &facts,
                            footer: Some(format!("{} · @{hash}", artifact.kind)),
                        },
                        is_selected,
                    );
                    if response.clicked() {
                        clicked = Some(artifact.stamp.clone());
                    }
                    if (i + 1) % columns == 0 {
                        ui.end_row();
                    }
                }
            });
        if let Some(stamp) = clicked {
            *selected = if selected.as_deref() == Some(stamp.as_str()) {
                None
            } else {
                Some(stamp)
            };
        }
        if let Some(stamp) = selected.clone() {
            if let Some(artifact) = rows.iter().find(|a| a.stamp == stamp) {
                ui.add_space(24.0);
                if detail(ui, model, artifact) {
                    *show = Some(stamp);
                }
            }
        }
    });
}

/// The selected artifact in full — its picture large, what the index
/// knows: its kind and name, where it lives, every stamp it cites, its
/// facts. The rich per-kind views (a fit's intervals, a certificate's
/// funnel) read the artifact itself and arrive with the tools that write
/// them.
fn detail(ui: &mut egui::Ui, model: &Model, artifact: &Artifact) -> bool {
    let (name, hash) = split_stamp(&artifact.stamp);
    let mut show = false;
    card(ui, None).show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        ui.horizontal_top(|ui| {
            if let Some(path) = model.preview_path(artifact) {
                let rect = egui::Rect::from_min_size(
                    ui.cursor().min,
                    egui::vec2(
                        DETAIL_PICTURE_WIDTH,
                        DETAIL_PICTURE_WIDTH / THUMBNAIL_ASPECT,
                    ),
                );
                if let Some(uri) = crate::widgets::preview_uri(ui, &path) {
                    egui::Image::new(uri).corner_radius(6.0).paint_at(ui, rect);
                }
                ui.advance_cursor_after_rect(rect);
                ui.add_space(16.0);
            }
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    icon_at(
                        ui,
                        Section::icon_for(&artifact.kind),
                        18.0,
                        ui.tokens().label_button_icon_color,
                    );
                    ui.label(
                        egui::RichText::new(name)
                            .text_style(DesignTokens::welcome_screen_h2())
                            .strong(),
                    );
                    tag(ui, &artifact.kind);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        // The artifact as itself: a robot in 3D, a recording as
                        // plots, a run's curves, a certificate's funnel — in the
                        // viewer, via the presenter (docs/76 §5.2).
                        if ui
                            .primary_button(("▶", " Show in viewer"))
                            .on_hover_text("Stream this artifact into the Live view")
                            .clicked()
                        {
                            show = true;
                        }
                    });
                });
                ui.horizontal(|ui| {
                    weak_body(ui, "version");
                    ui.label(
                        egui::RichText::new(hash)
                            .monospace()
                            .color(ui.visuals().weak_text_color()),
                    );
                });
                weak_body(ui, &artifact.path);
                if !artifact.cites.is_empty() {
                    ui.add_space(10.0);
                    ui.label(egui::RichText::new("Provenance").strong());
                    fact_grid(ui, ("cites", &artifact.stamp), &artifact.cites, true);
                }
            });
        });
    });
    // The artifact itself, in the field's terms: the sections the Python
    // side wrote for it (detail.rs). Falls back to the index's summary
    // for a kind that has no detail writer yet.
    match model
        .detail_path(artifact)
        .and_then(|p| crate::detail::Detail::load(&p))
    {
        Some(detail) => {
            ui.add_space(14.0);
            crate::detail::show(ui, &detail, &artifact.stamp);
        }
        None if !artifact.summary.is_empty() => {
            ui.add_space(14.0);
            card(ui, None).show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.label(egui::RichText::new("Summary").strong());
                fact_grid(ui, ("summary", &artifact.stamp), &artifact.summary, false);
            });
        }
        None => {}
    }
    show
}

/// A two-column key/value grid; stamps in monospace, `unrecorded` in the
/// warning colour when `stamps` is set.
fn fact_grid(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash + std::fmt::Debug,
    map: &serde_json::Map<String, serde_json::Value>,
    stamps: bool,
) {
    egui::Grid::new(id)
        .num_columns(2)
        .spacing([16.0, 4.0])
        .show(ui, |ui| {
            for (key, value) in map {
                // The index writes cites keys in the repo's older vocabulary
                // (frozen by tests); the reader sees the field's word.
                let shown = if stamps {
                    crate::detail::field_word(key)
                } else {
                    key
                };
                ui.label(egui::RichText::new(shown).color(ui.visuals().weak_text_color()));
                if stamps {
                    let text = value.as_str().unwrap_or_default();
                    if text == "unrecorded" {
                        ui.label(egui::RichText::new(text).color(ui.visuals().warn_fg_color));
                    } else {
                        ui.monospace(text);
                    }
                } else {
                    ui.label(render_value(value));
                }
                ui.end_row();
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kind_the_index_can_emit_has_a_section() {
        let kinds = [
            "robot",
            "recording",
            "task",
            "batch",
            "dataset",
            "run",
            "policy",
            "certificate",
            "deploy",
            "drift",
            "finding",
        ];
        for kind in kinds {
            assert!(section_of(kind).is_some(), "{kind} has no section");
        }
        // `fit` rides inside a robot bundle today (no fits section yet).
        assert!(section_of("fit").is_none());
    }

    #[test]
    fn stages_speak_the_fields_words() {
        assert_eq!(stage_label("asset onboarded"), "Asset");
        assert_eq!(stage_label("environment defined"), "Environment");
        assert_eq!(stage_label("drift monitored"), "Monitoring");
        assert_eq!(stage_label("new state"), "new state");
    }
}
