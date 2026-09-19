//! The pages behind the rail: Projects, Overview, one card grid per
//! section, and the empty/problem states. Built from Rerun's parts
//! (`widgets.rs`) and painted with its tokens and text styles, so the
//! Studio reads as one app.
//!
//! Vocabulary (decision 2026-09-09): the field's, not ours — Assets,
//! Environments, Recordings, Datasets, Experiments, Policies,
//! Evaluations, Deployments, Monitoring. The index's internal kind names
//! (`certificate`, `batch`, `task`, `run`…) never reach the screen: the
//! [`KINDS`] table maps each onto its page, its icon and the field's word
//! for one of it, and every painted kind goes through [`kind_word`].
//!
//! Every artifact that has a picture shows it (decision 2026-09-09: a
//! platform shows pictures, not hashes): a robot's rendered model, a
//! batch's first frame, a run's loss curve — written by the indexer into
//! `.index/previews/` and painted here the way Rerun paints its example
//! cards. An artifact without one shows its kind's icon, large and dim.

use re_ui::{icons, DesignTokens, UiExt as _};

use crate::control::{
    Event, BY_AGENT, BY_USER, EVENT_DESELECT, EVENT_OPEN, EVENT_SELECT, EVENT_SHOW, EVENT_TABLE,
    EVENT_TIME,
};
use crate::model::{
    ago, ago_iso, day_label, elapsed, now_epoch, render_value, short_time, split_stamp,
    summary_line, Artifact, Index, Job, Model, UNRECORDED,
};
use crate::widgets::{
    card, grid_columns, icon_at, tag, thumbnail, thumbnail_placeholder, CARD_INNER_MARGIN,
    CARD_RADIUS, THUMBNAIL_ASPECT,
};

/// The environment variable the cloud tools read for a control plane
/// (`rq_pipeline.cloud`): the Compute card reports whether it is set,
/// and never whether the endpoint answers — the Studio does not ask it.
const CLOUD_ENDPOINT_ENV: &str = "TRAINNR_ENDPOINT";

/// The artifact kinds the index writes (`rq_pipeline/project/kinds.py`,
/// `Kind`), as the Studio spells them once.
pub mod kind {
    pub const ROBOT: &str = "robot";
    pub const TASK: &str = "task";
    pub const RECORDING: &str = "recording";
    pub const BATCH: &str = "batch";
    pub const DATASET: &str = "dataset";
    pub const RUN: &str = "run";
    pub const POLICY: &str = "policy";
    pub const CERTIFICATE: &str = "certificate";
    pub const DEPLOY: &str = "deploy";
    pub const DRIFT: &str = "drift";
    pub const FINDING: &str = "finding";
}

/// One artifact kind: its index name, the page that lists it, its icon,
/// and the field's word for one of it (what a card, a tag or a refusal
/// prints — never the index name).
pub struct Kind {
    pub name: &'static str,
    pub section: Section,
    pub icon: &'static re_ui::Icon,
    pub word: &'static str,
}

/// Every kind the index can emit. `Section::kinds`, `for_kind`,
/// `icon_for`, `kind_word` and the KPI row all derive from this table.
pub const KINDS: &[Kind] = &[
    Kind {
        name: kind::ROBOT,
        section: Section::Robots,
        icon: &icons::ENTITY,
        word: "robot",
    },
    Kind {
        name: kind::TASK,
        section: Section::Environments,
        icon: &icons::VIEW_3D,
        word: "environment",
    },
    Kind {
        name: kind::RECORDING,
        section: Section::Recordings,
        icon: &icons::RECORDING,
        word: "recording",
    },
    Kind {
        name: kind::BATCH,
        section: Section::Datasets,
        icon: &icons::DATA_SOURCE,
        word: "generated dataset",
    },
    Kind {
        name: kind::DATASET,
        section: Section::Datasets,
        icon: &icons::DATASET,
        word: "dataset",
    },
    Kind {
        name: kind::RUN,
        section: Section::Experiments,
        icon: &icons::VIEW_TIMESERIES,
        word: "experiment",
    },
    Kind {
        name: kind::POLICY,
        section: Section::Policies,
        icon: &icons::COMPONENT_STATIC,
        word: "policy",
    },
    Kind {
        name: kind::CERTIFICATE,
        section: Section::Certificates,
        icon: &icons::SUCCESS,
        word: "evaluation",
    },
    Kind {
        name: kind::DEPLOY,
        section: Section::Deployments,
        icon: &icons::EXTERNAL_LINK,
        word: "deployment",
    },
    Kind {
        name: kind::DRIFT,
        section: Section::Monitoring,
        icon: &icons::LOOP,
        word: "drift record",
    },
    Kind {
        name: kind::FINDING,
        section: Section::Findings,
        icon: &icons::INFO,
        word: "finding",
    },
];

fn kind_entry(name: &str) -> Option<&'static Kind> {
    KINDS.iter().find(|k| k.name == name)
}

/// The field's word for an index kind name; a kind this build does not
/// know keeps its name (a newer index is not lied about).
pub fn kind_word(name: &str) -> &str {
    kind_entry(name).map_or(name, |k| k.word)
}

/// The page's content column, capped so a wide window does not stretch
/// cards into strips; Rerun's welcome screen does the same.
const MAX_CONTENT_WIDTH: f32 = 1280.0;
const PAGE_MARGIN: f32 = 32.0;
const GRID_GAP: f32 = 20.0;
/// A pipeline stage chip.
const STAGE_MIN_WIDTH: f32 = 116.0;
/// A chip's horizontal margins (10 + 10), and what its icon and spacing
/// add to the label's width when measuring it before placement.
const STAGE_CHIP_MARGIN: f32 = 20.0;
const STAGE_CHIP_EXTRA: f32 = STAGE_CHIP_MARGIN + 16.0 + 8.0 + 4.0;
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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
        ("SIMULATION", &[Section::Live]),
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
            Self::Live => "Simulator",
        }
    }

    /// The internal artifact kinds this section lists. Environments hold
    /// tasks (a task is a scene plus a referee); Datasets hold pressed
    /// batches and exported datasets alike; Experiments are training runs.
    /// The page's name as the control surface spells it (lowercase title);
    /// `Section::parse` reads the same word back.
    pub fn slug(self) -> String {
        self.title().to_lowercase().replace(' ', "-")
    }

    /// A page by its name, as an agent's `open` command spells it; the
    /// older internal name for Evaluations is accepted too.
    pub fn parse(name: &str) -> Option<Self> {
        let wanted = name.trim().to_lowercase();
        if wanted == "certificates" {
            return Some(Self::Certificates);
        }
        if wanted == "live" || wanted == "live view" {
            return Some(Self::Live); // the page's name until 2026-09-09
        }
        Self::RAIL
            .iter()
            .flat_map(|(_, items)| items.iter().copied())
            .find(|s| s.slug() == wanted)
    }

    /// The page that lists artifacts of this kind.
    pub fn for_kind(kind: &str) -> Option<Self> {
        kind_entry(kind).map(|k| k.section)
    }

    /// The index kinds this section lists (Datasets holds two: pressed
    /// batches and exported datasets alike; Experiments are training
    /// runs), from the [`KINDS`] table — no allocation, it runs per frame.
    pub fn kinds(self) -> impl Iterator<Item = &'static str> {
        KINDS
            .iter()
            .filter(move |k| k.section == self)
            .map(|k| k.name)
    }

    /// Whether the section lists artifacts at all (the rail's count).
    pub fn lists_artifacts(self) -> bool {
        self.kinds().next().is_some()
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

    /// The icon for an index kind (Datasets holds two); an unknown kind
    /// gets the empty-entity icon.
    pub fn icon_for(kind: &str) -> &'static re_ui::Icon {
        kind_entry(kind).map_or(&icons::ENTITY_EMPTY, |k| k.icon)
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
    Section::for_kind(kind)
}

// ---------------------------------------------------------------------
// Page frame and type

const SCROLL_TO_TOP: &str = "trainnr.page.scroll_to_top";

/// Ask the next page frame to start at the top — a page opened by name
/// (the agent's door, the sidebar) must not inherit the last scroll.
pub fn scroll_to_top(ctx: &egui::Context) {
    ctx.data_mut(|d| d.insert_temp(egui::Id::new(SCROLL_TO_TOP), true));
}

fn take_scroll_to_top(ctx: &egui::Context) -> bool {
    ctx.data_mut(|d| {
        let id = egui::Id::new(SCROLL_TO_TOP);
        let asked = d.get_temp::<bool>(id).unwrap_or(false);
        d.remove::<bool>(id);
        asked
    })
}

/// A centred, scrolling content column; every page draws inside one.
pub fn page<R>(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            if take_scroll_to_top(ui.ctx()) {
                let top = ui.cursor().min;
                ui.scroll_to_rect_animation(
                    egui::Rect::from_min_size(top, egui::vec2(1.0, 1.0)),
                    Some(egui::Align::TOP),
                    egui::style::ScrollAnimation::none(),
                );
            }
            // The column is capped and centred, and always keeps its
            // margin: in a narrow window the column gives way, not the edge.
            let available = ui.available_width();
            let inset = ((available - MAX_CONTENT_WIDTH) / 2.0).max(PAGE_MARGIN * 0.5);
            let width = (available - 2.0 * inset).min(MAX_CONTENT_WIDTH);
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
/// Where the user asked to go from inside a page: to an artifact by its
/// stamp (a lineage link), or back to where they came from.
#[derive(Default)]
pub struct Nav {
    pub open: Option<String>,
    pub back: bool,
    pub can_back: bool,
    /// Bring the drawer into view this frame (an artifact opened by the
    /// agent, a link, or a card far down the grid).
    pub scroll_to_detail: bool,
    /// The page was just opened: a lone artifact opens its own drawer.
    pub entered: bool,
}

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
    let salt = text.title.to_owned();
    picture_card_with_id(ui, width, picture, icon, text, selected, &salt)
}

/// A card whose egui id comes from `salt`, not its title: two versions
/// of one artifact in a row share a title, and a grid's cells share a Ui.
#[allow(clippy::too_many_arguments)]
fn picture_card_with_id(
    ui: &mut egui::Ui,
    width: f32,
    picture: Option<&std::path::Path>,
    icon: &re_ui::Icon,
    text: CardText<'_>,
    selected: bool,
    salt: &str,
) -> egui::Response {
    let tokens = ui.tokens();
    let thumb_h = width / THUMBNAIL_ASPECT;
    let rect = egui::Rect::from_min_size(
        ui.cursor().min,
        egui::vec2(width, thumb_h + CARD_DESCRIPTION_HEIGHT),
    );
    let response = ui.interact(
        rect,
        ui.id().with(salt).with(width as u32),
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
    let margin = f32::from(CARD_INNER_MARGIN);
    let desc = egui::Rect::from_min_size(
        egui::pos2(rect.min.x + margin, thumb_rect.max.y + 10.0),
        egui::vec2(width - 2.0 * margin, CARD_DESCRIPTION_HEIGHT - 20.0),
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
pub fn projects(ui: &mut egui::Ui, model: &mut Model) -> Option<std::path::PathBuf> {
    // The page is visited, not polled: a walk when it opens (throttled
    // inside), then the cached list.
    model.refresh_projects();
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

pub fn overview(ui: &mut egui::Ui, model: &Model, go_to: &mut Option<(Section, Option<String>)>) {
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

        best_by_condition(ui, model, index, go_to);

        // What is happening comes before what exists: the agent's moves
        // and its jobs, then the newest artifacts.
        ui.columns(2, |cols| {
            activity(&mut cols[0], &model.jobs, &model.events);
            compute(&mut cols[1], model);
        });
        ui.add_space(24.0);
        latest_pictures(ui, model, index, go_to);

        if !index.refused.is_empty() {
            ui.add_space(20.0);
            for r in &index.refused {
                ui.warning_label(format!("{} — {}", r.path, r.reason));
            }
        }
    });
}

/// The newest pictures in the project: what it looks like, one row.
fn latest_pictures(
    ui: &mut egui::Ui,
    model: &Model,
    index: &Index,
    go_to: &mut Option<(Section, Option<String>)>,
) {
    let mut with_pictures: Vec<&Artifact> = index
        .artifacts
        .iter()
        .filter(|a| a.preview.is_some())
        .collect();
    if with_pictures.is_empty() {
        return;
    }
    with_pictures.sort_by(|a, b| b.updated_epoch().total_cmp(&a.updated_epoch()));
    subheading(ui, "Latest");
    let (columns, width) = grid_columns(ui.available_width(), GRID_GAP);
    egui::Grid::new("latest_grid")
        .spacing(egui::vec2(GRID_GAP, GRID_GAP))
        .min_col_width(width)
        .max_col_width(width)
        .show(ui, |ui| {
            for artifact in with_pictures.iter().take(columns) {
                let (name, hash) = split_stamp(&artifact.stamp);
                let facts = summary_line(&artifact.summary)
                    .unwrap_or_else(|| kind_word(&artifact.kind).to_owned());
                let response = picture_card_with_id(
                    ui,
                    width,
                    model.preview_path(artifact).as_deref(),
                    Section::icon_for(&artifact.kind),
                    CardText {
                        title: name,
                        facts: &facts,
                        footer: Some(card_footer(artifact, hash)),
                    },
                    false,
                    &artifact.stamp,
                );
                if response.clicked() {
                    *go_to = section_of(&artifact.kind).map(|s| (s, Some(artifact.stamp.clone())));
                }
            }
        });
    ui.add_space(24.0);
}

/// The best evaluation under each condition, when the project judged
/// its policies under at least two: the comparison the study was run
/// for, answered on the front page.
fn best_by_condition(
    ui: &mut egui::Ui,
    model: &Model,
    index: &Index,
    go_to: &mut Option<(Section, Option<String>)>,
) {
    let evaluations = index.by_kind(kind::CERTIFICATE);
    if !crate::listing::matrix_available(&evaluations) {
        return;
    }
    let mut best: Vec<(String, &Artifact, f32)> = Vec::new();
    for a in &evaluations {
        let Some(condition) = crate::listing::condition_of(a) else {
            continue;
        };
        let Some((k, n)) = crate::listing::successes_of(a) else {
            continue;
        };
        if n == 0 {
            continue; // no trials: no rate to rank by
        }
        let rate = k as f32 / n as f32;
        match best.iter_mut().find(|(c, _, _)| *c == condition) {
            Some(entry) if rate > entry.2 => {
                entry.1 = a;
                entry.2 = rate;
            }
            Some(_) => {}
            None => best.push((condition, a, rate)),
        }
    }
    if best.is_empty() {
        return;
    }
    let (columns, width) = grid_columns(ui.available_width(), GRID_GAP);
    ui.horizontal(|ui| {
        subheading(ui, "Best policy by condition");
        if best.len() > columns {
            ui.add_space(8.0);
            if ui
                .add(egui::Link::new(
                    egui::RichText::new(format!("{} conditions · open the matrix", best.len()))
                        .color(ui.tokens().highlight_color),
                ))
                .clicked()
            {
                crate::listing::set_view(
                    ui.ctx(),
                    Section::Certificates,
                    crate::listing::View::Matrix,
                );
                *go_to = Some((Section::Certificates, None));
            }
        }
    });
    best.truncate(columns);
    egui::Grid::new("best_by_condition")
        .spacing(egui::vec2(GRID_GAP, GRID_GAP))
        .min_col_width(width)
        .max_col_width(width)
        .show(ui, |ui| {
            for (i, (condition, artifact, rate)) in best.iter().enumerate() {
                let policy = crate::listing::policy_of(artifact);
                let facts = format!("{policy} · {:.0}% success", rate * 100.0);
                let response = picture_card_with_id(
                    ui,
                    width,
                    model.preview_path(artifact).as_deref(),
                    Section::icon_for(&artifact.kind),
                    CardText {
                        title: condition,
                        facts: &facts,
                        footer: Some(split_stamp(&artifact.stamp).0.to_owned()),
                    },
                    false,
                    &artifact.stamp,
                );
                if response.clicked() {
                    *go_to = Some((Section::Certificates, Some(artifact.stamp.clone())));
                }
                if (i + 1) % columns == 0 {
                    ui.end_row();
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
    let first_missing = index.states.iter().position(|s| !s.present && s.needed);
    let proved = index.states.iter().filter(|s| s.present).count();
    let needed = index.states.iter().filter(|s| s.needed).count();
    card(ui, None).show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        ui.horizontal(|ui| {
            subheading(ui, "Sim-to-real pipeline");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                tag(ui, &format!("{proved} of {needed} stages"));
            });
        });
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(6.0, 8.0);
            for (i, state) in index.states.iter().enumerate() {
                let is_next = first_missing == Some(i);
                let (fill, stroke, text_color) = if !state.needed {
                    // A stage this loop never passes through: dim, struck by
                    // its note on hover, never "the next move".
                    (
                        egui::Color32::TRANSPARENT,
                        tokens.native_frame_stroke.color.linear_multiply(0.5),
                        ui.visuals().weak_text_color().linear_multiply(0.6),
                    )
                } else if state.present {
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
                // A frame never wraps itself in a wrapped row (only a label
                // does), so the chip is measured first and the row broken
                // when it would not fit — else it runs past the card.
                let label = stage_label(&state.name);
                let font = DesignTokens::welcome_screen_body().resolve(ui.style());
                let text_w = ui
                    .painter()
                    .layout_no_wrap(label.to_owned(), font, text_color)
                    .size()
                    .x;
                let chip_w = (text_w + STAGE_CHIP_EXTRA).max(STAGE_MIN_WIDTH + STAGE_CHIP_MARGIN);
                // In a wrapped row `available_width` is the whole row; what
                // remains is the distance from the cursor to the right edge.
                let remaining = ui.max_rect().right() - ui.cursor().min.x;
                if remaining < chip_w {
                    ui.end_row();
                }
                let response = egui::Frame::new()
                    .fill(fill)
                    .stroke(egui::Stroke::new(1.0, stroke))
                    .corner_radius(6.0)
                    .inner_margin(egui::Margin::symmetric(10, 6))
                    .show(ui, |ui| {
                        ui.set_min_width(STAGE_MIN_WIDTH);
                        ui.horizontal(|ui| {
                            let icon = if !state.needed {
                                &icons::REMOVE
                            } else if state.present {
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
                if let Some(note) = &state.note {
                    hover.push_str("\nnot needed: ");
                    hover.push_str(note);
                }
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
fn kpi_row(ui: &mut egui::Ui, index: &Index, go_to: &mut Option<(Section, Option<String>)>) {
    let episodes: u64 = index
        .by_kind(kind::BATCH)
        .iter()
        .filter_map(|a| a.summary.get("episodes").and_then(|v| v.as_u64()))
        .sum();
    let count_of = |section: Section| section.kinds().map(|k| index.count(k)).sum::<usize>();
    let cards: [(Section, String, String); 4] = [
        (
            Section::Robots,
            count_of(Section::Robots).to_string(),
            first_name(index, kind::ROBOT).unwrap_or_else(|| "no asset".into()),
        ),
        (
            Section::Datasets,
            count_of(Section::Datasets).to_string(),
            format!("{episodes} episodes"),
        ),
        (
            Section::Experiments,
            count_of(Section::Experiments).to_string(),
            first_name(index, kind::RUN).unwrap_or_else(|| "no experiment".into()),
        ),
        (
            Section::Certificates,
            count_of(Section::Certificates).to_string(),
            first_name(index, kind::CERTIFICATE).unwrap_or_else(|| "none yet".into()),
        ),
    ];
    let gap = GRID_GAP;
    let width = (ui.available_width() - gap * 3.0) / 4.0;
    let inner = width - 2.0 * f32::from(CARD_INNER_MARGIN);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(gap, gap);
        for (section, value, caption) in cards {
            let response = card(ui, None)
                .show(ui, |ui| {
                    ui.set_min_width(inner);
                    ui.set_max_width(inner);
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
                *go_to = Some((section, None));
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

/// What happened in this project, newest first: the agent's tools (the
/// job table) and what the human and the agent did in the window (the
/// event log), in one feed.
fn activity(ui: &mut egui::Ui, jobs: &[Job], events: &[Event]) {
    subheading(ui, "Recent activity");
    card(ui, None).show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        if jobs.is_empty() && events.is_empty() {
            weak_body(
                ui,
                "Nothing yet. Every tool your agent runs, and every move in this window, appears here.",
            );
            return;
        }
        let now = now_epoch();
        let mut lines: Vec<(f64, ActivityLine)> = jobs
            .iter()
            .map(|j| (j.started, ActivityLine::Job(j)))
            .chain(events.iter().map(|e| (e.epoch_seconds(), ActivityLine::Event(e))))
            .collect();
        lines.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (at, line) in lines.iter().take(ACTIVITY_ROWS) {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(ago(*at))
                        .small()
                        .color(ui.visuals().weak_text_color()),
                );
                match line {
                    ActivityLine::Job(job) => {
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
                    }
                    ActivityLine::Event(event) => {
                        let who = match event.by.as_str() {
                            BY_AGENT => "agent",
                            BY_USER => "you",
                            _ => "studio",
                        };
                        ui.label(egui::RichText::new(who).strong());
                        ui.label(
                            egui::RichText::new(event_words(event))
                                .color(ui.visuals().weak_text_color()),
                        );
                    }
                }
            });
        }
    });
}

enum ActivityLine<'a> {
    Job(&'a Job),
    Event(&'a Event),
}

/// An event in a sentence's tail: "opened Findings", "selected wide#3".
fn event_words(event: &Event) -> String {
    let name = |stamp: &Option<String>| {
        stamp
            .as_deref()
            .map(|s| split_stamp(s).0.to_owned())
            .unwrap_or_default()
    };
    let page = |slug: &Option<String>| {
        slug.as_deref()
            .and_then(Section::parse)
            .map(|s| s.title().to_owned())
            .unwrap_or_else(|| slug.clone().unwrap_or_default())
    };
    match event.kind.as_str() {
        EVENT_OPEN if event.project.is_some() => {
            format!(
                "opened project {}",
                event.project.clone().unwrap_or_default()
            )
        }
        EVENT_OPEN => format!("opened {}", page(&event.section)),
        EVENT_SELECT => format!("selected {}", name(&event.artifact)),
        EVENT_DESELECT => "closed the drawer".to_owned(),
        EVENT_SHOW => format!("showed {} in the viewer", name(&event.artifact)),
        EVENT_TABLE => match &event.table {
            Some(t) => format!("explored the {t} table"),
            None => "closed the table".to_owned(),
        },
        EVENT_TIME => "moved the time cursor".to_owned(),
        other => other.to_owned(),
    }
}

/// The cloud line of the Compute card: the endpoint this window's
/// environment names, or that none is named. Whether it answers is not
/// known here, and not claimed.
fn cloud_line() -> String {
    match std::env::var(CLOUD_ENDPOINT_ENV) {
        Ok(url) if !url.trim().is_empty() => {
            format!(
                "Cloud: {} (from {CLOUD_ENDPOINT_ENV}; reachability {UNRECORDED} here)",
                url.trim()
            )
        }
        _ => format!("Cloud: {CLOUD_ENDPOINT_ENV} is not set; jobs run on this machine."),
    }
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
        weak_body(ui, cloud_line());
    });
}

/// What a picture card says under its facts: the kind in the field's
/// word, the version hash, and when it last changed.
fn card_footer(artifact: &Artifact, hash: &str) -> String {
    format!(
        "{} · @{hash} · {}",
        kind_word(&artifact.kind),
        ago_iso(artifact.updated.as_deref())
    )
}

// ---------------------------------------------------------------------
// Section pages: a card grid, then the selected artifact's detail

/// One section as a grid of picture cards (Rerun's example-card shape):
/// the artifact's preview or its kind's icon, the name, its facts, the
/// version hash. Click to select; the detail drawer opens beneath.
/// The page's artifacts in the order it lists them: newest first, then
/// by stamp (the order the arrow keys walk).
pub fn ordered(model: &Model, section: Section) -> Vec<String> {
    let Some(index) = model.index() else {
        return Vec::new();
    };
    let mut rows: Vec<&Artifact> = section.kinds().flat_map(|k| index.by_kind(k)).collect();
    sort_newest(&mut rows);
    rows.into_iter().map(|a| a.stamp.clone()).collect()
}

fn sort_newest(rows: &mut [&Artifact]) {
    rows.sort_by(|a, b| {
        b.updated_epoch()
            .total_cmp(&a.updated_epoch())
            .then_with(|| a.stamp.cmp(&b.stamp))
    });
}

pub fn section(
    ui: &mut egui::Ui,
    model: &Model,
    section: Section,
    selected: &mut Option<String>,
    show: &mut Option<String>,
    explore: &mut Option<crate::detail::Section>,
    nav: &mut Nav,
) {
    let Some(index) = model.index() else {
        return;
    };
    let scroll_to_detail = nav.scroll_to_detail;
    let entered = nav.entered;
    let mut rows: Vec<&Artifact> = section.kinds().flat_map(|k| index.by_kind(k)).collect();
    // Newest first: what changed last is what the user came to see.
    sort_newest(&mut rows);
    page(ui, |ui| {
        let view = ui
            .horizontal(|ui| {
                if nav.can_back && ui.small_button("← back").clicked() {
                    nav.back = true;
                }
                heading(ui, section.title());
                ui.add_space(8.0);
                tag(ui, &rows.len().to_string());
                if rows.is_empty() {
                    crate::listing::View::Cards
                } else {
                    crate::listing::view_switch(
                        ui,
                        section,
                        crate::listing::matrix_available(&rows),
                    )
                }
            })
            .inner;
        ui.add_space(16.0);
        if rows.is_empty() {
            card(ui, None).show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                weak_body(ui, section.empty_hint());
            });
            return;
        }
        // A page with one artifact is that artifact: its drawer opens on
        // arrival (a click still closes it).
        if entered && selected.is_none() && rows.len() == 1 {
            *selected = Some(rows[0].stamp.clone());
        }
        let mut scroll = scroll_to_detail;
        let mut drawer = Drawer { show, explore, nav };
        match view {
            crate::listing::View::Cards => {
                card_rows(ui, model, &rows, selected, &mut scroll, &mut drawer);
            }
            crate::listing::View::Table | crate::listing::View::Matrix => {
                let clicked = if view == crate::listing::View::Table {
                    crate::listing::table(ui, section, &rows, selected.as_deref())
                } else {
                    crate::listing::matrix(ui, &rows, selected.as_deref())
                };
                if let Some(stamp) = clicked {
                    *selected = if selected.as_deref() == Some(stamp.as_str()) {
                        None
                    } else {
                        scroll = true;
                        Some(stamp)
                    };
                }
                if let Some(stamp) = selected.clone() {
                    if let Some(artifact) = rows.iter().find(|a| a.stamp == stamp) {
                        ui.add_space(GRID_GAP);
                        if scroll {
                            scroll_here(ui);
                        }
                        drawer.open(ui, model, artifact, &stamp);
                    }
                }
            }
        }
    });
}

/// The outputs a drawer can produce, threaded through the page.
struct Drawer<'a> {
    show: &'a mut Option<String>,
    explore: &'a mut Option<crate::detail::Section>,
    nav: &'a mut Nav,
}

impl Drawer<'_> {
    fn open(&mut self, ui: &mut egui::Ui, model: &Model, artifact: &Artifact, stamp: &str) {
        let (show_clicked, table) = detail(ui, model, artifact, self.nav);
        if show_clicked {
            *self.show = Some(stamp.to_owned());
        }
        if table.is_some() {
            *self.explore = table;
        }
    }
}

/// Instant, not animated: the agent's capture on the next frame must
/// already be looking at what was scrolled to.
fn scroll_here(ui: &mut egui::Ui) {
    let top = ui.cursor().min;
    ui.scroll_to_rect_animation(
        egui::Rect::from_min_size(top, egui::vec2(1.0, 1.0)),
        Some(egui::Align::TOP),
        egui::style::ScrollAnimation::none(),
    );
}

/// The cards, grouped by the day they last changed, one grid per row so
/// the drawer can sit right under the row that holds the selected card —
/// never below hundreds of cards.
fn card_rows(
    ui: &mut egui::Ui,
    model: &Model,
    rows: &[&Artifact],
    selected: &mut Option<String>,
    scroll: &mut bool,
    drawer: &mut Drawer<'_>,
) {
    let (columns, width) = grid_columns(ui.available_width(), GRID_GAP);
    let mut groups: Vec<(String, Vec<&Artifact>)> = Vec::new();
    for a in rows {
        let label = a
            .updated
            .as_deref()
            .and_then(crate::model::epoch_of)
            .map(day_label)
            .unwrap_or_else(|| "undated".to_owned());
        match groups.last_mut() {
            Some((day, members)) if *day == label => members.push(a),
            _ => groups.push((label, vec![a])),
        }
    }
    let mut clicked: Option<String> = None;
    let mut row_index = 0usize;
    for (day, members) in &groups {
        if row_index > 0 {
            ui.add_space(GRID_GAP);
        }
        ui.label(
            egui::RichText::new(day)
                .text_style(DesignTokens::welcome_screen_tag())
                .color(ui.visuals().weak_text_color()),
        );
        ui.add_space(6.0);
        for row in members.chunks(columns) {
            if row_index > 0 {
                ui.add_space(GRID_GAP);
            }
            egui::Grid::new(("section_grid", row_index))
                .spacing(egui::vec2(GRID_GAP, GRID_GAP))
                .min_col_width(width)
                .max_col_width(width)
                .show(ui, |ui| {
                    for artifact in row {
                        let (name, hash) = split_stamp(&artifact.stamp);
                        let facts = summary_line(&artifact.summary)
                            .unwrap_or_else(|| artifact.path.clone());
                        let is_selected = selected.as_deref() == Some(artifact.stamp.as_str());
                        let response = picture_card_with_id(
                            ui,
                            width,
                            model.preview_path(artifact).as_deref(),
                            Section::icon_for(&artifact.kind),
                            CardText {
                                title: name,
                                facts: &facts,
                                footer: Some(card_footer(artifact, hash)),
                            },
                            is_selected,
                            &artifact.stamp,
                        );
                        if response.clicked() {
                            clicked = Some(artifact.stamp.clone());
                        }
                    }
                    ui.end_row();
                });
            row_index += 1;
            // A click on this row settles the selection before the drawer
            // is placed, so the drawer opens under the card in the same frame.
            if let Some(stamp) = clicked.take() {
                *selected = if selected.as_deref() == Some(stamp.as_str()) {
                    None
                } else {
                    *scroll = true;
                    Some(stamp)
                };
            }
            let Some(stamp) = selected.clone() else {
                continue;
            };
            let Some(artifact) = row.iter().find(|a| a.stamp == stamp) else {
                continue;
            };
            ui.add_space(GRID_GAP);
            if *scroll {
                scroll_here(ui);
            }
            drawer.open(ui, model, artifact, &stamp);
        }
    }
}

/// The selected artifact in full — its picture large, what the index
/// knows: its kind and name, where it lives, every stamp it cites, its
/// facts. The rich per-kind views (a fit's intervals, a certificate's
/// funnel) read the artifact itself and arrive with the tools that write
/// them.
fn detail(
    ui: &mut egui::Ui,
    model: &Model,
    artifact: &Artifact,
    nav: &mut Nav,
) -> (bool, Option<crate::detail::Section>) {
    let (name, hash) = split_stamp(&artifact.stamp);
    let mut show = false;
    let mut explore = None;
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
                    tag(ui, kind_word(&artifact.kind));
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
                ui.horizontal(|ui| {
                    weak_body(ui, "added");
                    weak_body(ui, ago_iso(artifact.created.as_deref()));
                    weak_body(ui, "· updated");
                    weak_body(ui, ago_iso(artifact.updated.as_deref()));
                });
                if !artifact.cites.is_empty() {
                    ui.add_space(10.0);
                    ui.label(egui::RichText::new("Provenance").strong());
                    lineage_grid(ui, ("cites", &artifact.stamp), model, &artifact.cites, nav);
                }
                if !artifact.cited_by.is_empty() {
                    ui.add_space(10.0);
                    ui.label(egui::RichText::new("Used by").strong());
                    used_by(ui, model, artifact, nav);
                }
            });
        });
    });
    // The artifact itself, in the field's terms: the sections the Python
    // side wrote for it (detail.rs), parsed once and re-read when the
    // file moves. Falls back to the index's summary for a kind that has
    // no detail writer yet.
    match model.detail(artifact) {
        Some(detail) => {
            ui.add_space(14.0);
            explore = crate::detail::show(ui, &detail, &artifact.stamp);
        }
        None if !artifact.summary.is_empty() => {
            ui.add_space(14.0);
            card(ui, None).show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.label(egui::RichText::new("Summary").strong());
                fact_grid(ui, ("summary", &artifact.stamp), &artifact.summary);
            });
        }
        None => {}
    }
    (show, explore)
}

/// The cites as a two-column grid; a stamp that names an artifact in
/// this project is a link that opens it (the back button returns).
fn lineage_grid(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash + std::fmt::Debug,
    model: &Model,
    map: &serde_json::Map<String, serde_json::Value>,
    nav: &mut Nav,
) {
    egui::Grid::new(id)
        .num_columns(2)
        .spacing([16.0, 4.0])
        .show(ui, |ui| {
            for (key, value) in map {
                ui.label(
                    egui::RichText::new(crate::detail::field_word(key))
                        .color(ui.visuals().weak_text_color()),
                );
                let text = value.as_str().unwrap_or_default();
                stamp_link(ui, model, text, false, nav);
                ui.end_row();
            }
        });
}

/// A stamp as a link when the project holds it (`short`: its name only,
/// the version on hover), in monospace otherwise; `unrecorded` in the
/// warning colour.
fn stamp_link(ui: &mut egui::Ui, model: &Model, stamp: &str, short: bool, nav: &mut Nav) {
    if stamp == UNRECORDED {
        ui.label(egui::RichText::new(stamp).color(ui.visuals().warn_fg_color));
    } else if model.artifact(stamp).is_some() {
        let (name, hash) = split_stamp(stamp);
        let text = if short { name } else { stamp };
        if ui
            .add(egui::Link::new(
                egui::RichText::new(text)
                    .monospace()
                    .color(ui.tokens().highlight_color),
            ))
            .on_hover_text(format!("open {name} (version {hash})"))
            .clicked()
        {
            nav.open = Some(stamp.to_owned());
        }
    } else {
        ui.monospace(stamp);
    }
}

/// What was made from this artifact, grouped by kind, each a link.
fn used_by(ui: &mut egui::Ui, model: &Model, artifact: &Artifact, nav: &mut Nav) {
    let mut groups: Vec<(String, Vec<&Artifact>)> = Vec::new();
    for stamp in &artifact.cited_by {
        let Some(a) = model.artifact(stamp) else {
            continue;
        };
        match groups.iter_mut().find(|(kind, _)| *kind == a.kind) {
            Some((_, members)) => members.push(a),
            None => groups.push((a.kind.clone(), vec![a])),
        }
    }
    // One wrapped row per kind (a grid cannot size a wrapped cell, and
    // rows would overlap): the count and the kind, then the links.
    for (kind, members) in &groups {
        let word = Section::for_kind(kind)
            .map(|s| s.title().to_lowercase())
            .unwrap_or_else(|| kind.clone());
        ui.horizontal_wrapped(|ui| {
            ui.label(
                egui::RichText::new(format!("{} {word}", members.len()))
                    .color(ui.visuals().weak_text_color()),
            );
            ui.add_space(8.0);
            for a in members.iter().take(USED_BY_SHOWN) {
                stamp_link(ui, model, &a.stamp, true, nav);
            }
            if members.len() > USED_BY_SHOWN {
                weak_body(ui, format!("… and {} more", members.len() - USED_BY_SHOWN));
            }
        });
    }
}

/// How many links a "Used by" row shows before folding.
const USED_BY_SHOWN: usize = 12;

/// A two-column key/value grid of an artifact's summary (the cites have
/// their own grid, `lineage_grid`, whose values are links).
fn fact_grid(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash + std::fmt::Debug,
    map: &serde_json::Map<String, serde_json::Value>,
) {
    egui::Grid::new(id)
        .num_columns(2)
        .spacing([16.0, 4.0])
        .show(ui, |ui| {
            for (key, value) in map {
                ui.label(egui::RichText::new(key).color(ui.visuals().weak_text_color()));
                ui.label(render_value(value));
                ui.end_row();
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_parse_from_their_names_and_kinds_find_their_page() {
        assert_eq!(Section::parse("robots"), Some(Section::Robots));
        assert_eq!(Section::parse(" Evaluations "), Some(Section::Certificates));
        assert_eq!(Section::parse("certificates"), Some(Section::Certificates));
        assert_eq!(Section::parse("live"), Some(Section::Live));
        assert_eq!(Section::parse("simulator"), Some(Section::Live));
        assert_eq!(Section::Live.slug(), "simulator");
        assert_eq!(Section::parse("dance"), None);
        assert_eq!(Section::for_kind(kind::BATCH), Some(Section::Datasets));
        assert_eq!(
            Section::for_kind(kind::CERTIFICATE),
            Some(Section::Certificates)
        );
        assert_eq!(Section::for_kind("nothing"), None);
        assert_eq!(Section::Overview.slug(), "overview");
    }

    #[test]
    fn every_kind_the_index_can_emit_has_a_section_an_icon_and_a_word() {
        // The kinds `rq_pipeline/project/kinds.py` writes, by name.
        let written = [
            kind::ROBOT,
            kind::RECORDING,
            kind::TASK,
            kind::BATCH,
            kind::DATASET,
            kind::RUN,
            kind::POLICY,
            kind::CERTIFICATE,
            kind::DEPLOY,
            kind::DRIFT,
            kind::FINDING,
        ];
        assert_eq!(KINDS.len(), written.len(), "one table row per kind");
        for name in written {
            assert!(section_of(name).is_some(), "{name} has no section");
            assert!(
                !std::ptr::eq(Section::icon_for(name), &icons::ENTITY_EMPTY),
                "{name} has no icon"
            );
            assert_ne!(kind_word(name), "", "{name} has no word");
        }
        // The index's names never reach the screen where the field has a word.
        assert_eq!(kind_word(kind::CERTIFICATE), "evaluation");
        assert_eq!(kind_word(kind::BATCH), "generated dataset");
        assert_eq!(kind_word(kind::RUN), "experiment");
        assert_eq!(kind_word("something-newer"), "something-newer");
        // A section's kinds come from the same table.
        assert_eq!(
            Section::Datasets.kinds().collect::<Vec<_>>(),
            vec![kind::BATCH, kind::DATASET]
        );
        assert!(!Section::Overview.lists_artifacts());
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
