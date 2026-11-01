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
    Event, BY_AGENT, BY_USER, EVENT_DESELECT, EVENT_KEY, EVENT_OPEN, EVENT_SELECT, EVENT_SHOW,
    EVENT_TABLE, EVENT_TIME,
};
use crate::model::{
    ago, ago_iso, day_label, elapsed, now_epoch, render_value, short_time, split_stamp,
    summary_line, Artifact, Index, Job, Model, ProjectSummary, BASIS_OWN, UNRECORDED,
};
use crate::widgets::{
    card, grid_columns, icon_at, tag, thumbnail, thumbnail_placeholder, weak_body,
    CARD_INNER_MARGIN, CARD_RADIUS, THUMBNAIL_ASPECT,
};

/// The artifact kinds the index writes (`trainnr/project/kinds.py`,
/// `Kind`), as the Studio spells them once.
pub mod kind {
    pub const ROBOT: &str = "robot";
    pub const TASK: &str = "task";
    pub const SCENE: &str = "scene";
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
        name: kind::SCENE,
        section: Section::Environments,
        icon: &icons::VIEW_3D,
        word: "scene",
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
        // A histogram, not a check: an evaluation is a measurement, and a
        // check on a 0 of 40 read as a pass (design review, 2026-10-12).
        icon: &icons::VIEW_HISTOGRAM,
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
/// The "›" between two stage chips with its spacing, measured.
const STAGE_SEPARATOR_WIDTH: f32 = 20.0;
/// How many activity rows the Overview shows.
const ACTIVITY_ROWS: usize = 8;
/// Icon sizes: the rail and cards; Rerun's own are 16 and 22.
pub const RAIL_ICON: f32 = 18.0;
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
            Self::Certificates => &icons::VIEW_HISTOGRAM,
            Self::Deployments => &icons::EXTERNAL_LINK,
            Self::Monitoring => &icons::LOOP,
            Self::Findings => &icons::VIEW_TEXT,
            Self::Live => &icons::PLAY,
        }
    }

    /// The icon for an index kind (Datasets holds two); an unknown kind
    /// gets the empty-entity icon.
    pub fn icon_for(kind: &str) -> &'static re_ui::Icon {
        kind_entry(kind).map_or(&icons::ENTITY_EMPTY, |k| k.icon)
    }
}

/// The eight loop states, in the field's words, keyed by the index's
/// state names (`trainnr/project/index.py::STATES`).
pub fn stage_label(name: &str) -> &str {
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

/// A drawer action with its own fill and edge: the light theme gives an
/// inactive button no frame, so these read as text beside the filled
/// "Show in viewer" (design review, 2026-10-12).
fn framed_button<'a>(ui: &egui::Ui, label: &'a str) -> egui::Button<'a> {
    let palette = crate::theme::palette(ui);
    egui::Button::new(label)
        .fill(palette.canvas)
        .stroke(egui::Stroke::new(1.0, palette.edge))
        .frame_when_inactive(true)
}

/// The drift verdict that asks for a new fit (`fleet/drift.py`,
/// `DRIFTED_WORD`), and the button and tool its recommendation names.
const DRIFTED: &str = "drifted";
const DRIFT_REMEDY: (&str, &str) = ("Re-identify dynamics", "identify_system");

/// The action an artifact's own verdict recommends, as a button's label
/// and the tool its form runs.
fn remedy(artifact: &Artifact) -> Option<(&'static str, &'static str)> {
    let drifted = artifact.kind == kind::DRIFT
        && artifact.summary.get("verdict").and_then(|v| v.as_str()) == Some(DRIFTED);
    drifted.then_some(DRIFT_REMEDY)
}

/// The index's next move as a person reads it: the tool names it carries
/// for an agent ("(onboard_robot)") left out.
fn without_tool_names(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find(" (") {
        let after = &rest[open + 2..];
        match after.find(')') {
            Some(close)
                if after[..close].contains('_')
                    && after[..close]
                        .chars()
                        .all(|c| c.is_ascii_lowercase() || c == '_') =>
            {
                out.push_str(&rest[..open]);
                rest = &after[close + 1..];
            }
            _ => {
                out.push_str(&rest[..open + 2]);
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
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

pub fn subheading(ui: &mut egui::Ui, title: &str) {
    ui.label(
        egui::RichText::new(title)
            .text_style(DesignTokens::welcome_screen_example_title())
            .strong(),
    );
    ui.add_space(8.0);
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
    /// A viewport scene asked for from the drawer (a deployment played or
    /// replayed in the Studio's own MuJoCo viewport).
    pub scene: Option<String>,
    /// The page opened its lone artifact's drawer on arrival: the Studio's
    /// own move, logged as such rather than as the human's click
    /// (the feed said "you selected go2-walk" for it until 2026-09-28).
    pub auto_selected: bool,
    /// A form button was clicked: the tool whose form to open.
    pub form: Option<String>,
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
/// of one artifact in a row share a title, and a row's cards share a Ui.
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
    let palette = crate::theme::palette(ui);
    ui.painter().rect_filled(rect, CARD_RADIUS, palette.surface);
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
        egui::Stroke::new(1.0, palette.hairline)
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
                .weak(),
        )
        .truncate(),
    );
    if let Some(footer) = text.footer {
        child.add(
            egui::Label::new(
                egui::RichText::new(footer)
                    .text_style(DesignTokens::welcome_screen_tag())
                    .weak(),
            )
            .truncate(),
        );
    }
    ui.advance_cursor_after_rect(rect);
    response
}

/// One row of cards `width` wide, `GRID_GAP` apart. Rows, not an
/// egui::Grid: a new Grid hides its first frame and discards it (egui
/// grid.rs, `request_discard`), so a page drew twice on every first
/// visit. The gap tops up `item_spacing.x`, which the cards' own title
/// rows inherit and must keep. A row further than a screen's height from
/// the view takes its place and builds nothing: no card is laid out, and
/// none asks for its picture, until the row nears the view.
fn card_row<T>(ui: &mut egui::Ui, row: &[T], width: f32, mut card: impl FnMut(&mut egui::Ui, &T)) {
    ui.horizontal(|ui| {
        let size = egui::vec2(
            ui.available_width(),
            width / THUMBNAIL_ASPECT + CARD_DESCRIPTION_HEIGHT,
        );
        let rect = egui::Rect::from_min_size(ui.cursor().min, size);
        if !ui.is_rect_visible(rect.expand2(egui::vec2(0.0, ui.clip_rect().height()))) {
            ui.allocate_space(size);
            return;
        }
        for (i, item) in row.iter().enumerate() {
            if i > 0 {
                ui.add_space((GRID_GAP - ui.spacing().item_spacing.x).max(0.0));
            }
            card(ui, item);
        }
    });
}

/// `items` in rows of `columns` cards `width` wide, `GRID_GAP` apart both
/// ways.
fn card_grid<T>(
    ui: &mut egui::Ui,
    items: &[T],
    columns: usize,
    width: f32,
    mut card: impl FnMut(&mut egui::Ui, &T),
) {
    ui.scope(|ui| {
        ui.spacing_mut().item_spacing.y = GRID_GAP;
        for row in items.chunks(columns.max(1)) {
            card_row(ui, row, width, &mut card);
        }
    });
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
/// picture, its name, its stage progress and artifact count; then the
/// sample projects. Returns what was clicked: a project root to switch
/// to, or a sample to open.
pub fn projects(
    ui: &mut egui::Ui,
    model: &mut Model,
    samples: &crate::onboarding::ActionState,
) -> Option<crate::onboarding::Pick> {
    // The page is visited, not polled: a walk when it opens (throttled
    // inside), then the cached list.
    model.refresh_projects();
    let list = model.projects();
    let mut picked = None;
    page(ui, |ui| {
        ui.horizontal(|ui| {
            heading(ui, "Projects");
            ui.add_space(8.0);
            tag(ui, &list.len().to_string());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if samples.is_running(crate::onboarding::NEW_PROJECT) {
                    ui.label(samples.busy_word("Creating…"));
                    ui.spinner();
                } else {
                    let busy = samples.running.is_some();
                    if crate::onboarding::primary_button(ui, "New project", !busy).clicked() {
                        picked = Some(crate::onboarding::Pick::NewProject);
                    }
                }
            });
        });
        crate::onboarding::failure_line(
            ui,
            samples,
            crate::onboarding::NEW_PROJECT,
            "It was not created",
        );
        ui.add_space(4.0);
        weak_body(
            ui,
            "One project per robot effort. Make one here or ask your agent to, or open one below.",
        );
        ui.add_space(20.0);
        if list.is_empty() {
            crate::onboarding::empty_state(
                ui,
                Section::Projects.icon(),
                &Section::Projects.guide(),
            );
        } else {
            picked =
                project_grid(ui, list, &model.project_root).map(crate::onboarding::Pick::Project);
        }
        ui.add_space(28.0);
        subheading(ui, "Samples");
        ui.add_space(2.0);
        weak_body(
            ui,
            "Finished projects to explore before your own: open one and every page has \
             something on it.",
        );
        ui.add_space(12.0);
        for sample in crate::onboarding::SAMPLES {
            if crate::onboarding::sample_card(ui, sample, samples, list, &model.project_root) {
                picked = Some(crate::onboarding::Pick::Sample(sample.name));
            }
        }
    });
    picked
}

/// The projects as a grid of picture cards, the open one marked; the
/// root of the one clicked, when it is another.
pub fn project_grid(
    ui: &mut egui::Ui,
    list: &[ProjectSummary],
    open: &std::path::Path,
) -> Option<std::path::PathBuf> {
    let mut switch_to = None;
    let (columns, width) = grid_columns(ui.available_width(), GRID_GAP);
    card_grid(ui, list, columns, width, |ui, project| {
        let current = project.root == open;
        let facts = if project.stages > 0 {
            format!(
                "{} of {} stages · {} artifacts",
                project.stages_proved, project.stages, project.artifacts
            )
        } else {
            "not indexed yet".to_owned()
        };
        let footer = (!project.indexed.is_empty())
            .then(|| format!("updated {}", short_time(&project.indexed)));
        let response = picture_card(
            ui,
            width,
            if ui.visuals().dark_mode {
                project.preview.as_deref()
            } else {
                project
                    .preview_light
                    .as_deref()
                    .or(project.preview.as_deref())
            },
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
    });
    switch_to
}

// ---------------------------------------------------------------------
// Overview

/// `open_run`: a chip on the Compute card was clicked (its job id); the
/// shell opens the Running now panel on it.
pub fn overview(
    ui: &mut egui::Ui,
    model: &Model,
    forms: &crate::forms::Catalog,
    go_to: &mut Option<(Section, Option<String>)>,
    open_run: &mut Option<String>,
    open_form: &mut Option<String>,
) {
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
            weak_body(ui, crate::widgets::home_relative(&index.root));
            weak_body(ui, format!("· updated {}", short_time(&index.indexed)));
        });
        ui.add_space(24.0);

        // Nothing yet: what to do first, above the stages it will light
        // (the empty pipeline led the page until 2026-10-12).
        // A template's plan stands in for them, and stays (docs/86 §6).
        if let Some(plan) = &index.template {
            *open_form = crate::onboarding::plan_card(ui, plan, forms, &|k| index.count(k));
            ui.add_space(24.0);
        } else if index.artifacts.is_empty() {
            *open_form = crate::onboarding::start_here(ui, forms, &|k| index.count(k));
            ui.add_space(24.0);
        }
        pipeline_strip(ui, index);
        ui.add_space(24.0);

        // The best policy per condition leads a project with results. The
        // four count tiles above it repeated the rail's counts and were
        // dropped in the 2026-10-12 design review.
        if !index.artifacts.is_empty() {
            best_by_condition(ui, model, index, go_to);
        }

        // What is happening comes before what exists: the agent's moves
        // and its jobs, then the newest artifacts.
        ui.columns(2, |cols| {
            activity(&mut cols[0], &model.jobs, &model.events);
            compute(&mut cols[1], model, open_run);
        });
        ui.add_space(24.0);
        latest_pictures(ui, model, index, go_to);

        if !index.refused.is_empty() {
            ui.add_space(20.0);
            for r in &index.refused {
                ui.warning_label(format!("{}: {}", r.path, r.reason));
            }
        }
    });
}

/// Work still being written (the index's `in_progress`): a spinner, the
/// folder and its stage. An hourglass emoji led this line until the
/// 2026-10-12 design review.
fn in_progress_row(ui: &mut egui::Ui, p: &crate::model::InProgress) {
    ui.horizontal(|ui| {
        ui.add(egui::Spinner::new().size(12.0));
        weak_body(ui, format!("{} · {}", p.path, p.stage));
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
    let latest = &with_pictures[..with_pictures.len().min(columns)];
    card_row(ui, latest, width, |ui, artifact| {
        let (name, hash) = split_stamp(&artifact.stamp);
        let facts =
            summary_line(&artifact.summary).unwrap_or_else(|| kind_word(&artifact.kind).to_owned());
        let response = picture_card_with_id(
            ui,
            width,
            model
                .preview_path(artifact, !ui.visuals().dark_mode)
                .as_deref(),
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
                    egui::RichText::new(format!(
                        "{} · open the matrix",
                        crate::widgets::count_word(best.len(), "condition")
                    ))
                    .color(crate::theme::palette(ui).link),
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
    card_grid(
        ui,
        &best,
        columns,
        width,
        |ui, (condition, artifact, rate)| {
            let policy = crate::listing::policy_of(artifact);
            let facts = format!("{policy} · {:.0}% success", rate * 100.0);
            let response = picture_card_with_id(
                ui,
                width,
                model
                    .preview_path(artifact, !ui.visuals().dark_mode)
                    .as_deref(),
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
        },
    );
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
                        crate::theme::palette(ui).hairline,
                        ui.visuals().weak_text_color().gamma_multiply(0.7),
                    )
                } else if state.present && state.basis.as_deref().is_some_and(|b| b != BASIS_OWN) {
                    // Proved on a basis that is not the operator's own robot (a
                    // public log, a simulation): outlined in the accent, never
                    // the success fill.
                    (
                        egui::Color32::TRANSPARENT,
                        tokens.highlight_color,
                        tokens.strong_fg_color,
                    )
                } else if state.present {
                    (
                        tokens.alert_success.fill,
                        tokens.alert_success.icon,
                        tokens.strong_fg_color,
                    )
                } else if is_next {
                    // The edge in the highlight, the words in the accent
                    // as text: the highlight is a background token and
                    // read 1.4:1 as text (design review, 2026-10-12).
                    (
                        egui::Color32::TRANSPARENT,
                        tokens.highlight_color,
                        crate::theme::palette(ui).accent_text,
                    )
                } else {
                    (
                        egui::Color32::TRANSPARENT,
                        crate::theme::palette(ui).hairline,
                        ui.visuals().weak_text_color(),
                    )
                };
                // A frame never wraps itself in a wrapped row (only a label
                // does), so the chip is measured first and the row broken
                // when it would not fit — else it runs past the card.
                let label = match state.basis.as_deref() {
                    Some(basis) if state.present && basis != BASIS_OWN => {
                        format!("{} · {basis}", stage_label(&state.name))
                    }
                    _ => stage_label(&state.name).to_owned(),
                };
                let font = DesignTokens::welcome_screen_body().resolve(ui.style());
                let text_w = ui
                    .painter()
                    .layout_no_wrap(label.to_owned(), font, text_color)
                    .size()
                    .x;
                let chip_w = (text_w + STAGE_CHIP_EXTRA).max(STAGE_MIN_WIDTH + STAGE_CHIP_MARGIN);
                // In a wrapped row `available_width` is the whole row; what
                // remains is the distance from the cursor to the right edge.
                // The "›" goes before a chip on the same row, never at the
                // end of one: it was left dangling where the row broke
                // (design review, 2026-10-12).
                let remaining = ui.max_rect().right() - ui.cursor().min.x;
                let separator_w = if i > 0 { STAGE_SEPARATOR_WIDTH } else { 0.0 };
                if remaining < chip_w + separator_w {
                    ui.end_row();
                } else if i > 0 {
                    ui.weak("›");
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
                                egui::RichText::new(label.as_str())
                                    .text_style(DesignTokens::welcome_screen_body())
                                    .strong()
                                    .color(text_color),
                            );
                        });
                    })
                    .response;
                let mut hover = state.name.clone();
                if let Some(basis) = &state.basis {
                    hover.push_str("\nbasis: ");
                    hover.push_str(basis);
                }
                if let Some(note) = &state.note {
                    // A skipped stage's note says why; a needed one's says
                    // what is missing (a task declared, not yet accepted).
                    hover.push_str(if state.needed { "\n" } else { "\nnot needed: " });
                    hover.push_str(note);
                }
                if !state.proved_by.is_empty() {
                    hover.push_str("\nproved by\n");
                    hover.push_str(&state.proved_by.join("\n"));
                }
                response.on_hover_text(hover);
            }
        });
        ui.add_space(10.0);
        match &index.next_move {
            Some(next) => {
                // The move is a sentence naming doors and formats: it wraps
                // inside the card instead of widening the whole page past
                // the window (seen on a fresh project, 2026-09-28).
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        egui::RichText::new("Next")
                            .text_style(DesignTokens::welcome_screen_body())
                            .strong()
                            .color(crate::theme::palette(ui).accent_text),
                    );
                    ui.label(
                        egui::RichText::new(without_tool_names(next))
                            .text_style(DesignTokens::welcome_screen_body()),
                    );
                });
            }
            None => {
                // Every stage has a result; whether the policy is good is
                // the evaluation's rate, not this line (2026-10-04 review).
                ui.label("Every stage has a result. The evaluation's success rate says how good the policy is.");
            }
        }
    });
}

/// What happened in this project, newest first: the agent's tools (the
/// job table) and what the human and the agent did in the window (the
/// event log), in one feed.
fn activity(ui: &mut egui::Ui, jobs: &[Job], events: &[Event]) {
    subheading(ui, "Recent activity");
    card(ui, None).show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        let now = now_epoch();
        let mut lines: Vec<(f64, ActivityLine)> = jobs
            .iter()
            .map(|j| (j.started, ActivityLine::Job(j)))
            .chain(
                events
                    .iter()
                    .filter(|e| worth_a_line(e))
                    .map(|e| (e.epoch_seconds(), ActivityLine::Event(e))),
            )
            .collect();
        if lines.is_empty() {
            weak_body(
                ui,
                "Nothing yet. Every tool your agent runs, and what it does in this window, appears here.",
            );
            return;
        }
        lines.sort_by(|a, b| b.0.total_cmp(&a.0));
        // A run of page opens is one move: the newest stands for it (an
        // agent walking the pages filled every row, 2026-10-12).
        lines.dedup_by(|older, newer| is_page_open(&older.1) && is_page_open(&newer.1));
        for (at, line) in lines.iter().take(ACTIVITY_ROWS) {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(ago(*at))
                        .small()
                        .weak(),
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
                                .weak(),
                        );
                    }
                    ActivityLine::Event(event) => {
                        let who = match event.by.as_str() {
                            BY_AGENT => "agent",
                            BY_USER => "you",
                            _ => "studio",
                        };
                        ui.label(egui::RichText::new(who).strong());
                        ui.weak(event_words(event));
                    }
                }
            });
        }
    });
}

/// Whether an event belongs in Recent activity: what the agent did in the
/// window, and the person's own moves that change something (a project
/// opened, a recording shown). The person's page opens, selections,
/// closed drawers and time scrubs filled the feed with what they had
/// just done themselves (design review, 2026-10-12).
fn worth_a_line(event: &Event) -> bool {
    match event.kind.as_str() {
        EVENT_DESELECT | EVENT_TIME | EVENT_KEY => false,
        EVENT_TABLE if event.table.is_none() => false,
        EVENT_OPEN if event.project.is_some() => true,
        EVENT_OPEN | EVENT_SELECT | EVENT_TABLE => event.by == BY_AGENT,
        _ => true,
    }
}

/// Whether a feed line is a page being opened (not a project).
fn is_page_open(line: &ActivityLine) -> bool {
    matches!(line, ActivityLine::Event(e) if e.kind == EVENT_OPEN && e.project.is_none())
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

/// The cloud line of the Compute card. It named a TRAINNR_ENDPOINT that
/// no tool reads (fresh-install audit, 2026-10-08); jobs run on this
/// machine, and a full training run rents a GPU through the cloud tool.
fn cloud_line(ui: &mut egui::Ui) {
    weak_body(
        ui,
        "Jobs run on this machine. For a full training run, your agent can rent a cloud GPU.",
    )
    .on_hover_text(format!("With {CLOUD_TOOL}"));
}

/// The tool that rents a GPU for a full training run (README, Platforms).
const CLOUD_TOOL: &str = "tools/cloud-gpu.py";

/// Where work runs: local by default; a rented machine when the cloud
/// tools are configured. Honest about what it does not know.
fn compute(ui: &mut egui::Ui, model: &Model, open_run: &mut Option<String>) {
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
            ui.label(egui::RichText::new("This machine").strong());
            tag(ui, std::env::consts::ARCH);
            tag(ui, std::env::consts::OS);
        });
        ui.add_space(4.0);
        // The same runs as the top bar's Running now panel, as the same
        // chips: every run in the job table, whoever started it; a click
        // opens the panel on that run, which has its buttons.
        let now = crate::model::now_epoch();
        let rows = crate::running::panel_rows(&model.jobs, now, crate::running::FINISHED_KEPT);
        if running == 0 && rows.is_empty() {
            weak_body(ui, "Nothing running");
        }
        if !rows.is_empty() {
            match crate::running::chip_strip(ui, &rows, now, None, false) {
                Some(crate::running::StripClick::Chip(id)) => *open_run = Some(id),
                Some(crate::running::StripClick::More) => {
                    *open_run = rows.first().map(|j| j.id.clone());
                }
                None => {}
            }
        }
        // What a running chain is filling in, with its stage (the index's
        // `in_progress`): the folder is work, not a warning, until the
        // record lands and it becomes a card.
        if let Some(index) = model.index() {
            for p in &index.in_progress {
                in_progress_row(ui, p);
            }
        }
        ui.add_space(6.0);
        cloud_line(ui);
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

#[allow(clippy::too_many_arguments)] // one page, one argument per thing it reads or asks
pub fn section(
    ui: &mut egui::Ui,
    model: &Model,
    section: Section,
    selected: &mut Option<String>,
    show: &mut Option<String>,
    explore: &mut Option<crate::detail::Section>,
    nav: &mut Nav,
    actions: (&[&crate::forms::Form], &crate::onboarding::ActionState),
) {
    let (forms, state) = actions;
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
                let view = if rows.is_empty() {
                    crate::listing::View::Cards
                } else {
                    crate::listing::view_switch(
                        ui,
                        section,
                        crate::listing::matrix_available(&rows),
                    )
                };
                // The page's own actions, the agent's tools as buttons.
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Drawn right to left in reverse, so they read left to
                    // right in the registry's order, the first one filled:
                    // Export, Run gate, Pre-flight is the workflow's order
                    // (it read backwards until 2026-10-12).
                    for (i, form) in forms.iter().enumerate().rev() {
                        if state.is_running(&form.tool) {
                            ui.label(state.busy_word("Running…"));
                            ui.spinner();
                            continue;
                        }
                        let missing = crate::forms::missing_input(form, &|k| index.count(k));
                        let free = state.running.is_none() && missing.is_none();
                        let button = if i == 0 {
                            crate::onboarding::primary_button(ui, &form.label, free)
                        } else {
                            crate::onboarding::secondary_button(ui, &form.label, free)
                        };
                        let hover = missing.unwrap_or_else(|| form.summary.clone());
                        if button
                            .on_hover_text(&hover)
                            .on_disabled_hover_text(&hover)
                            .clicked()
                        {
                            nav.form = Some(form.tool.clone());
                        }
                    }
                });
                view
            })
            .inner;
        for form in forms {
            crate::onboarding::failure_line(ui, state, &form.tool, &form.label);
            crate::onboarding::done_line(ui, state, &form.tool);
        }
        // Work this page's kinds are still becoming (a capture listening, a
        // scene being captured), from the index's `in_progress`.
        for p in index
            .in_progress
            .iter()
            .filter(|p| section.kinds().any(|k| k == p.kind))
        {
            in_progress_row(ui, p);
        }
        ui.add_space(16.0);
        if rows.is_empty() {
            crate::onboarding::empty_state(ui, section.icon(), &section.guide());
            return;
        }
        // A page with one artifact is that artifact: its drawer opens on
        // arrival (a click still closes it).
        if entered && selected.is_none() && rows.len() == 1 {
            *selected = Some(rows[0].stamp.clone());
            nav.auto_selected = true;
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

/// The cards, grouped by the day they last changed, one row at a time so
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
                .weak(),
        );
        ui.add_space(6.0);
        for row in members.chunks(columns) {
            if row_index > 0 {
                ui.add_space(GRID_GAP);
            }
            card_row(ui, row, width, |ui, artifact| {
                let (name, hash) = split_stamp(&artifact.stamp);
                let facts =
                    summary_line(&artifact.summary).unwrap_or_else(|| artifact.path.clone());
                // An evaluation is told apart by its condition: the
                // policy repeats on every card (design review,
                // 2026-10-12), so the condition is the title.
                let (title, facts) = if artifact.kind == kind::CERTIFICATE {
                    (facts, name.to_owned())
                } else {
                    (name.to_owned(), facts)
                };
                let is_selected = selected.as_deref() == Some(artifact.stamp.as_str());
                let response = picture_card_with_id(
                    ui,
                    width,
                    model
                        .preview_path(artifact, !ui.visuals().dark_mode)
                        .as_deref(),
                    Section::icon_for(&artifact.kind),
                    CardText {
                        title: &title,
                        facts: &facts,
                        footer: Some(card_footer(artifact, hash)),
                    },
                    is_selected,
                    &artifact.stamp,
                );
                if response.clicked() {
                    clicked = Some(artifact.stamp.clone());
                }
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
            if let Some(path) = model.preview_path(artifact, !ui.visuals().dark_mode) {
                let rect = egui::Rect::from_min_size(
                    ui.cursor().min,
                    egui::vec2(
                        DETAIL_PICTURE_WIDTH,
                        DETAIL_PICTURE_WIDTH / THUMBNAIL_ASPECT,
                    ),
                );
                // Decoded off the UI thread like every card picture
                // (pictures.rs); a file URI here drew the loader's red
                // triangle in the drawer (stranger test, 2026-10-03).
                crate::widgets::thumbnail(ui, &path, rect);
                ui.advance_cursor_after_rect(rect);
                ui.add_space(16.0);
            }
            ui.vertical(|ui| {
                // The buttons take their width first, from the right; the
                // name truncates into what is left, so a long stamp never
                // runs under "Show in viewer" (2026-09-27).
                ui.horizontal(|ui| {
                    ui.set_width(ui.available_width());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        // The drawer's actions at one height, padded alike.
                        ui.spacing_mut().button_padding = egui::vec2(10.0, 5.0);
                        let mut buttons = |ui: &mut egui::Ui| {
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
                            // A deployment in the MuJoCo viewport: live, or a
                            // gate trial / pre-flight segment replayed.
                            let scenes = crate::model::viewport_scenes(artifact);
                            if let Some(((_, live), replays)) = scenes.split_first() {
                                if !replays.is_empty() {
                                    // Framed like a button, as Show is: the
                                    // drawer's actions read as one set.
                                    let replay = framed_button(ui, "Replay in simulator");
                                    egui::containers::menu::MenuButton::from_button(replay).ui(
                                        ui,
                                        |ui| {
                                            for (label, scene) in replays {
                                                if ui.button(label).clicked() {
                                                    nav.scene = Some(scene.clone());
                                                    ui.close();
                                                }
                                            }
                                        },
                                    );
                                }
                                if ui
                                    .add(framed_button(ui, "Play in simulator"))
                                    .on_hover_text(
                                        "The exported policy live in the Simulator's MuJoCo \
                                     viewport: drive it with WASD",
                                    )
                                    .clicked()
                                {
                                    nav.scene = Some(live.clone());
                                }
                            }
                            // What a finding asks for, one click away: a
                            // drift check's recommendation named a tool in
                            // prose only (design review, 2026-10-12).
                            if let Some((label, tool)) = remedy(artifact) {
                                if ui
                                    .add(framed_button(ui, label))
                                    .on_hover_text("Opens the form that runs it")
                                    .clicked()
                                {
                                    nav.form = Some(tool.to_owned());
                                }
                            }
                        };
                        buttons(ui);
                        ui.add_space(6.0);
                        tag(ui, kind_word(&artifact.kind));
                        ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                            icon_at(
                                ui,
                                Section::icon_for(&artifact.kind),
                                18.0,
                                ui.tokens().label_button_icon_color,
                            );
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(name)
                                        .text_style(DesignTokens::welcome_screen_h2())
                                        .strong(),
                                )
                                .truncate(),
                            )
                            .on_hover_text(name);
                        });
                    });
                });
                ui.horizontal(|ui| {
                    weak_body(ui, "version");
                    ui.label(egui::RichText::new(hash).monospace().weak());
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
                ui.weak(crate::detail::field_word(key));
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
        ui.label(egui::RichText::new(stamp).color(crate::theme::palette(ui).warn));
    } else if model.artifact(stamp).is_some() {
        let (name, hash) = split_stamp(stamp);
        let text = if short { name } else { stamp };
        if ui
            .add(egui::Link::new(
                egui::RichText::new(text)
                    .monospace()
                    .color(crate::theme::palette(ui).link),
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
        let count = crate::widgets::count_word(members.len(), kind_word(kind));
        ui.horizontal_wrapped(|ui| {
            ui.weak(count);
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
const USED_BY_SHOWN: usize = 4;

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
                ui.weak(key);
                ui.label(render_value(value));
                ui.end_row();
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_next_move_reads_without_tool_names() {
        let next = "bring in the robot's model (onboard_robot), or its telemetry \
                    (ingest_recording) or a public log of it (ingest_public_log)";
        assert_eq!(
            without_tool_names(next),
            "bring in the robot's model, or its telemetry or a public log of it"
        );
        let plain = "an ONNX export (with its manifest) and the gate";
        assert_eq!(without_tool_names(plain), plain, "prose in brackets stays");
    }

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
        // The kinds `trainnr/project/kinds.py` writes, by name.
        let written = [
            kind::ROBOT,
            kind::RECORDING,
            kind::TASK,
            kind::SCENE,
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
