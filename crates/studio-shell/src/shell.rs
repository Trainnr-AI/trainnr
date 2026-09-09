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
use crate::pages::{self, Section, RAIL_ICON};
use crate::widgets::icon_at;

/// The rail's width: wide enough for "Environments" plus a count.
const RAIL_WIDTH: f32 = 196.0;
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
    /// The presenter process for the open project (`tools/studio-present.py`),
    /// spawned on first Show and killed with the shell or on a project switch.
    presenter: Option<std::process::Child>,
    /// Set when the user asked to show an artifact; the frame loop
    /// switches to Live once the viewer has a recording.
    pub show_requested: bool,
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
    repo_root: std::path::PathBuf,
}

impl Shell {
    pub fn new(model: Model, repo_root: std::path::PathBuf) -> Self {
        Self {
            model,
            section: Section::Overview,
            selected: None,
            switch_to: None,
            presenter: None,
            show_requested: false,
            shown: None,
            last_shown: None,
            table: None,
            scroll_to_detail: false,
            history: Vec::new(),
            entered: true,
            palette: None,
            dismissed_failure: None,
            repo_root,
        }
    }

    /// Show an artifact in the viewer: make sure the presenter runs for
    /// this project, then write the intent it watches for.
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
        if let Some(child) = self.presenter.as_mut() {
            if matches!(child.try_wait(), Ok(None)) {
                return;
            }
        }
        let pipeline = self.repo_root.join("pipeline");
        let script = self.repo_root.join("tools").join("studio-present.py");
        let mut command = std::process::Command::new("uv");
        command
            .current_dir(&pipeline)
            .args(["run", "--extra", "sim", "--extra", "viz", "python"])
            .arg(&script)
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
        // Its own process group, so the whole tree can be reaped (the
        // viewport's spawn does the same).
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt as _;
            command.process_group(0);
        }
        self.presenter = command.spawn().ok();
    }

    fn kill_presenter(&mut self) {
        if let Some(mut child) = self.presenter.take() {
            #[cfg(unix)]
            {
                // `-s TERM -- -PGID`: the group, wrapper and grandchild
                // alike (viewport.rs says why the `--` matters).
                let _ = std::process::Command::new("kill")
                    .args(["-s", "TERM", "--", &format!("-{}", child.id())])
                    .status();
            }
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    /// The top bar: wordmark, project, jobs, and the viewer's panel
    /// toggles (meaningful on the Live view; harmless elsewhere).
    pub fn top_bar(&mut self, ui: &mut egui::Ui, viewer_buttons: impl FnOnce(&mut egui::Ui)) {
        let tokens = ui.tokens();
        egui::Frame::new()
            .fill(tokens.top_bar_color)
            .inner_margin(egui::Margin::symmetric(8, 6))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.horizontal(|ui| {
                    #[cfg(target_os = "macos")]
                    ui.add_space(TRAFFIC_LIGHTS_INSET);
                    ui.label(egui::RichText::new("●").color(tokens.highlight_color));
                    ui.label(egui::RichText::new("trainnr").strong().size(15.0));
                    ui.add_space(12.0);
                    // The project switcher: the current project's name;
                    // click for the list, or open the Projects page.
                    let name = self.model.name();
                    let projects = self.model.projects();
                    let response = ui.add(
                        egui::Button::new(
                            egui::RichText::new(format!("{name}  ▾"))
                                .text_style(DesignTokens::welcome_screen_body()),
                        )
                        .frame(false),
                    );
                    egui::Popup::menu(&response).show(|ui| {
                        ui.set_min_width(220.0);
                        for project in &projects {
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
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        viewer_buttons(ui);
                        ui.add_space(8.0);
                        let running = self.model.running_jobs();
                        if running > 0 {
                            ui.small_icon(&icons::PLAY, Some(tokens.highlight_color));
                            ui.label(
                                egui::RichText::new(format!("{running} running"))
                                    .small()
                                    .color(tokens.highlight_color),
                            );
                        } else {
                            ui.label(
                                egui::RichText::new("idle")
                                    .small()
                                    .color(ui.visuals().weak_text_color()),
                            );
                        }
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

    fn rail_item(&mut self, ui: &mut egui::Ui, item: Section) {
        let tokens = ui.tokens();
        let selected = self.section == item;
        let count = self
            .model
            .index()
            .map(|index| item.kinds().iter().map(|k| index.count(k)).sum::<usize>())
            .unwrap_or(0);
        let fill = if selected {
            tokens.selection_bg_fill
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
                    let tint = if selected {
                        tokens.strong_fg_color
                    } else {
                        tokens.label_button_icon_color
                    };
                    icon_at(ui, item.icon(), RAIL_ICON, tint);
                    let text = egui::RichText::new(item.title());
                    ui.label(if selected { text.strong() } else { text });
                    if !item.kinds().is_empty() {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(
                                egui::RichText::new(count.to_string())
                                    .small()
                                    .color(ui.visuals().weak_text_color()),
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
        if let Some(root) = self.switch_to.take() {
            self.kill_presenter();
            self.model.switch(root);
            self.selected = None;
            self.section = Section::Overview;
        }
        self.model.refresh();
        ui.ctx().request_repaint_after(crate::model::RELOAD_EVERY);
        if let Some(problem) = self.model.problem.clone() {
            if self.section != Section::Projects {
                pages::problem_page(ui, &problem);
                return;
            }
        }
        self.presenter_failure(ui);
        match self.section {
            Section::Projects => {
                if let Some(root) = pages::projects(ui, &self.model) {
                    self.switch_to = Some(root);
                }
            }
            Section::Overview => {
                let mut go_to = None;
                pages::overview(ui, &self.model, &mut go_to);
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
                pages::section(
                    ui,
                    &self.model,
                    section,
                    &mut self.selected,
                    &mut show,
                    &mut explore,
                    &mut nav,
                );
                if let Some(stamp) = show {
                    self.show(&stamp);
                }
                if let Some(table) = explore {
                    self.table = Some(crate::detail::TableView::new(table));
                }
                if let Some(stamp) = nav.open {
                    self.navigate(&stamp);
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
        self.keys(ui.ctx());
        if let Some(palette) = self.palette.as_mut() {
            let (chosen, close) = crate::palette::show(ui.ctx(), palette, self.model.index());
            if close {
                self.palette = None;
            }
            match chosen {
                Some(crate::palette::Hit::Page(section)) => {
                    self.section = section;
                    self.selected = None;
                    self.history.clear();
                    self.entered = true;
                    pages::scroll_to_top(ui.ctx());
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
        let typing = ctx.memory(|m| m.focused().is_some());
        let (palette_key, escape, left, right) = ctx.input_mut(|i| {
            (
                i.consume_key(egui::Modifiers::COMMAND, egui::Key::K),
                !typing && i.consume_key(egui::Modifiers::NONE, egui::Key::Escape),
                !typing && i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowLeft),
                !typing && i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight),
            )
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
    fn presenter_failure(&mut self, ui: &mut egui::Ui) {
        let Some(status) = self.model.present_status.clone() else {
            return;
        };
        let (Some(stamp), Some(error)) = (status.stamp, status.error) else {
            return;
        };
        if self.shown.as_deref() != Some(stamp.as_str())
            && self.last_shown.as_deref() != Some(stamp.as_str())
        {
            return;
        }
        if self.dismissed_failure.as_ref() == Some(&(stamp.clone(), error.clone())) {
            return;
        }
        egui::Frame::new()
            .fill(ui.visuals().warn_fg_color.linear_multiply(0.12))
            .inner_margin(egui::Margin::symmetric(12, 8))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.small_icon(&icons::WARNING, Some(ui.visuals().warn_fg_color));
                    ui.label(
                        egui::RichText::new(format!(
                            "The viewer could not show {}: {error}",
                            crate::model::split_stamp(&stamp).0
                        ))
                        .color(ui.visuals().warn_fg_color),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("dismiss").clicked() {
                            self.dismissed_failure = Some((stamp.clone(), error.clone()));
                        }
                    });
                });
            });
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
            .detail_path(artifact)
            .and_then(|p| crate::detail::Detail::load(&p))
            .ok_or_else(|| format!("{stamp} has no detail file"))?;
        let wanted = title.trim().to_lowercase();
        let section = detail
            .sections
            .into_iter()
            .find(|s| s.kind == "table" && s.title.to_lowercase() == wanted)
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
