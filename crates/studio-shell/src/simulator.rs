//! The Simulator page's controls, shaped by what a person does in a
//! simulator, in order of how often: watch the scene; pause, step,
//! reset, change speed; poke the robot; flip a debug overlay; look up a
//! fact about the model. So (Prakhar's calls, 2026-09-09):
//!
//! - the picture is the page — nothing permanent sits beside it;
//! - a **transport bar** under the picture, the video-player shape and
//!   the shape of Rerun's own timeline right below it: play or pause,
//!   step, ten steps, reset, the keyframes, speed, the scene picker, and
//!   three chips that never move — sim time, real-time factor, frames;
//!   Space, → and R do what they do in `simulate`;
//! - two **modes**, said plainly: the scene runs itself, or you drive;
//! - an **Inspect drawer** over the right of the picture — Control,
//!   Joints, Physics — sliders read-only while the scene runs itself,
//!   editable in drive mode, grouped by the name's prefix;
//! - an **overlay toolbar** in the picture's corner: contacts, forces,
//!   joints, inertia, transparent, shadows — one click, a lit state — and
//!   the camera's named views beside them;
//! - the **agent, visible**: what it just pressed shows on the bar.
//!
//! Every number is drawn at a fixed width, and the frame rate is refreshed
//! once a second, because a value that changes width thirty times a
//! second moves everything beside it (the flicker, 2026-09-09).
//! Nothing here touches MuJoCo: the stream's status says what is, the
//! wire says what the human asked, and the physics process decides
//! (docs/76 §10.2).

use re_ui::UiExt as _;

use crate::viewport::{SimModel, SimStatus, ViewportFeed, PREVIEW_TASKS, WALK_TASK};

/// What the bar asked the page to do with the viewport itself.
pub enum Action {
    Spawn(String),
    Stop,
}

/// Which world the camera keeps in view, in a many-worlds scene.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Follow {
    #[default]
    None,
    World(u32),
    /// The lowest reward right now.
    Worst,
    /// A world whose episode just ended (fallen, timed out).
    Failing,
    /// Every world in turn, a few seconds each.
    Cycle,
}

const CYCLE_EVERY: std::time::Duration = std::time::Duration::from_secs(4);

fn follow_state(ctx: &egui::Context) -> Follow {
    ctx.data(|d| {
        d.get_temp(egui::Id::new("simulator-follow"))
            .unwrap_or_default()
    })
}

pub fn set_follow(ctx: &egui::Context, follow: Follow) {
    ctx.data_mut(|d| d.insert_temp(egui::Id::new("simulator-follow"), follow));
}

/// The rule's answer this frame: which world index to follow.
fn follow_target(ctx: &egui::Context, rule: Follow, worlds: &[crate::viewport::SimWorld]) -> i32 {
    match rule {
        Follow::None => -1,
        Follow::World(w) => w as i32,
        Follow::Worst => worlds
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.reward.total_cmp(&b.1.reward))
            .map_or(-1, |(i, _)| i as i32),
        Follow::Failing => worlds.iter().position(|w| w.done).map_or(-1, |i| i as i32),
        Follow::Cycle => {
            // The clock is read BEFORE the memory lock: egui's context is
            // one lock, and reading input inside `data_mut` deadlocked the
            // frame (the hang, 2026-09-09).
            let now = ctx.input(|i| i.time);
            let n = worlds.len().max(1) as u64;
            let started: f64 = ctx.data_mut(|d| {
                *d.get_temp_mut_or_insert_with(egui::Id::new("simulator-cycle-start"), || now)
            });
            ctx.request_repaint_after(CYCLE_EVERY);
            (((now - started) / CYCLE_EVERY.as_secs_f64()) as u64 % n) as i32
        }
    }
}

/// Apply the follow rule: send the world to the renderer when it changes.
pub fn apply_follow(ctx: &egui::Context, viewport: &mut ViewportFeed) {
    let (Some(status), _) = viewport.report() else {
        return;
    };
    if status.worlds.is_empty() {
        return;
    }
    let target = follow_target(ctx, follow_state(ctx), &status.worlds);
    if target != status.follow as i32 {
        viewport.send_follow(target);
    }
}

// Sliders never clamp on display: egui would otherwise pull a value that
// sits outside its nominal range back inside and report it as a change —
// which took manual control of the scene before anyone touched anything.
/// A joint or actuator with no declared range still needs a slider.
const UNLIMITED_HINGE: [f64; 2] = [-std::f64::consts::PI, std::f64::consts::PI];
const UNLIMITED_LINEAR: [f64; 2] = [-1.0, 1.0];
/// Speed slider bounds (simulate's Speed goes further; the stream clamps).
const SPEED_RANGE: std::ops::RangeInclusive<f32> = 0.1..=4.0;
/// The drawer's width over the picture, and a value's fixed width.
const DRAWER_WIDTH: f32 = 340.0;
const VALUE_WIDTH: f32 = 64.0;
const CHIP_WIDTH: f32 = 92.0;
/// The overlay toggles: the label shown, MuJoCo's flag name, and whether
/// it is a rendering flag (`mjtRndFlag`) rather than a visualization one.
const OVERLAYS: &[(&str, &str, bool)] = &[
    ("contacts", "contactpoint", false),
    ("forces", "contactforce", false),
    ("joints", "joint", false),
    ("inertia", "inertia", false),
    ("transparent", "transparent", false),
    ("shadows", "shadow", true),
];
/// The camera's named views, in the wire's order (`VIEW_PRESETS`).
const VIEWS: &[(&str, u8)] = &[("Front", 1), ("Side", 2), ("Top", 3), ("Reset view", 0)];
/// Rendering flags MuJoCo turns on by default (after mjv_defaultScene).
const RND_DEFAULT_ON: &[&str] = &["shadow", "reflection", "skybox", "haze", "cullface"];

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum Tab {
    #[default]
    Control,
    Joints,
    Physics,
}

/// Open the drawer on a tab by name, or close it — the agent's door.
pub fn inspect(ctx: &egui::Context, what: &str) -> Result<(), String> {
    let (_, tab) = drawer_state(ctx);
    match what {
        "close" | "" => set_drawer(ctx, false, tab),
        "control" => set_drawer(ctx, true, Tab::Control),
        "joints" => set_drawer(ctx, true, Tab::Joints),
        "physics" => set_drawer(ctx, true, Tab::Physics),
        other => {
            return Err(format!(
                "inspect {other:?}: one of control, joints, physics, close"
            ));
        }
    }
    Ok(())
}

/// The drawer's open state and tab live in egui's memory, keyed here.
fn drawer_state(ctx: &egui::Context) -> (bool, Tab) {
    ctx.data(|d| {
        (
            d.get_temp(egui::Id::new("simulator-drawer-open"))
                .unwrap_or(false),
            d.get_temp(egui::Id::new("simulator-drawer-tab"))
                .unwrap_or_default(),
        )
    })
}

fn set_drawer(ctx: &egui::Context, open: bool, tab: Tab) {
    ctx.data_mut(|d| {
        d.insert_temp(egui::Id::new("simulator-drawer-open"), open);
        d.insert_temp(egui::Id::new("simulator-drawer-tab"), tab);
    });
}

// -- the transport bar ---------------------------------------------------------

/// Under the picture. Returns what the page should do with the viewport.
pub fn transport(ui: &mut egui::Ui, viewport: &mut ViewportFeed) -> Option<Action> {
    let mut action = None;
    let (status, model) = viewport.report();
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 8.0;
        // The scene picker: a named menu, the empty state's only door.
        let current = viewport.task().unwrap_or("Scene");
        menu(ui, current, |ui| {
            for task in PREVIEW_TASKS {
                if ui.button(*task).clicked() {
                    action = Some(Action::Spawn((*task).to_owned()));
                    ui.close();
                }
            }
            if ui
                .button(WALK_TASK)
                .on_hover_text("the newest trained walk policy, live")
                .clicked()
            {
                action = Some(Action::Spawn(WALK_TASK.to_owned()));
                ui.close();
            }
            if viewport.is_active() {
                ui.separator();
                if ui.button("Stop the scene").clicked() {
                    action = Some(Action::Stop);
                    ui.close();
                }
            }
        });
        let Some(status) = status else {
            if viewport.is_active() {
                ui.label(egui::RichText::new("starting…").color(ui.visuals().weak_text_color()));
            }
            return;
        };
        ui.separator();
        // Play / pause, step, reset — with their keys in the tooltips.
        if status.paused {
            if ui
                .small_icon_button(&re_ui::icons::PLAY, "Run  (Space)")
                .clicked()
            {
                viewport.send_run(true);
            }
        } else if ui
            .small_icon_button(&re_ui::icons::PAUSE, "Pause  (Space)")
            .clicked()
        {
            viewport.send_run(false);
        }
        if ui
            .small_icon_button(
                &re_ui::icons::ARROW_RIGHT,
                "One step  (→); pauses, you drive",
            )
            .clicked()
        {
            viewport.send_step(1);
        }
        if ui.small("×10").on_hover_text("ten steps").clicked() {
            viewport.send_step(10);
        }
        if ui
            .small_icon_button(
                &re_ui::icons::RESET,
                "Reset  (R): the scene from its start, running itself",
            )
            .clicked()
        {
            viewport.send_reset(None);
        }
        if let Some(model) = model.as_ref().filter(|m| !m.keyframes.is_empty()) {
            menu(ui, "keyframe", |ui| {
                for (k, name) in model.keyframes.iter().enumerate() {
                    if ui.button(name).clicked() {
                        viewport.send_reset(Some(k as u32));
                        ui.close();
                    }
                }
            });
        }
        ui.separator();
        // Speed: a compact slider and its value, fixed width.
        ui.label(egui::RichText::new("speed").color(ui.visuals().weak_text_color()));
        let mut speed = status.speed as f32;
        ui.spacing_mut().slider_width = 90.0;
        let response = ui.add(
            egui::Slider::new(&mut speed, SPEED_RANGE)
                .clamping(egui::SliderClamping::Never)
                .logarithmic(true)
                .show_value(false),
        );
        mono(ui, format!("{speed:>5.2}×"), 52.0);
        if response.drag_stopped() || (response.changed() && !response.dragged()) {
            viewport.send_speed(speed);
        }
        ui.separator();
        // The mode, said plainly.
        let mode = if status.manual {
            "you drive"
        } else {
            "runs itself"
        };
        let hint = if status.manual {
            "your sliders drive the scene; click to hand it back to its own motion (from its start)"
        } else {
            "the scene runs its own motion; click, or move a slider, to drive it yourself"
        };
        if toggle(ui, status.manual, mode, hint) {
            viewport.send_manual(!status.manual);
        }
        if !status.worlds.is_empty() {
            ui.separator();
            worlds_row(ui, viewport, &status);
        }
        // The chips: never move.
        ui.separator();
        chip(ui, "t", format!("{:>8.2} s", status.time));
        chip(ui, "rtf", format!("{:>5.2}", status.rtf));
        chip(
            ui,
            "fps",
            format!("{:>4.0}", viewport.fps_settled().unwrap_or(0.0)),
        );
        // The agent, visible.
        if let Some(what) = viewport.flashing() {
            ui.label(
                egui::RichText::new(format!("agent · {what}"))
                    .small()
                    .color(ui.visuals().hyperlink_color),
            );
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(200));
        }
        // Inspect, at the right end.
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let (open, tab) = drawer_state(ui.ctx());
            if toggle(ui, open, "Inspect", "Control, Joints, Physics  (I)") {
                set_drawer(ui.ctx(), !open, tab);
            }
        });
    });
    action
}

/// A quiet toggle: text that brightens and underlines when on — never a
/// solid block of accent colour behind a word.
fn toggle(ui: &mut egui::Ui, on: bool, text: &str, hint: &str) -> bool {
    let color = if on {
        ui.visuals().strong_text_color()
    } else {
        ui.visuals().weak_text_color()
    };
    let label = egui::RichText::new(text).color(color);
    let response = ui
        .add(egui::Button::new(label).frame(false))
        .on_hover_text(hint);
    if on {
        let rect = response.rect;
        let stroke = egui::Stroke::new(1.5, ui.visuals().hyperlink_color);
        ui.painter()
            .hline(rect.x_range().shrink(2.0), rect.bottom() - 1.0, stroke);
    }
    response.clicked()
}

/// A menu whose arrow is an icon, not a glyph the font may lack.
fn menu<R>(
    ui: &mut egui::Ui,
    text: &str,
    add: impl FnOnce(&mut egui::Ui) -> R,
) -> egui::InnerResponse<Option<R>> {
    ui.menu_image_text_button(re_ui::icons::DROPDOWN_ARROW.as_image(), text, add)
}

/// Many worlds: a dot per world (green running, red fallen, the followed
/// one ringed) — click one to follow it — and the follow rule as a menu.
fn worlds_row(ui: &mut egui::Ui, viewport: &mut ViewportFeed, status: &SimStatus) {
    let rule = follow_state(ui.ctx());
    let label = match rule {
        Follow::None => "follow".to_owned(),
        Follow::World(w) => format!("w{w}"),
        Follow::Worst => "worst".to_owned(),
        Follow::Failing => "failing".to_owned(),
        Follow::Cycle => "cycle".to_owned(),
    };
    menu(ui, &label, |ui| {
        for (name, choice) in [
            ("none", Follow::None),
            ("worst reward", Follow::Worst),
            ("a failing world", Follow::Failing),
            ("cycle through", Follow::Cycle),
        ] {
            if ui.button(name).clicked() {
                set_follow(ui.ctx(), choice);
                ui.close();
            }
        }
    });
    let followed = status.follow;
    ui.spacing_mut().item_spacing.x = 3.0;
    for (i, world) in status.worlds.iter().enumerate() {
        let (rect, response) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::click());
        let color = if world.done {
            egui::Color32::from_rgb(220, 60, 60)
        } else {
            egui::Color32::from_rgb(70, 200, 110)
        };
        ui.painter().circle_filled(rect.center(), 4.0, color);
        if i as i64 == followed {
            ui.painter().circle_stroke(
                rect.center(),
                5.5,
                egui::Stroke::new(1.5, ui.visuals().strong_text_color()),
            );
        }
        let response = response.on_hover_text(format!(
            "w{i}: reward {:.2}{}",
            world.reward,
            if world.done { ", ended" } else { "" }
        ));
        if response.clicked() {
            set_follow(ui.ctx(), Follow::World(i as u32));
        }
    }
    ui.spacing_mut().item_spacing.x = 8.0;
    let n = status.worlds.len();
    ui.label(
        egui::RichText::new(format!("{n} worlds"))
            .small()
            .color(ui.visuals().weak_text_color()),
    );
    let _ = viewport;
}

fn chip(ui: &mut egui::Ui, key: &str, value: String) {
    ui.label(
        egui::RichText::new(key)
            .small()
            .color(ui.visuals().weak_text_color()),
    );
    mono(ui, value, CHIP_WIDTH - 24.0);
}

/// A number at a fixed width, so the layout never moves with it.
fn mono(ui: &mut egui::Ui, text: String, width: f32) {
    ui.add_sized(
        [width, ui.spacing().interact_size.y],
        egui::Label::new(egui::RichText::new(text).monospace()),
    );
}

// -- the overlay toolbar and the camera views, in the picture's corner ---------

pub fn overlays(ctx: &egui::Context, picture: egui::Rect, viewport: &mut ViewportFeed) {
    let (status, model) = viewport.report();
    let (Some(status), Some(model)) = (status, model) else {
        return;
    };
    egui::Area::new(egui::Id::new("simulator-overlays"))
        .order(egui::Order::Foreground)
        .fixed_pos(picture.right_top() + egui::vec2(-8.0, 8.0))
        .pivot(egui::Align2::RIGHT_TOP)
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(ui.visuals().panel_fill.gamma_multiply(0.85))
                .corner_radius(6.0)
                .inner_margin(4.0)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        for (label, flag, rendering) in OVERLAYS {
                            let table = if *rendering {
                                &model.rnd_flags
                            } else {
                                &model.vis_flags
                            };
                            let Some(index) = table.iter().position(|f| f == flag) else {
                                continue;
                            };
                            let map = if *rendering { &status.rnd } else { &status.vis };
                            let default_on = if *rendering {
                                if *flag == "shadow" {
                                    status.shadows
                                } else {
                                    RND_DEFAULT_ON.contains(flag)
                                }
                            } else {
                                false
                            };
                            let on = map.get(&index.to_string()).copied().unwrap_or(default_on);
                            if toggle(ui, on, label, &format!("MuJoCo {flag}")) {
                                if *rendering {
                                    viewport.send_rnd(index as u32, !on);
                                } else {
                                    viewport.send_vis(index as u32, !on);
                                }
                            }
                        }
                        ui.separator();
                        ui.menu_image_button(re_ui::icons::VIEW_3D.as_image(), |ui| {
                            for (name, preset) in VIEWS {
                                if ui.button(*name).clicked() {
                                    viewport.send_view(*preset);
                                    ui.close();
                                }
                            }
                        })
                        .response
                        .on_hover_text("camera: front, side, top, reset");
                    });
                });
        });
}

// -- the Inspect drawer --------------------------------------------------------

pub fn drawer(ctx: &egui::Context, picture: egui::Rect, viewport: &mut ViewportFeed) {
    let (open, tab) = drawer_state(ctx);
    if !open {
        return;
    }
    let (status, model) = viewport.report();
    let (Some(status), Some(model)) = (status, model) else {
        return;
    };
    let height = picture.height() - 56.0;
    egui::Area::new(egui::Id::new("simulator-drawer"))
        .order(egui::Order::Foreground)
        .fixed_pos(picture.right_bottom() + egui::vec2(-8.0, -8.0))
        .pivot(egui::Align2::RIGHT_BOTTOM)
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(ui.visuals().panel_fill.gamma_multiply(0.92))
                .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
                .corner_radius(8.0)
                .inner_margin(10.0)
                .show(ui, |ui| {
                    ui.set_width(DRAWER_WIDTH);
                    ui.set_max_height(height);
                    let mut tab = tab;
                    ui.horizontal(|ui| {
                        for (t, name) in [
                            (Tab::Control, "Control"),
                            (Tab::Joints, "Joints"),
                            (Tab::Physics, "Physics"),
                        ] {
                            if toggle(ui, tab == t, name, "") {
                                tab = t;
                            }
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .small_icon_button(&re_ui::icons::CLOSE_SMALL, "Close")
                                .clicked()
                            {
                                set_drawer(ui.ctx(), false, tab);
                            }
                        });
                    });
                    if tab != drawer_state(ui.ctx()).1 {
                        set_drawer(ui.ctx(), true, tab);
                    }
                    ui.add_space(4.0);
                    if tab != Tab::Physics {
                        mode_line(ui, viewport, &status);
                    }
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, true])
                        .show(ui, |ui| match tab {
                            Tab::Control => control(ui, viewport, &status, &model),
                            Tab::Joints => joints(ui, viewport, &status, &model),
                            Tab::Physics => physics(ui, &model),
                        });
                });
        });
}

/// One line that says which mode the sliders are in, and flips it.
fn mode_line(ui: &mut egui::Ui, viewport: &mut ViewportFeed, status: &SimStatus) {
    ui.horizontal(|ui| {
        if status.manual {
            ui.label(egui::RichText::new("You drive.").strong());
            if ui
                .small("hand it back")
                .on_hover_text("the scene's own motion, from its start")
                .clicked()
            {
                viewport.send_manual(false);
            }
        } else {
            ui.label(
                egui::RichText::new("The scene runs itself — values shown, not editable.")
                    .color(ui.visuals().weak_text_color()),
            );
            if ui
                .small("drive")
                .on_hover_text("take the controls from here")
                .clicked()
            {
                viewport.send_manual(true);
            }
        }
    });
    ui.add_space(4.0);
}

/// Rows grouped by the name's prefix before the first `/` (an arm, a leg).
fn groups<T>(items: &[T], name: impl Fn(&T) -> &str) -> Vec<(String, Vec<(usize, &T)>)> {
    let mut out: Vec<(String, Vec<(usize, &T)>)> = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let n = name(item);
        let group = n.split_once('/').map_or("", |(g, _)| g).to_owned();
        match out.iter_mut().find(|(g, _)| *g == group) {
            Some((_, rows)) => rows.push((i, item)),
            None => out.push((group, vec![(i, item)])),
        }
    }
    out
}

fn short(name: &str) -> &str {
    name.split_once('/').map_or(name, |(_, s)| s)
}

/// One row: the name, a slider or a bar, the value at a fixed width.
/// Returns a moved-to value and whether a drag just ended.
fn row(
    ui: &mut egui::Ui,
    name: &str,
    range: [f64; 2],
    value: f64,
    editable: bool,
) -> (Option<f64>, bool) {
    let mut v = value;
    let mut moved = None;
    let mut stopped = false;
    ui.horizontal(|ui| {
        ui.add_sized(
            [96.0, ui.spacing().interact_size.y],
            egui::Label::new(egui::RichText::new(short(name)).small()).truncate(),
        )
        .on_hover_text(name);
        let width = DRAWER_WIDTH - 96.0 - VALUE_WIDTH - 24.0;
        if editable {
            ui.spacing_mut().slider_width = width;
            let response = ui.add(
                egui::Slider::new(&mut v, range[0]..=range[1])
                    .clamping(egui::SliderClamping::Never)
                    .show_value(false),
            );
            if response.changed() {
                moved = Some(v);
            }
            stopped = response.drag_stopped();
        } else {
            // A quiet bar: the value's place in its range, in a muted
            // fill — not the accent of something you could drag.
            let span = (range[1] - range[0]).max(1e-9);
            let frac = ((v - range[0]) / span).clamp(0.0, 1.0) as f32;
            ui.add_sized(
                [width, 8.0],
                egui::ProgressBar::new(frac)
                    .fill(ui.visuals().weak_text_color().gamma_multiply(0.55)),
            );
        }
        mono(ui, format!("{v:>8.3}"), VALUE_WIDTH);
    });
    (moved, stopped)
}

fn control(ui: &mut egui::Ui, viewport: &mut ViewportFeed, status: &SimStatus, model: &SimModel) {
    if status.manual
        && ui
            .small("Clear all")
            .on_hover_text("every actuator to 0")
            .clicked()
    {
        for (i, _) in model.actuators.iter().enumerate() {
            viewport.send_ctrl(i as u32, 0.0);
            viewport.stop_editing(1, i);
        }
    }
    for (group, rows) in groups(&model.actuators, |a| &a.name) {
        if !group.is_empty() {
            ui.label(egui::RichText::new(&group).small().strong());
        }
        for (i, actuator) in rows {
            let range = if actuator.limited {
                actuator.range
            } else {
                UNLIMITED_LINEAR
            };
            let echoed = status.ctrl.get(i).copied().unwrap_or(0.0);
            let shown = viewport.slider_value(1, i, echoed);
            let (moved, stopped) = row(ui, &actuator.name, range, shown, status.manual);
            if let Some(v) = moved {
                viewport.send_ctrl(i as u32, v as f32);
            }
            if stopped {
                viewport.stop_editing(1, i);
            }
        }
    }
}

fn joints(ui: &mut egui::Ui, viewport: &mut ViewportFeed, status: &SimStatus, model: &SimModel) {
    for (group, rows) in groups(&model.joints, |j| &j.name) {
        if !group.is_empty() {
            ui.label(egui::RichText::new(&group).small().strong());
        }
        for (_, joint) in rows {
            let range = if joint.limited {
                joint.range
            } else if joint.kind == "hinge" {
                UNLIMITED_HINGE
            } else {
                UNLIMITED_LINEAR
            };
            let echoed = status.qpos.get(joint.qpos).copied().unwrap_or(0.0);
            let shown = viewport.slider_value(0, joint.qpos, echoed);
            let (moved, stopped) = row(ui, &joint.name, range, shown, status.manual);
            if let Some(v) = moved {
                viewport.send_qpos(joint.qpos as u32, v as f32);
            }
            if stopped {
                viewport.stop_editing(0, joint.qpos);
            }
        }
    }
}

/// The model's facts, looked up rarely: simulate's Physics section.
fn physics(ui: &mut egui::Ui, model: &SimModel) {
    egui::Grid::new("simulator_physics")
        .num_columns(2)
        .min_col_width(110.0)
        .spacing([10.0, 4.0])
        .show(ui, |ui| {
            for (key, value) in [
                ("timestep", format!("{} s", model.timestep)),
                ("integrator", model.integrator.clone()),
                (
                    "solver",
                    format!("{} · {} iterations", model.solver, model.iterations),
                ),
                (
                    "gravity",
                    format!(
                        "{}, {}, {} m/s²",
                        model.gravity[0], model.gravity[1], model.gravity[2]
                    ),
                ),
                ("bodies", model.nbody.to_string()),
                ("geoms", model.ngeom.to_string()),
                ("actuators", model.actuators.len().to_string()),
                ("sliding joints", model.joints.len().to_string()),
                (
                    "keyframes",
                    if model.keyframes.is_empty() {
                        "none".to_owned()
                    } else {
                        model.keyframes.join(", ")
                    },
                ),
            ] {
                ui.label(egui::RichText::new(key).color(ui.visuals().weak_text_color()));
                ui.monospace(value);
                ui.end_row();
            }
        });
}

// -- keys ------------------------------------------------------------------------

/// Space runs or pauses, → steps once, R resets — unless a text field has
/// the keyboard.
pub fn shortcuts(ctx: &egui::Context, viewport: &mut ViewportFeed) {
    if !viewport.is_active() || ctx.egui_wants_keyboard_input() {
        return;
    }
    let (status, _) = viewport.report();
    let Some(status) = status else { return };
    let (space, right, reset, inspect_key) = ctx.input(|i| {
        (
            i.key_pressed(egui::Key::Space),
            i.key_pressed(egui::Key::ArrowRight),
            i.key_pressed(egui::Key::R),
            i.key_pressed(egui::Key::I),
        )
    });
    if inspect_key {
        let (open, tab) = drawer_state(ctx);
        set_drawer(ctx, !open, tab);
    }
    if space {
        viewport.send_run(status.paused);
    }
    if right {
        viewport.send_step(1);
    }
    if reset {
        viewport.send_reset(None);
    }
}
