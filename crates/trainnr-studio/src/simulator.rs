//! The Simulator page's controls, shaped by what a person does in a
//! simulator, in order of how often: watch the scene; pause, step,
//! reset, change speed; poke the robot; flip a debug overlay; look up a
//! fact about the model. So (the operator's calls, 2026-09-09):
//!
//! - the picture is the page — nothing permanent sits beside it;
//! - a **transport bar** under the picture, the video-player shape and
//!   the shape of Rerun's own timeline right below it: play or pause,
//!   step, ten steps, reset, the keyframes, speed, the scene picker, and
//!   three chips that never move — sim time, real-time factor, frames;
//!   Space, → and R do what they do in `simulate`; W A S D walk the
//!   camera over the ground, Q and E lower and raise it, Shift hurries;
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

use crate::viewport::{
    SimModel, SimStatus, SliderKind, ViewportFeed, PREVIEW_TASKS, SPEED_RANGE, VIEW_PRESETS,
    WALK_TASK,
};

/// What the bar asked the page to do with the viewport itself.
pub enum Action {
    Spawn(String),
    Stop,
    /// The viewport alone on the page, or back (the `f` key, the button).
    ToggleFullscreen,
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
/// The slider's span: the part of the stream's [`SPEED_RANGE`] a thumb
/// can set with any precision (the agent's door reaches the whole range).
const SPEED_SLIDER_RANGE: std::ops::RangeInclusive<f32> = 0.1..=4.0;
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
/// Rendering flags MuJoCo turns on by default (after mjv_defaultScene).
const RND_DEFAULT_ON: &[&str] = &["shadow", "reflection", "skybox", "haze", "cullface"];
/// MuJoCo draws groups below this number by default (`mjv_defaultOption`).
const MJ_DEFAULT_GROUPS_ON: usize = 3;
/// The room a flag cell keeps from the drawer's edge.
const FLAG_CELL_INSET: f32 = 10.0;

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum Tab {
    #[default]
    Control,
    Joints,
    Physics,
    /// Every MuJoCo visualization and rendering flag, and the group
    /// masks - `simulate`'s Visualization, Rendering and Group enable
    /// sections in one list.
    Visuals,
    /// A walk scene's commanded twist: forward, left, turn sliders for
    /// the followed world, on mjlab's own joystick override.
    Commands,
}

impl Tab {
    /// Every tab: the door's word for it, the label the drawer shows. One
    /// table feeds the door, its refusal, the drawer and the tooltip.
    const ALL: &'static [(Tab, &'static str, &'static str)] = &[
        (Tab::Control, "control", "Control"),
        (Tab::Joints, "joints", "Joints"),
        (Tab::Physics, "physics", "Physics"),
        (Tab::Visuals, "visuals", "Visuals"),
        (Tab::Commands, "commands", "Commands"),
    ];
    /// The door's word that closes the drawer.
    const CLOSE: &'static str = "close";

    fn by_slug(slug: &str) -> Option<Tab> {
        Tab::ALL
            .iter()
            .find(|(_, s, _)| *s == slug)
            .map(|(tab, _, _)| *tab)
    }

    /// Whether this tab exists for the scene: the Commands tab needs a
    /// commanded twist to show.
    fn offered(self, model: &SimModel) -> bool {
        self != Tab::Commands || model.twist_ranges.is_some()
    }

    /// The door's choices, for a refusal or a tooltip.
    fn words() -> String {
        let mut words: Vec<&str> = Tab::ALL.iter().map(|(_, s, _)| *s).collect();
        words.push(Tab::CLOSE);
        words.join(", ")
    }

    fn labels() -> String {
        Tab::ALL
            .iter()
            .map(|(_, _, l)| *l)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// The twist axes as the door and the rows name them, in wire order.
/// No slash in a row's name: `short` would take the part after it.
const TWIST_AXES: [(&str, &str); 3] = [("vx", "forward"), ("vy", "left"), ("wz", "turn")];
/// Which world a commanded twist goes to: the followed one (the follow
/// rule's answer now, not the echoed one), else w0.
fn command_world(ctx: &egui::Context, status: &SimStatus) -> i32 {
    follow_target(ctx, follow_state(ctx), &status.worlds).max(0)
}

/// What the human holds: as last sent from here, else as the stream
/// echoes it (a hold begun by another Studio session), else nothing.
fn held_twist(viewport: &ViewportFeed, status: &SimStatus) -> Option<(i32, [f64; 3])> {
    viewport
        .twist_held()
        .map(|(w, t)| (w, t.map(f64::from)))
        .or_else(|| status.twist.as_ref().map(|t| (t.world as i32, t.value)))
}

/// The door's `command`, one axis by name with a value, or `own`: the
/// twist it asks for, checked and not yet sent: `None` hands the
/// worlds back to their policy (`own`), else the world and the held
/// twist with one axis replaced. Checked apart from the sending so a
/// refusal changes nothing (`Command::Simulator` applies every field
/// only once all of them passed).
pub fn twist_from_door(
    ctx: &egui::Context,
    viewport: &ViewportFeed,
    axis: &str,
    value: Option<f32>,
) -> Result<Option<(i32, [f32; 3])>, String> {
    let (status, model) = viewport.report();
    let (Some(status), Some(model)) = (status, model) else {
        return Err("the scene is not described yet".into());
    };
    if model.twist_ranges.is_none() {
        return Err("commands need a walk scene; this scene has none".into());
    }
    if axis == "own" {
        return Ok(None);
    }
    let index = TWIST_AXES
        .iter()
        .position(|(name, _)| *name == axis)
        .ok_or_else(|| format!("no command axis {axis:?}: vx, vy, wz or own"))?;
    let value = value.ok_or("a command axis needs a `value`")?;
    let (world, held) =
        held_twist(viewport, &status).unwrap_or_else(|| (command_world(ctx, &status), [0.0; 3]));
    let mut twist = held.map(|v| v as f32);
    twist[index] = value;
    Ok(Some((world, twist)))
}

/// Send what `twist_from_door` checked.
pub fn apply_twist_from_door(
    ctx: &egui::Context,
    viewport: &mut ViewportFeed,
    twist: Option<(i32, [f32; 3])>,
) {
    match twist {
        None => viewport.send_twist(-1, [0.0; 3]),
        Some((world, twist)) => {
            viewport.send_twist(world, twist);
            set_drawer(ctx, true, Tab::Commands);
        }
    }
}

/// The drawer state `inspect` asks for, checked and not yet applied:
/// `None` closes it, `Some(tab)` opens it there.
pub fn inspect_tab(what: &str) -> Result<Option<Tab>, String> {
    if what.is_empty() || what == Tab::CLOSE {
        return Ok(None);
    }
    Tab::by_slug(what)
        .map(Some)
        .ok_or_else(|| format!("inspect {what:?}: one of {}", Tab::words()))
}

/// Apply what `inspect_tab` checked.
pub fn apply_inspect(ctx: &egui::Context, tab: Option<Tab>) {
    let (_, current) = drawer_state(ctx);
    match tab {
        None => set_drawer(ctx, false, current),
        Some(wanted) => set_drawer(ctx, true, wanted),
    }
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
/// `deployments`: (name, live scene) for every deployment the project
/// holds, offered beside the previews.
pub fn transport(
    ui: &mut egui::Ui,
    viewport: &mut ViewportFeed,
    deployments: &[(String, String)],
) -> Option<Action> {
    let mut action = None;
    let (status, model) = viewport.report();
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 8.0;
        // The scene picker: a named menu, the empty state's only door.
        // In the empty state the picker is the page's one door: it says so.
        let current = viewport.task().unwrap_or("Choose a scene");
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
            if !deployments.is_empty() {
                ui.separator();
                for (name, scene) in deployments {
                    if ui
                        .button(format!("deploy · {name}"))
                        .on_hover_text("the exported policy, live: drive it with WASD")
                        .clicked()
                    {
                        action = Some(Action::Spawn(scene.clone()));
                        ui.close();
                    }
                }
            }
            if viewport.is_active() {
                ui.separator();
                if ui.button("Stop the scene").clicked() {
                    action = Some(Action::Stop);
                    ui.close();
                }
            }
        });
        if let Some(caption) = model.as_ref().and_then(|m| m.caption.as_deref()) {
            ui.label(egui::RichText::new(caption).color(ui.visuals().weak_text_color()));
        }
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
            egui::Slider::new(&mut speed, SPEED_SLIDER_RANGE)
                .clamping(egui::SliderClamping::Never)
                .logarithmic(true)
                .show_value(false),
        );
        mono(ui, format!("{speed:>5.2}×"), 52.0);
        if response.drag_stopped() || (response.changed() && !response.dragged()) {
            viewport.send_speed(speed.clamp(*SPEED_RANGE.start(), *SPEED_RANGE.end()));
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
        // No frame yet is a dash, never a zero the reader could take for
        // a measured rate.
        chip(
            ui,
            "fps",
            viewport
                .fps_settled()
                .map_or_else(|| format!("{:>4}", "–"), |f| format!("{f:>4.0}")),
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
        // Inspect and full screen, at the right end.
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .small_icon_button(&re_ui::icons::CHROME_MAXIMIZE, "Viewport full screen  (F)")
                .clicked()
            {
                action = Some(Action::ToggleFullscreen);
            }
            let (open, tab) = drawer_state(ui.ctx());
            if toggle(ui, open, "Inspect", &format!("{}  (I)", Tab::labels())) {
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
        // Rerun's own status colours, so the dots match every other
        // success and error mark in the window.
        let color = if world.done {
            ui.visuals().error_fg_color
        } else {
            ui.tokens().alert_success.icon
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
                            let Some(index) = flag_index(&model, flag, *rendering) else {
                                continue;
                            };
                            let on = flag_on(&status, index, flag, *rendering);
                            if toggle(ui, on, label, &format!("MuJoCo {flag}")) {
                                send_flag(viewport, index, *rendering, !on);
                            }
                        }
                        ui.separator();
                        ui.menu_image_button(re_ui::icons::VIEW_3D.as_image(), |ui| {
                            for view in VIEW_PRESETS {
                                if ui.button(view.label).clicked() {
                                    viewport.send_view(view.index);
                                    ui.close();
                                }
                            }
                        })
                        .response
                        .on_hover_text(format!(
                            "camera: {}",
                            crate::viewport::view_preset_names().join(", ")
                        ));
                    });
                });
        });
}

/// A flag's wire index in the model's table for its kind.
fn flag_index(model: &SimModel, flag: &str, rendering: bool) -> Option<usize> {
    let table = if rendering {
        &model.rnd_flags
    } else {
        &model.vis_flags
    };
    table.iter().position(|f| f == flag)
}

/// A flag as rendered — the status echoes every flag's value; before the
/// first status the defaults are MuJoCo's (rendering flags mostly on,
/// visualization flags off), and shadows are whatever the stream chose.
fn flag_on(status: &SimStatus, index: usize, flag: &str, rendering: bool) -> bool {
    let table = if rendering {
        &status.rnd_table
    } else {
        &status.vis_table
    };
    let default_on = match (rendering, flag) {
        (true, "shadow") => status.shadows,
        (true, _) => RND_DEFAULT_ON.contains(&flag),
        (false, _) => false,
    };
    table.get(index).copied().flatten().unwrap_or(default_on)
}

fn send_flag(viewport: &mut ViewportFeed, index: usize, rendering: bool, on: bool) {
    if rendering {
        viewport.send_rnd(index as u32, on);
    } else {
        viewport.send_vis(index as u32, on);
    }
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
                        for (t, _, name) in Tab::ALL.iter().filter(|(t, _, _)| t.offered(&model)) {
                            if toggle(ui, tab == *t, name, "") {
                                tab = *t;
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
                    if matches!(tab, Tab::Control | Tab::Joints) {
                        mode_line(ui, viewport, &status);
                    }
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, true])
                        .show(ui, |ui| match tab {
                            Tab::Control => control(ui, viewport, &status, &model),
                            Tab::Joints => joints(ui, viewport, &status, &model),
                            Tab::Physics => physics(ui, &model),
                            Tab::Visuals => visuals(ui, viewport, &status, &model),
                            Tab::Commands => commands(ui, viewport, &status, &model),
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
            viewport.stop_editing(SliderKind::Actuator, i);
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
            // An actuator the status has not echoed yet has no value to
            // show; a row of 0.000 would read as one.
            let Some(echoed) = status.ctrl.get(i).copied() else {
                continue;
            };
            let shown = viewport.slider_value(SliderKind::Actuator, i, echoed);
            let (moved, stopped) = row(ui, &actuator.name, range, shown, status.manual);
            if let Some(v) = moved {
                viewport.send_ctrl(i as u32, v as f32);
            }
            if stopped {
                viewport.stop_editing(SliderKind::Actuator, i);
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
            let Some(echoed) = status.qpos.get(joint.qpos).copied() else {
                continue; // not echoed yet: no value to show
            };
            let shown = viewport.slider_value(SliderKind::Joint, joint.qpos, echoed);
            let (moved, stopped) = row(ui, &joint.name, range, shown, status.manual);
            if let Some(v) = moved {
                viewport.send_qpos(joint.qpos as u32, v as f32);
            }
            if stopped {
                viewport.stop_editing(SliderKind::Joint, joint.qpos);
            }
        }
    }
}

/// The model's facts, looked up rarely: simulate's Physics section.
/// The walk's twist for one world: three sliders bounded by the task's
/// own command ranges. Moving one takes the commands for the followed
/// world (else w0) on mjlab's joystick override; "hand back" returns
/// them to the task's sampler. The rows show what the human holds.
fn commands(ui: &mut egui::Ui, viewport: &mut ViewportFeed, status: &SimStatus, model: &SimModel) {
    let Some(ranges) = model.twist_ranges else {
        return;
    };
    let held = held_twist(viewport, status);
    let world = held.map_or_else(|| command_world(ui.ctx(), status), |(w, _)| w);
    ui.horizontal(|ui| {
        if let Some((held_world, _)) = held {
            ui.label(egui::RichText::new(format!("You command w{held_world}.")).strong());
            if ui
                .small("hand back")
                .on_hover_text("the task's own commands again")
                .clicked()
            {
                viewport.send_twist(-1, [0.0; 3]);
            }
        } else {
            ui.label(
                egui::RichText::new(format!(
                    "The task commands. A slider commands w{world} (the followed world)."
                ))
                .color(ui.visuals().weak_text_color()),
            );
        }
    });
    ui.label(
        egui::RichText::new("metres per second forward and left, radians per second turn")
            .small()
            .color(ui.visuals().weak_text_color()),
    );
    let values = held.map_or([0.0; 3], |(_, v)| v);
    let mut twist = values;
    let mut moved = false;
    let mut stopped = None;
    for (i, (name, unit)) in TWIST_AXES.iter().enumerate() {
        let shown = viewport.slider_value(SliderKind::Twist, i, values[i]);
        let (change, done) = row(ui, &format!("{name} {unit}"), ranges[i], shown, true);
        twist[i] = change.unwrap_or(shown);
        if let Some(v) = change {
            viewport.start_editing(SliderKind::Twist, i, v);
            moved = true;
        }
        if done {
            stopped = Some(i);
        }
    }
    if moved {
        viewport.send_twist(world, twist.map(|v| v as f32));
    }
    if let Some(i) = stopped {
        viewport.stop_editing(SliderKind::Twist, i);
    }
}

/// The whole switchboard: every visualization flag, every rendering
/// flag, and the group masks, each a checkbox that sends its bit — the
/// stream's status echoes what is rendered, so a box shows the truth.
fn visuals(ui: &mut egui::Ui, viewport: &mut ViewportFeed, status: &SimStatus, model: &SimModel) {
    for (heading, rendering, table) in [
        ("Visualization", false, &model.vis_flags),
        ("Rendering", true, &model.rnd_flags),
    ] {
        ui.label(egui::RichText::new(heading).strong());
        flag_grid(ui, heading, table, |ui, index, flag| {
            let mut on = flag_on(status, index, flag, rendering);
            if ui.checkbox(&mut on, flag).changed() {
                send_flag(viewport, index, rendering, on);
            }
        });
        ui.add_space(6.0);
    }
    if model.groups.is_empty() {
        return;
    }
    ui.label(egui::RichText::new("Groups").strong())
        .on_hover_text("which group numbers of each kind are drawn (MuJoCo's group enable)");
    egui::Grid::new("simulator_groups")
        .num_columns(1 + model.ngroup as usize)
        .spacing([6.0, 4.0])
        .show(ui, |ui| {
            ui.label("");
            for g in 0..model.ngroup {
                ui.label(egui::RichText::new(g.to_string()).weak());
            }
            ui.end_row();
            for (kind_index, kind) in model.groups.iter().enumerate() {
                ui.label(kind);
                let mask = status.groups.get(kind);
                for g in 0..model.ngroup as usize {
                    let reported = mask.and_then(|m| m.get(g).copied());
                    let mut on = reported.unwrap_or(g < MJ_DEFAULT_GROUPS_ON);
                    // Until the stream reports the mask the box shows MuJoCo's
                    // documented default and takes no click: never a value
                    // the window invented.
                    let box_ = ui.add_enabled(reported.is_some(), egui::Checkbox::new(&mut on, ""));
                    if reported.is_none() {
                        box_.on_hover_text("unrecorded until the scene reports its groups");
                    } else if box_.changed() {
                        if let (Ok(kind), Ok(group)) = (u8::try_from(kind_index), u8::try_from(g)) {
                            viewport.send_group(kind, group, on);
                        }
                    }
                }
                ui.end_row();
            }
        });
}

/// Flags in the stream's index order, FLAG_COLUMNS to a row: MuJoCo has
/// thirty-odd visualization flags, and the drawer is a column.
const FLAG_COLUMNS: usize = 3;

fn flag_grid(
    ui: &mut egui::Ui,
    id: &str,
    table: &[String],
    mut cell: impl FnMut(&mut egui::Ui, usize, &str),
) {
    egui::Grid::new(format!("simulator_flags_{id}"))
        .num_columns(FLAG_COLUMNS)
        .min_col_width(DRAWER_WIDTH / FLAG_COLUMNS as f32 - FLAG_CELL_INSET)
        .spacing([4.0, 1.0])
        .show(ui, |ui| {
            for (index, flag) in table.iter().enumerate() {
                cell(ui, index, flag);
                if index % FLAG_COLUMNS == FLAG_COLUMNS - 1 {
                    ui.end_row();
                }
            }
        });
}

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
                // Hinge and slide joints: the ones with one scalar (MuJoCo's
                // free and ball joints have none, and are not listed).
                (
                    "scalar joints (hinge, slide)",
                    model.joints.len().to_string(),
                ),
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

/// Shift on a pan key moves the camera this much faster.
const PAN_HURRY: f32 = 3.0;

/// A pan key pressed this frame and what decided its fate, for the
/// event log.
pub struct KeyPress {
    pub key: String,
    /// Whether the keys were the picture's (`wants_keys`).
    pub picture: bool,
    pub focus: &'static str,
    pub pointer: Option<egui::Pos2>,
}

const PAN_KEYS: [egui::Key; 6] = [
    egui::Key::W,
    egui::Key::A,
    egui::Key::S,
    egui::Key::D,
    egui::Key::Q,
    egui::Key::E,
];

/// The pan key pressed this frame, if one was — recorded whether or
/// not it acted, so a silent key can be explained from the log. The
/// first press only: egui reports a held key's auto-repeat as presses
/// too, and a held W wrote thirty lines a second. Never while a text
/// field has the keyboard: the log is not a keystroke recorder.
fn pan_key_press(ctx: &egui::Context, viewport: &ViewportFeed) -> Option<KeyPress> {
    if crate::keys::typing(ctx) {
        return None;
    }
    let key = crate::keys::first_press_of(ctx, &PAN_KEYS);
    let pointer = ctx.input(|i| i.pointer.latest_pos());
    key.map(|key| KeyPress {
        key: format!("{key:?}"),
        picture: viewport.wants_keys(ctx),
        focus: viewport.focus_owner(ctx),
        pointer,
    })
}

/// Space runs or pauses, → steps once, R resets, W A S D Q E pan the
/// camera while held — when the keys are the picture's (`wants_keys`).
/// Returns the pan key pressed this frame for the event log.
pub fn shortcuts(ctx: &egui::Context, viewport: &mut ViewportFeed) -> Option<KeyPress> {
    let press = pan_key_press(ctx, viewport);
    if !viewport.wants_keys(ctx) {
        return press;
    }
    pan_keys(ctx, viewport);
    let (status, _) = viewport.report();
    let Some(status) = status else { return press };
    // First presses only: a held Space flipped run/pause at repeat rate.
    let space = crate::keys::first_press(ctx, egui::Key::Space);
    let right = crate::keys::first_press(ctx, egui::Key::ArrowRight);
    let reset = crate::keys::first_press(ctx, egui::Key::R);
    let inspect_key = crate::keys::first_press(ctx, egui::Key::I);
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
    press
}

/// The first-person walk every 3D tool has: W/S along the view, A/D
/// across it, Q/E down and up, for as long as the key is held. Each
/// frame sends the seconds the keys were down (`stable_dt`), so the
/// speed is the stream's to set and the frame rate does not change it.
/// A held key raises no event, so the frame asks for the next one
/// itself. Panning releases a followed world: the follow would put the
/// lookat straight back.
fn pan_keys(ctx: &egui::Context, viewport: &mut ViewportFeed) {
    use egui::Key::{A, D, E, Q, S, W};
    let (forward, right, up, hurry, dt) = ctx.input(|i| {
        let axis = |plus: egui::Key, minus: egui::Key| {
            f32::from(i.key_down(plus)) - f32::from(i.key_down(minus))
        };
        (
            axis(W, S),
            axis(D, A),
            axis(E, Q),
            i.modifiers.shift,
            i.stable_dt,
        )
    });
    if forward == 0.0 && right == 0.0 && up == 0.0 {
        return;
    }
    let seconds = if hurry { dt * PAN_HURRY } else { dt };
    if follow_state(ctx) != Follow::None {
        set_follow(ctx, Follow::None);
    }
    viewport.send_pan(forward * seconds, right * seconds, up * seconds);
    ctx.request_repaint();
}
