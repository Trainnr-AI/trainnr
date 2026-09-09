//! The Simulator page's control panel: MuJoCo's own `simulate` sections
//! — Simulation (run, step, reset, keyframes, speed), Physics (the clock
//! and the model's facts), Joint and Control sliders, Visualization and
//! Rendering flags — drawn with Rerun's own widgets over the stream's
//! status messages; every change is one tagged message on the viewport's
//! wire (docs/76 §10.2, docs/e2e-research/74 §3 for the sections'
//! provenance).
//!
//! The panel lives INSIDE the viewport strip, beside the picture, not at
//! the window's edge: the Rerun viewer below owns the window's right edge
//! for its selection panel. Every number is drawn at a fixed width and
//! the frame rate is refreshed once a second: a value that changes width
//! ten times a second moves everything beside it — "the frame number is
//! changing constantly", the flicker Prakhar named on 2026-09-09.
//!
//! Nothing here touches MuJoCo: the stream's status says what is, the
//! wire says what the human asked, and the physics process decides.

use re_ui::UiExt as _;

use crate::viewport::{SimModel, SimStatus, ViewportFeed};

// Sliders never clamp on display: egui would otherwise pull a value that
// sits outside its nominal range (a joint past its limit, an unlimited
// actuator) back inside and report it as a change — which took manual
// control of the scene before anyone touched anything (2026-09-09).
/// A joint or actuator with no declared range still needs a slider.
const UNLIMITED_HINGE: [f64; 2] = [-std::f64::consts::PI, std::f64::consts::PI];
const UNLIMITED_LINEAR: [f64; 2] = [-1.0, 1.0];
/// Speed slider bounds (simulate's Speed goes further; the stream clamps).
const SPEED_RANGE: std::ops::RangeInclusive<f32> = 0.1..=4.0;
/// The panel's width inside the strip, and the value column's width.
pub const PANEL_WIDTH: f32 = 330.0;
const VALUE_WIDTH: f32 = 64.0;
const INNER_WIDTH: f32 = PANEL_WIDTH - 24.0;
/// Rendering flags MuJoCo turns on by default (after mjv_defaultScene).
const RND_DEFAULT_ON: &[&str] = &["shadow", "reflection", "skybox", "haze", "cullface"];
/// The visualization flags shown; the rest of `mjtVisFlag` stay reachable
/// through the agent's door by name.
const VIS_SHOWN: &[&str] = &[
    "contactpoint",
    "contactforce",
    "joint",
    "actuator",
    "constraint",
    "inertia",
    "com",
    "transparent",
    "perturbforce",
    "camera",
    "light",
    "tendon",
];
const RND_SHOWN: &[&str] = &["shadow", "reflection", "skybox", "fog", "wireframe"];

pub fn controls(ui: &mut egui::Ui, viewport: &mut ViewportFeed) {
    let (status, model) = viewport.report();
    let Some(status) = status else {
        ui.add_space(8.0);
        ui.info_label("Waiting for the simulator's first status…");
        return;
    };
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.set_min_width(INNER_WIDTH);
            ui.collapsing_header("Simulation", true, |ui| {
                simulation(ui, viewport, &status, model.as_ref());
            });
            ui.collapsing_header("Physics", true, |ui| {
                physics(ui, viewport, &status, model.as_ref());
            });
            if let Some(model) = model.as_ref() {
                ui.collapsing_header(format!("Control ({})", model.actuators.len()), true, |ui| {
                    control(ui, viewport, &status, model)
                });
                ui.collapsing_header(format!("Joint ({})", model.joints.len()), false, |ui| {
                    joint(ui, viewport, &status, model);
                });
                ui.collapsing_header("Visualization", false, |ui| {
                    visualization(ui, viewport, &status, model);
                });
                ui.collapsing_header("Rendering", false, |ui| {
                    rendering(ui, viewport, &status, model);
                });
            }
        });
}

/// Run, pause, step, reset, keyframes, speed, manual.
fn simulation(
    ui: &mut egui::Ui,
    viewport: &mut ViewportFeed,
    status: &SimStatus,
    model: Option<&SimModel>,
) {
    ui.horizontal(|ui| {
        if status.paused {
            if ui.small_icon_button(&re_ui::icons::PLAY, "Run").clicked() {
                viewport.send_run(true);
            }
        } else if ui
            .small_icon_button(&re_ui::icons::PAUSE, "Pause")
            .clicked()
        {
            viewport.send_run(false);
        }
        if ui
            .small("Step")
            .on_hover_text("one physics step; pauses and takes manual control")
            .clicked()
        {
            viewport.send_step(1);
        }
        if ui.small("×10").on_hover_text("ten physics steps").clicked() {
            viewport.send_step(10);
        }
        if ui
            .small_icon_button(
                &re_ui::icons::RESET,
                "Reset: the scene's initial state, its own motion",
            )
            .clicked()
        {
            viewport.send_reset(None);
        }
        let mut manual = status.manual;
        if ui
            .re_checkbox(&mut manual, "manual")
            .on_hover_text("the sliders drive the scene; off returns it to its own motion")
            .changed()
        {
            viewport.send_manual(manual);
        }
    });
    if let Some(model) = model.filter(|m| !m.keyframes.is_empty()) {
        ui.horizontal_wrapped(|ui| {
            weak(ui, "keyframe");
            for (k, name) in model.keyframes.iter().enumerate() {
                if ui
                    .small(name)
                    .on_hover_text("reset to this keyframe")
                    .clicked()
                {
                    viewport.send_reset(Some(k as u32));
                }
            }
        });
    }
    ui.horizontal(|ui| {
        weak(ui, "speed");
        let mut speed = status.speed as f32;
        ui.spacing_mut().slider_width = INNER_WIDTH - VALUE_WIDTH - 70.0;
        let response = ui.add(
            egui::Slider::new(&mut speed, SPEED_RANGE)
                .clamping(egui::SliderClamping::Never)
                .logarithmic(true)
                .show_value(false),
        );
        mono(ui, format!("{speed:>5.2}×"));
        if response.drag_stopped() || (response.changed() && !response.dragged()) {
            viewport.send_speed(speed);
        }
    });
}

/// The clock and the model's facts: simulate's Physics section, read-only.
fn physics(
    ui: &mut egui::Ui,
    viewport: &ViewportFeed,
    status: &SimStatus,
    model: Option<&SimModel>,
) {
    egui::Grid::new("simulator_physics")
        .num_columns(2)
        .min_col_width(110.0)
        .spacing([10.0, 3.0])
        .show(ui, |ui| {
            fact(ui, "sim time", format!("{:>9.3} s", status.time));
            fact(ui, "real-time factor", format!("{:>9.2}", status.rtf));
            fact(
                ui,
                "on screen",
                format!("{:>5.0} fps", viewport.fps_settled().unwrap_or(0.0)),
            );
            fact(ui, "render", format!("{:>7.1} ms", status.render_ms));
            fact(
                ui,
                "shadows",
                if status.shadows { "on" } else { "off (budget)" }.to_owned(),
            );
            if let Some(m) = model {
                fact(ui, "timestep", format!("{} s", m.timestep));
                fact(ui, "integrator", m.integrator.clone());
                fact(
                    ui,
                    "solver",
                    format!("{} · {} iter", m.solver, m.iterations),
                );
                fact(
                    ui,
                    "gravity",
                    format!("{}, {}, {}", m.gravity[0], m.gravity[1], m.gravity[2]),
                );
                fact(ui, "bodies · geoms", format!("{} · {}", m.nbody, m.ngeom));
            }
        });
}

fn fact(ui: &mut egui::Ui, key: &str, value: String) {
    weak(ui, key);
    mono(ui, value);
    ui.end_row();
}

fn weak(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).color(ui.visuals().weak_text_color()));
}

/// A number at a fixed width, so the layout never moves with it.
fn mono(ui: &mut egui::Ui, text: String) {
    ui.add_sized(
        [VALUE_WIDTH, ui.spacing().interact_size.y],
        egui::Label::new(egui::RichText::new(text).monospace()),
    );
}

/// One slider row: the name above, the slider and its value beside.
/// Returns the new value when the human moved it, and whether the drag
/// just ended.
fn slider_row(ui: &mut egui::Ui, name: &str, range: [f64; 2], value: f64) -> (Option<f64>, bool) {
    let mut v = value;
    ui.label(egui::RichText::new(name).small());
    let (changed, stopped) = ui
        .horizontal(|ui| {
            ui.spacing_mut().slider_width = INNER_WIDTH - VALUE_WIDTH - 24.0;
            let response = ui.add(
                egui::Slider::new(&mut v, range[0]..=range[1])
                    .clamping(egui::SliderClamping::Never)
                    .show_value(false),
            );
            mono(ui, format!("{v:>8.3}"));
            (response.changed(), response.drag_stopped())
        })
        .inner;
    (changed.then_some(v), stopped)
}

/// Control: a slider per actuator in its control range.
fn control(ui: &mut egui::Ui, viewport: &mut ViewportFeed, status: &SimStatus, model: &SimModel) {
    if ui.small("Clear all").clicked() {
        for (i, _) in model.actuators.iter().enumerate() {
            viewport.send_ctrl(i as u32, 0.0);
            viewport.stop_editing(1, i);
        }
    }
    for (i, actuator) in model.actuators.iter().enumerate() {
        let range = if actuator.limited {
            actuator.range
        } else {
            UNLIMITED_LINEAR
        };
        let echoed = status.ctrl.get(i).copied().unwrap_or(0.0);
        let shown = viewport.slider_value(1, i, echoed);
        let (moved, stopped) = slider_row(ui, &actuator.name, range, shown);
        if let Some(v) = moved {
            viewport.send_ctrl(i as u32, v as f32);
        }
        if stopped {
            viewport.stop_editing(1, i);
        }
    }
}

/// Joint: a slider per hinge or slide joint (free and ball have none).
fn joint(ui: &mut egui::Ui, viewport: &mut ViewportFeed, status: &SimStatus, model: &SimModel) {
    for joint in &model.joints {
        let range = if joint.limited {
            joint.range
        } else if joint.kind == "hinge" {
            UNLIMITED_HINGE
        } else {
            UNLIMITED_LINEAR
        };
        let echoed = status.qpos.get(joint.qpos).copied().unwrap_or(0.0);
        let shown = viewport.slider_value(0, joint.qpos, echoed);
        let (moved, stopped) = slider_row(ui, &joint.name, range, shown);
        if let Some(v) = moved {
            viewport.send_qpos(joint.qpos as u32, v as f32);
        }
        if stopped {
            viewport.stop_editing(0, joint.qpos);
        }
    }
}

/// Visualization: MuJoCo's `mjtVisFlag` names, off unless turned on.
fn visualization(
    ui: &mut egui::Ui,
    viewport: &mut ViewportFeed,
    status: &SimStatus,
    model: &SimModel,
) {
    for name in VIS_SHOWN {
        let Some(index) = model.vis_flags.iter().position(|f| f == name) else {
            continue;
        };
        let mut on = status.vis.get(&index.to_string()).copied().unwrap_or(false);
        if ui.re_checkbox(&mut on, *name).changed() {
            viewport.send_vis(index as u32, on);
        }
    }
}

/// Rendering: MuJoCo's `mjtRndFlag` names, at MuJoCo's defaults.
fn rendering(ui: &mut egui::Ui, viewport: &mut ViewportFeed, status: &SimStatus, model: &SimModel) {
    for name in RND_SHOWN {
        let Some(index) = model.rnd_flags.iter().position(|f| f == name) else {
            continue;
        };
        let default_on = if *name == "shadow" {
            status.shadows
        } else {
            RND_DEFAULT_ON.contains(name)
        };
        let mut on = status
            .rnd
            .get(&index.to_string())
            .copied()
            .unwrap_or(default_on);
        if ui.re_checkbox(&mut on, *name).changed() {
            viewport.send_rnd(index as u32, on);
        }
    }
}
