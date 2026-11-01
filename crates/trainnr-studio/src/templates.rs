//! Project templates, as the New project dialog offers them: mirrored
//! from `trainnr/templates.py::TEMPLATES` (checked equal by
//! `test_studio_mirrors`). A template is a plan from raw video to a robot
//! policy with NVIDIA's models by default (docs/86 §6); its steps reach
//! the Overview through the index, not from here.

/// A template's card in the New project dialog.
pub struct TemplateCard {
    pub name: &'static str,
    pub title: &'static str,
    pub summary: &'static str,
    pub robot: &'static str,
    pub vendor: &'static str,
}

pub const TEMPLATES: &[TemplateCard] = &[
    TemplateCard {
        name: "humanoid-motion-from-video",
        title: "Humanoid motion from video",
        summary: "Film a person moving; a humanoid learns the motion in simulation, tracking it without falling.",
        robot: "Unitree G1",
        vendor: "NVIDIA",
    },
    TemplateCard {
        name: "manipulation-from-human-video",
        title: "Manipulation from human video",
        summary: "Film hands doing a task; the arm replays each motion in simulation and keeps the ones that succeed.",
        robot: "SO-101",
        vendor: "NVIDIA",
    },
    TemplateCard {
        name: "more-from-your-demonstrations",
        title: "More from your own demonstrations",
        summary: "Vary how your teleoperated episodes look, then measure whether the variations help.",
        robot: "SO-101",
        vendor: "NVIDIA",
    },
];
