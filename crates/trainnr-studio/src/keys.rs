//! The keyboard as the Studio reads it: one answer to "was this key
//! pressed this frame" and one to "is a text field typing".
//!
//! egui counts a held key's auto-repeat as presses (`key_pressed` and
//! `consume_key` both), so a held Space flipped run/pause at repeat rate
//! and a held W wrote thirty log lines a second (2026-09-12/13). Every
//! shortcut in the crate asks here for the FIRST press only. Two
//! definitions of "typing" (`memory.focused()` vs `text_edit_focused`)
//! went dead against each other once the picture took focus on click;
//! there is one now.

/// Whether `key` went down this frame — the first press, never a repeat.
pub fn first_press(ctx: &egui::Context, key: egui::Key) -> bool {
    ctx.input(|i| i.events.iter().any(|event| is_first_press(event, key)))
}

/// The event that counts as a press of `key`.
pub fn is_first_press(event: &egui::Event, key: egui::Key) -> bool {
    matches!(
        event,
        egui::Event::Key {
            key: pressed_key,
            pressed: true,
            repeat: false,
            ..
        } if *pressed_key == key
    )
}

/// The first of `keys` pressed this frame, if any.
pub fn first_press_of(ctx: &egui::Context, keys: &[egui::Key]) -> Option<egui::Key> {
    ctx.input(|i| {
        i.events
            .iter()
            .find_map(|event| keys.iter().copied().find(|key| is_first_press(event, *key)))
    })
}

/// Whether a text field has the keyboard: the shortcuts stand aside.
pub fn typing(ctx: &egui::Context) -> bool {
    ctx.text_edit_focused()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key_event(key: egui::Key, repeat: bool) -> egui::Event {
        egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat,
            modifiers: egui::Modifiers::NONE,
        }
    }

    #[test]
    fn a_repeat_is_not_a_press() {
        assert!(is_first_press(
            &key_event(egui::Key::Space, false),
            egui::Key::Space
        ));
        assert!(!is_first_press(
            &key_event(egui::Key::Space, true),
            egui::Key::Space
        ));
        assert!(!is_first_press(
            &key_event(egui::Key::R, false),
            egui::Key::Space
        ));
        let release = egui::Event::Key {
            key: egui::Key::Space,
            physical_key: None,
            pressed: false,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        };
        assert!(!is_first_press(&release, egui::Key::Space));
    }
}
