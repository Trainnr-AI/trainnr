//! The window's chrome is ours on every platform that lets a client draw
//! it, so the title bar is the app's colour everywhere and only the
//! buttons differ: the Mac keeps its traffic lights (a full-size content
//! view with the title bar hidden, as Rerun does), Windows and Linux get
//! caption buttons drawn by re_ui at the right of our top bar. Asked for
//! 2026-09-12: "make the OS window border standard for Windows, Mac and
//! Linux, just the difference in close and minimize buttons, the app's
//! colour - the border is grey in WSL" (that grey was WSLg's window
//! manager's frame).
//!
//! Without native decorations the windowing system stops offering resize
//! borders; the invisible edge and corner zones below give them back
//! through `ViewportCommand::BeginResize`, the way Rerun's own viewer
//! does (`re_viewer::app::ui`, private there).

use egui::{CursorIcon, ResizeDirection, ViewportCommand};

const EDGE: f32 = 6.0;
const CORNER: f32 = 16.0;

/// Whether this platform draws the chrome itself (Windows and Linux).
/// The Mac answers false and keeps its traffic lights.
pub fn custom_chrome() -> bool {
    re_ui::supports_custom_decorations(egui::os::OperatingSystem::default())
}

/// The app's ground under the whole window: every gap between panels
/// is this colour, never the desktop behind a transparent window.
pub fn paint_ground(ui: &egui::Ui) {
    let fill = ui.visuals().panel_fill;
    ui.painter().rect_filled(ui.max_rect(), 0.0, fill);
}

/// A title-bar region: press-and-drag moves the window, a double click
/// toggles maximized. Call before the region's widgets so they win input.
pub fn title_bar_interaction(ui: &mut egui::Ui, rect: egui::Rect, id: egui::Id) {
    let response = ui.interact(rect, id, egui::Sense::click());
    if response.double_clicked() {
        let maximized = ui.input(|i| i.viewport().maximized.unwrap_or(false));
        ui.send_viewport_cmd(ViewportCommand::Maximized(!maximized));
    } else if response.is_pointer_button_down_on() {
        ui.send_viewport_cmd(ViewportCommand::StartDrag);
    }
}

/// Invisible resize zones along the window's edges and corners.
pub fn resize_handles(ui: &egui::Ui) {
    let (fullscreen, maximized) = ui.ctx().input(|i| {
        let v = i.viewport();
        (v.fullscreen.unwrap_or(false), v.maximized.unwrap_or(false))
    });
    if fullscreen || maximized {
        return;
    }
    let rect = ui.max_rect();
    let (l, t, r, b) = (rect.left(), rect.top(), rect.right(), rect.bottom());
    let zones = [
        (
            egui::Rect::from_min_max(egui::pos2(l, t), egui::pos2(l + CORNER, t + CORNER)),
            ResizeDirection::NorthWest,
            CursorIcon::ResizeNwSe,
        ),
        (
            egui::Rect::from_min_max(egui::pos2(r - CORNER, t), egui::pos2(r, t + CORNER)),
            ResizeDirection::NorthEast,
            CursorIcon::ResizeNeSw,
        ),
        (
            egui::Rect::from_min_max(egui::pos2(l, b - CORNER), egui::pos2(l + CORNER, b)),
            ResizeDirection::SouthWest,
            CursorIcon::ResizeNeSw,
        ),
        (
            egui::Rect::from_min_max(egui::pos2(r - CORNER, b - CORNER), egui::pos2(r, b)),
            ResizeDirection::SouthEast,
            CursorIcon::ResizeNwSe,
        ),
        (
            egui::Rect::from_min_max(egui::pos2(l + CORNER, t), egui::pos2(r - CORNER, t + EDGE)),
            ResizeDirection::North,
            CursorIcon::ResizeVertical,
        ),
        (
            egui::Rect::from_min_max(egui::pos2(l + CORNER, b - EDGE), egui::pos2(r - CORNER, b)),
            ResizeDirection::South,
            CursorIcon::ResizeVertical,
        ),
        (
            egui::Rect::from_min_max(egui::pos2(l, t + CORNER), egui::pos2(l + EDGE, b - CORNER)),
            ResizeDirection::West,
            CursorIcon::ResizeHorizontal,
        ),
        (
            egui::Rect::from_min_max(egui::pos2(r - EDGE, t + CORNER), egui::pos2(r, b - CORNER)),
            ResizeDirection::East,
            CursorIcon::ResizeHorizontal,
        ),
    ];
    for (i, (zone, direction, cursor)) in zones.iter().enumerate() {
        let response = ui.interact(
            *zone,
            ui.id().with(("window-resize", i)),
            egui::Sense::drag(),
        );
        if response.hovered() || response.dragged() {
            ui.ctx().set_cursor_icon(*cursor);
        }
        if response.drag_started() {
            ui.ctx()
                .send_viewport_cmd(ViewportCommand::BeginResize(*direction));
        }
    }
}

/// The frame's invisible margin under WSLg's window manager, which
/// places a new frameless window with that margin above the screen.
const WSLG_FRAME_MARGIN: f32 = 32.0;

/// Keep the window's outer edge on the screen: WSLg's window manager
/// opened it 21 px above the top (2026-09-12), and a frame partly off
/// the screen is where pointer offsets came from. Cheap enough to run
/// every frame; it sends a command only when something is off-screen.
pub fn keep_on_screen(ctx: &egui::Context) {
    let outer = ctx.input(|i| i.viewport().outer_rect);
    let Some(outer) = outer else {
        return;
    };
    let (x, y) = (outer.min.x, outer.min.y);
    if x < 0.0 || y < 0.0 {
        ctx.send_viewport_cmd(ViewportCommand::OuterPosition(egui::pos2(
            x.max(WSLG_FRAME_MARGIN),
            y.max(WSLG_FRAME_MARGIN),
        )));
    }
}
