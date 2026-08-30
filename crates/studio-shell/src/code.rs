//! The code panel: a syntax-highlighted editor over the repo, toggled
//! from the header — file tree on the left, editor filling the rest.
//!
//! NOT an embedded Lapce (or Zed): both are whole applications on their
//! own UI frameworks (Floem, GPUI) with their own event loops — there is
//! nothing to call inside an egui frame. The Rerun embed worked ONLY
//! because Rerun's viewer is itself an egui app. This panel instead uses
//! what the tree already ships: `egui_extras::syntax_highlighting`
//! (syntect under the `syntect` feature — Python, XML/MJCF, TOML, Rust,
//! everything this repo is made of), memoized per frame by egui itself.
//!
//! The Studio's editing story stays honest: this is for reading agent
//! output and making focused edits without leaving the window; deep
//! editing sessions still belong in a real editor next door.

use std::path::{Path, PathBuf};

use egui_extras::syntax_highlighting::{highlight, CodeTheme};

/// Directories the tree never descends into — build output, VCS
/// internals, caches. Everything else is the product.
const SKIP_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    "__pycache__",
    ".venv",
    ".ruff_cache",
    ".pytest_cache",
];

/// Refuse to open files past this size — the editor is for source, and
/// a multi-megabyte recording opened as text freezes the layouter.
const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;

/// The file-tree strip's starting width inside the panel.
const FILE_TREE_WIDTH: f32 = 220.0;

/// One file open in the editor.
struct OpenFile {
    path: PathBuf,
    content: String,
    /// Edited in the panel and not yet saved.
    dirty: bool,
    /// What syntect keys highlighting on — the file extension.
    language: String,
}

/// The panel's state: the repo root it browses, and whichever file is
/// open. The tree itself is stateless — directories are read on demand
/// each frame for expanded nodes only (immediate mode; a collapsed
/// node's body never runs).
pub struct CodePanel {
    root: PathBuf,
    open: Option<OpenFile>,
    /// A load/save failure, shown until the next successful action.
    error: Option<String>,
}

impl CodePanel {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            open: None,
            error: None,
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui) {
        // Cmd+S (Ctrl+S elsewhere) saves the open file — consumed here,
        // before any widget, same pattern as the composer's Enter.
        if ui.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::S)) {
            self.save();
        }

        egui::Panel::left("code_file_tree")
            .resizable(true)
            .default_size(FILE_TREE_WIDTH)
            .show(ui, |ui| {
                egui::ScrollArea::both()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let root = self.root.clone();
                        self.tree(ui, &root);
                    });
            });

        self.editor(ui);
    }

    /// One directory level of the file tree. Recursion happens through
    /// egui: an expanded directory's `CollapsingHeader` body calls back
    /// in for its children.
    fn tree(&mut self, ui: &mut egui::Ui, dir: &Path) {
        let mut entries: Vec<(bool, PathBuf)> = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                let name = file_name(p);
                !(p.is_dir() && SKIP_DIRS.contains(&name)) && name != ".DS_Store"
            })
            .map(|p| (p.is_dir(), p))
            .collect();
        // Directories first, then files, both alphabetical — every file
        // browser's order, because it is the readable one.
        entries.sort_by(|a, b| (!a.0, file_name(&a.1)).cmp(&(!b.0, file_name(&b.1))));

        for (is_dir, path) in entries {
            let name = file_name(&path).to_owned();
            if is_dir {
                egui::CollapsingHeader::new(&name)
                    .id_salt(&path)
                    .show(ui, |ui| self.tree(ui, &path));
            } else {
                let open = self.open.as_ref().is_some_and(|f| f.path == path);
                if ui.selectable_label(open, &name).clicked() {
                    self.load(path);
                }
            }
        }
    }

    fn editor(&mut self, ui: &mut egui::Ui) {
        if let Some(error) = &self.error {
            use re_ui::UiExt as _;
            ui.error_label(error);
        }
        let Some(file) = &self.open else {
            ui.centered_and_justified(|ui| {
                ui.weak("pick a file");
            });
            return;
        };

        // Header row reads state and records intent; the actions run
        // after the closure, when no borrow of `self.open` is live.
        let rel = file
            .path
            .strip_prefix(&self.root)
            .unwrap_or(&file.path)
            .display()
            .to_string();
        let dirty = file.dirty;
        let (mut do_save, mut do_reload) = (false, false);
        ui.horizontal(|ui| {
            ui.monospace(rel);
            if dirty {
                ui.strong("●");
                do_save = ui.button("Save").clicked();
            }
            // Agents edit files too — offer the disk's truth back.
            do_reload = ui.button("Reload").clicked();
        });
        if do_save {
            self.save();
        }
        if do_reload {
            if let Some(path) = self.open.as_ref().map(|f| f.path.clone()) {
                self.load(path);
            }
        }

        let theme = CodeTheme::from_memory(ui.ctx(), ui.style());
        let Some(file) = &mut self.open else { return };
        let language = file.language.clone();
        let mut layouter = |ui: &egui::Ui, buf: &dyn egui::TextBuffer, wrap_width: f32| {
            let mut job = highlight(ui.ctx(), ui.style(), &theme, buf.as_str(), &language);
            job.wrap.max_width = wrap_width;
            ui.fonts_mut(|f| f.layout_job(job))
        };
        egui::ScrollArea::both()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let response = ui.add(
                    egui::TextEdit::multiline(&mut file.content)
                        .font(egui::TextStyle::Monospace)
                        .code_editor()
                        .desired_width(f32::INFINITY)
                        .layouter(&mut layouter),
                );
                if response.changed() {
                    file.dirty = true;
                }
            });
    }

    fn load(&mut self, path: PathBuf) {
        match std::fs::metadata(&path) {
            Ok(meta) if meta.len() > MAX_FILE_BYTES => {
                self.error = Some(format!(
                    "{} is {} bytes — past the {MAX_FILE_BYTES}-byte editor limit",
                    file_name(&path),
                    meta.len()
                ));
                return;
            }
            Err(err) => {
                self.error = Some(format!("{}: {err}", file_name(&path)));
                return;
            }
            Ok(_) => {}
        }
        match std::fs::read_to_string(&path) {
            Ok(content) => {
                let language = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("txt")
                    .to_owned();
                self.open = Some(OpenFile {
                    path,
                    content,
                    dirty: false,
                    language,
                });
                self.error = None;
            }
            Err(err) => {
                // Binary or unreadable — named, not silently ignored.
                self.error = Some(format!("{}: {err}", file_name(&path)));
            }
        }
    }

    fn save(&mut self) {
        if let Some(file) = &mut self.open {
            if file.dirty {
                match std::fs::write(&file.path, &file.content) {
                    Ok(()) => {
                        file.dirty = false;
                        self.error = None;
                    }
                    Err(err) => self.error = Some(format!("save failed: {err}")),
                }
            }
        }
    }
}

fn file_name(path: &Path) -> &str {
    path.file_name().and_then(|n| n.to_str()).unwrap_or("?")
}
