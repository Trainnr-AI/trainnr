//! Forms over the agent's tools: a page's buttons, and the dialog each
//! opens (the desktop app plan, docs/85, 2026-10-11).
//!
//! Which tools have a form, on which page, and what each argument is
//! beyond its type (an artifact to pick, a file to choose, one of a few
//! words) is the Python side's registry (`trainnr/forms.py`), printed by
//! `trainnr forms` with every field filled in from the tool's own input
//! schema, the one an agent reads. The Studio loads it once per session,
//! draws the fields, and runs a submitted form as `trainnr call <tool>
//! --args '<json>'`: a person in the app and an agent over MCP run the
//! same tool, with the same checks and the same refusals.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};

use re_ui::DesignTokens;
use serde::Deserialize;

use crate::pages::Section;

/// `trainnr/forms.py::SCHEMA`; the family before the `/` must match.
pub const SCHEMA: &str = "trainnr-forms/1";
/// A file field's extension meaning any file (`trainnr/forms.py::ANY_FILE`).
const ANY_FILE: &str = "*";
/// The form dialog's width.
const DIALOG_WIDTH: f32 = 460.0;

/// One argument of a form, as `trainnr forms` describes it.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Field {
    pub name: String,
    pub label: String,
    #[serde(rename = "type")]
    pub kind_of_value: String,
    #[serde(default)]
    pub required: bool,
    #[serde(default)]
    pub default: Option<serde_json::Value>,
    #[serde(default)]
    pub description: String,
    /// An artifact kind of the open project, picked from a list.
    #[serde(default)]
    pub kind: Option<String>,
    /// A file to choose, by these extensions (`*`: any file).
    #[serde(default)]
    pub files: Vec<String>,
    /// A folder may be chosen too.
    #[serde(default)]
    pub folder: bool,
    /// One of these words.
    #[serde(default)]
    pub choices: Vec<String>,
}

/// A tool as a button on a page.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Form {
    pub tool: String,
    pub page: String,
    pub label: String,
    #[serde(default)]
    pub summary: String,
    pub fields: Vec<Field>,
}

#[derive(Deserialize)]
struct Listing {
    schema: String,
    forms: Vec<Form>,
}

/// `trainnr forms`'s output as forms, refused when its schema family is
/// not this one.
pub fn parse(text: &str) -> Result<Vec<Form>, String> {
    let listing: Listing =
        serde_json::from_str(text).map_err(|e| format!("unreadable forms: {e}"))?;
    let family = |s: &str| s.split('/').next().unwrap_or_default().to_owned();
    if family(&listing.schema) != family(SCHEMA) {
        return Err(format!(
            "forms schema {:?} is not {SCHEMA:?}",
            listing.schema
        ));
    }
    Ok(listing.forms)
}

/// The forms this session has: loaded once, from `trainnr forms`.
#[derive(Default)]
pub enum Catalog {
    #[default]
    Unloaded,
    Loading {
        child: std::process::Child,
        out: PathBuf,
        log: PathBuf,
    },
    Ready(Vec<Form>),
    Failed(String),
}

impl Catalog {
    /// Start `trainnr forms`, its output into `dir`; a no-op once started.
    pub fn load(&mut self, dir: &std::path::Path) {
        if !matches!(self, Self::Unloaded) {
            return;
        }
        // This window's own, by its process id, removed once read.
        let out = dir.join(format!("studio-forms-{}.json", std::process::id()));
        let log = dir.join(format!("studio-forms-{}.log", std::process::id()));
        let started = output_files(dir, &out, &log).and_then(|(printed, errors)| {
            crate::spawn::trainnr_command()
                .arg("forms")
                .stdin(std::process::Stdio::null())
                .stdout(printed)
                .stderr(errors)
                .spawn()
                .map_err(|e| e.to_string())
        });
        *self = match started {
            Ok(child) => Self::Loading { child, out, log },
            Err(why) => Self::Failed(format!("trainnr forms did not start: {why}")),
        };
    }

    /// The load, when it has ended: the forms, or why there are none.
    pub fn poll(&mut self) {
        let Self::Loading { child, out, log } = self else {
            return;
        };
        let ended = match child.try_wait() {
            Ok(None) => return,
            Ok(Some(status)) => status.success(),
            Err(_) => false,
        };
        let text = std::fs::read_to_string(&*out).unwrap_or_default();
        let next = match (ended, parse(&text)) {
            (true, Ok(forms)) => Self::Ready(forms),
            (true, Err(why)) => Self::Failed(why),
            (false, _) => Self::Failed(last_line(log)),
        };
        let _ = std::fs::remove_file(&*out);
        let _ = std::fs::remove_file(&*log);
        *self = next;
    }

    /// The forms to show: the loaded ones, else the ones built in, so a
    /// page has its buttons before the engine is set up (a click sets it
    /// up) and when `trainnr forms` could not run.
    fn forms(&self) -> &[Form] {
        match self {
            Self::Ready(forms) => forms,
            _ => built_in(),
        }
    }

    /// This page's forms, in the registry's order.
    pub fn for_page(&self, section: Section) -> Vec<&Form> {
        self.forms()
            .iter()
            .filter(|f| Section::parse(&f.page) == Some(section))
            .collect()
    }

    /// Why there are no forms, when loading them failed.
    pub fn problem(&self) -> Option<&str> {
        match self {
            Self::Failed(why) => Some(why),
            _ => None,
        }
    }

    /// The form that runs `tool`.
    pub fn form(&self, tool: &str) -> Option<&Form> {
        self.forms().iter().find(|f| f.tool == tool)
    }
}

/// A command's two output files, created in `dir`, never through a link
/// (`control::refuse_links`): a planted link in a projects home would
/// otherwise have the Studio write wherever it points.
pub fn output_files(
    dir: &std::path::Path,
    out: &std::path::Path,
    log: &std::path::Path,
) -> Result<(std::fs::File, std::fs::File), String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    crate::control::refuse_links(out)?;
    crate::control::refuse_links(log)?;
    let create = |p: &std::path::Path| std::fs::File::create(p).map_err(|e| e.to_string());
    Ok((create(out)?, create(log)?))
}

/// Why a form cannot run yet: a required field picks an artifact kind the
/// project holds none of ("Needs a robot in this project first"), so
/// the button stays off instead of opening a form that cannot run
/// (design review, 2026-10-12). `count` is the project's count of a kind.
pub fn missing_input(form: &Form, count: &dyn Fn(&str) -> usize) -> Option<String> {
    form.fields
        .iter()
        .filter(|f| f.required)
        .filter_map(|f| f.kind.as_deref())
        .find(|kind| count(kind) == 0)
        .map(|kind| {
            format!(
                "Needs a {} in this project first",
                crate::pages::kind_word(kind)
            )
        })
}

/// The forms this build was made with (`forms.json`, written by
/// `trainnr forms` and pinned to it by `tests/test_studio_mirrors.py`):
/// the built-in tools' forms, without a plugin's.
pub fn built_in() -> &'static [Form] {
    static BUILT_IN: std::sync::OnceLock<Vec<Form>> = std::sync::OnceLock::new();
    BUILT_IN.get_or_init(|| parse(include_str!("forms.json")).unwrap_or_default())
}

/// A file's last line that says something, or a word that it said nothing.
pub fn last_line(path: &std::path::Path) -> String {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| {
            text.lines()
                .rev()
                .find(|l| !l.trim().is_empty())
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "it exited without saying why".to_owned())
}

/// A form being filled in: what is typed or picked so far.
pub struct Open {
    pub form: Form,
    values: BTreeMap<String, String>,
    flags: BTreeMap<String, bool>,
    /// A file dialog open for a field: its answer arrives on the channel
    /// (the dialog runs on its own thread, so the window keeps drawing).
    picking: Option<(String, Receiver<Option<PathBuf>>)>,
}

impl Open {
    pub fn new(form: Form) -> Self {
        let mut values = BTreeMap::new();
        let mut flags = BTreeMap::new();
        for field in &form.fields {
            match (&field.default, field.kind_of_value.as_str()) {
                (Some(serde_json::Value::Bool(b)), _) => {
                    flags.insert(field.name.clone(), *b);
                }
                (_, "boolean") => {
                    flags.insert(field.name.clone(), false);
                }
                (Some(serde_json::Value::String(s)), _) => {
                    values.insert(field.name.clone(), s.clone());
                }
                (Some(value @ serde_json::Value::Number(_)), _) => {
                    values.insert(field.name.clone(), value.to_string());
                }
                _ => {
                    values.insert(field.name.clone(), String::new());
                }
            }
        }
        Self {
            form,
            values,
            flags,
            picking: None,
        }
    }

    /// Set a field's text, as typing or picking does.
    #[cfg(test)]
    pub fn set(&mut self, name: &str, value: &str) {
        self.values.insert(name.to_owned(), value.to_owned());
    }

    /// The tool's arguments as JSON: typed by the schema, empty optional
    /// fields left out (the tool's default applies), a missing required
    /// field or a number that is not one refused by its label.
    pub fn arguments(&self) -> Result<serde_json::Value, String> {
        let mut out = serde_json::Map::new();
        for field in &self.form.fields {
            if field.kind_of_value == "boolean" {
                let flag = self.flags.get(&field.name).copied().unwrap_or(false);
                out.insert(field.name.clone(), flag.into());
                continue;
            }
            let text = self.values.get(&field.name).map_or("", |v| v.trim());
            if text.is_empty() {
                if field.required {
                    return Err(format!("{} is needed", field.label));
                }
                continue;
            }
            let value = match field.kind_of_value.as_str() {
                "integer" => text
                    .parse::<i64>()
                    .map(serde_json::Value::from)
                    .map_err(|_| format!("{} is a whole number", field.label))?,
                "number" => text
                    .parse::<f64>()
                    .map(serde_json::Value::from)
                    .map_err(|_| format!("{} is a number", field.label))?,
                _ => serde_json::Value::from(text),
            };
            out.insert(field.name.clone(), value);
        }
        Ok(serde_json::Value::Object(out))
    }

    /// Start a file (or folder) dialog for a field on its own thread.
    fn pick(&mut self, field: &Field, folder: bool) {
        let (send, receive) = channel();
        let extensions = field.files.clone();
        let label = field.label.clone();
        std::thread::spawn(move || {
            let dialog = rfd::FileDialog::new().set_title(&label);
            let any = extensions.is_empty() || extensions.iter().any(|e| e == ANY_FILE);
            let chosen = if folder {
                dialog.pick_folder()
            } else if any {
                dialog.pick_file()
            } else {
                dialog.add_filter(&label, &extensions).pick_file()
            };
            let _ = send.send(chosen);
        });
        self.picking = Some((field.name.clone(), receive));
    }

    /// A file dialog's answer, when it has come.
    fn take_pick(&mut self) {
        let Some((name, receive)) = &self.picking else {
            return;
        };
        if let Ok(chosen) = receive.try_recv() {
            if let Some(path) = chosen {
                self.values.insert(name.clone(), path.display().to_string());
            }
            self.picking = None;
        }
    }
}

/// What the form dialog answered this frame.
pub enum Answer {
    Open,
    Submit(serde_json::Value),
    Cancel,
}

/// The form as a dialog: a field per argument, Run and Cancel. `artifacts`
/// names the open project's artifacts of a kind, for the pickers.
pub fn dialog(
    ctx: &egui::Context,
    open: &mut Open,
    artifacts: &dyn Fn(&str) -> Vec<String>,
) -> Answer {
    open.take_pick();
    if open.picking.is_some() {
        ctx.request_repaint_after(std::time::Duration::from_millis(200));
    }
    let mut answer = Answer::Open;
    let mut problem: Option<String> = None;
    let modal = egui::Modal::new(egui::Id::new(("form", &open.form.tool))).show(ctx, |ui| {
        let palette = crate::theme::palette(ui);
        ui.set_width(DIALOG_WIDTH);
        ui.label(
            egui::RichText::new(&open.form.label)
                .text_style(DesignTokens::welcome_screen_h2())
                .strong()
                .color(ui.visuals().strong_text_color()),
        );
        ui.add_space(4.0);
        ui.label(egui::RichText::new(&open.form.summary).color(palette.text));
        ui.add_space(12.0);
        let fields = open.form.fields.clone();
        for field in &fields {
            field_row(ui, open, field, artifacts);
            ui.add_space(10.0);
        }
        let ready = open.arguments();
        if let Err(why) = &ready {
            problem = Some(why.clone());
        }
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let run = crate::onboarding::primary_button(ui, &open.form.label, ready.is_ok());
            if run.clicked() {
                if let Ok(arguments) = ready {
                    answer = Answer::Submit(arguments);
                }
            }
            if ui.button("Cancel").clicked() {
                answer = Answer::Cancel;
            }
            if let Some(why) = &problem {
                ui.label(egui::RichText::new(why).color(palette.muted));
            }
        });
        ui.add_space(6.0);
        ui.label(
            egui::RichText::new(format!(
                "Runs the {} tool, as your agent would.",
                open.form.tool
            ))
            .small()
            .color(palette.muted),
        );
    });
    if modal.should_close() && matches!(answer, Answer::Open) && open.picking.is_none() {
        answer = Answer::Cancel;
    }
    answer
}

/// One field: its label, and the widget its hint asks for.
fn field_row(
    ui: &mut egui::Ui,
    open: &mut Open,
    field: &Field,
    artifacts: &dyn Fn(&str) -> Vec<String>,
) {
    let palette = crate::theme::palette(ui);
    let mark = if field.required { "" } else { "  (optional)" };
    let label = ui.label(
        egui::RichText::new(format!("{}{mark}", field.label))
            .strong()
            .color(palette.text),
    );
    if !field.description.is_empty() {
        // Visible, not only on hover: "smoke (minutes, saves nothing) or
        // full" is what a person needs before starting a long run (design
        // review, 2026-10-12).
        label.on_hover_text(&field.description);
        if field.kind_of_value != "boolean" {
            ui.label(
                egui::RichText::new(&field.description)
                    .small()
                    .color(palette.muted),
            );
        }
    }
    if field.kind_of_value == "boolean" {
        let flag = open.flags.entry(field.name.clone()).or_insert(false);
        ui.checkbox(flag, &field.description);
        return;
    }
    let value = open.values.entry(field.name.clone()).or_default();
    if let Some(kind) = &field.kind {
        let names = artifacts(kind);
        if names.is_empty() {
            ui.label(
                egui::RichText::new(format!("No {kind} in this project yet.")).color(palette.muted),
            );
            return;
        }
        let shown = if value.is_empty() {
            "Choose…"
        } else {
            value.as_str()
        };
        egui::ComboBox::from_id_salt(("field", &field.name))
            .selected_text(shown.to_owned())
            .width(DIALOG_WIDTH - 20.0)
            .show_ui(ui, |ui| {
                for name in names {
                    ui.selectable_value(value, name.clone(), name);
                }
            });
        return;
    }
    if !field.choices.is_empty() {
        egui::ComboBox::from_id_salt(("field", &field.name))
            .selected_text(value.clone())
            .show_ui(ui, |ui| {
                for choice in &field.choices {
                    ui.selectable_value(value, choice.clone(), choice);
                }
            });
        return;
    }
    let mut pick: Option<bool> = None;
    ui.horizontal(|ui| {
        let width = if field.files.is_empty() {
            DIALOG_WIDTH - 20.0
        } else {
            DIALOG_WIDTH - 190.0
        };
        ui.add(
            egui::TextEdit::singleline(value)
                .desired_width(width)
                .hint_text(
                    field
                        .default
                        .as_ref()
                        .map(ToString::to_string)
                        .unwrap_or_default(),
                ),
        );
        if !field.files.is_empty() || field.folder {
            if ui.button("File…").clicked() {
                pick = Some(false);
            }
            if field.folder && ui.button("Folder…").clicked() {
                pick = Some(true);
            }
        }
    });
    if let Some(folder) = pick {
        open.pick(field, folder);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LISTED: &str = r#"{"schema": "trainnr-forms/1", "forms": [
        {"tool": "identify_system", "page": "robots", "label": "Identify dynamics",
         "summary": "System identification.", "fields": [
            {"name": "robot", "label": "Robot", "type": "string", "required": true,
             "kind": "robot", "files": [], "folder": false, "choices": []},
            {"name": "trials", "label": "Trials", "type": "integer", "required": false,
             "default": 40},
            {"name": "full", "label": "Full", "type": "boolean", "default": false}]}]}"#;

    #[test]
    fn a_listing_reads_and_a_foreign_one_is_refused() {
        let forms = parse(LISTED).expect("reads");
        assert_eq!(forms[0].tool, "identify_system");
        assert_eq!(forms[0].fields[0].kind.as_deref(), Some("robot"));
        assert!(parse(r#"{"schema": "someone-else/1", "forms": []}"#).is_err());
        assert!(parse("{not json").is_err());
    }

    #[test]
    fn arguments_are_typed_and_a_missing_one_is_named() {
        let form = parse(LISTED).expect("reads").remove(0);
        let mut open = Open::new(form);
        assert_eq!(open.arguments(), Err("Robot is needed".to_owned()));
        open.set("robot", "go2@0e7e123a7de7");
        assert_eq!(
            open.arguments(),
            Ok(serde_json::json!({"robot": "go2@0e7e123a7de7", "trials": 40, "full": false}))
        );
        open.set("trials", "many");
        assert_eq!(open.arguments(), Err("Trials is a whole number".to_owned()));
        open.set("trials", " ");
        assert_eq!(
            open.arguments(),
            Ok(serde_json::json!({"robot": "go2@0e7e123a7de7", "full": false})),
            "an empty optional field takes the tool's default"
        );
    }

    /// Every kind of field draws, headless, and Escape cancels: a picker
    /// over the project's artifacts, an empty one, a number, a flag, a
    /// file and a folder, and a choice.
    #[test]
    fn a_form_with_every_kind_of_field_draws_and_escape_cancels() {
        let text = r#"{"schema": "trainnr-forms/1", "forms": [
            {"tool": "t", "page": "robots", "label": "Do it", "summary": "S.", "fields": [
              {"name": "robot", "label": "Robot", "type": "string", "required": true, "kind": "robot"},
              {"name": "task", "label": "Task", "type": "string", "kind": "task"},
              {"name": "n", "label": "N", "type": "integer", "default": 3},
              {"name": "on", "label": "On", "type": "boolean", "default": true},
              {"name": "src", "label": "Src", "type": "string", "files": ["*"], "folder": true},
              {"name": "how", "label": "How", "type": "string", "choices": ["a", "b"]}]}]}"#;
        let form = parse(text).expect("reads").remove(0);
        let mut open = Open::new(form);
        let ctx = egui::Context::default();
        re_ui::apply_style_and_install_loaders(&ctx);
        let artifacts = |kind: &str| -> Vec<String> {
            if kind == "robot" {
                vec!["go2@5003bf617b5f".to_owned()]
            } else {
                Vec::new()
            }
        };
        let frame = |input: egui::RawInput, open: &mut Open| {
            let mut answer = Answer::Open;
            let mut output = ctx.run_ui(input, |ui| answer = dialog(ui.ctx(), open, &artifacts));
            output.textures_delta.clear(); // no renderer here to apply them
            answer
        };
        for _ in 0..3 {
            assert!(matches!(frame(Default::default(), &mut open), Answer::Open));
        }
        let escape = egui::RawInput {
            events: vec![egui::Event::Key {
                key: egui::Key::Escape,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..Default::default()
        };
        assert!(matches!(frame(escape, &mut open), Answer::Cancel));
        assert_eq!(
            open.arguments(),
            Err("Robot is needed".to_owned()),
            "the required picker starts empty"
        );
    }

    #[test]
    fn a_form_that_needs_a_missing_artifact_says_which() {
        let form = parse(LISTED).expect("reads").remove(0);
        let none = |_: &str| 0;
        assert_eq!(
            missing_input(&form, &none).as_deref(),
            Some("Needs a robot in this project first")
        );
        let one = |_: &str| 1;
        assert_eq!(missing_input(&form, &one), None);
    }

    #[test]
    fn a_page_lists_its_own_forms() {
        let catalog = Catalog::Ready(parse(LISTED).expect("reads"));
        assert_eq!(catalog.for_page(Section::Robots).len(), 1);
        assert!(catalog.for_page(Section::Recordings).is_empty());
        assert!(catalog.form("identify_system").is_some());
        let built_in = Catalog::Unloaded.for_page(Section::Robots);
        assert!(!built_in.is_empty(), "the built-in forms stand in");
    }
}
