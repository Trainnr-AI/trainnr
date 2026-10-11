//! Connecting a coding agent to trainnr, from the app (the desktop app
//! plan, docs/85, phase 3, 2026-10-11).
//!
//! The agent runs `trainnr mcp` over stdio from the engine this Studio set
//! up (or the checkout it runs from). On macOS and Linux it is reached
//! through a launcher at a fixed path, `<trainnr home>/engine/bin/trainnr`,
//! rewritten whenever the engine changes, so an agent's entry survives an
//! update (Paper's `~/.paper/bin/paper` is the same idea); on Windows the
//! entry names the engine's Python, because a `.cmd` launcher does not
//! start reliably from every client. Each client is added by its own
//! button, which is the asking: Claude Code and Codex through their own
//! `mcp add`, Cursor by adding one entry to its `mcp.json`, every other
//! entry left as it was. Nothing is written at launch.

use std::path::{Path, PathBuf};

/// The MCP server's name in every client's configuration.
pub const SERVER_NAME: &str = "trainnr";

/// The command an agent's entry runs: a program, its arguments, and the
/// environment it needs (the engine's uv, for the jobs).
#[derive(Clone, Debug, PartialEq)]
pub struct ServerCommand {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
}

impl ServerCommand {
    /// One line a person can paste into a client that is not listed.
    pub fn line(&self) -> String {
        let quote = |s: &str| {
            if s.contains(' ') {
                format!("\"{s}\"")
            } else {
                s.to_owned()
            }
        };
        let env: Vec<String> = self
            .env
            .iter()
            .map(|(k, v)| format!("{k}={}", quote(v)))
            .collect();
        let mut parts = env;
        parts.push(quote(&self.program.display().to_string()));
        parts.extend(self.args.iter().map(|a| quote(a)));
        parts.join(" ")
    }

    /// The entry as most clients' `mcpServers` hold it.
    pub fn json_entry(&self) -> serde_json::Value {
        let env: serde_json::Map<String, serde_json::Value> = self
            .env
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::from(v.clone())))
            .collect();
        serde_json::json!({
            "type": "stdio",
            "command": self.program.display().to_string(),
            "args": self.args,
            "env": env,
        })
    }
}

/// The Python of the environment under `root` (a checkout or the
/// engine's runtime), where `uv sync` made it.
fn venv_python(root: &Path) -> PathBuf {
    let venv = root.join(crate::spawn::PIPELINE_DIR).join(".venv");
    if cfg!(windows) {
        venv.join("Scripts").join("python.exe")
    } else {
        venv.join("bin").join("python")
    }
}

/// The command an agent runs, once there is an environment to run it in:
/// on macOS and Linux the launcher (written here, pointing at `root`), on
/// Windows the environment's Python itself.
pub fn server_command(root: &Path, engine_home: &Path, uv: &Path) -> Result<ServerCommand, String> {
    let python = venv_python(root);
    if !python.is_file() {
        return Err(format!("no environment at {} yet", python.display()));
    }
    let env = vec![(crate::engine::UV_ENV.to_owned(), uv.display().to_string())];
    if cfg!(windows) {
        return Ok(ServerCommand {
            program: python,
            args: vec![
                "-m".into(),
                crate::spawn::TRAINNR_CLI_MODULE.into(),
                "mcp".into(),
            ],
            env,
        });
    }
    let launcher = write_launcher(engine_home, &python, uv)?;
    Ok(ServerCommand {
        program: launcher,
        args: vec!["mcp".into()],
        env,
    })
}

/// `<engine home>/bin/trainnr`: runs the `trainnr` command in `python`'s
/// environment with the engine's uv named, whatever version is set up.
fn write_launcher(engine_home: &Path, python: &Path, uv: &Path) -> Result<PathBuf, String> {
    let bin = engine_home.join("bin");
    std::fs::create_dir_all(&bin).map_err(|e| e.to_string())?;
    let launcher = bin.join(SERVER_NAME);
    crate::control::refuse_links(&launcher)?;
    let quoted = |p: &Path| format!("'{}'", p.display().to_string().replace('\'', "'\\''"));
    let body = format!(
        "#!/bin/sh\n# Written by the trainnr Studio: the trainnr command from the engine.\n\
         {}=${{{}:-{}}} exec {} -m {} \"$@\"\n",
        crate::engine::UV_ENV,
        crate::engine::UV_ENV,
        quoted(uv),
        quoted(python),
        crate::spawn::TRAINNR_CLI_MODULE,
    );
    std::fs::write(&launcher, body).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(&launcher, std::fs::Permissions::from_mode(0o755))
            .map_err(|e| e.to_string())?;
    }
    Ok(launcher)
}

/// A coding agent the panel lists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Client {
    ClaudeCode,
    Codex,
    Cursor,
}

pub const CLIENTS: [Client; 3] = [Client::ClaudeCode, Client::Codex, Client::Cursor];

impl Client {
    pub fn label(self) -> &'static str {
        match self {
            Self::ClaudeCode => "Claude Code",
            Self::Codex => "Codex",
            Self::Cursor => "Cursor",
        }
    }

    /// What the button does, said before it does it.
    pub fn what(self) -> &'static str {
        match self {
            Self::ClaudeCode => "Runs `claude mcp add --scope user trainnr …`: trainnr in every Claude Code project.",
            Self::Codex => "Runs `codex mcp add trainnr …`: trainnr in Codex's global configuration.",
            Self::Cursor => "Adds a `trainnr` entry to ~/.cursor/mcp.json, leaving every other entry as it is.",
        }
    }

    /// Whether this machine has the client (its command, or its folder).
    pub fn present(self, home: &Path) -> bool {
        match self {
            Self::ClaudeCode => on_path("claude") || home.join(".claude").is_dir(),
            Self::Codex => on_path("codex") || codex_home(home).is_dir(),
            Self::Cursor => home.join(".cursor").is_dir(),
        }
    }

    /// Whether the client already has trainnr: its MCP entry, or (Claude
    /// Code) the trainnr plugin.
    pub fn configured(self, home: &Path) -> bool {
        let json = |path: PathBuf| -> Option<serde_json::Value> {
            serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
        };
        match self {
            Self::ClaudeCode => {
                let user = json(home.join(".claude.json"))
                    .is_some_and(|c| c["mcpServers"].get(SERVER_NAME).is_some());
                let plugin = json(
                    home.join(".claude")
                        .join("plugins")
                        .join("installed_plugins.json"),
                )
                .is_some_and(|p| {
                    let plugins = p.get("plugins").unwrap_or(&p);
                    plugins
                        .as_object()
                        .is_some_and(|m| m.keys().any(|k| k.starts_with("trainnr@")))
                });
                user || plugin
            }
            Self::Codex => std::fs::read_to_string(codex_home(home).join("config.toml"))
                .is_ok_and(|t| t.contains(&format!("[mcp_servers.{SERVER_NAME}]"))),
            Self::Cursor => json(home.join(".cursor").join("mcp.json"))
                .is_some_and(|c| c["mcpServers"].get(SERVER_NAME).is_some()),
        }
    }

    /// Add trainnr to the client; the answer is what it said, or why not.
    pub fn add(self, home: &Path, command: &ServerCommand) -> Result<String, String> {
        match self {
            Self::ClaudeCode => {
                run_add("claude", &["mcp", "add", "--scope", "user"], "-e", command)
            }
            Self::Codex => run_add("codex", &["mcp", "add"], "--env", command),
            Self::Cursor => add_to_json(&home.join(".cursor").join("mcp.json"), command),
        }
    }
}

fn codex_home(home: &Path) -> PathBuf {
    std::env::var_os("CODEX_HOME")
        .filter(|v| !v.is_empty())
        .map_or_else(|| home.join(".codex"), PathBuf::from)
}

fn on_path(name: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|dir| {
            dir.join(name).is_file()
                || dir.join(format!("{name}.exe")).is_file()
                || dir.join(format!("{name}.cmd")).is_file()
        })
    })
}

/// `<cli> <add…> trainnr <env flag> K=V … -- <program> <args…>`: the
/// client's own command writes its own configuration.
fn run_add(
    cli: &str,
    add: &[&str],
    env_flag: &str,
    command: &ServerCommand,
) -> Result<String, String> {
    let mut run = std::process::Command::new(cli);
    run.args(add).arg(SERVER_NAME);
    for (key, value) in &command.env {
        run.arg(env_flag).arg(format!("{key}={value}"));
    }
    run.arg("--").arg(&command.program).args(&command.args);
    let done = run
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|e| format!("{cli} did not start: {e}"))?;
    let said = |bytes: &[u8]| {
        String::from_utf8_lossy(bytes)
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("")
            .trim()
            .to_owned()
    };
    if done.status.success() {
        Ok(said(&done.stdout))
    } else {
        let why = said(&done.stderr);
        Err(if why.is_empty() {
            said(&done.stdout)
        } else {
            why
        })
    }
}

/// Add the `trainnr` entry to a JSON file's `mcpServers`, every other key
/// kept; a file that is not JSON is refused, never overwritten.
fn add_to_json(path: &Path, command: &ServerCommand) -> Result<String, String> {
    let mut config: serde_json::Value = match std::fs::read_to_string(path) {
        Ok(text) if !text.trim().is_empty() => serde_json::from_str(&text).map_err(|e| {
            format!(
                "{} is not JSON ({e}); add the entry by hand",
                path.display()
            )
        })?,
        _ => serde_json::json!({}),
    };
    let Some(root) = config.as_object_mut() else {
        return Err(format!("{} does not hold an object", path.display()));
    };
    let servers = root
        .entry("mcpServers")
        .or_insert_with(|| serde_json::json!({}));
    let Some(servers) = servers.as_object_mut() else {
        return Err(format!("{}: mcpServers is not an object", path.display()));
    };
    servers.insert(SERVER_NAME.to_owned(), command.json_entry());
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    crate::control::refuse_links(path)?;
    let body = serde_json::to_string_pretty(&config).map_err(|e| e.to_string())?;
    crate::control::write_atomic(path, &(body + "\n")).map_err(|e| e.to_string())?;
    Ok(format!("added to {}", path.display()))
}

/// What the panel asked for this frame.
pub enum Ask {
    Open,
    Close,
    Add(Client),
    SetUpEngine,
}

/// Where adding to a client stands: the one being added, and each
/// client's last answer.
#[derive(Default)]
pub struct Panel {
    pub adding: Option<Client>,
    pub answers: Vec<(Client, Result<String, String>)>,
    receive: Option<std::sync::mpsc::Receiver<(Client, Result<String, String>)>>,
    /// Each client: on this machine, already has trainnr. Looked up when
    /// the panel opens and after an add, never per frame: a PATH search
    /// crosses the Windows folders under WSL, seconds a frame (measured
    /// 2026-10-11).
    facts: Vec<(Client, bool, bool)>,
}

impl Panel {
    /// Add trainnr to `client` on its own thread (a client's own command
    /// takes a second or two).
    pub fn start(&mut self, client: Client, home: PathBuf, command: ServerCommand) {
        let (send, receive) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = send.send((client, client.add(&home, &command)));
        });
        self.adding = Some(client);
        self.receive = Some(receive);
    }

    /// Look each client up again (the panel opening, an add done).
    pub fn refresh(&mut self, home: &Path) {
        self.facts = CLIENTS
            .iter()
            .map(|&c| (c, c.present(home), c.configured(home)))
            .collect();
    }

    fn fact(&self, client: Client) -> (bool, bool) {
        self.facts
            .iter()
            .find(|(c, _, _)| *c == client)
            .map_or((false, false), |&(_, present, configured)| {
                (present, configured)
            })
    }

    /// The answer, when it has come; `home` is looked up again with it.
    pub fn poll(&mut self, home: &Path) {
        let Some(receive) = &self.receive else {
            return;
        };
        if let Ok((client, answer)) = receive.try_recv() {
            self.answers.retain(|(c, _)| *c != client);
            self.answers.push((client, answer));
            self.adding = None;
            self.receive = None;
            self.refresh(home);
        }
    }

    fn answer(&self, client: Client) -> Option<&Result<String, String>> {
        self.answers
            .iter()
            .find(|(c, _)| *c == client)
            .map(|(_, a)| a)
    }
}

/// The panel: each agent with its state and its button, and a line to
/// copy for any other. `command` is `None` until the engine is set up;
/// `engine` says what its setup is doing.
pub fn panel(
    ctx: &egui::Context,
    state: &Panel,
    command: Option<&ServerCommand>,
    engine: &crate::engine::Setup,
) -> Ask {
    if state.receive.is_some() {
        ctx.request_repaint_after(std::time::Duration::from_millis(200));
    }
    let mut ask = Ask::Open;
    let modal = egui::Modal::new(egui::Id::new("connect-agent")).show(ctx, |ui| {
        let palette = crate::theme::palette(ui);
        ui.set_width(PANEL_WIDTH);
        ui.label(
            egui::RichText::new("Connect your coding agent")
                .text_style(re_ui::DesignTokens::welcome_screen_h2())
                .strong()
                .color(ui.visuals().strong_text_color()),
        );
        ui.add_space(6.0);
        ui.label(
            egui::RichText::new(
                "Your agent runs trainnr's tools over MCP and works in this window with you. \
                 Each button changes that agent's settings and nothing else; restart the \
                 agent afterwards.",
            )
            .color(palette.text),
        );
        ui.add_space(14.0);
        let Some(command) = command else {
            match engine {
                crate::engine::Setup::Working(step) => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(egui::RichText::new(step).color(palette.text));
                    });
                }
                crate::engine::Setup::Failed(why) => {
                    ui.label(
                        egui::RichText::new(format!("The engine did not set up: {why}"))
                            .color(ui.visuals().warn_fg_color),
                    );
                    if crate::onboarding::primary_button(ui, "Try again", true).clicked() {
                        ask = Ask::SetUpEngine;
                    }
                }
                _ => {
                    ui.label(
                        egui::RichText::new(
                            "The agent runs trainnr from its engine, which sets up first \
                             (about 750 MB on disk, once).",
                        )
                        .color(palette.text),
                    );
                    ui.add_space(8.0);
                    if crate::onboarding::primary_button(ui, "Set up the engine", true).clicked() {
                        ask = Ask::SetUpEngine;
                    }
                }
            }
            ui.add_space(12.0);
            if ui.button("Close").clicked() {
                ask = Ask::Close;
            }
            return;
        };
        for client in CLIENTS {
            let (present, configured) = state.fact(client);
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(client.label())
                        .strong()
                        .color(palette.text),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if state.adding == Some(client) {
                        ui.spinner();
                    } else if configured {
                        ui.label(egui::RichText::new("added").color(palette.text));
                    } else if present {
                        let free = state.adding.is_none();
                        let label = format!("Add to {}", client.label());
                        if crate::onboarding::secondary_button(ui, &label, free)
                            .on_hover_text(client.what())
                            .clicked()
                        {
                            ask = Ask::Add(client);
                        }
                    } else {
                        ui.label(egui::RichText::new("not on this machine").color(palette.muted));
                    }
                });
            });
            match state.answer(client) {
                Some(Ok(said)) if !said.is_empty() => {
                    ui.label(egui::RichText::new(said).small().color(palette.text));
                }
                Some(Err(why)) => {
                    ui.label(
                        egui::RichText::new(why)
                            .small()
                            .color(ui.visuals().warn_fg_color),
                    );
                }
                _ => {}
            }
            ui.add_space(8.0);
        }
        ui.add_space(6.0);
        ui.label(egui::RichText::new("Another agent: its MCP entry runs").color(palette.text));
        ui.add_space(4.0);
        crate::onboarding::command_chip(ui, &command.line());
        ui.add_space(12.0);
        if ui.button("Close").clicked() {
            ask = Ask::Close;
        }
    });
    if modal.should_close() && matches!(ask, Ask::Open) {
        ask = Ask::Close;
    }
    ask
}

/// The panel's width.
const PANEL_WIDTH: f32 = 520.0;

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("studio-connect-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn command() -> ServerCommand {
        ServerCommand {
            program: PathBuf::from("/e/bin/trainnr"),
            args: vec!["mcp".into()],
            env: vec![("TRAINNR_UV".into(), "/e/bin/uv".into())],
        }
    }

    /// Claude Code, for real, in a throwaway home: the panel's add, then
    /// Claude Code's own check that it starts the server. By hand, alone:
    /// `TRAINNR_TEST_ENGINE_ROOT=<a runtime with its .venv>
    /// TRAINNR_TEST_CLIENT_HOME=<empty dir> cargo test --release --
    /// --ignored connecting_claude_code --nocapture`.
    #[test]
    #[ignore = "runs the claude CLI and the MCP server; run by hand"]
    #[cfg(unix)]
    fn connecting_claude_code_for_real() {
        let root = PathBuf::from(std::env::var("TRAINNR_TEST_ENGINE_ROOT").expect("a runtime"));
        let home = PathBuf::from(std::env::var("TRAINNR_TEST_CLIENT_HOME").expect("a home"));
        std::env::set_var("HOME", &home); // the client writes here, never the user's own
        let command = server_command(&root, &home.join("engine"), &crate::engine::uv_program())
            .expect("a command");
        eprintln!("entry: {}", command.line());
        assert!(!Client::ClaudeCode.configured(&home));
        let said = Client::ClaudeCode.add(&home, &command).expect("added");
        eprintln!("claude: {said}");
        assert!(Client::ClaudeCode.configured(&home));
        let check = std::process::Command::new("claude")
            .args(["mcp", "get", SERVER_NAME])
            .output()
            .expect("claude runs");
        let report = String::from_utf8_lossy(&check.stdout);
        eprintln!("{report}");
        assert!(
            report.contains("Connected"),
            "Claude Code started the server"
        );
    }

    /// The panel draws, headless, before the engine and after it, and
    /// Escape closes it.
    #[test]
    fn the_panel_draws_before_and_after_the_engine_and_escape_closes_it() {
        let ctx = egui::Context::default();
        re_ui::apply_style_and_install_loaders(&ctx);
        let home = scratch("panel");
        let state = Panel::default();
        let frame = |input: egui::RawInput, command: Option<&ServerCommand>| {
            let mut ask = Ask::Open;
            let engine = crate::engine::Setup::Idle;
            let mut output =
                ctx.run_ui(input, |ui| ask = panel(ui.ctx(), &state, command, &engine));
            output.textures_delta.clear(); // no renderer here to apply them
            ask
        };
        assert!(matches!(frame(Default::default(), None), Ask::Open));
        let made = command();
        assert!(matches!(frame(Default::default(), Some(&made)), Ask::Open));
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
        assert!(matches!(frame(escape, Some(&made)), Ask::Close));
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn cursors_file_gains_one_entry_and_keeps_the_rest() {
        let home = scratch("cursor");
        let file = home.join(".cursor").join("mcp.json");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(
            &file,
            r#"{"mcpServers": {"other": {"command": "x"}}, "theme": 1}"#,
        )
        .unwrap();
        assert!(!Client::Cursor.configured(&home));
        Client::Cursor.add(&home, &command()).expect("adds");
        let written: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        assert_eq!(written["mcpServers"]["other"]["command"], "x");
        assert_eq!(written["theme"], 1);
        assert_eq!(
            written["mcpServers"]["trainnr"]["command"],
            "/e/bin/trainnr"
        );
        assert_eq!(
            written["mcpServers"]["trainnr"]["env"]["TRAINNR_UV"],
            "/e/bin/uv"
        );
        assert!(Client::Cursor.configured(&home));
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn a_file_that_is_not_json_is_refused_and_left_alone() {
        let home = scratch("cursor-bad");
        let file = home.join(".cursor").join("mcp.json");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, "{ broken").unwrap();
        assert!(Client::Cursor.add(&home, &command()).is_err());
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "{ broken");
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn claude_codes_entry_or_plugin_reads_as_configured() {
        let home = scratch("claude");
        assert!(!Client::ClaudeCode.configured(&home));
        std::fs::write(
            home.join(".claude.json"),
            r#"{"mcpServers": {"trainnr": {}}}"#,
        )
        .unwrap();
        assert!(Client::ClaudeCode.configured(&home));
        std::fs::write(home.join(".claude.json"), "{}").unwrap();
        let plugins = home.join(".claude").join("plugins");
        std::fs::create_dir_all(&plugins).unwrap();
        std::fs::write(
            plugins.join("installed_plugins.json"),
            r#"{"plugins": {"trainnr@trainnr": []}}"#,
        )
        .unwrap();
        assert!(Client::ClaudeCode.configured(&home), "the plugin counts");
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    #[cfg(unix)]
    fn the_launcher_runs_the_trainnr_command_in_the_engines_python() {
        let home = scratch("launcher");
        let root = home.join("runtime");
        let python = venv_python(&root);
        std::fs::create_dir_all(python.parent().unwrap()).unwrap();
        std::fs::write(&python, "").unwrap();
        let made =
            server_command(&root, &home.join("engine"), Path::new("/e/bin/uv")).expect("made");
        assert_eq!(
            made.program,
            home.join("engine").join("bin").join("trainnr")
        );
        assert_eq!(made.args, ["mcp"]);
        let script = std::fs::read_to_string(&made.program).unwrap();
        assert!(script.contains(&python.display().to_string()));
        assert!(script.contains("-m trainnr.cli"));
        assert!(server_command(&home.join("none"), &home, Path::new("uv")).is_err());
        let _ = std::fs::remove_dir_all(&home);
    }
}
