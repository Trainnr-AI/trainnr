//! The agent panel: a real agent-client-protocol client, not a mock.
//!
//! Spawns Claude Code's own ACP adapter as a subprocess, via the crate's
//! own blessed constructor (`AcpAgent::claude_agent()`, which runs
//! `npx -y @agentclientprotocol/claude-agent-acp@latest` — NOT
//! `@zed-industries/claude-code-acp`; that name is the project's earlier
//! org, dead-ended on first run here with "Query closed before response
//! received" until this crate's own source pointed at the current one),
//! over the exact client API the crate ships its own example for
//! (`examples/yolo_one_shot_client.rs`, read from the crate's source at
//! `cargo add` time rather than guessed). Every prompt the user sends
//! starts a real Claude Code turn under the caller's own account; this
//! module does not mock or cache that, by design — the operator chose the
//! real thing over a free stub, and this is that choice.
//!
//! `agent-client-protocol`'s connection API is `tokio`-based; `eframe`'s
//! `run_native` owns the main thread with its own event loop, so the
//! connection runs on a second OS thread with its own single-threaded
//! runtime, talking back to the UI thread the same way `viewport.rs` talks
//! back from its reader thread: a shared, lock-protected `Vec`, woken by
//! `egui::Context::request_repaint`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;

use agent_client_protocol::schema::v1::{
    ContentBlock, InitializeRequest, McpServer, McpServerStdio, NewSessionRequest,
    PermissionOptionId, PromptRequest, RequestPermissionOutcome, RequestPermissionRequest,
    RequestPermissionResponse, SelectedPermissionOutcome, SessionNotification, SessionUpdate,
    TextContent,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{AcpAgent, Agent, ConnectionTo, LineDirection, Responder};
use tokio::sync::mpsc::{self, UnboundedSender};
use tokio::sync::oneshot;

/// One entry in the transcript shown in the panel.
pub enum AgentLine {
    User(String),
    /// The agent's reply text. `push` grows this line in place as more
    /// chunks stream in (adjacency-based, see `push`'s comment) rather
    /// than starting a new "claude" bubble every few words — a real
    /// session showed exactly that fragmentation before this existed.
    AgentText(String),
    /// A tool call, plan update, or anything else this pass doesn't render
    /// richly yet — Debug-formatted rather than dropped, so nothing this
    /// early client doesn't specifically handle goes silently missing.
    Other(String),
    /// Routine state ("starting…", "connected") — `re_ui::Alert::info`.
    Status(String),
    /// A real failure (spawn failure, connection dropped with an error) —
    /// `re_ui::Alert::error`, not lumped in with routine `Status` text.
    Error(String),
}

impl AgentLine {
    /// Plain text for lines where two in a row with identical text are
    /// noise, not information — `Other`/`Status`/`Error` only.
    /// `User`/`AgentText` are real conversation and can legitimately repeat.
    fn dedup_key(&self) -> Option<&str> {
        match self {
            AgentLine::Other(text) | AgentLine::Status(text) | AgentLine::Error(text) => Some(text),
            AgentLine::User(_) | AgentLine::AgentText(_) => None,
        }
    }
}

/// A tool call awaiting a yes/no from the user, shown as buttons in the
/// panel instead of decided automatically.
pub struct PendingPermission {
    pub tool_title: String,
    pub options: Vec<(PermissionOptionId, String)>,
    reply: oneshot::Sender<PermissionOptionId>,
}

/// How long `Drop` waits for the connection thread to tear the agent
/// subprocess down. The ACP crate kills the agent's process group when
/// the connection drops, with its own ~1 s shutdown grace — 3 s covers
/// that with room. Past it, the close proceeds anyway (see `Drop`).
const SHUTDOWN_WAIT: std::time::Duration = std::time::Duration::from_secs(3);

/// Owns the background thread running the ACP connection, the shared
/// transcript it appends to, and any permission request currently waiting
/// on the user. Dropping this tears the agent subprocess down: closing
/// `outgoing` ends `run_session`'s prompt loop, which drops the
/// connection, which is what makes the ACP crate kill the agent's whole
/// process group (`npx` wrapper included). `Drop` waits for that —
/// same no-orphans guarantee `ViewportFeed` gives the render side.
pub struct AgentSession {
    transcript: Arc<Mutex<Vec<AgentLine>>>,
    /// `Option` so `Drop` can close the channel while `&mut self`.
    outgoing: Option<UnboundedSender<String>>,
    pending: Arc<Mutex<Option<PendingPermission>>>,
    /// Signalled by the connection thread after `run_session` returns —
    /// i.e. after the crate's teardown of the agent subprocess has run.
    done: std::sync::mpsc::Receiver<()>,
}

impl AgentSession {
    pub fn spawn(ctx: &egui::Context) -> Self {
        let transcript = Arc::new(Mutex::new(Vec::new()));
        let pending = Arc::new(Mutex::new(None));
        let (outgoing, rx) = mpsc::unbounded_channel::<String>();
        let (done_tx, done) = std::sync::mpsc::channel();

        let transcript_for_thread = Arc::clone(&transcript);
        let pending_for_thread = Arc::clone(&pending);
        let ctx_for_thread = ctx.clone();
        thread::spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(err) => {
                    push(
                        &transcript_for_thread,
                        &ctx_for_thread,
                        AgentLine::Error(format!("could not start a tokio runtime: {err}")),
                    );
                    return;
                }
            };
            runtime.block_on(run_session(
                transcript_for_thread,
                pending_for_thread,
                rx,
                ctx_for_thread,
            ));
            // After run_session: the connection (and with it the agent
            // subprocess) is torn down. Tell Drop it can stop waiting.
            let _ = done_tx.send(());
        });

        Self {
            transcript,
            outgoing: Some(outgoing),
            pending,
            done,
        }
    }

    /// Enqueues a prompt for the background connection to send. Silently
    /// dropped if the connection thread has already ended — the next
    /// `drain_into` call will show why, via its own `AgentLine::Status`.
    pub fn send(&self, text: String) {
        if let Some(outgoing) = &self.outgoing {
            let _ = outgoing.send(text);
        }
    }

    /// Moves every transcript line accumulated since the last call into
    /// `out`, in order. Draining rather than cloning keeps the panel from
    /// re-walking an ever-growing `Vec` every frame.
    ///
    /// The merge/dedup rules run HERE too, across the drain boundary —
    /// not only in `push`. `push` merges against the shared buffer, but
    /// this method empties that buffer every frame, so a slow-streaming
    /// turn (one chunk per frame) always found it empty and every chunk
    /// opened a fresh "claude" bubble. Seen live on the first long
    /// subagent report; the fast-chunk case that tested green earlier
    /// had simply been batching between frames.
    pub fn drain_into(&self, out: &mut Vec<AgentLine>) {
        let mut shared = self.transcript.lock().expect("not poisoned");
        for line in shared.drain(..) {
            if let AgentLine::AgentText(text) = &line {
                if let Some(AgentLine::AgentText(last)) = out.last_mut() {
                    last.push_str(text);
                    continue;
                }
            }
            if let Some(key) = line.dedup_key() {
                if Some(key) == out.last().and_then(AgentLine::dedup_key) {
                    continue;
                }
            }
            out.push(line);
        }
    }

    /// The tool title and button labels for a pending permission request,
    /// if one is waiting — `None` means nothing to show right now.
    pub fn pending_permission(&self) -> Option<(String, Vec<(PermissionOptionId, String)>)> {
        let guard = self.pending.lock().expect("not poisoned");
        guard
            .as_ref()
            .map(|p| (p.tool_title.clone(), p.options.clone()))
    }

    /// Answers the pending permission request with `option_id`. A no-op if
    /// nothing is pending or it was already answered.
    pub fn resolve_permission(&self, option_id: PermissionOptionId) {
        if let Some(pending) = self.pending.lock().expect("not poisoned").take() {
            let _ = pending.reply.send(option_id);
        }
    }
}

impl Drop for AgentSession {
    fn drop(&mut self) {
        // Cancel any permission request still waiting on a click —
        // dropping its oneshot sender makes the handler answer Cancelled,
        // unblocking a turn that would otherwise wait forever and keep
        // the prompt loop (below) from ever noticing the closed channel.
        drop(self.pending.lock().expect("not poisoned").take());

        // Closing the channel ends run_session's prompt loop; the
        // connection then drops, and the ACP crate kills the agent's
        // process group. Wait for the thread to confirm.
        drop(self.outgoing.take());
        if self.done.recv_timeout(SHUTDOWN_WAIT).is_err() {
            // A turn was mid-flight (the loop only checks the channel
            // between prompts), so teardown hasn't run yet. Proceeding
            // means the agent subprocess may outlive us — the one case
            // this Drop doesn't close, accepted over hanging the window
            // close indefinitely.
            eprintln!("agent session did not shut down within {SHUTDOWN_WAIT:?}; proceeding");
        }
    }
}

fn push(transcript: &Arc<Mutex<Vec<AgentLine>>>, ctx: &egui::Context, line: AgentLine) {
    let mut guard = transcript.lock().expect("not poisoned");

    // Consecutive chunks grow one line in place instead of starting a new
    // "claude" bubble every few words. Adjacency, not `message_id`
    // equality: a live session showed this adapter doesn't set
    // `message_id` on every chunk, so a strict-equality check missed real
    // mid-sentence continuations. Two `AgentText` chunks with nothing
    // else between them are, in practice, always the same reply — a real
    // turn boundary has a tool call, a user message, or a status line in
    // between, never silence.
    if let AgentLine::AgentText(text) = &line {
        if let Some(AgentLine::AgentText(last_text)) = guard.last_mut() {
            last_text.push_str(text);
            drop(guard);
            ctx.request_repaint();
            return;
        }
    }

    if let Some(key) = line.dedup_key() {
        if Some(key) == guard.last().and_then(AgentLine::dedup_key) {
            return; // identical status/other line as last time — nothing new to show
        }
    }
    guard.push(line);
    drop(guard);
    ctx.request_repaint();
}

/// The instrument's own MCP server (tools/mcp-server.py), handed to the
/// agent at session creation over ACP's `mcp_servers` — the panel's
/// Claude queries bundles, actuators, tasks, engines and runs through
/// the same seams the pipeline acts on, instead of grepping for them.
/// Stdio transport: the one every ACP agent MUST support.
fn robotiq_mcp_server() -> McpServer {
    let repo_root = crate::repo_root();
    McpServer::Stdio(McpServerStdio::new("robotiq", "uv").args(vec![
        "run".to_string(),
        "--directory".to_string(),
        repo_root.join("pipeline").display().to_string(),
        "--extra".to_string(),
        "sim".to_string(),
        "--extra".to_string(),
        "mcp".to_string(),
        "python".to_string(),
        repo_root.join("tools/mcp-server.py").display().to_string(),
    ]))
}

async fn run_session(
    transcript: Arc<Mutex<Vec<AgentLine>>>,
    pending: Arc<Mutex<Option<PendingPermission>>>,
    mut prompts: mpsc::UnboundedReceiver<String>,
    ctx: egui::Context,
) {
    // The adapter's stderr in the transcript was bring-up diagnostics
    // (it's how the stale package name and the session ids were seen);
    // with the connection stable it's noise in a product panel. Off by
    // default, one env var to bring it back when diagnosing:
    //     STUDIO_ACP_DEBUG=1 cargo run
    let show_stderr = std::env::var_os("STUDIO_ACP_DEBUG").is_some();
    let debug_transcript = Arc::clone(&transcript);
    let debug_ctx = ctx.clone();
    let agent = AcpAgent::claude_agent().with_debug(move |line, direction| {
        if show_stderr && direction == LineDirection::Stderr {
            push(
                &debug_transcript,
                &debug_ctx,
                AgentLine::Other(format!("[stderr] {line}")),
            );
        }
    });

    // Read the real command back from the agent rather than restating it —
    // `claude_agent()` is the crate's own black-box convenience
    // constructor, and a literal copy of what it happens to run today
    // would silently go stale the moment that changes upstream.
    let command_line = {
        let config = agent.config();
        std::iter::once(config.command().display().to_string())
            .chain(config.arguments().iter().cloned())
            .collect::<Vec<_>>()
            .join(" ")
    };
    push(
        &transcript,
        &ctx,
        AgentLine::Status(format!("starting {command_line}…")),
    );

    let notify_transcript = Arc::clone(&transcript);
    let notify_ctx = ctx.clone();

    let result = agent_client_protocol::Client
        .builder()
        .on_receive_notification(
            move |notification: SessionNotification, _cx| {
                let transcript = Arc::clone(&notify_transcript);
                let ctx = notify_ctx.clone();
                async move {
                    let line = describe_update(notification.update);
                    // `describe_update` returns an empty `Other` for
                    // updates deliberately not worth a transcript row
                    // (e.g. the user's own message echoed back).
                    if !matches!(&line, AgentLine::Other(text) if text.is_empty()) {
                        push(&transcript, &ctx, line);
                    }
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .on_receive_request(
            {
                let pending = Arc::clone(&pending);
                let ctx = ctx.clone();
                move |request: RequestPermissionRequest,
                      responder: Responder<RequestPermissionResponse>,
                      _connection| {
                    let pending = Arc::clone(&pending);
                    let ctx = ctx.clone();
                    async move {
                        if request.options.is_empty() {
                            return responder.respond(RequestPermissionResponse::new(
                                RequestPermissionOutcome::Cancelled,
                            ));
                        }
                        let tool_title = request
                            .tool_call
                            .fields
                            .title
                            .unwrap_or_else(|| "a tool call".to_string());
                        let options = request
                            .options
                            .iter()
                            .map(|opt| (opt.option_id.clone(), opt.name.clone()))
                            .collect();
                        let (reply, answer) = oneshot::channel();
                        *pending.lock().expect("not poisoned") = Some(PendingPermission {
                            tool_title,
                            options,
                            reply,
                        });
                        ctx.request_repaint();

                        // The UI thread answers via `resolve_permission`; a
                        // dropped sender (the window closed mid-request)
                        // falls back to cancelling rather than hanging.
                        let outcome = match answer.await {
                            Ok(option_id) => RequestPermissionOutcome::Selected(
                                SelectedPermissionOutcome::new(option_id),
                            ),
                            Err(_) => RequestPermissionOutcome::Cancelled,
                        };
                        responder.respond(RequestPermissionResponse::new(outcome))
                    }
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .connect_with(agent, {
            let transcript = Arc::clone(&transcript);
            let ctx = ctx.clone();
            async move |connection: ConnectionTo<Agent>| {
                connection
                    .send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;

                let session = connection
                    .send_request(
                        NewSessionRequest::new(
                            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")),
                        )
                        .mcp_servers(vec![robotiq_mcp_server()]),
                    )
                    .block_task()
                    .await?;
                let session_id = session.session_id;

                push(
                    &transcript,
                    &ctx,
                    AgentLine::Status("connected".to_string()),
                );

                while let Some(text) = prompts.recv().await {
                    push(&transcript, &ctx, AgentLine::User(text.clone()));
                    connection
                        .send_request(PromptRequest::new(
                            session_id.clone(),
                            vec![ContentBlock::Text(TextContent::new(text))],
                        ))
                        .block_task()
                        .await?;
                }
                Ok(())
            }
        })
        .await;

    if let Err(err) = result {
        push(
            &transcript,
            &ctx,
            AgentLine::Error(format!("ACP connection ended: {err}")),
        );
    }
}

/// Turns one protocol update into a transcript line. Most variants get a
/// short, hand-picked summary rather than their full `Debug` — the first
/// real session showed why: `AvailableCommandsUpdate` alone dumped several
/// dozen command descriptions into the panel as a single wall of text.
/// Anything not special-cased below still shows something (the variant
/// name only), never a silent drop and never that wall again.
fn describe_update(update: SessionUpdate) -> AgentLine {
    match update {
        SessionUpdate::AgentMessageChunk(chunk) => match chunk.content {
            ContentBlock::Text(text) => AgentLine::AgentText(text.text),
            other => AgentLine::Other(format!("{other:?}")),
        },
        SessionUpdate::AgentThoughtChunk(chunk) => match chunk.content {
            ContentBlock::Text(text) => AgentLine::Other(format!("(thinking) {}", text.text)),
            _ => AgentLine::Other("(thinking)".to_string()),
        },
        SessionUpdate::UserMessageChunk(_) => {
            // Already shown when the user hit Send — the agent echoing it
            // back is the same content twice, not new information.
            AgentLine::Other(String::new())
        }
        SessionUpdate::ToolCall(tool_call) => AgentLine::Other(format!(
            "tool: {} [{:?}]",
            tool_call.title, tool_call.status
        )),
        SessionUpdate::ToolCallUpdate(update) => {
            AgentLine::Other(format!("tool update: {:?}", update.tool_call_id))
        }
        SessionUpdate::AvailableCommandsUpdate(update) => AgentLine::Other(format!(
            "{} slash commands available",
            update.available_commands.len()
        )),
        SessionUpdate::UsageUpdate(usage) => {
            AgentLine::Other(format!("tokens: {} / {}", usage.used, usage.size))
        }
        SessionUpdate::SessionInfoUpdate(info) => match info.title {
            agent_client_protocol::schema::MaybeUndefined::Value(title) => {
                AgentLine::Other(format!("session: {title}"))
            }
            _ => AgentLine::Other(String::new()), // unset/cleared — nothing to show
        },
        other => AgentLine::Other(format!("{other:?}")),
    }
}
