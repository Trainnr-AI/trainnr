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
//! The transcript model here is the panel's first-class rework (Zed and
//! Cursor studied for the interaction ideas, never the code — Zed is
//! GPL): tool calls are CARDS keyed by `ToolCallId` that mutate in place
//! as updates stream in, instead of the earlier flat lines where every
//! `ToolCallUpdate` printed another "tool update: ToolCallId(…)" row;
//! the agent's plan is one checklist card that replaces itself; token
//! usage and turn state are panel chrome (read via `usage()` /
//! `turn_active()`), not transcript rows.
//!
//! `agent-client-protocol`'s connection API is `tokio`-based; `eframe`'s
//! `run_native` owns the main thread with its own event loop, so the
//! connection runs on a second OS thread with its own single-threaded
//! runtime, talking back to the UI thread the same way `viewport.rs` talks
//! back from its reader thread: a shared, lock-protected `Vec`, woken by
//! `egui::Context::request_repaint`.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use agent_client_protocol::schema::v1::{
    CancelNotification, ContentBlock, Diff, InitializeRequest, McpServer, McpServerStdio,
    NewSessionRequest, PermissionOptionId, PlanEntry, PromptRequest, RequestPermissionOutcome,
    RequestPermissionRequest, RequestPermissionResponse, SelectedPermissionOutcome,
    SessionNotification, SessionUpdate, TextContent, ToolCallContent, ToolCallId, ToolCallStatus,
    ToolCallUpdate, ToolKind,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{AcpAgent, Agent, ConnectionTo, LineDirection, Responder};
use tokio::sync::mpsc::{self, UnboundedSender};
use tokio::sync::oneshot;

/// One entry in the transcript shown in the panel.
#[derive(Debug, PartialEq)]
pub enum TranscriptItem {
    /// What the user typed, plus which pipeline specialist it was routed
    /// to (shown as a small chip — the wire carries the full "Use the …
    /// subagent:" wrapper, the transcript shows the human's own words).
    User {
        text: String,
        specialist: Option<String>,
    },
    /// The agent's reply, as markdown. Grown in place as more chunks
    /// stream in (adjacency-based — see `apply`) rather than starting a
    /// new "claude" bubble every few words; a real session showed exactly
    /// that fragmentation before this existed.
    Agent(String),
    /// The agent's reasoning, streamed the same way but rendered
    /// collapsed and dimmed — present when wanted, never in the way.
    Thought(String),
    /// A tool call as a live card: created by `SessionUpdate::ToolCall`,
    /// then MUTATED by every `ToolCallUpdate` with the same id.
    Tool(ToolCard),
    /// The agent's current plan — ONE card that replaces itself in place
    /// (the protocol sends the full entry list every time), so the
    /// checklist ticks over where it first appeared.
    Plan(Vec<PlanEntry>),
    /// Routine state ("starting…", "connected") — rendered quietly.
    Status(String),
    /// A real failure (spawn failure, connection dropped with an error).
    Error(String),
}

/// The live state of one tool call. Everything the renderer needs and
/// nothing else — the protocol types stay at the boundary.
#[derive(Debug, PartialEq)]
pub struct ToolCard {
    pub id: ToolCallId,
    pub title: String,
    pub kind: ToolKind,
    pub status: ToolCallStatus,
    pub detail: Vec<ToolDetail>,
}

/// Tool-call content the panel renders richly: file edits become real
/// ± diffs (computed UI-side with `similar`), everything else is text.
#[derive(Debug, PartialEq)]
pub enum ToolDetail {
    Text(String),
    Diff {
        path: String,
        old: String,
        new: String,
    },
}

/// What the connection thread hands the UI thread: either a new item, or
/// a patch for a tool card already in the transcript. The protocol's own
/// `ToolCallUpdate` is the patch type verbatim — mirroring its five
/// optional fields into a local struct would be a second copy to drift.
/// (Boxed: `ToolCallUpdate` is ~4× the size of the other variants, and
/// clippy is right that every op would pay for the largest.)
#[derive(Debug)]
pub enum TranscriptOp {
    Push(TranscriptItem),
    UpdateTool(Box<ToolCallUpdate>),
}

/// A tool call awaiting a yes/no from the user, shown as buttons in the
/// panel instead of decided automatically.
pub struct PendingPermission {
    pub tool_title: String,
    pub options: Vec<(PermissionOptionId, String)>,
    reply: oneshot::Sender<PermissionOptionId>,
}

/// What the UI sends the connection thread.
enum Outgoing {
    Prompt(String),
    /// Stop the in-flight turn (ACP `session/cancel`). Harmless if
    /// nothing is in flight.
    Cancel,
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
    transcript: Arc<Mutex<Vec<TranscriptOp>>>,
    /// `Option` so `Drop` can close the channel while `&mut self`.
    outgoing: Option<UnboundedSender<Outgoing>>,
    pending: Arc<Mutex<Option<PendingPermission>>>,
    /// True while a prompt turn is in flight — drives the footer spinner
    /// and the Stop button.
    turn_active: Arc<AtomicBool>,
    /// Latest context-window numbers (`used`, `size`), shown in the
    /// footer rather than as transcript rows — the protocol streams one
    /// per chunk and a transcript of token counts is noise.
    usage: Arc<Mutex<Option<(u64, u64)>>>,
    /// One line of session state for the panel HEADER — "starting…",
    /// "connected", then the session's own title once the agent names
    /// it. These used to open the transcript as debris rows; a
    /// conversation should start with the conversation.
    status: Arc<Mutex<String>>,
    /// Signalled by the connection thread after `run_session` returns —
    /// i.e. after the crate's teardown of the agent subprocess has run.
    done: std::sync::mpsc::Receiver<()>,
    ctx: egui::Context,
}

impl AgentSession {
    pub fn spawn(ctx: &egui::Context) -> Self {
        let transcript = Arc::new(Mutex::new(Vec::new()));
        let pending = Arc::new(Mutex::new(None));
        let turn_active = Arc::new(AtomicBool::new(false));
        let usage = Arc::new(Mutex::new(None));
        let status = Arc::new(Mutex::new("starting…".to_owned()));
        let (outgoing, rx) = mpsc::unbounded_channel::<Outgoing>();
        let (done_tx, done) = std::sync::mpsc::channel();

        let transcript_for_thread = Arc::clone(&transcript);
        let pending_for_thread = Arc::clone(&pending);
        let turn_for_thread = Arc::clone(&turn_active);
        let usage_for_thread = Arc::clone(&usage);
        let status_for_thread = Arc::clone(&status);
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
                        TranscriptItem::Error(format!("could not start a tokio runtime: {err}")),
                    );
                    return;
                }
            };
            runtime.block_on(run_session(Shared {
                transcript: transcript_for_thread,
                pending: pending_for_thread,
                turn_active: turn_for_thread,
                usage: usage_for_thread,
                status: status_for_thread,
                ctx: ctx_for_thread,
                prompts: rx,
            }));
            // After run_session: the connection (and with it the agent
            // subprocess) is torn down. Tell Drop it can stop waiting.
            let _ = done_tx.send(());
        });

        Self {
            transcript,
            outgoing: Some(outgoing),
            pending,
            turn_active,
            usage,
            status,
            done,
            ctx: ctx.clone(),
        }
    }

    /// Enqueues a prompt for the background connection to send, and shows
    /// it in the transcript IMMEDIATELY — a prompt sent mid-turn queues
    /// behind the running one (Zed's behavior), and a message that
    /// vanishes until its turn starts looks lost, not queued.
    ///
    /// With a `specialist`, the WIRE carries the "Use the … subagent:"
    /// wrapper that makes Claude Code dispatch to that agent, while the
    /// transcript shows the user's own words plus a routing chip — the
    /// plumbing works without being read back to the person who typed.
    pub fn send(&self, text: String, specialist: Option<&str>) {
        if let Some(outgoing) = &self.outgoing {
            let wire = match specialist {
                Some(name) => format!("Use the {name} subagent: {text}"),
                None => text.clone(),
            };
            push(
                &self.transcript,
                &self.ctx,
                TranscriptItem::User {
                    text,
                    specialist: specialist.map(str::to_owned),
                },
            );
            let _ = outgoing.send(Outgoing::Prompt(wire));
        }
    }

    /// Stops the in-flight turn. Also answers any pending permission
    /// request as cancelled — per the protocol, `session/cancel` obliges
    /// the client to resolve outstanding permission requests itself.
    pub fn cancel(&self) {
        drop(self.pending.lock().expect("not poisoned").take());
        if let Some(outgoing) = &self.outgoing {
            let _ = outgoing.send(Outgoing::Cancel);
        }
    }

    /// True while a prompt turn is streaming.
    pub fn turn_active(&self) -> bool {
        self.turn_active.load(Ordering::Relaxed)
    }

    /// Latest (used, size) context-window tokens, if the agent sent any.
    pub fn usage(&self) -> Option<(u64, u64)> {
        *self.usage.lock().expect("not poisoned")
    }

    /// One line of session state for the panel header.
    pub fn status_line(&self) -> String {
        self.status.lock().expect("not poisoned").clone()
    }

    /// Moves every op accumulated since the last call into `out`,
    /// applying `apply`'s rules on the way. Draining rather than cloning
    /// keeps the panel from re-walking an ever-growing `Vec` every frame
    /// — and the rules MUST run on this side of the boundary: the shared
    /// buffer empties every frame, so merging there only ever covered
    /// chunks that batched between frames (a slow-streaming turn
    /// fragmented into dozens of bubbles until this moved).
    pub fn drain_into(&self, out: &mut Vec<TranscriptItem>) {
        let mut shared = self.transcript.lock().expect("not poisoned");
        for op in shared.drain(..) {
            apply(out, op);
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

        // Closing the channel ends run_session's prompt loop (which also
        // cancels any in-flight turn — see the select loop); the
        // connection then drops, and the ACP crate kills the agent's
        // process group. Wait for the thread to confirm.
        drop(self.outgoing.take());
        if self.done.recv_timeout(SHUTDOWN_WAIT).is_err() {
            // Teardown hasn't run yet. Proceeding means the agent
            // subprocess may outlive us — the one case this Drop doesn't
            // close, accepted over hanging the window close indefinitely.
            eprintln!("agent session did not shut down within {SHUTDOWN_WAIT:?}; proceeding");
        }
    }
}

/// Applies one op to the transcript under its rules — THE single home
/// for them (tested below):
///
/// - Adjacent `Agent` (and separately `Thought`) chunks are one bubble
///   growing — merged by ADJACENCY, not `message_id` equality: this
///   adapter doesn't set `message_id` on every chunk (measured live),
///   while a real turn boundary always has a tool card, user message or
///   status line in between, never silence.
/// - A `Status`/`Error` line identical to the previous one is noise, not
///   information, and is dropped; `User`/`Agent` can legitimately repeat
///   and never are.
/// - A `Plan` REPLACES the existing plan card in place — the protocol
///   sends the full entry list every time, and a transcript of stale
///   checklists buries the live one.
/// - `UpdateTool` patches the matching card wherever it sits; an update
///   for an id never seen becomes a new card (the protocol allows it).
pub fn apply(out: &mut Vec<TranscriptItem>, op: TranscriptOp) {
    let item = match op {
        TranscriptOp::Push(item) => item,
        TranscriptOp::UpdateTool(update) => {
            let update = *update;
            let card = out.iter_mut().rev().find_map(|item| match item {
                TranscriptItem::Tool(card) if card.id == update.tool_call_id => Some(card),
                _ => None,
            });
            match card {
                Some(card) => patch_card(card, update),
                None => {
                    let mut card = ToolCard {
                        id: update.tool_call_id.clone(),
                        title: String::new(),
                        kind: ToolKind::Other,
                        status: ToolCallStatus::Pending,
                        detail: Vec::new(),
                    };
                    patch_card(&mut card, update);
                    out.push(TranscriptItem::Tool(card));
                }
            }
            return;
        }
    };

    match &item {
        TranscriptItem::Agent(text) => {
            if let Some(TranscriptItem::Agent(last)) = out.last_mut() {
                last.push_str(text);
                return;
            }
        }
        TranscriptItem::Thought(text) => {
            if let Some(TranscriptItem::Thought(last)) = out.last_mut() {
                last.push_str(text);
                return;
            }
        }
        TranscriptItem::Plan(entries) => {
            if let Some(TranscriptItem::Plan(existing)) = out
                .iter_mut()
                .rev()
                .find(|i| matches!(i, TranscriptItem::Plan(_)))
            {
                *existing = entries.clone();
                return;
            }
        }
        TranscriptItem::Status(text) | TranscriptItem::Error(text) => {
            if matches!(out.last(),
                Some(TranscriptItem::Status(last) | TranscriptItem::Error(last)) if last == text)
            {
                return;
            }
        }
        TranscriptItem::User { .. } | TranscriptItem::Tool(_) => {}
    }
    out.push(item);
}

/// Folds a protocol `ToolCallUpdate` into a card. Content, when present,
/// REPLACES the card's detail — that is the field's protocol semantics.
fn patch_card(card: &mut ToolCard, update: ToolCallUpdate) {
    let fields = update.fields;
    if let Some(title) = fields.title {
        card.title = title;
    }
    if let Some(kind) = fields.kind {
        card.kind = kind;
    }
    if let Some(status) = fields.status {
        card.status = status;
    }
    if let Some(content) = fields.content {
        card.detail = detail_from(content);
    }
}

/// Protocol tool-call content → what the panel renders. Diffs keep their
/// texts verbatim (the renderer computes the ± lines); text blocks pass
/// through; a terminal embed becomes its id (no terminal support yet —
/// named rather than dropped).
fn detail_from(content: Vec<ToolCallContent>) -> Vec<ToolDetail> {
    content
        .into_iter()
        .map(|entry| match entry {
            ToolCallContent::Content(c) => match c.content {
                ContentBlock::Text(text) => ToolDetail::Text(text.text),
                other => ToolDetail::Text(format!("{other:?}")),
            },
            ToolCallContent::Diff(Diff {
                path,
                old_text,
                new_text,
                ..
            }) => ToolDetail::Diff {
                path: path.display().to_string(),
                old: old_text.unwrap_or_default(),
                new: new_text,
            },
            ToolCallContent::Terminal(term) => {
                ToolDetail::Text(format!("terminal {:?}", term.terminal_id))
            }
            // Future content kinds: named, never silently dropped.
            other => ToolDetail::Text(format!("{other:?}")),
        })
        .collect()
}

fn push(transcript: &Arc<Mutex<Vec<TranscriptOp>>>, ctx: &egui::Context, item: TranscriptItem) {
    push_op(transcript, ctx, TranscriptOp::Push(item));
}

fn push_op(transcript: &Arc<Mutex<Vec<TranscriptOp>>>, ctx: &egui::Context, op: TranscriptOp) {
    // Plain append: the merge and patch rules live in ONE place —
    // `apply`, run on the UI side by `drain_into` — because they must run
    // there anyway (this shared buffer empties every frame, so any
    // merging done here only ever covered chunks that happened to batch
    // between frames; for a while the rules lived in both places, two
    // copies drifting).
    let mut guard = transcript.lock().expect("not poisoned");
    guard.push(op);
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

/// Everything `run_session` needs, in one bag — six loose parameters
/// crossed the clippy too-many-arguments line and read worse.
struct Shared {
    transcript: Arc<Mutex<Vec<TranscriptOp>>>,
    pending: Arc<Mutex<Option<PendingPermission>>>,
    turn_active: Arc<AtomicBool>,
    usage: Arc<Mutex<Option<(u64, u64)>>>,
    status: Arc<Mutex<String>>,
    ctx: egui::Context,
    prompts: mpsc::UnboundedReceiver<Outgoing>,
}

/// Sets the header's one-line session state.
fn set_status(status: &Arc<Mutex<String>>, ctx: &egui::Context, line: impl Into<String>) {
    *status.lock().expect("not poisoned") = line.into();
    ctx.request_repaint();
}

async fn run_session(shared: Shared) {
    let Shared {
        transcript,
        pending,
        turn_active,
        usage,
        status,
        ctx,
        mut prompts,
    } = shared;

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
                TranscriptItem::Status(format!("[stderr] {line}")),
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
    set_status(&status, &ctx, format!("starting {command_line}…"));

    let notify_transcript = Arc::clone(&transcript);
    let notify_usage = Arc::clone(&usage);
    let notify_status = Arc::clone(&status);
    let notify_ctx = ctx.clone();

    let result = agent_client_protocol::Client
        .builder()
        .on_receive_notification(
            move |notification: SessionNotification, _cx| {
                let transcript = Arc::clone(&notify_transcript);
                let usage = Arc::clone(&notify_usage);
                let status = Arc::clone(&notify_status);
                let ctx = notify_ctx.clone();
                async move {
                    // Usage and the session title are panel chrome, not
                    // conversation — footer slot and header line, never
                    // transcript rows.
                    match &notification.update {
                        SessionUpdate::UsageUpdate(u) => {
                            *usage.lock().expect("not poisoned") = Some((u.used, u.size));
                            ctx.request_repaint();
                            return Ok(());
                        }
                        SessionUpdate::SessionInfoUpdate(info) => {
                            if let agent_client_protocol::schema::MaybeUndefined::Value(title) =
                                &info.title
                            {
                                set_status(&status, &ctx, title.clone());
                            }
                            return Ok(());
                        }
                        _ => {}
                    }
                    if let Some(op) = op_for(notification.update) {
                        push_op(&transcript, &ctx, op);
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
                        // dropped sender (Stop clicked, or the window closed
                        // mid-request) falls back to cancelling rather than
                        // hanging.
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
            let turn_active = Arc::clone(&turn_active);
            let status = Arc::clone(&status);
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

                set_status(&status, &ctx, "connected");

                // The prompt loop, with the Stop button's requirement
                // built in: while a turn is in flight, the channel is
                // STILL read — Cancel becomes `session/cancel` on the
                // wire (the agent then ends the turn with a Cancelled
                // stop reason), and further prompts queue behind the
                // turn instead of being lost.
                let mut queued: VecDeque<String> = VecDeque::new();
                let mut channel_open = true;
                loop {
                    // A closed channel means the window is going away:
                    // stop BEFORE starting anything still queued.
                    if !channel_open {
                        break;
                    }
                    let text = match queued.pop_front() {
                        Some(text) => text,
                        None => match prompts.recv().await {
                            Some(Outgoing::Prompt(text)) => text,
                            Some(Outgoing::Cancel) => continue, // nothing in flight
                            None => break,
                        },
                    };

                    turn_active.store(true, Ordering::Relaxed);
                    let request = PromptRequest::new(
                        session_id.clone(),
                        vec![ContentBlock::Text(TextContent::new(text))],
                    );
                    let fut = connection.send_request(request).block_task();
                    tokio::pin!(fut);
                    let outcome = loop {
                        tokio::select! {
                            outcome = &mut fut => break outcome,
                            message = prompts.recv(), if channel_open => match message {
                                Some(Outgoing::Prompt(text)) => queued.push_back(text),
                                Some(Outgoing::Cancel) => {
                                    let _ = connection.send_notification(
                                        CancelNotification::new(session_id.clone()),
                                    );
                                }
                                None => {
                                    // Window closing: cancel the turn so
                                    // the response (Cancelled) arrives and
                                    // teardown isn't stuck behind a long
                                    // turn — the wait Drop used to lose.
                                    channel_open = false;
                                    let _ = connection.send_notification(
                                        CancelNotification::new(session_id.clone()),
                                    );
                                }
                            },
                        }
                    };
                    turn_active.store(false, Ordering::Relaxed);
                    ctx.request_repaint();
                    outcome?;
                }
                Ok(())
            }
        })
        .await;

    turn_active.store(false, Ordering::Relaxed);
    if let Err(err) = result {
        push(
            &transcript,
            &ctx,
            TranscriptItem::Error(format!("ACP connection ended: {err}")),
        );
    }
}

/// Turns one protocol update into a transcript op — or `None` for
/// updates that deliberately have no transcript presence (the user's own
/// message echoed back; the slash-command list, which belongs in an
/// input autocomplete this panel doesn't have yet). Usage never reaches
/// here (routed to the footer slot by the notification handler).
/// Anything not special-cased still shows something (a truncated debug),
/// never a silent drop — the first real session's
/// `AvailableCommandsUpdate` wall is why summaries exist at all.
fn op_for(update: SessionUpdate) -> Option<TranscriptOp> {
    let item = match update {
        SessionUpdate::AgentMessageChunk(chunk) => match chunk.content {
            ContentBlock::Text(text) => TranscriptItem::Agent(text.text),
            other => TranscriptItem::Status(format!("{other:?}")),
        },
        SessionUpdate::AgentThoughtChunk(chunk) => match chunk.content {
            ContentBlock::Text(text) => TranscriptItem::Thought(text.text),
            _ => return None,
        },
        SessionUpdate::UserMessageChunk(_) => return None,
        SessionUpdate::ToolCall(call) => TranscriptItem::Tool(ToolCard {
            id: call.tool_call_id,
            title: call.title,
            kind: call.kind,
            status: call.status,
            detail: detail_from(call.content),
        }),
        SessionUpdate::ToolCallUpdate(update) => {
            return Some(TranscriptOp::UpdateTool(Box::new(update)))
        }
        SessionUpdate::Plan(plan) => TranscriptItem::Plan(plan.entries),
        SessionUpdate::AvailableCommandsUpdate(_) => return None,
        // SessionInfoUpdate and UsageUpdate are intercepted by the
        // notification handler (header line / footer slot) before this
        // function runs; listed here so the fallback below never
        // Debug-prints them if that routing ever changes.
        SessionUpdate::SessionInfoUpdate(_) => return None,
        SessionUpdate::UsageUpdate(_) => return None,
        other => {
            let mut text = format!("{other:?}");
            text.truncate(200);
            TranscriptItem::Status(text)
        }
    };
    Some(TranscriptOp::Push(item))
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_client_protocol::schema::v1::ToolCallUpdateFields;

    fn drain(ops: Vec<TranscriptOp>) -> Vec<TranscriptItem> {
        let mut out = Vec::new();
        for op in ops {
            apply(&mut out, op);
        }
        out
    }

    fn agent(text: &str) -> TranscriptOp {
        TranscriptOp::Push(TranscriptItem::Agent(text.into()))
    }

    fn tool(id: &str, title: &str) -> TranscriptOp {
        TranscriptOp::Push(TranscriptItem::Tool(ToolCard {
            id: id.to_string().into(),
            title: title.into(),
            kind: ToolKind::Read,
            status: ToolCallStatus::Pending,
            detail: Vec::new(),
        }))
    }

    #[test]
    fn adjacent_agent_text_chunks_grow_one_bubble() {
        let out = drain(vec![
            agent("Builds clean"),
            agent(" (the crate is its own workspace)."),
        ]);
        assert_eq!(
            out,
            vec![TranscriptItem::Agent(
                "Builds clean (the crate is its own workspace).".into()
            )]
        );
    }

    #[test]
    fn a_tool_card_between_chunks_is_a_turn_boundary() {
        let out = drain(vec![
            agent("first reply"),
            tool("call-1", "Read File"),
            agent("second reply"),
        ]);
        assert_eq!(out.len(), 3, "the card must split the bubbles: {out:?}");
    }

    #[test]
    fn a_tool_update_mutates_the_card_in_place() {
        // THE fix this rework exists for: updates used to print
        // "tool update: ToolCallId(…)" rows; now they patch the card.
        let out = drain(vec![
            tool("call-1", "Read File"),
            agent("reading…"),
            TranscriptOp::UpdateTool(Box::new(ToolCallUpdate::new(
                "call-1",
                ToolCallUpdateFields::new().status(ToolCallStatus::Completed),
            ))),
        ]);
        assert_eq!(out.len(), 2, "no new row for the update: {out:?}");
        let TranscriptItem::Tool(card) = &out[0] else {
            panic!("card missing: {out:?}");
        };
        assert_eq!(card.status, ToolCallStatus::Completed);
        assert_eq!(card.title, "Read File", "unset fields stay untouched");
    }

    #[test]
    fn an_update_for_an_unseen_id_becomes_a_new_card() {
        // The protocol permits updates for calls the client never saw
        // created; they must not vanish.
        let out = drain(vec![TranscriptOp::UpdateTool(Box::new(
            ToolCallUpdate::new(
                "ghost",
                ToolCallUpdateFields::new()
                    .title("Late title".to_string())
                    .status(ToolCallStatus::InProgress),
            ),
        ))]);
        let TranscriptItem::Tool(card) = &out[0] else {
            panic!("card missing: {out:?}");
        };
        assert_eq!(card.title, "Late title");
        assert_eq!(card.status, ToolCallStatus::InProgress);
    }

    #[test]
    fn a_new_plan_replaces_the_old_card_in_place() {
        use agent_client_protocol::schema::v1::{PlanEntryPriority, PlanEntryStatus};
        let entry = |text: &str, status| {
            PlanEntry::new(text.to_string(), PlanEntryPriority::Medium, status)
        };
        let out = drain(vec![
            TranscriptOp::Push(TranscriptItem::Plan(vec![entry(
                "read the file",
                PlanEntryStatus::InProgress,
            )])),
            agent("working through it"),
            TranscriptOp::Push(TranscriptItem::Plan(vec![entry(
                "read the file",
                PlanEntryStatus::Completed,
            )])),
        ]);
        assert_eq!(out.len(), 2, "one plan card, not a stack: {out:?}");
        let TranscriptItem::Plan(entries) = &out[0] else {
            panic!("plan missing: {out:?}");
        };
        assert_eq!(entries[0].status, PlanEntryStatus::Completed);
    }

    #[test]
    fn repeated_status_noise_collapses_but_conversation_never_does() {
        let status = |text: &str| TranscriptOp::Push(TranscriptItem::Status(text.into()));
        let user = |text: &str| {
            TranscriptOp::Push(TranscriptItem::User {
                text: text.into(),
                specialist: None,
            })
        };
        let out = drain(vec![
            status("connected"),
            status("connected"),
            user("hi"),
            user("hi"),
        ]);
        // One status line survives; BOTH user lines do — a person may
        // genuinely say the same thing twice.
        assert_eq!(
            out,
            vec![
                TranscriptItem::Status("connected".into()),
                TranscriptItem::User {
                    text: "hi".into(),
                    specialist: None,
                },
                TranscriptItem::User {
                    text: "hi".into(),
                    specialist: None,
                },
            ]
        );
    }

    #[test]
    fn non_adjacent_duplicate_status_lines_both_survive() {
        // Dedup is last-line-only by design: the same status an hour
        // later is real information again.
        let status = |text: &str| TranscriptOp::Push(TranscriptItem::Status(text.into()));
        let out = drain(vec![
            status("session: rework"),
            status("connected"),
            status("session: rework"),
        ]);
        assert_eq!(out.len(), 3);
    }
}
