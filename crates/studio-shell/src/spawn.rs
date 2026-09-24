//! Where the checkout is, and how the Studio spawns the Python side.
//!
//! Every subprocess this window starts (the presenter, the render
//! stream, the walk view) is built here, once: the same `uv run` line,
//! the same working directory rule, the same process-group discipline
//! and the same teardown. The repo root is resolved at run time — a
//! binary copied to another machine, or a checkout that moved, must
//! still find its pipeline.

use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::OnceLock;

/// Names the checkout explicitly; wins over every guess below.
pub const REPO_ENV: &str = "TRAINNR_REPO";
/// The file whose presence marks a checkout: the pipeline's manifest.
const REPO_MARKER: &str = "pipeline/pyproject.toml";
/// The Python packages, relative to the repo root.
pub const PIPELINE_DIR: &str = "pipeline";
pub const WALK_PACKAGE_DIR: &str = "rq_mjlab";
/// The scripts the Studio runs, relative to the repo root.
pub const TOOLS_DIR: &str = "tools";
pub const PRESENTER_SCRIPT: &str = "studio-present.py";
pub const RENDER_STREAM_SCRIPT: &str = "studio-render-stream.py";
/// The pipeline extras a Studio subprocess needs: MuJoCo (`sim`) and
/// the Rerun SDK (`viz`, so a script can narrate into this window).
// `deploy`: onnxruntime, for a deployment's policy in the viewport.
const PIPELINE_EXTRAS: &[&str] = &["sim", "viz", "deploy"];

/// The WSL-only environment for MuJoCo's offscreen GL (the box's
/// documented gotcha: without these it falls back to llvmpipe, the
/// software rasterizer at ~300 ms/frame). Mirrors `pipeline/wsl.env`.
#[cfg(target_os = "linux")]
const WSL_GALLIUM_DRIVER: &str = "d3d12";
#[cfg(target_os = "linux")]
const WSL_LIB_DIR: &str = "/usr/lib/wsl/lib";

/// Whether this process runs under WSL. Both marks are checked: the
/// distro name the shell exports, and the interop file the kernel keeps.
pub fn on_wsl() -> bool {
    std::env::var_os("WSL_DISTRO_NAME").is_some()
        || Path::new("/proc/sys/fs/binfmt_misc/WSLInterop").exists()
}

/// The checkout, resolved once: `$TRAINNR_REPO`, else the nearest
/// ancestor of the executable or of the working directory that holds
/// `pipeline/pyproject.toml`, else — for `cargo run` from a checkout —
/// the crate's own manifest directory two levels down from the root.
pub fn repo_root() -> &'static Path {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        let root = resolve_repo_root(
            std::env::var_os(REPO_ENV).map(PathBuf::from),
            std::env::current_exe().ok(),
            std::env::current_dir().ok(),
        );
        rerun::external::re_log::info!("repo root: {}", root.display());
        root
    })
}

/// The rule behind `repo_root`, testable without touching the process.
fn resolve_repo_root(
    named: Option<PathBuf>,
    exe: Option<PathBuf>,
    cwd: Option<PathBuf>,
) -> PathBuf {
    if let Some(root) = named.filter(|p| is_repo_root(p)) {
        return root;
    }
    for start in [exe, cwd].into_iter().flatten() {
        if let Some(root) = start.ancestors().find(|p| is_repo_root(p)) {
            return root.to_path_buf();
        }
    }
    // `crates/studio-shell` is two directories under the repo root in a
    // checkout; a copied binary never reaches this line with a live path.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .map(Path::to_path_buf)
        .unwrap_or_default()
}

fn is_repo_root(path: &Path) -> bool {
    path.join(REPO_MARKER).is_file()
}

/// `uv run --extra sim --extra viz python tools/<script>` from the
/// pipeline directory, in its own process group so `kill_tree` reaps
/// the python grandchild with the `uv` wrapper. The caller adds the
/// script's arguments and the stdio it wants.
pub fn pipeline_command(script: &str) -> Command {
    let root = repo_root();
    let mut command = Command::new("uv");
    command.arg("run");
    for extra in PIPELINE_EXTRAS {
        command.args(["--extra", extra]);
    }
    command
        .arg("python")
        .arg(root.join(TOOLS_DIR).join(script))
        .current_dir(root.join(PIPELINE_DIR));
    own_process_group(&mut command);
    mujoco_environment(&mut command);
    command
}

/// `uv run --offline python -m <module>` from the walk package's
/// directory: the rq_mjlab venv (torch, warp, mjlab), not the pipeline's.
pub fn walk_command(module: &str) -> Command {
    let mut command = Command::new("uv");
    command
        .args(["run", "--offline", "python", "-m", module])
        .current_dir(repo_root().join(WALK_PACKAGE_DIR));
    own_process_group(&mut command);
    mujoco_environment(&mut command);
    command
}

/// Its own process group, so the whole tree can be reaped: killing only
/// the `uv` wrapper left the python grandchild alive and flooding the
/// ingest channel (the zombie stream, 2026-09-01).
fn own_process_group(command: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        command.process_group(0);
    }
    #[cfg(not(unix))]
    {
        let _ = command;
    }
}

/// MuJoCo's offscreen GL: EGL on Linux (macOS must not get it) unless
/// the caller's environment already chose one (a headless box on
/// osmesa, `pipeline/wsl.env`); under WSL the D3D12 driver with its
/// library directory PREPENDED to whatever the caller's loader path
/// already holds — a cloud box's CUDA paths stay. Harmless on native
/// Linux, where the WSL block is skipped.
fn mujoco_environment(command: &mut Command) {
    #[cfg(target_os = "linux")]
    {
        for (name, default) in [("MUJOCO_GL", "egl"), ("OMP_NUM_THREADS", "1")] {
            if std::env::var_os(name).is_none_or(|v| v.is_empty()) {
                command.env(name, default);
            }
        }
        if on_wsl() {
            let lib_path = match std::env::var_os("LD_LIBRARY_PATH") {
                Some(existing) if !existing.is_empty() => {
                    let mut joined = std::ffi::OsString::from(WSL_LIB_DIR);
                    joined.push(":");
                    joined.push(existing);
                    joined
                }
                _ => std::ffi::OsString::from(WSL_LIB_DIR),
            };
            command
                .env("GALLIUM_DRIVER", WSL_GALLIUM_DRIVER)
                .env("LD_LIBRARY_PATH", lib_path);
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = command;
    }
}

/// How long a child gets to leave on TERM before the KILL: the Python
/// sides close their Rerun stream into the Studio's own server on TERM
/// (`leave_cleanly_on_term` in pipeline/rq_pipeline/viz.py); a KILL in
/// the same instant cut the stream and the server logged an h2 error at
/// every close (2026-09-12). Only a child that ignores TERM waits it out.
#[cfg(unix)]
const TERM_GRACE: std::time::Duration = std::time::Duration::from_millis(1500);
#[cfg(unix)]
const TERM_POLL: std::time::Duration = std::time::Duration::from_millis(25);

/// Signal a child's whole process group (the child leads it; see
/// `spawn_*` above). Always with the `--`: without it, procps kill can
/// re-parse a negative pgid as a signal spec plus a DIFFERENT pid —
/// measured 2026-09-01, and the mis-signaled process was the Studio
/// itself.
#[cfg(unix)]
fn signal_group(child: &Child, signal: &str) {
    let _ = Command::new("kill")
        .args(["-s", signal, "--", &format!("-{}", child.id())])
        .status();
}

/// End a spawned tree, blocking: TERM to the group, up to `TERM_GRACE`
/// for the tree to leave on its own, then KILL to the whole group (a
/// grandchild that ignored TERM must not outlive its parent), then the
/// child itself, then reap it.
pub fn kill_tree(child: &mut Child) {
    #[cfg(unix)]
    {
        signal_group(child, "TERM");
        let deadline = std::time::Instant::now() + TERM_GRACE;
        while std::time::Instant::now() < deadline {
            if matches!(child.try_wait(), Ok(Some(_))) {
                return;
            }
            std::thread::sleep(TERM_POLL);
        }
        signal_group(child, "KILL");
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// End a spawned tree without holding the caller: TERM goes to the
/// group now (so a quit that ends the process right after still sent
/// it), and the grace, the KILL and the reap run on their own thread.
/// The 1.5 s grace on the UI thread froze the window on every scene
/// change and doubled at quit (2026-09-13).
pub fn end_tree(child: Child) {
    #[cfg(unix)]
    signal_group(&child, "TERM");
    // The child sits in a slot both sides can reach: the thread takes it
    // to reap, and if no thread can be had this thread takes it back and
    // reaps the blocking way, rather than dropping it unreaped.
    let slot = std::sync::Arc::new(std::sync::Mutex::new(Some(child)));
    let theirs = std::sync::Arc::clone(&slot);
    let reap = std::thread::Builder::new()
        .name("end-tree".to_owned())
        .spawn(move || {
            if let Some(mut child) = theirs.lock().ok().and_then(|mut s| s.take()) {
                kill_tree(&mut child);
            }
        });
    if reap.is_err() {
        if let Some(mut child) = slot.lock().ok().and_then(|mut s| s.take()) {
            kill_tree(&mut child);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_repo_root_is_named_walked_up_to_or_the_checkouts_own() {
        let checkout = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .expect("two above the crate")
            .to_path_buf();
        assert!(is_repo_root(&checkout), "the crate sits in a checkout");
        // A named root that is one wins.
        assert_eq!(
            resolve_repo_root(Some(checkout.clone()), None, None),
            checkout
        );
        // A named path that is not a checkout is ignored, and the
        // executable's ancestors are walked instead.
        let deep = checkout.join("crates").join("studio-shell").join("src");
        assert_eq!(
            resolve_repo_root(Some(std::env::temp_dir()), Some(deep.clone()), None),
            checkout
        );
        // The working directory is the next guess.
        assert_eq!(resolve_repo_root(None, None, Some(deep)), checkout);
        // Nothing named, nothing found: the checkout this was built in.
        assert_eq!(
            resolve_repo_root(None, Some(std::env::temp_dir()), None),
            checkout
        );
    }

    #[test]
    fn the_pipeline_line_is_one_uv_run_from_the_pipeline_directory() {
        let command = pipeline_command(PRESENTER_SCRIPT);
        let args: Vec<String> = command
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert_eq!(args[0], "run");
        assert!(args.contains(&"--extra".to_owned()) && args.contains(&"sim".to_owned()));
        assert!(args.last().unwrap().ends_with(PRESENTER_SCRIPT));
        assert!(command
            .get_current_dir()
            .is_some_and(|d| d.ends_with(PIPELINE_DIR)));
        let walk = walk_command("rq_mjlab.walk_view");
        assert!(walk
            .get_current_dir()
            .is_some_and(|d| d.ends_with(WALK_PACKAGE_DIR)));
    }
}
