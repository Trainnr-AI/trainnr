//! The engine: the Python side the Studio runs its tools with, set up the
//! first time an action needs it when no checkout is there (the desktop
//! app plan, docs/85, 2026-10-11).
//!
//! A Studio downloaded from a release has no repository beside it, and
//! every tool, job and stream runs from one. The engine is that, made
//! lean: the release's runtime archive (`tools/runtime-archive.py`: the
//! Python packages, the job scripts and the robot library, about 9 MB),
//! unpacked under `<trainnr home>/engine/`, then `uv sync` of the core
//! (Python, the tools, MuJoCo). `uv` comes from PATH when it is there and
//! is downloaded otherwise, at a pinned version checked by its SHA-256.
//! Nothing is fetched before a button needs it, and each piece after the
//! core (the viewer stream, the trainer) arrives the first time it is
//! used, by `uv run` itself.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sha2::{Digest as _, Sha256};

/// A different runtime archive to install (a URL or a local file): for a
/// mirror, a pre-release, and the tests.
pub const RUNTIME_ENV: &str = "TRAINNR_RUNTIME_URL";
/// The uv the engine runs (`trainnr/paths.py::UV_ENV`), handed to every
/// process the Studio starts so the jobs run the same one.
pub const UV_ENV: &str = "TRAINNR_UV";
/// Under the user's trainnr home (`model::user_home`).
const ENGINE_DIR: &str = "engine";
/// Written last, once the core is synced: an engine without it is
/// unfinished and is set up again.
const READY_FILE: &str = ".ready";
/// The release's runtime archive, by version (`tools/runtime-archive.py`).
const RUNTIME_URL: &str =
    "https://github.com/Trainnr-AI/trainnr/releases/download/v{version}/trainnr-runtime-v{version}.tar.gz";
/// The runtime's folder inside the archive.
const RUNTIME_NAME: &str = "trainnr-runtime-v{version}";
/// The core the first action needs: the tools and the MCP server.
pub const CORE_EXTRAS: &[&str] = &["sim", "mcp"];
/// No download the engine makes is larger than this.
const MAX_DOWNLOAD: u64 = 300_000_000;
const TIMEOUT: Duration = Duration::from_secs(600);

/// uv as the engine downloads it when none is on PATH: the version, and
/// each platform's archive with the SHA-256 astral publishes beside it
/// (read 2026-10-11 from github.com/astral-sh/uv/releases/tag/0.13.0).
const UV_VERSION: &str = "0.13.0";
struct UvPin {
    archive: &'static str,
    sha256: &'static str,
}
const UV_PINS: &[(&str, UvPin)] = &[
    (
        "x86_64-linux",
        UvPin {
            archive: "uv-x86_64-unknown-linux-gnu.tar.gz",
            sha256: "1468ebd5a5541121837c5a2817b9972ba6090fa6caa3d142620850a47fb75154",
        },
    ),
    (
        "aarch64-linux",
        UvPin {
            archive: "uv-aarch64-unknown-linux-gnu.tar.gz",
            sha256: "3ccfb6af6e242433eb552f7d9676abd5412c6595c497e990d9c8cb7b5bd4d2c3",
        },
    ),
    (
        "aarch64-macos",
        UvPin {
            archive: "uv-aarch64-apple-darwin.tar.gz",
            sha256: "a9c1b29002cf3c83f07fa9cd8a887a3be0107d90e23189721221e7257db8e3d6",
        },
    ),
    (
        "x86_64-macos",
        UvPin {
            archive: "uv-x86_64-apple-darwin.tar.gz",
            sha256: "5f44dcbde809b632f47c36fadb241cb4d6f9af71d0c8f172b5d2026d3dde742c",
        },
    ),
    (
        "x86_64-windows",
        UvPin {
            archive: "uv-x86_64-pc-windows-msvc.zip",
            sha256: "088962f9e7b7bd9ea740c04c650b2a21c8928c345bd99ac24350dc924dba656c",
        },
    ),
];

fn this_platform() -> String {
    format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)
}

fn exe(name: &str) -> String {
    format!("{name}{}", std::env::consts::EXE_SUFFIX)
}

/// `<trainnr home>/engine`, when there is a home.
pub fn engine_home() -> Option<PathBuf> {
    crate::model::user_home().map(|home| home.join(ENGINE_DIR))
}

/// The runtime this Studio's version runs, where it unpacks.
fn runtime_dir(home: &Path) -> PathBuf {
    home.join(RUNTIME_NAME.replace("{version}", env!("CARGO_PKG_VERSION")))
}

/// The engine's runtime, once it is set up and synced.
pub fn installed_runtime() -> Option<PathBuf> {
    let dir = runtime_dir(&engine_home()?);
    dir.join(READY_FILE).is_file().then_some(dir)
}

/// The uv to run: `$TRAINNR_UV`, else one on PATH, else the engine's own
/// (downloaded), else the bare name for the OS to find.
pub fn uv_program() -> PathBuf {
    existing_uv(engine_home().as_deref()).unwrap_or_else(|| PathBuf::from("uv"))
}

/// A uv already here, in the one order the setup and every run share:
/// `$TRAINNR_UV`, else PATH, else the engine's own from an earlier setup
/// (so a retry after a failed sync never downloads it again).
fn existing_uv(home: Option<&Path>) -> Option<PathBuf> {
    if let Some(named) = std::env::var_os(UV_ENV).filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(named));
    }
    on_path("uv").or_else(|| {
        home.map(|h| h.join("bin").join(exe("uv")))
            .filter(|p| p.is_file())
    })
}

fn on_path(name: &str) -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    std::env::split_paths(&paths)
        .map(|dir| dir.join(exe(name)))
        .find(|p| p.is_file())
}

/// Where the engine's setup stands, as the buttons show it.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Setup {
    #[default]
    Idle,
    /// What it is doing now, in a few words.
    Working(String),
    Ready,
    Failed(String),
}

/// The engine's setup, on its own thread; `state` is what it is doing.
#[derive(Default)]
pub struct Installer {
    state: Arc<Mutex<Setup>>,
    /// The running setup's downloads and its cancel.
    transfer: Arc<Transfer>,
}

impl Installer {
    /// What the setup is doing now.
    pub fn state(&self) -> Setup {
        self.state.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// The running setup's progress, for the window to draw.
    pub fn transfer(&self) -> Arc<Transfer> {
        Arc::clone(&self.transfer)
    }

    /// Start setting the engine up, unless it is under way or done.
    pub fn start(&mut self) {
        {
            let Ok(mut state) = self.state.lock() else {
                return;
            };
            if matches!(*state, Setup::Working(_) | Setup::Ready) {
                return;
            }
            *state = Setup::Working("Setting up the engine…".to_owned());
        }
        self.transfer = Arc::new(Transfer::default());
        let transfer = Arc::clone(&self.transfer);
        let state = Arc::clone(&self.state);
        std::thread::Builder::new()
            .name("trainnr-engine-setup".to_owned())
            .spawn(move || {
                let say = |words: &str| {
                    if let Ok(mut s) = state.lock() {
                        *s = Setup::Working(words.to_owned());
                    }
                };
                let outcome = match engine_home() {
                    Some(home) => install(&home, &say, &transfer),
                    None => Err("no home folder to set the engine up in".to_owned()),
                };
                if let Ok(mut s) = state.lock() {
                    *s = match outcome {
                        Ok(_) => Setup::Ready,
                        // Cancelled by the person: back to not started,
                        // with nothing to apologise for.
                        Err(_) if transfer.is_cancelled() => Setup::Idle,
                        Err(why) => Setup::Failed(why),
                    };
                }
            })
            .ok();
    }
}

/// Set the engine up in `home`: uv, the runtime, the core. Returns the
/// runtime's folder. `say` names each step for the window.
pub fn install(home: &Path, say: &dyn Fn(&str), transfer: &Transfer) -> Result<PathBuf, String> {
    std::fs::create_dir_all(home).map_err(|e| format!("{}: {e}", home.display()))?;
    let uv = match existing_uv(Some(home)) {
        Some(uv) => uv,
        None => {
            say(&format!("Downloading uv {UV_VERSION} (about 16 MB)…"));
            install_uv(home, transfer)?
        }
    };
    let runtime = runtime_dir(home);
    if !runtime.join(crate::spawn::REPO_MARKER).is_file() {
        say("Downloading the runtime (about 9 MB)…");
        install_runtime(home, transfer)?;
    }
    say("Installing Python and the core tools (about 750 MB on disk, once)…");
    transfer.begin(0); // uv reports no size: the bar runs without one
    let mut sync = std::process::Command::new(&uv);
    // `--locked`: the lock file's versions and hashes, or an error; never
    // a new resolution from PyPI on the day (review, 2026-10-11).
    sync.arg("sync")
        .arg("--locked")
        .arg("--project")
        .arg(runtime.join(crate::spawn::PIPELINE_DIR));
    for extra in CORE_EXTRAS {
        sync.args(["--extra", extra]);
    }
    let done = output_unless_cancelled(&mut sync, transfer)?;
    if !done.status.success() {
        let said = String::from_utf8_lossy(&done.stderr);
        let last = said
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("");
        return Err(format!("uv sync failed: {last}"));
    }
    std::fs::write(runtime.join(READY_FILE), UV_VERSION).map_err(|e| e.to_string())?;
    remove_older_runtimes(home, &runtime);
    Ok(runtime)
}

/// Run `command` to its end like `Command::output`, but stop it when
/// `transfer` is cancelled: `uv sync` takes minutes, and Cancel must not
/// wait for it. What uv installed so far stays, and the next sync goes on
/// from there.
fn output_unless_cancelled(
    command: &mut std::process::Command,
    transfer: &Transfer,
) -> Result<std::process::Output, String> {
    let mut child = command
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("uv did not start: {e}"))?;
    // Drain both pipes on their own threads so a full pipe never stalls uv.
    let drain = |pipe: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut out = Vec::new();
            if let Some(mut pipe) = pipe {
                let _ = pipe.read_to_end(&mut out);
            }
            out
        })
    };
    let stdout = drain(
        child
            .stdout
            .take()
            .map(|p| Box::new(p) as Box<dyn Read + Send>),
    );
    let stderr = drain(
        child
            .stderr
            .take()
            .map(|p| Box::new(p) as Box<dyn Read + Send>),
    );
    let status = loop {
        if transfer.is_cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            return Err(CANCELLED.to_owned());
        }
        match child.try_wait().map_err(|e| e.to_string())? {
            Some(status) => break status,
            None => std::thread::sleep(CANCEL_POLL),
        }
    };
    Ok(std::process::Output {
        status,
        stdout: stdout.join().unwrap_or_default(),
        stderr: stderr.join().unwrap_or_default(),
    })
}

/// How often a long step looks at its cancel flag.
const CANCEL_POLL: Duration = Duration::from_millis(200);

/// Every other runtime under `home`: an earlier version's, left by an
/// update, about 600 MB each. Agents' entries go through the launcher,
/// which the Studio points at the new one (`connect.rs`).
fn remove_older_runtimes(home: &Path, current: &Path) {
    let prefix = RUNTIME_NAME.replace("{version}", "");
    let Ok(entries) = std::fs::read_dir(home) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let named = entry.file_name().to_string_lossy().starts_with(&prefix);
        if named && path != current && path.is_dir() && !path.is_symlink() {
            let _ = std::fs::remove_dir_all(&path);
        }
    }
}

/// The pinned uv for this platform, downloaded, checked and unpacked into
/// `<home>/bin`.
fn install_uv(home: &Path, transfer: &Transfer) -> Result<PathBuf, String> {
    let platform = this_platform();
    let pin = UV_PINS
        .iter()
        .find(|(p, _)| *p == platform)
        .map(|(_, pin)| pin)
        .ok_or_else(|| format!("no uv build is pinned for {platform}; install uv and retry"))?;
    let url = format!(
        "https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{}",
        pin.archive
    );
    let bytes = download_with(&url, transfer)?;
    check_sha256(&bytes, pin.sha256, &url)?;
    let bin = home.join("bin");
    std::fs::create_dir_all(&bin).map_err(|e| e.to_string())?;
    let target = bin.join(exe("uv"));
    if pin.archive.ends_with(".zip") {
        // Windows: the archive is a zip; Windows' own tar (bsdtar, in every
        // Windows 10 and 11) reads it, and the bytes are already checked.
        let staged = home.join(pin.archive);
        std::fs::write(&staged, &bytes).map_err(|e| e.to_string())?;
        let status = std::process::Command::new("tar")
            .arg("-xf")
            .arg(&staged)
            .arg("-C")
            .arg(&bin)
            .status()
            .map_err(|e| format!("tar did not start: {e}"))?;
        let _ = std::fs::remove_file(&staged);
        if !status.success() {
            return Err(format!("could not unpack {}", pin.archive));
        }
    } else {
        // The unix archives hold `uv-<target>/uv` and `uvx`: uv alone is kept.
        let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(&bytes[..]));
        for entry in archive.entries().map_err(|e| e.to_string())? {
            let mut entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path().map_err(|e| e.to_string())?.into_owned();
            if path.file_name().is_some_and(|n| n == "uv") && entry.header().entry_type().is_file()
            {
                let mut body = Vec::new();
                entry.read_to_end(&mut body).map_err(|e| e.to_string())?;
                std::fs::write(&target, body).map_err(|e| e.to_string())?;
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt as _;
                    std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o755))
                        .map_err(|e| e.to_string())?;
                }
            }
        }
    }
    target
        .is_file()
        .then(|| target.clone())
        .ok_or_else(|| format!("{} held no uv", pin.archive))
}

/// The runtime archive: from `$TRAINNR_RUNTIME_URL` (a URL or a file), else
/// the release's, checked against the `.sha256` beside it, unpacked into
/// `home`.
fn install_runtime(home: &Path, transfer: &Transfer) -> Result<(), String> {
    let version = env!("CARGO_PKG_VERSION");
    let source = std::env::var(RUNTIME_ENV)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| RUNTIME_URL.replace("{version}", version));
    let (bytes, listed) = if source.starts_with("https://") {
        let bytes = download_with(&source, transfer)?;
        let listed = String::from_utf8(download(&format!("{source}.sha256"))?)
            .map_err(|_| "the checksum file is not text".to_owned())?;
        (bytes, listed)
    } else {
        let bytes = std::fs::read(&source).map_err(|e| format!("{source}: {e}"))?;
        let listed = std::fs::read_to_string(format!("{source}.sha256"))
            .map_err(|e| format!("{source}.sha256: {e}"))?;
        (bytes, listed)
    };
    let expected = listed.split_whitespace().next().unwrap_or_default();
    check_sha256(&bytes, expected, &source)?;
    let staging = home.join(format!(".unpacking-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);
    unpack_tar_gz(&bytes, &staging)?;
    let name = RUNTIME_NAME.replace("{version}", version);
    let unpacked = staging.join(&name);
    if !unpacked.join(crate::spawn::REPO_MARKER).is_file() {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(format!(
            "{source}: no {name}/{} inside",
            crate::spawn::REPO_MARKER
        ));
    }
    let target = runtime_dir(home);
    let _ = std::fs::remove_dir_all(&target);
    std::fs::rename(&unpacked, &target).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_dir_all(&staging);
    Ok(())
}

/// A download over HTTPS only (a redirect away from it is refused), at
/// most `MAX_DOWNLOAD` bytes.
pub fn download(url: &str) -> Result<Vec<u8>, String> {
    download_with(url, &Transfer::default())
}

/// [`download`], counting into `transfer` as the bytes arrive and
/// stopping with [`CANCELLED`] when it is cancelled.
pub fn download_with(url: &str, transfer: &Transfer) -> Result<Vec<u8>, String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .https_only(true)
        .timeout_global(Some(TIMEOUT))
        .build()
        .into();
    let mut reply = agent.get(url).call().map_err(|e| format!("{url}: {e}"))?;
    transfer.begin(reply.body().content_length().unwrap_or(0));
    let mut reader = reply.body_mut().with_config().limit(MAX_DOWNLOAD).reader();
    let mut bytes = Vec::new();
    let mut chunk = vec![0_u8; DOWNLOAD_CHUNK];
    loop {
        if transfer.is_cancelled() {
            return Err(CANCELLED.to_owned());
        }
        let n = reader.read(&mut chunk).map_err(|e| format!("{url}: {e}"))?;
        if n == 0 {
            return Ok(bytes);
        }
        bytes.extend_from_slice(&chunk[..n]);
        transfer.add(n as u64);
    }
}

/// What a cancelled download or setup answers with.
pub const CANCELLED: &str = "Cancelled";
/// How much a download reads between looks at its cancel flag.
const DOWNLOAD_CHUNK: usize = 64 * 1024;

/// A long fetch's progress and its cancel, shared by the thread doing
/// the work and the window drawing it. The sample and the engine had a
/// spinner and no way out until the 2026-10-12 design review.
#[derive(Default, Debug)]
pub struct Transfer {
    received: AtomicU64,
    /// The size, when the server said it; 0 when unknown (and while a
    /// step that is not a download runs).
    total: AtomicU64,
    cancelled: AtomicBool,
}

impl Transfer {
    /// A new download (or a step with no size, at 0) starts.
    pub fn begin(&self, total: u64) {
        self.received.store(0, Ordering::Relaxed);
        self.total.store(total, Ordering::Relaxed);
    }

    fn add(&self, n: u64) {
        self.received.fetch_add(n, Ordering::Relaxed);
    }

    /// Ask the work to stop at its next look.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// The share done, when the size is known.
    pub fn fraction(&self) -> Option<f32> {
        let total = self.total.load(Ordering::Relaxed);
        (total > 0).then(|| (self.received.load(Ordering::Relaxed) as f64 / total as f64) as f32)
    }

    /// "12 of 48 MB", "12 MB" with no size, nothing before a byte.
    pub fn amount(&self) -> Option<String> {
        let mb = |b: u64| b as f64 / 1e6;
        let received = self.received.load(Ordering::Relaxed);
        let total = self.total.load(Ordering::Relaxed);
        match (received, total) {
            (0, 0) => None,
            (r, 0) => Some(format!("{:.0} MB", mb(r))),
            (r, t) => Some(format!("{:.0} of {:.0} MB", mb(r), mb(t))),
        }
    }
}

/// `bytes` must hash to `expected` (hex SHA-256), or nothing is unpacked.
pub fn check_sha256(bytes: &[u8], expected: &str, what: &str) -> Result<(), String> {
    let digest = Sha256::digest(bytes);
    let got: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    if got.eq_ignore_ascii_case(expected.trim()) {
        Ok(())
    } else {
        Err(format!(
            "{what}: SHA-256 {} is not the listed {}",
            &got[..12],
            expected.get(..12).unwrap_or(expected)
        ))
    }
}

/// Unpack a tar.gz into `dir`: plain files and folders only, every one
/// inside `dir`; a link, a device or a path that leaves is refused and
/// nothing is kept.
pub fn unpack_tar_gz(bytes: &[u8], dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(bytes));
    archive.set_preserve_permissions(true);
    let result = (|| {
        for entry in archive.entries().map_err(|e| e.to_string())? {
            let mut entry = entry.map_err(|e| e.to_string())?;
            let kind = entry.header().entry_type();
            let path = entry.path().map_err(|e| e.to_string())?.into_owned();
            if !(kind.is_file() || kind.is_dir()) {
                return Err(format!("{}: not a file or a folder", path.display()));
            }
            let plain = path
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_)));
            if !plain {
                return Err(format!("{}: leaves the folder", path.display()));
            }
            if !entry.unpack_in(dir).map_err(|e| e.to_string())? {
                return Err(format!("{}: leaves the folder", path.display()));
            }
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_dir_all(dir);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tar_gz(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut builder = tar::Builder::new(Vec::new());
        for (name, body) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(body.len() as u64);
            header.set_mode(0o644);
            header.set_entry_type(tar::EntryType::Regular);
            // The name written as raw bytes: the crate refuses to write a
            // `..`, and a hostile archive is exactly one that has it.
            let raw = &mut header.as_gnu_mut().expect("gnu").name;
            raw[..name.len()].copy_from_slice(name.as_bytes());
            header.set_cksum();
            builder.append(&header, *body).expect("append");
        }
        let raw = builder.into_inner().expect("tar");
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        std::io::Write::write_all(&mut gz, &raw).expect("gz");
        gz.finish().expect("gz")
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("studio-engine-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn a_digest_that_is_not_the_listed_one_is_refused() {
        let bytes = b"runtime";
        let good: String = Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert!(check_sha256(bytes, &good, "x").is_ok());
        assert!(check_sha256(bytes, &good.to_uppercase(), "x").is_ok());
        assert!(check_sha256(b"tampered", &good, "x").is_err());
    }

    #[test]
    fn an_archive_unpacks_and_one_that_leaves_is_refused_whole() {
        let dir = scratch("unpack");
        let good = tar_gz(&[
            ("rt/trainnr/pyproject.toml", b"[project]"),
            ("rt/tools/a.py", b"x"),
        ]);
        unpack_tar_gz(&good, &dir).expect("unpacks");
        assert!(dir.join("rt/trainnr/pyproject.toml").is_file());
        let _ = std::fs::remove_dir_all(&dir);
        let bad = tar_gz(&[("rt/ok.txt", b"x"), ("../escaped.txt", b"x")]);
        assert!(unpack_tar_gz(&bad, &dir).is_err());
        assert!(!dir.exists(), "nothing is kept of a refused archive");
        assert!(!std::env::temp_dir().join("escaped.txt").exists());
    }

    #[test]
    fn the_runtime_installs_from_a_local_archive_checked_by_its_digest() {
        let home = scratch("runtime");
        std::fs::create_dir_all(&home).unwrap();
        let name = RUNTIME_NAME.replace("{version}", env!("CARGO_PKG_VERSION"));
        let marker = format!("{name}/{}", crate::spawn::REPO_MARKER);
        let bytes = tar_gz(&[(&marker, b"[project]")]);
        let archive = home.join("rt.tar.gz");
        std::fs::write(&archive, &bytes).unwrap();
        let digest: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        std::fs::write(
            home.join("rt.tar.gz.sha256"),
            format!("{digest}  rt.tar.gz\n"),
        )
        .unwrap();
        std::env::set_var(RUNTIME_ENV, &archive);
        let installed = install_runtime(&home, &Transfer::default());
        std::fs::write(home.join("rt.tar.gz.sha256"), "0000  rt.tar.gz\n").unwrap();
        let tampered = install_runtime(&home, &Transfer::default());
        std::env::remove_var(RUNTIME_ENV);
        installed.expect("installs");
        assert!(runtime_dir(&home).join(crate::spawn::REPO_MARKER).is_file());
        assert!(
            tampered.is_err(),
            "a digest that does not match unpacks nothing"
        );
        let _ = std::fs::remove_dir_all(&home);
    }

    /// The whole setup on this machine, from nothing: uv downloaded when
    /// none is on PATH, the runtime unpacked, the core synced. Network and
    /// minutes, so it runs on request:
    /// `TRAINNR_TEST_ENGINE_HOME=<empty dir> TRAINNR_RUNTIME_URL=<archive>
    /// cargo test --release -- --ignored a_fresh_engine --nocapture`.
    #[test]
    #[ignore = "downloads uv and Python packages; run by hand"]
    fn a_fresh_engine_sets_up_from_nothing() {
        let home = PathBuf::from(std::env::var("TRAINNR_TEST_ENGINE_HOME").expect("a home"));
        let started = std::time::Instant::now();
        let say = |step: &str| eprintln!("{:>6.1}s  {step}", started.elapsed().as_secs_f32());
        let runtime = install(&home, &say, &Transfer::default()).expect("sets up");
        say("ready");
        assert!(runtime.join(READY_FILE).is_file());
        assert!(runtime.join(crate::spawn::REPO_MARKER).is_file());
    }

    #[test]
    fn every_platform_the_release_builds_has_a_pinned_uv() {
        for platform in ["x86_64-linux", "aarch64-macos", "x86_64-windows"] {
            let pin = UV_PINS.iter().find(|(p, _)| *p == platform);
            let (_, pin) = pin.unwrap_or_else(|| panic!("no uv pinned for {platform}"));
            assert_eq!(pin.sha256.len(), 64, "{platform}");
        }
    }

    #[test]
    fn a_transfer_reports_its_share_and_its_amount() {
        let t = Transfer::default();
        assert_eq!((t.fraction(), t.amount()), (None, None), "nothing yet");
        t.begin(48_000_000);
        t.add(12_000_000);
        assert_eq!(t.fraction(), Some(0.25));
        assert_eq!(t.amount().as_deref(), Some("12 of 48 MB"));
        t.begin(0);
        t.add(3_000_000);
        assert_eq!(t.fraction(), None, "no size: no share");
        assert_eq!(t.amount().as_deref(), Some("3 MB"));
        assert!(!t.is_cancelled());
        t.cancel();
        assert!(t.is_cancelled());
    }

    /// Cancel stops a long step at once, rather than at its end.
    #[cfg(unix)]
    #[test]
    fn a_cancelled_step_stops_its_process() {
        let t = Transfer::default();
        t.cancel();
        let started = std::time::Instant::now();
        let mut sleep = std::process::Command::new("sleep");
        sleep.arg("30");
        let outcome = output_unless_cancelled(&mut sleep, &t);
        assert_eq!(outcome.err().as_deref(), Some(CANCELLED));
        assert!(started.elapsed() < Duration::from_secs(5));
        let mut quick = std::process::Command::new("true");
        let done = output_unless_cancelled(&mut quick, &Transfer::default()).expect("runs");
        assert!(done.status.success());
    }
}
