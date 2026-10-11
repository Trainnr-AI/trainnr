//! Whether a newer trainnr is out: at most one look a day at this
//! repository's latest release, in the background, and a word in the
//! status bar when there is one (the desktop app plan, docs/85, phase 4,
//! 2026-10-11). `TRAINNR_NO_UPDATE_CHECK=1` turns it off; nothing is
//! downloaded or installed by it.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Turns the check off.
pub const OFF_ENV: &str = "TRAINNR_NO_UPDATE_CHECK";
/// The newest release, as GitHub's API names it (pre-releases excluded).
const LATEST_API: &str = "https://api.github.com/repos/Trainnr-AI/trainnr/releases/latest";
/// Where a person downloads it.
pub const RELEASES_PAGE: &str = "https://github.com/Trainnr-AI/trainnr/releases/latest";
/// The last look and its answer, under the user's trainnr home.
const MEMO_FILE: &str = "update-check.json";
/// At most one look a day.
const EVERY: Duration = Duration::from_secs(24 * 3600);

/// A newer version than this build, when one is out.
#[derive(Default)]
pub struct Check {
    newer: Arc<Mutex<Option<String>>>,
}

impl Check {
    /// Look in the background, unless it is off or was done today.
    pub fn start() -> Self {
        let check = Self::default();
        let off = std::env::var(OFF_ENV).is_ok_and(|v| !v.trim().is_empty() && v != "0");
        let Some(memo) = crate::model::user_home().map(|h| h.join(MEMO_FILE)) else {
            return check;
        };
        if off {
            return check;
        }
        let newer = Arc::clone(&check.newer);
        std::thread::Builder::new()
            .name("trainnr-update-check".to_owned())
            .spawn(move || {
                let latest = remembered(&memo).or_else(|| {
                    let latest = latest_release().ok()?;
                    remember(&memo, &latest);
                    Some(latest)
                });
                if let Some(latest) = latest {
                    if is_newer(&latest, env!("CARGO_PKG_VERSION")) {
                        if let Ok(mut slot) = newer.lock() {
                            *slot = Some(latest);
                        }
                    }
                }
            })
            .ok();
        check
    }

    /// The newer version, once the look has found one.
    pub fn newer(&self) -> Option<String> {
        self.newer.lock().ok().and_then(|s| s.clone())
    }
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or_default()
}

/// The answer of a look made within the day, if there was one.
fn remembered(memo: &Path) -> Option<String> {
    let text = std::fs::read_to_string(memo).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    let checked = value["checked"].as_f64()?;
    (now() - checked < EVERY.as_secs_f64())
        .then(|| value["latest"].as_str().map(str::to_owned))
        .flatten()
}

fn remember(memo: &Path, latest: &str) {
    if crate::control::refuse_links(memo).is_err() {
        return;
    }
    if let Some(dir) = memo.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let body = serde_json::json!({ "checked": now(), "latest": latest }).to_string();
    let _ = crate::control::write_atomic(memo, &body);
}

/// The latest release's version, without its `v`.
fn latest_release() -> Result<String, String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .https_only(true)
        .timeout_global(Some(Duration::from_secs(10)))
        .user_agent(concat!("trainnr-studio/", env!("CARGO_PKG_VERSION")))
        .build()
        .into();
    let mut reply = agent
        .get(LATEST_API)
        .header("Accept", "application/vnd.github+json")
        .call()
        .map_err(|e| e.to_string())?;
    let text = reply
        .body_mut()
        .with_config()
        .limit(1_000_000)
        .read_to_string()
        .map_err(|e| e.to_string())?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    value["tag_name"]
        .as_str()
        .map(|t| t.trim_start_matches('v').to_owned())
        .ok_or_else(|| "no tag_name".to_owned())
}

/// Whether `candidate` is a later release than `current`: their numbers
/// compared part by part; a pre-release (`-rc.1`) is before its release.
pub fn is_newer(candidate: &str, current: &str) -> bool {
    fn parts(v: &str) -> (Vec<u64>, bool) {
        let (numbers, pre) = v.split_once('-').map_or((v, false), |(n, _)| (n, true));
        let numbers = numbers.split('.').map(|p| p.parse().unwrap_or(0)).collect();
        (numbers, pre)
    }
    let (new, new_pre) = parts(candidate.trim_start_matches('v'));
    let (old, old_pre) = parts(current.trim_start_matches('v'));
    match new.cmp(&old) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => old_pre && !new_pre,
    }
}

/// The memo's path, for the tests.
#[cfg(test)]
fn memo_in(dir: &Path) -> std::path::PathBuf {
    dir.join(MEMO_FILE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_compare_part_by_part_and_a_candidate_precedes_its_release() {
        assert!(is_newer("0.2.0", "0.1.0"));
        assert!(is_newer("v0.10.0", "0.9.3"));
        assert!(is_newer("1.0.0", "0.99.99"));
        assert!(is_newer("0.1.0", "0.1.0-rc.2"));
        assert!(!is_newer("0.1.0", "0.1.0"));
        assert!(!is_newer("0.1.0-rc.3", "0.1.0"));
        assert!(!is_newer("0.0.9", "0.1.0"));
    }

    #[test]
    fn a_look_is_remembered_for_a_day() {
        let dir = std::env::temp_dir().join(format!("studio-update-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let memo = memo_in(&dir);
        assert_eq!(remembered(&memo), None);
        remember(&memo, "0.2.0");
        assert_eq!(remembered(&memo).as_deref(), Some("0.2.0"));
        let stale = serde_json::json!({ "checked": now() - 2.0 * 86400.0, "latest": "0.2.0" });
        std::fs::write(&memo, stale.to_string()).unwrap();
        assert_eq!(remembered(&memo), None, "a day old: look again");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
