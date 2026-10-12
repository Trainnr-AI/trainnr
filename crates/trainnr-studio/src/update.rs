//! Whether a newer trainnr is out, when the person asks: the version in
//! the status bar is a button that looks at this repository's latest
//! release (the desktop app plan, docs/85, phase 4, 2026-10-11). Nothing
//! looks on its own: the repository's rule is that no network call runs
//! without the user asking (AGENTS.md), and the first version of this
//! looked once a day by itself (review, 2026-10-11). Nothing is
//! downloaded or installed here; a newer release links its page.

use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The newest release, as GitHub's API names it (pre-releases excluded).
const LATEST_API: &str = "https://api.github.com/repos/Trainnr-AI/trainnr/releases/latest";
/// Where a person downloads it.
pub const RELEASES_PAGE: &str = "https://github.com/Trainnr-AI/trainnr/releases/latest";
/// This build's version.
pub const THIS_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Where a look stands.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum State {
    #[default]
    Idle,
    Looking,
    UpToDate,
    Newer(String),
    Failed(String),
}

/// The look, on its own thread once asked for.
#[derive(Default)]
pub struct Check {
    state: Arc<Mutex<State>>,
}

impl Check {
    pub fn state(&self) -> State {
        self.state.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// Look now (the person clicked); a look under way is left to finish.
    pub fn look(&self) {
        {
            let Ok(mut state) = self.state.lock() else {
                return;
            };
            if *state == State::Looking {
                return;
            }
            *state = State::Looking;
        }
        let state = Arc::clone(&self.state);
        std::thread::Builder::new()
            .name("trainnr-update-check".to_owned())
            .spawn(move || {
                let answer = match latest_release() {
                    Ok(latest) if is_newer(&latest, THIS_VERSION) => State::Newer(latest),
                    Ok(_) => State::UpToDate,
                    Err(why) => State::Failed(why),
                };
                if let Ok(mut s) = state.lock() {
                    *s = answer;
                }
            })
            .ok();
    }
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
    fn nothing_looks_until_asked() {
        assert_eq!(Check::default().state(), State::Idle);
    }
}
