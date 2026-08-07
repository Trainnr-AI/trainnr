//! Command-line parsing for the vision binaries.
//!
//! Lives in the library, not in a `main.rs`, for one reason: argument
//! parsing is where a typo turns into "the robot loaded the wrong model
//! and nobody noticed", and it is trivially testable — but only if it does
//! not read `std::env::args()` directly.
//!
//! Hence [`Args::parse_from`], which takes an iterator. `main` passes the
//! real arguments; tests pass whatever they like.

use anyhow::Result;
use std::path::PathBuf;

use crate::detect::DetectorModel;

/// What the `chase` binary was asked to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    /// Closed-set detector to load. Ignored when [`Args::find`] is set.
    pub model: DetectorModel,
    /// Comma-separated phrases for the open-vocabulary detector.
    /// `Some` switches `chase` away from the closed-set model entirely.
    pub find: Option<String>,
    /// Explicit camera index. `None` means "look for the Brio by name",
    /// which is the reliable path on macOS.
    pub camera: Option<u32>,
    /// Write every frame's perception result here — see
    /// [`crate::session`].
    pub record: Option<PathBuf>,
    /// Drive the control loop from a recording instead of a camera. No
    /// camera is opened and no detector is loaded.
    pub replay: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            model: DetectorModel::DFineN,
            find: None,
            camera: None,
            record: None,
            replay: None,
        }
    }
}

/// A flag's value as a path, or a clear error naming the flag.
///
/// Factored out because `--record --replay x` must not silently record to
/// a file called `--replay`; the same slip cost two debugging runs on the
/// HIL host.
fn path_value<I, S>(it: &mut I, flag: &str) -> Result<PathBuf>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    let v = it
        .next()
        .ok_or_else(|| anyhow::anyhow!("{flag} needs a path"))?;
    let v = v.as_ref();
    if v.starts_with('-') {
        anyhow::bail!("{flag} needs a path, got the flag {v:?}");
    }
    Ok(PathBuf::from(v))
}

impl Args {
    /// Parse arguments (excluding the program name).
    ///
    /// Unknown flags starting with `-` are an error rather than being
    /// silently treated as a camera index: `--modle deimv2-s` must not
    /// quietly run the default model.
    pub fn parse_from<I, S>(args: I) -> Result<Args>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut parsed = Args::default();
        let mut it = args.into_iter();

        while let Some(raw) = it.next() {
            match raw.as_ref() {
                "--model" => {
                    let name = it
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("--model needs a value"))?;
                    let name = name.as_ref().to_string();
                    parsed.model = DetectorModel::from_name(&name).ok_or_else(|| {
                        anyhow::anyhow!(
                            "unknown model {name:?}. Available:\n{}",
                            DetectorModel::help()
                        )
                    })?;
                }
                "--find" => {
                    let phrase = it
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("--find needs a phrase"))?;
                    parsed.find = Some(phrase.as_ref().to_string());
                }
                "--record" => {
                    parsed.record = Some(path_value(&mut it, "--record")?);
                }
                "--replay" => {
                    parsed.replay = Some(path_value(&mut it, "--replay")?);
                }
                other if other.starts_with('-') => {
                    anyhow::bail!("unknown flag {other:?}");
                }
                other => {
                    parsed.camera = Some(
                        other
                            .parse()
                            .map_err(|_| anyhow::anyhow!("not a camera index: {other:?}"))?,
                    );
                }
            }
        }
        Ok(parsed)
    }

    /// The phrases to hand an open-vocabulary detector, split and trimmed.
    pub fn phrases(&self) -> Vec<&str> {
        match &self.find {
            Some(f) => f
                .split(',')
                .map(|p| p.trim())
                .filter(|p| !p.is_empty())
                .collect(),
            None => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_replay_take_paths() {
        let a = Args::parse_from(["--record", "run.perc"]).unwrap();
        assert_eq!(a.record.unwrap().to_str(), Some("run.perc"));
        let a = Args::parse_from(["--replay", "run.perc"]).unwrap();
        assert_eq!(a.replay.unwrap().to_str(), Some("run.perc"));
    }

    /// The HIL host briefly read `--record run.wire` as a positional UF2
    /// path. Same shape of mistake, caught here instead.
    #[test]
    fn a_flag_is_not_accepted_as_a_paths_value() {
        let err = Args::parse_from(["--record", "--replay"]).unwrap_err();
        assert!(err.to_string().contains("got the flag"), "{err}");
        assert!(Args::parse_from(["--record"]).is_err(), "missing value");
    }

    /// A camera index must still parse when flags are present — the
    /// positional argument and the flags share one loop.
    #[test]
    fn a_camera_index_survives_alongside_flags() {
        let a = Args::parse_from(["--record", "r.perc", "2"]).unwrap();
        assert_eq!(a.camera, Some(2));
        assert_eq!(a.record.unwrap().to_str(), Some("r.perc"));
    }

    #[test]
    fn no_arguments_gives_the_documented_defaults() {
        let a = Args::parse_from(Vec::<String>::new()).unwrap();
        assert_eq!(a.model, DetectorModel::DFineN);
        assert_eq!(a.find, None);
        assert_eq!(a.camera, None);
    }

    #[test]
    fn every_model_name_is_selectable_from_the_cli() {
        // Guards the registry and the CLI drifting apart: a model added to
        // DetectorModel::ALL but unreachable by flag is a silent dead end.
        for m in DetectorModel::ALL {
            let a = Args::parse_from(["--model", m.name()]).unwrap();
            assert_eq!(a.model, m, "could not select {}", m.name());
        }
    }

    #[test]
    fn an_unknown_model_is_rejected_and_lists_the_alternatives() {
        let err = Args::parse_from(["--model", "yolov8n"]).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("yolov8n"), "error omits the bad name: {msg}");
        assert!(msg.contains("d-fine-n"), "error omits the options: {msg}");
    }

    #[test]
    fn a_flag_missing_its_value_is_an_error() {
        assert!(Args::parse_from(["--model"]).is_err());
        assert!(Args::parse_from(["--find"]).is_err());
    }

    #[test]
    fn find_switches_to_open_vocabulary() {
        let a = Args::parse_from(["--find", "red mug"]).unwrap();
        assert_eq!(a.find.as_deref(), Some("red mug"));
    }

    #[test]
    fn a_bare_number_is_a_camera_index() {
        let a = Args::parse_from(["1"]).unwrap();
        assert_eq!(a.camera, Some(1));
    }

    #[test]
    fn a_mistyped_flag_is_rejected_rather_than_ignored() {
        // The failure this prevents: `--modle deimv2-s` silently running
        // D-FINE-N, and a benchmark that measures the wrong model.
        let err = Args::parse_from(["--modle", "deimv2-s"]).unwrap_err();
        assert!(format!("{err}").contains("--modle"));
    }

    #[test]
    fn a_non_numeric_bare_argument_is_rejected() {
        assert!(Args::parse_from(["camera2"]).is_err());
    }

    #[test]
    fn flags_combine() {
        let a = Args::parse_from(["--model", "deimv2-s", "2"]).unwrap();
        assert_eq!(a.model, DetectorModel::Deimv2S);
        assert_eq!(a.camera, Some(2));
    }

    #[test]
    fn later_flags_win_over_earlier_ones() {
        let a = Args::parse_from(["--model", "d-fine-s", "--model", "deimv2-n"]).unwrap();
        assert_eq!(a.model, DetectorModel::Deimv2N);
    }

    // ---- phrases ----

    #[test]
    fn phrases_are_empty_without_find() {
        assert!(Args::default().phrases().is_empty());
    }

    #[test]
    fn phrases_split_on_commas_and_trim() {
        let a = Args::parse_from(["--find", "red mug, blue cube ,pen"]).unwrap();
        assert_eq!(a.phrases(), vec!["red mug", "blue cube", "pen"]);
    }

    #[test]
    fn a_single_phrase_needs_no_comma() {
        let a = Args::parse_from(["--find", "bottle"]).unwrap();
        assert_eq!(a.phrases(), vec!["bottle"]);
    }

    #[test]
    fn empty_phrases_are_dropped_rather_than_sent_to_the_model() {
        // "cup,,pen" and a trailing comma must not produce an empty
        // prompt — Grounding DINO errors on an empty vocabulary entry.
        let a = Args::parse_from(["--find", "cup,,pen,"]).unwrap();
        assert_eq!(a.phrases(), vec!["cup", "pen"]);
    }
}
