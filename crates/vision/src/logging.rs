//! Logging setup — and the two-gate trap it exists to prevent.
//!
//! # Why this module exists
//!
//! Debugging why CoreML gave no speedup cost an hour, most of it spent
//! staring at *no output at all*. The cause was three requirements that
//! must ALL hold, where missing any one produces identical silence:
//!
//! | # | Requirement | If missing |
//! |---|---|---|
//! | 1 | a `tracing` subscriber installed in `main` | events go nowhere |
//! | 2 | `ORT_LOG` — the ONNX Runtime **C++** log level | C++ discards them first |
//! | 3 | `RUST_LOG` — the Rust subscriber's filter | subscriber drops them |
//!
//! ONNX Runtime is a C++ library. Its diagnostics originate in C++, are
//! filtered *there*, and only survivors cross into Rust via a callback
//! that re-emits them as `tracing` events. `ORT_LOG` defaults to `error`,
//! so setting only `RUST_LOG=ort=debug` filters an already-empty stream —
//! which looks exactly like "nothing is wrong".
//!
//! [`init`] handles #1 and warns loudly about the #2/#3 mismatch.
//!
//! # Seeing execution-provider diagnostics
//!
//! ```sh
//! tools/debug-inference.sh cargo run --release -p vision --bin bench
//! ```
//!
//! or by hand:
//!
//! ```sh
//! ORT_LOG=verbose RUST_LOG=ort=trace cargo run --release -p vision --bin bench
//! ```
//!
//! Look for `VerifyEachNodeIsAssignedToAnEp` — it reports exactly which
//! provider took which nodes, and is the ground truth for "did my
//! accelerator actually do anything".

use std::sync::Once;

static INIT: Once = Once::new();

/// Install the tracing subscriber. Safe to call more than once.
///
/// Defaults to `warn` when `RUST_LOG` is unset, so normal runs stay quiet
/// but real problems still surface.
pub fn init() {
    INIT.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
        tracing_subscriber::fmt().with_env_filter(filter).init();

        warn_on_half_configured_logging();
    });
}

/// Is the caller asking for ORT logs through `RUST_LOG` while leaving the
/// C++ gate (`ORT_LOG`) shut?
///
/// The exact mistake that wasted an hour, expressed as a pure function so
/// it can be tested without touching process environment — env vars are
/// global mutable state, and tests that set them race each other.
pub fn is_half_configured(rust_log: &str, ort_log: Option<&str>) -> bool {
    let wants_ort = rust_log.contains("ort") || rust_log.contains("trace");
    wants_ort && ort_log.is_none()
}

/// Catch the specific mistake that wasted an hour: asking for ort logs
/// through `RUST_LOG` while leaving the C++ gate shut.
fn warn_on_half_configured_logging() {
    let rust_log = std::env::var("RUST_LOG").unwrap_or_default();
    let ort_log = std::env::var("ORT_LOG").ok();

    if is_half_configured(&rust_log, ort_log.as_deref()) {
        eprintln!(
            "\n\
             ── logging note ──────────────────────────────────────────────\n\
             RUST_LOG asks for ort logs, but ORT_LOG is unset (default:\n\
             `error`). ONNX Runtime filters in C++ BEFORE anything reaches\n\
             Rust, so you will see nothing from the execution providers.\n\
             \n\
             Add ORT_LOG too:   ORT_LOG=verbose RUST_LOG=ort=trace ...\n\
             Or just run:       tools/debug-inference.sh <command>\n\
             ──────────────────────────────────────────────────────────────\n"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asking_for_ort_without_the_cpp_gate_is_flagged() {
        // The hour-wasting configuration.
        assert!(is_half_configured("ort=info", None));
        assert!(is_half_configured("ort=trace", None));
        assert!(is_half_configured("warn,ort=debug", None));
    }

    #[test]
    fn setting_both_gates_is_not_flagged() {
        assert!(!is_half_configured("ort=trace", Some("verbose")));
        // Even ORT_LOG=error counts as "set" — the user chose it.
        assert!(!is_half_configured("ort=trace", Some("error")));
    }

    #[test]
    fn a_bare_trace_request_is_flagged_too() {
        // `RUST_LOG=trace` implies ort logs without naming ort.
        assert!(is_half_configured("trace", None));
    }

    #[test]
    fn ordinary_filters_are_left_alone() {
        // No mention of ort or trace: nothing to warn about, and warning
        // anyway would train people to ignore the message.
        assert!(!is_half_configured("", None));
        assert!(!is_half_configured("warn", None));
        assert!(!is_half_configured("info", None));
        assert!(!is_half_configured("vision=debug", None));
    }

    #[test]
    fn init_is_idempotent() {
        // Called by every binary; a second call must not panic on the
        // already-installed global subscriber.
        init();
        init();
    }
}
