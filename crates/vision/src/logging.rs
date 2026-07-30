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

/// Catch the specific mistake that wasted an hour: asking for ort logs
/// through `RUST_LOG` while leaving the C++ gate shut.
fn warn_on_half_configured_logging() {
    let rust_log = std::env::var("RUST_LOG").unwrap_or_default();
    let wants_ort = rust_log.contains("ort") || rust_log.contains("trace");
    let ort_log_set = std::env::var("ORT_LOG").is_ok();

    if wants_ort && !ort_log_set {
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
