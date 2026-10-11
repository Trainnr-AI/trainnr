//! Opening a sample with no engine: the Studio downloads it, checks it,
//! unpacks it into the projects home and opens it, all by itself (the
//! desktop app plan, docs/85, 2026-10-11). A new user goes from the
//! download to a finished project on screen with nothing else fetched.
//!
//! The archive is the one `trainnr sample open` fetches (the Hugging Face
//! dataset at a pinned commit, `trainnr/samples.py`), and it carries its
//! index (`tools/make-sample.py`), so no Python has to run: only the
//! index's `root`, which the archive leaves empty, is written in here.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};

use crate::onboarding::SampleCard;

/// The index file inside a project (`trainnr/project/locate.py`).
const INDEX_FILE: &str = ".index/project.json";

/// Open `sample` in `home` on its own thread; the answer arrives on the
/// channel: the project's folder, or why it did not open.
pub fn open_in_background(
    sample: &'static SampleCard,
    home: PathBuf,
) -> Receiver<Result<PathBuf, String>> {
    let (send, receive) = channel();
    std::thread::Builder::new()
        .name("trainnr-sample".to_owned())
        .spawn(move || {
            let _ = send.send(open(sample, &home));
        })
        .ok();
    receive
}

/// The sample in `home`: downloaded, checked and unpacked the first time,
/// the folder that is there after that; made the current project, so an
/// agent session opens it too.
pub fn open(sample: &SampleCard, home: &Path) -> Result<PathBuf, String> {
    let target = home.join(sample.project);
    if !target.join(crate::model::MANIFEST_FILE).is_file() {
        if target.exists() {
            return Err(format!(
                "{} exists and is not the sample; move it to open the sample",
                target.display()
            ));
        }
        let bytes = crate::engine::download(sample.url)?;
        unpack(sample, &bytes, home)?;
    }
    remember(home, &target)?;
    Ok(target)
}

/// Check the archive (its size, then its SHA-256) and unpack it as the
/// sample's folder in `home`, with the index's `root` written in.
fn unpack(sample: &SampleCard, bytes: &[u8], home: &Path) -> Result<(), String> {
    if bytes.len() as u64 != sample.bytes {
        return Err(format!(
            "{}: {} bytes, listed {}",
            sample.url,
            bytes.len(),
            sample.bytes
        ));
    }
    crate::engine::check_sha256(bytes, sample.sha256, sample.url)?;
    let staging = home.join(format!(".unpacking-{}-{}", sample.name, std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);
    crate::engine::unpack_tar_gz(bytes, &staging)?;
    let inside = staging.join(sample.project);
    let target = home.join(sample.project);
    let moved = if inside.join(crate::model::MANIFEST_FILE).is_file() {
        std::fs::rename(&inside, &target).map_err(|e| e.to_string())
    } else {
        Err(format!(
            "{}: no {}/project.json inside",
            sample.url, sample.project
        ))
    };
    let _ = std::fs::remove_dir_all(&staging);
    moved?;
    write_root(&target)
}

/// The index's `root`: the folder's own path, as `write_index` writes it.
fn write_root(project: &Path) -> Result<(), String> {
    let path = project.join(INDEX_FILE);
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut index: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    index["root"] = serde_json::Value::from(project.display().to_string());
    let body = serde_json::to_string_pretty(&index).map_err(|e| e.to_string())?;
    crate::control::write_atomic(&path, &body).map_err(|e| e.to_string())
}

/// The projects home's `.current`: the project the tools act on next.
fn remember(home: &Path, project: &Path) -> Result<(), String> {
    let current = home.join(crate::model::CURRENT_FILE);
    crate::control::refuse_links(&current)?;
    std::fs::write(&current, format!("{}\n", project.display())).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Digest as _;

    fn archive(project: &str) -> Vec<u8> {
        let mut builder = tar::Builder::new(Vec::new());
        let files: [(String, &[u8]); 2] = [
            (
                format!("{project}/project.json"),
                br#"{"schema": "trainnr-project/1"}"#,
            ),
            (
                format!("{project}/{INDEX_FILE}"),
                br#"{"root": "", "artifacts": []}"#,
            ),
        ];
        for (name, body) in files {
            let mut header = tar::Header::new_gnu();
            header.set_size(body.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_data(&mut header, name, body)
                .expect("append");
        }
        let raw = builder.into_inner().expect("tar");
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        std::io::Write::write_all(&mut gz, &raw).expect("gz");
        gz.finish().expect("gz")
    }

    fn card(bytes: &[u8]) -> SampleCard {
        let digest: String = sha2::Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        SampleCard {
            name: "demo",
            title: "Demo",
            summary: "",
            project: "demo-sample",
            bytes: bytes.len() as u64,
            sha256: Box::leak(digest.into_boxed_str()),
            url: "https://example.invalid/demo.tar.gz",
        }
    }

    #[test]
    fn a_checked_archive_unpacks_with_its_root_written_in() {
        let home = std::env::temp_dir().join(format!("studio-sample-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        std::fs::create_dir_all(&home).unwrap();
        let bytes = archive("demo-sample");
        let sample = card(&bytes);
        unpack(&sample, &bytes, &home).expect("unpacks");
        let project = home.join("demo-sample");
        let index: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(project.join(INDEX_FILE)).unwrap())
                .unwrap();
        assert_eq!(index["root"], project.display().to_string());
        remember(&home, &project).expect("remembers");
        assert!(std::fs::read_to_string(home.join(".current"))
            .unwrap()
            .contains("demo-sample"));
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn a_wrong_size_or_digest_unpacks_nothing() {
        let home = std::env::temp_dir().join(format!("studio-sample-bad-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        std::fs::create_dir_all(&home).unwrap();
        let bytes = archive("demo-sample");
        let mut sample = card(&bytes);
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(unpack(&sample, &longer, &home).is_err(), "size");
        sample.sha256 = "00";
        assert!(unpack(&sample, &bytes, &home).is_err(), "digest");
        assert!(!home.join("demo-sample").exists());
        let _ = std::fs::remove_dir_all(&home);
    }
}
