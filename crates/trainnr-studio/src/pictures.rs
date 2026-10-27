//! Card pictures, decoded and shrunk off the UI thread.
//!
//! egui's image loader read a 674x500 PNG, decoded it and uploaded the
//! full texture on the UI thread, per card, the first time a page showed
//! it: fifty-three cards cost a 2.5 s frame on WSLg's graphics layer
//! (2026-10-03, "when I click on a card it takes time to reflect"). Here a
//! thread reads, decodes and resamples each picture to the pixel size its
//! card draws, the UI thread uploads that small texture when it arrives, and
//! the page paints at once with the pictures filling in.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};

/// A picture at one size: ready as a texture, or failed and never retried
/// for that version and size.
enum Picture {
    Ready(egui::TextureHandle),
    Failed,
}

type Decoded = (String, [u32; 2], Option<egui::ColorImage>);

/// Per key (the preview's versioned URI), its pictures by size in pixels,
/// each with when it was last drawn; the keys decoding; the channel the
/// decoded pictures arrive on.
pub struct Pictures {
    ready: HashMap<String, Vec<([u32; 2], Picture, f64)>>,
    pending: HashSet<String>,
    tx: mpsc::Sender<Decoded>,
    rx: mpsc::Receiver<Decoded>,
}

impl Default for Pictures {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            ready: HashMap::new(),
            pending: HashSet::new(),
            tx,
            rx,
        }
    }
}

/// The one cache of the window, kept in egui's temp data.
pub fn shared(ctx: &egui::Context) -> Arc<Mutex<Pictures>> {
    let id = egui::Id::new("card-pictures");
    ctx.data_mut(|d| {
        d.get_temp_mut_or_insert_with(id, || Arc::new(Mutex::new(Pictures::default())))
            .clone()
    })
}

impl Pictures {
    /// The texture for `key` at `size` pixels, the size it is drawn at, so
    /// it is shown one texel to one pixel. Until that size arrives, or if
    /// it failed, the key's texture at another size stands in (a resized
    /// window never flashes empty), or None. One decode per key runs at a
    /// time; a size not drawn for a second (left by a resize) is dropped.
    pub fn get(
        &mut self,
        ctx: &egui::Context,
        key: &str,
        path: &std::path::Path,
        size: [u32; 2],
    ) -> Option<egui::TextureHandle> {
        let now = ctx.input(|i| i.time);
        self.receive(ctx, now);
        let mut stand_in = None;
        if let Some(sizes) = self.ready.get_mut(key) {
            sizes.retain(|(s, _, used)| *s == size || now - *used < 1.0);
            let mut failed = false;
            if let Some((_, picture, used)) = sizes.iter_mut().find(|(s, _, _)| *s == size) {
                *used = now;
                match picture {
                    Picture::Ready(texture) => return Some(texture.clone()),
                    Picture::Failed => failed = true,
                }
            }
            if let Some((_, Picture::Ready(texture), used)) = sizes
                .iter_mut()
                .find(|(_, p, _)| matches!(p, Picture::Ready(_)))
            {
                *used = now;
                stand_in = Some(texture.clone());
            }
            if failed {
                return stand_in;
            }
        }
        if self.pending.insert(key.to_owned()) {
            let tx = self.tx.clone();
            let ctx = ctx.clone();
            let owned = key.to_owned();
            let path: PathBuf = path.to_owned();
            let spawned = std::thread::Builder::new()
                .name("trainnr-picture".to_owned())
                .spawn(move || {
                    let image = decode_to_fill(&path, size);
                    let _ = tx.send((owned, size, image));
                    ctx.request_repaint();
                });
            if spawned.is_err() {
                self.pending.remove(key);
            }
        }
        stand_in
    }

    /// Drop a key whose file changed; the next `get` decodes the new one,
    /// and a decode of the old one still running is dropped when it lands.
    pub fn forget(&mut self, key: &str) {
        self.ready.remove(key);
        self.pending.remove(key);
    }

    fn receive(&mut self, ctx: &egui::Context, now: f64) {
        while let Ok((key, size, image)) = self.rx.try_recv() {
            if !self.pending.remove(&key) {
                continue; // forgotten while it decoded
            }
            let picture = match image {
                Some(image) => {
                    Picture::Ready(ctx.load_texture(&key, image, egui::TextureOptions::LINEAR))
                }
                None => Picture::Failed,
            };
            let sizes = self.ready.entry(key).or_default();
            sizes.retain(|(s, _, _)| *s != size);
            sizes.push((size, picture, now));
        }
    }
}

/// Read and decode a PNG and resample it to exactly `size`, cropped to
/// fill it (the card's shape), with a sharp filter: a fast shrink to fit
/// and a stretch on the GPU blurred the small text in the pictures. None
/// when the file is unreadable or not an image.
fn decode_to_fill(path: &std::path::Path, size: [u32; 2]) -> Option<egui::ColorImage> {
    let decoded = image::load_from_memory(&std::fs::read(path).ok()?).ok()?;
    let rgba = decoded
        .resize_to_fill(
            size[0].max(1),
            size[1].max(1),
            image::imageops::FilterType::CatmullRom,
        )
        .to_rgba8();
    Some(egui::ColorImage::from_rgba_unmultiplied(
        [rgba.width() as usize, rgba.height() as usize],
        rgba.as_raw(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One frame at `time` seconds, calling `get` for `key` at `size`.
    /// The key is marked as decoding first, so `get` starts no thread and
    /// only what the test sends arrives.
    fn frame(
        ctx: &egui::Context,
        pictures: &mut Pictures,
        time: f64,
        size: [u32; 2],
    ) -> Option<[usize; 2]> {
        let mut got = None;
        let raw = egui::RawInput {
            time: Some(time),
            ..Default::default()
        };
        let mut output = ctx.run_ui(raw, |ui| {
            pictures.pending.insert("k".to_owned());
            got = pictures
                .get(ui.ctx(), "k", std::path::Path::new("none"), size)
                .map(|t| t.size());
        });
        output.textures_delta.clear(); // no renderer here to apply them
        got
    }

    fn arrive(pictures: &Pictures, size: [u32; 2], ok: bool) {
        let image = ok.then(|| {
            egui::ColorImage::filled([size[0] as usize, size[1] as usize], egui::Color32::GRAY)
        });
        pictures.tx.send(("k".to_owned(), size, image)).unwrap();
    }

    #[test]
    fn a_failed_size_shows_another_and_a_size_left_unused_goes() {
        let ctx = egui::Context::default();
        let mut pictures = Pictures::default();
        arrive(&pictures, [10, 10], true);
        assert_eq!(frame(&ctx, &mut pictures, 0.0, [10, 10]), Some([10, 10]));
        arrive(&pictures, [20, 20], false);
        assert_eq!(
            frame(&ctx, &mut pictures, 0.1, [20, 20]),
            Some([10, 10]),
            "the failed size falls back to the good one"
        );
        arrive(&pictures, [30, 30], true);
        assert_eq!(frame(&ctx, &mut pictures, 0.2, [30, 30]), Some([30, 30]));
        assert_eq!(frame(&ctx, &mut pictures, 2.0, [30, 30]), Some([30, 30]));
        let sizes: Vec<_> = pictures.ready["k"].iter().map(|(s, _, _)| *s).collect();
        assert_eq!(
            sizes,
            vec![[30, 30]],
            "the sizes not drawn for a second went"
        );
    }

    #[test]
    fn a_decode_that_lands_after_forget_is_dropped() {
        let ctx = egui::Context::default();
        let mut pictures = Pictures::default();
        pictures.pending.insert("k".to_owned());
        pictures.forget("k");
        arrive(&pictures, [10, 10], true);
        pictures.receive(&ctx, 0.0);
        assert!(!pictures.ready.contains_key("k"));
    }

    #[test]
    fn a_picture_is_decoded_to_the_exact_size_it_is_drawn_at() {
        let path = std::env::temp_dir().join(format!("studio-picture-{}.png", std::process::id()));
        image::RgbaImage::new(674, 500).save(&path).unwrap();
        for size in [[300, 223], [337, 140], [900, 900], [1, 1]] {
            let image = decode_to_fill(&path, size).expect("decodes");
            assert_eq!(image.size, [size[0] as usize, size[1] as usize]);
        }
        assert!(decode_to_fill(&path.with_extension("missing"), [10, 10]).is_none());
        let _ = std::fs::remove_file(&path);
    }
}
