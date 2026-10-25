//! Card pictures, decoded and shrunk off the UI thread.
//!
//! egui's image loader read a 674x500 PNG, decoded it and uploaded the
//! full texture on the UI thread, per card, the first time a page showed
//! it: fifty-three cards cost a 2.5 s frame on WSLg's graphics layer
//! (2026-10-03, "when I click on a card it takes time to reflect"). Here a
//! thread reads, decodes and resamples each picture to the pixel size its
//! card draws, the UI thread uploads that small texture when it arrives, and
//! the page paints at once with the pictures filling in.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};

/// A picture by its key (the preview's versioned URI) and its size in
/// pixels: ready as a texture, or failed and never retried for that version.
enum Picture {
    Ready(egui::TextureHandle),
    Failed,
}

type Decoded = (String, [u32; 2], Option<egui::ColorImage>);

/// Textures by key and size, with when each was last asked for; the size
/// decoding per key; the channel the decoded pictures arrive on.
pub struct Pictures {
    ready: HashMap<(String, [u32; 2]), (Picture, f64)>,
    pending: HashMap<String, [u32; 2]>,
    tx: mpsc::Sender<Decoded>,
    rx: mpsc::Receiver<Decoded>,
}

impl Default for Pictures {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            ready: HashMap::new(),
            pending: HashMap::new(),
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
    /// it is shown one texel to one pixel. Until that size arrives, the
    /// key's texture at another size stands in (a resized window never
    /// flashes empty), or None; one decode per key runs at a time.
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
        for ((k, s), (picture, used)) in &mut self.ready {
            if k != key {
                continue;
            }
            match picture {
                Picture::Failed => return None,
                Picture::Ready(texture) if *s == size => {
                    *used = now;
                    return Some(texture.clone());
                }
                Picture::Ready(texture) => stand_in = Some(texture.clone()),
            }
        }
        if !self.pending.contains_key(key) {
            self.pending.insert(key.to_owned(), size);
            let tx = self.tx.clone();
            let ctx = ctx.clone();
            let key = key.to_owned();
            let path: PathBuf = path.to_owned();
            std::thread::Builder::new()
                .name("trainnr-picture".to_owned())
                .spawn(move || {
                    let image = decode_to_fill(&path, size);
                    let _ = tx.send((key, size, image));
                    ctx.request_repaint();
                })
                .ok();
        }
        stand_in
    }

    /// Drop a key whose file changed; the next `get` decodes the new one.
    pub fn forget(&mut self, key: &str) {
        self.ready.retain(|(k, _), _| k != key);
        self.pending.remove(key);
    }

    /// Takes the decoded pictures in; a key's other sizes not drawn in the
    /// last second go (the old card size after a resize).
    fn receive(&mut self, ctx: &egui::Context, now: f64) {
        while let Ok((key, size, image)) = self.rx.try_recv() {
            self.pending.remove(&key);
            self.ready
                .retain(|(k, s), (_, used)| k != &key || *s == size || now - *used < 1.0);
            let picture = match image {
                Some(image) => {
                    Picture::Ready(ctx.load_texture(&key, image, egui::TextureOptions::LINEAR))
                }
                None => Picture::Failed,
            };
            self.ready.insert((key, size), (picture, now));
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
