//! Card pictures, decoded and shrunk off the UI thread.
//!
//! egui's image loader read a 674x500 PNG, decoded it and uploaded the
//! full texture on the UI thread, per card, the first time a page showed
//! it: fifty-three cards cost a 2.5 s frame on WSLg's graphics layer
//! (2026-10-03, "when I click on a card it takes time to reflect"). Here a
//! thread reads, decodes and shrinks each picture to the size its card
//! draws, the UI thread uploads that small texture when it arrives, and
//! the page paints at once with the pictures filling in.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};

/// A picture by its key (the preview's versioned URI): ready as a
/// texture, or failed and never retried for that version.
enum Picture {
    Ready(egui::TextureHandle),
    Failed,
}

/// Textures by key; pictures still decoding; the channel they arrive on.
pub struct Pictures {
    ready: HashMap<String, Picture>,
    pending: HashSet<String>,
    tx: mpsc::Sender<(String, Option<egui::ColorImage>)>,
    rx: mpsc::Receiver<(String, Option<egui::ColorImage>)>,
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
    /// The texture for `key`, when it has arrived; otherwise starts the
    /// decode (once per key) and answers None. `max` is the size in
    /// pixels the card draws at: the picture is shrunk to fit it.
    pub fn get(
        &mut self,
        ctx: &egui::Context,
        key: &str,
        path: &std::path::Path,
        max: [u32; 2],
    ) -> Option<egui::TextureHandle> {
        self.receive(ctx);
        match self.ready.get(key) {
            Some(Picture::Ready(texture)) => return Some(texture.clone()),
            Some(Picture::Failed) => return None,
            None => {}
        }
        if self.pending.insert(key.to_owned()) {
            let tx = self.tx.clone();
            let ctx = ctx.clone();
            let key = key.to_owned();
            let path: PathBuf = path.to_owned();
            std::thread::Builder::new()
                .name("trainnr-picture".to_owned())
                .spawn(move || {
                    let image = decode_and_shrink(&path, max);
                    let _ = tx.send((key, image));
                    ctx.request_repaint();
                })
                .ok();
        }
        None
    }

    /// Drop a key whose file changed; the next `get` decodes the new one.
    pub fn forget(&mut self, key: &str) {
        self.ready.remove(key);
        self.pending.remove(key);
    }

    fn receive(&mut self, ctx: &egui::Context) {
        while let Ok((key, image)) = self.rx.try_recv() {
            self.pending.remove(&key);
            let picture = match image {
                Some(image) => {
                    Picture::Ready(ctx.load_texture(&key, image, egui::TextureOptions::LINEAR))
                }
                None => Picture::Failed,
            };
            self.ready.insert(key, picture);
        }
    }
}

/// Read, decode and shrink a PNG to fit `max` (never enlarged); None when
/// the file is unreadable or not an image.
fn decode_and_shrink(path: &std::path::Path, max: [u32; 2]) -> Option<egui::ColorImage> {
    let bytes = std::fs::read(path).ok()?;
    let decoded = image::load_from_memory(&bytes).ok()?;
    let (w, h) = (decoded.width(), decoded.height());
    let shrunk = if w > max[0] || h > max[1] {
        decoded.thumbnail(max[0].max(1), max[1].max(1))
    } else {
        decoded
    };
    let rgba = shrunk.to_rgba8();
    let size = [rgba.width() as usize, rgba.height() as usize];
    Some(egui::ColorImage::from_rgba_unmultiplied(
        size,
        rgba.as_raw(),
    ))
}
