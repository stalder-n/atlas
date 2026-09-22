// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4-Flash-Vision-Exp image pipeline — a pure-Rust, GPU-free port of the
//! checkpoint's own `inference/image_processor.py` plus the two visibility helpers in
//! `inference/model.py` (`get_image_visible`, `get_window_topk_idxs_visible`), at revision
//! `86f746b36186f0e567729a5c06a8c918caba82a9`.
//!
//! Nothing here touches CUDA, a GPU handle, the network or the filesystem (SBIO): the server
//! resolves a remote image to a `data:` URI before [`preprocess::preprocess_image_deepseek`]
//! is called.
//!
//! # Where the truth comes from
//!
//! `fixtures/*.json` is the REFERENCE'S OWN OUTPUT, produced by `gen_layout_goldens.py`
//! beside them: the generator fetches `inference/{image_processor,vision,model}.py` from
//! Hugging Face at the pinned revision (falling back to a local cache), imports them, and
//! records what they return. Each fixture header carries the revision, whether HF or the
//! cache served it, and a sha256 per reference file, so a regeneration diff names exactly
//! which input moved. No expectation in this tree is hand-derived.
//!
//! # SSOT
//!
//! [`block::build_image_block`] is the ONLY source of the image-token layout. Server-side
//! prompt expansion, the model-side embedding splice and the attention visibility bounds all
//! read it; none of them recomputes the arithmetic.

pub mod block;
pub mod geometry;
pub mod preprocess;
pub mod resample;
pub mod visibility;

#[cfg(test)]
pub(crate) mod testutil {
    //! The deterministic source image the resampling and preprocessing goldens are
    //! generated from. The formula is duplicated VERBATIM in `gen_layout_goldens.py`
    //! (`gradient_rgb`) — change one, change both, and regenerate the fixtures.
    //!
    //!   `r = (7x + 13y) % 256`, `g = (x² + 3y) % 256`, `b = (x ^ y) % 256`

    /// Row-major RGB8, `w * h * 3` bytes.
    pub(crate) fn gradient_rgb(w: usize, h: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                out.push(((7 * x + 13 * y) % 256) as u8);
                out.push(((x * x + 3 * y) % 256) as u8);
                out.push(((x ^ y) % 256) as u8);
            }
        }
        out
    }

    /// The same image encoded as a `data:image/png;base64,…` URI, for the preprocess tests.
    pub(crate) fn gradient_png_data_uri(w: usize, h: usize) -> String {
        use base64::Engine as _;
        use image::ImageEncoder as _;
        let rgb = gradient_rgb(w, h);
        let mut png = Vec::new();
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(&rgb, w as u32, h as u32, image::ExtendedColorType::Rgb8)
            .expect("png encode");
        format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&png)
        )
    }
}
