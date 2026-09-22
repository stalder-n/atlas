// SPDX-License-Identifier: AGPL-3.0-only

//! `build_image_block` and the placeholder-expansion loop of `prepare_vl_inputs`.
//!
//! ★ SSOT: this module is the ONLY source of the DeepSeek image-token layout. Server-side
//! prompt expansion, the model-side embedding splice and the attention visibility bounds all
//! read it; none of them recomputes the arithmetic.

use anyhow::{Result, bail};

use super::geometry::COMPRESS_PAD_TO;

pub const IMAGE_START: u8 = 0;
pub const IMAGE_PAD: u8 = 1;
pub const IMAGE: u8 = 2;
pub const IMAGE_NEWLINE: u8 = 3;
pub const IMAGE_END: u8 = 4;
/// `range(5)` in the reference.
pub const NUM_SENTINEL_TYPES: u32 = 5;

/// `vocab_size + ty` — the id that travels in the token stream. Never samplable: `lm_head`
/// is `[vocab_size]`.
pub fn sentinel_id(vocab_size: u32, ty: u8) -> u32 {
    debug_assert!(
        (ty as u32) < NUM_SENTINEL_TYPES,
        "unknown sentinel type {ty}"
    );
    vocab_size.saturating_add(ty as u32)
}

/// `Some(ty)` iff `vocab_size <= tok < vocab_size + 5`.
pub fn sentinel_type(vocab_size: u32, tok: u32) -> Option<u8> {
    let off = tok.checked_sub(vocab_size)?;
    (off < NUM_SENTINEL_TYPES).then_some(off as u8)
}

pub fn is_sentinel(vocab_size: u32, tok: u32) -> bool {
    sentinel_type(vocab_size, tok).is_some()
}

/// One expanded image block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsImageBlock {
    /// The final token-TYPE sequence, including the leading compress pad, START, the
    /// row-pair-interleaved body, the alignment pad and END.
    pub types: Vec<u8>,
    /// `perm[j]` = the aligner row that fills the j-th `IMAGE` slot of `types`.
    pub perm: Vec<u32>,
}

/// The shape numbers every helper here shares, derived exactly as the reference does.
struct BlockShape {
    /// `COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO` — the only place `start_pos`
    /// appears, hence "only the residue matters".
    compress_pad: usize,
    /// `n_llm_h % 2` — an odd grid is padded up to an even row count.
    pad_h: usize,
    /// `n_llm_h + pad_h`, always even.
    rows: usize,
    /// `n_llm_w + 1` — the grid columns plus the row's trailing NEWLINE slot.
    row_len: usize,
    /// `rows // 2 * row_len % 2 * 2`, left-associative exactly as in Python.
    pad_last: usize,
}

impl BlockShape {
    fn new(n_llm_h: usize, n_llm_w: usize, start_pos: usize) -> Self {
        let pad_h = n_llm_h % 2;
        let rows = n_llm_h + pad_h;
        let row_len = n_llm_w + 1;
        Self {
            compress_pad: COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO,
            pad_h,
            rows,
            row_len,
            pad_last: (rows / 2 * row_len) % 2 * 2,
        }
    }

    /// The reference's row-pair interleave: the flat index range `0 .. rows * row_len`,
    /// viewed as `(rows // 2, 2, row_len)`, transposed on axes `(1, 2)` and flattened.
    ///
    /// After the transpose the axes are `(pair, column, row-within-pair)`, and flattening
    /// walks them in that order, so element `(p, c, r)` carries the ORIGINAL flat index
    /// `p * 2 * row_len + r * row_len + c`. Emitting that triple loop is the same tensor
    /// without torch — and it is what makes consecutive IMAGE slots VERTICAL neighbours,
    /// which is what lets ratio-4 KV compression pool 2×2 spatial blocks.
    fn order(&self) -> Vec<usize> {
        let mut order = Vec::with_capacity(self.rows * self.row_len);
        for p in 0..self.rows / 2 {
            for c in 0..self.row_len {
                for r in 0..2 {
                    order.push(p * 2 * self.row_len + r * self.row_len + c);
                }
            }
        }
        order
    }
}

/// `image_idx[order]` filtered to `>= 0` — the aligner row for each IMAGE slot, in the
/// order the slots appear after the row-pair interleave.
fn permutation(n_llm_h: usize, n_llm_w: usize, shape: &BlockShape) -> Vec<u32> {
    // `image_idx` is -1 everywhere except the top-left n_llm_h × n_llm_w sub-block of the
    // (rows, row_len) grid, which is filled row-major with 0..n_llm_h*n_llm_w.
    let mut image_idx = vec![-1i64; shape.rows * shape.row_len];
    for r in 0..n_llm_h {
        for c in 0..n_llm_w {
            image_idx[r * shape.row_len + c] = (r * n_llm_w + c) as i64;
        }
    }
    shape
        .order()
        .into_iter()
        .map(|o| image_idx[o])
        .filter(|&v| v >= 0)
        .map(|v| v as u32)
        .collect()
}

/// Exact port of the reference's `build_image_block`.
pub fn build_image_block(n_llm_h: usize, n_llm_w: usize, start_pos: usize) -> DsImageBlock {
    let shape = BlockShape::new(n_llm_h, n_llm_w, start_pos);

    // The PRE-permutation type row pattern: `([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) *
    // n_llm_h + [IMAGE_PAD] * (row_len * pad_h)`.
    let mut base = Vec::with_capacity(shape.rows * shape.row_len);
    for _ in 0..n_llm_h {
        base.extend(std::iter::repeat_n(IMAGE, n_llm_w));
        base.push(IMAGE_NEWLINE);
    }
    base.extend(std::iter::repeat_n(IMAGE_PAD, shape.row_len * shape.pad_h));

    let order = shape.order();
    let perm = permutation(n_llm_h, n_llm_w, &shape);

    let mut types = Vec::with_capacity(block_len(n_llm_h, n_llm_w, start_pos));
    types.extend(std::iter::repeat_n(IMAGE_PAD, shape.compress_pad));
    types.push(IMAGE_START);
    types.extend(order.into_iter().map(|o| base[o]));
    types.extend(std::iter::repeat_n(IMAGE_PAD, shape.pad_last));
    types.push(IMAGE_END);

    DsImageBlock { types, perm }
}

/// `build_image_block(..).types.len()`, in closed form.
///
/// Equals `geometry::grid_tokens(..).2 + compress_pad`: `grid_tokens` counts
/// `rows * row_len + 2 + pad_last`, and the block adds only the leading compress pad.
/// `block_tests.rs` pins that equality against the reference for all 576 cases.
pub fn block_len(n_llm_h: usize, n_llm_w: usize, start_pos: usize) -> usize {
    let s = BlockShape::new(n_llm_h, n_llm_w, start_pos);
    s.compress_pad + 1 + s.rows * s.row_len + s.pad_last + 1
}

/// The permutation alone — it does not depend on `start_pos`.
pub fn perm_only(n_llm_h: usize, n_llm_w: usize) -> Vec<u32> {
    permutation(n_llm_h, n_llm_w, &BlockShape::new(n_llm_h, n_llm_w, 0))
}

/// Sequential expansion: every `placeholder` occurrence becomes the sentinel ids of one
/// block, built with `start_pos` = the number of tokens already emitted.
///
/// The reference's `prepare_vl_inputs` uses `len(tokens)` — the length of the OUTPUT built
/// so far, not the input index — so an image's compress pad depends on everything expanded
/// before it. Getting that wrong misaligns every later block by up to three tokens.
pub fn expand_placeholders(
    tokens: &[u32],
    placeholder: u32,
    vocab_size: u32,
    grids: &[(usize, usize)],
) -> Result<(Vec<u32>, Vec<DsImageBlock>)> {
    let num_placeholders = tokens.iter().filter(|&&t| t == placeholder).count();
    if num_placeholders != grids.len() {
        bail!(
            "found {num_placeholders} image placeholder tokens but got {} images",
            grids.len()
        );
    }
    // Every sentinel must stay representable; `sentinel_id` saturates rather than wrapping,
    // so an overflow here would silently collapse distinct types onto `u32::MAX`.
    if vocab_size.checked_add(NUM_SENTINEL_TYPES - 1).is_none() {
        bail!("vocab_size {vocab_size} + {NUM_SENTINEL_TYPES} sentinel ids overflows u32");
    }

    let mut out: Vec<u32> = Vec::with_capacity(tokens.len());
    let mut blocks = Vec::with_capacity(grids.len());
    for &tok in tokens {
        if tok != placeholder {
            out.push(tok);
            continue;
        }
        // `blocks.len()` is the number of placeholders already consumed, and the count was
        // checked above — indexing by it keeps the loop total without an `expect`.
        let Some(&(n_llm_h, n_llm_w)) = grids.get(blocks.len()) else {
            bail!("ran out of image grids at placeholder {}", blocks.len());
        };
        let block = build_image_block(n_llm_h, n_llm_w, out.len());
        out.extend(block.types.iter().map(|&ty| sentinel_id(vocab_size, ty)));
        blocks.push(block);
    }
    Ok((out, blocks))
}

#[cfg(test)]
#[path = "block_tests.rs"]
mod tests;
