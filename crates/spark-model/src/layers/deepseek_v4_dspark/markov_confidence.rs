// SPDX-License-Identifier: AGPL-3.0-only

//! DSpark head: `hc_head → norm → lm_head → greedy Markov chain → confidence`.
//!
//! Reference (`inference/model.py`, `DSparkBlock.forward_head`, l. 909-923):
//!
//! ```text
//! x      = hc_head(h, hc_head_fn, hc_head_scale, hc_head_base)   # [γ, dim]
//! logits = head(norm(x))                                         # [γ, vocab]
//! output_ids[0] = last_token
//! for i in range(γ):
//!     bias, embed  = markov_head(output_ids[i])
//!     logits[i]   += bias
//!     output_ids[i + 1] = sample(logits[i])
//! confidence = confidence_head(x, markov_embed)
//! ```
//!
//! Three details are load-bearing and are the reason this is NOT
//! `dflash_head/markov.rs` with different pointers:
//!
//! 1. **Row 0 IS biased**, by the ANCHOR token (`output_ids[0] == last_token`).
//!    There is no "first row is unbiased" arm and no lever for one.
//! 2. **Confidence reads the PRE-`norm` `x`**, not the normed rows DFlash
//!    feeds its head.
//! 3. **`confidence_head.proj` has NO bias** — DeepSeek's is
//!    `Linear(dim + rank, 1, bias=False)`. DFlash's has one.
//!
//! And one more, from the loader side: the chain indexes the FULL vocabulary,
//! because `markov_w1`/`markov_w2` are `[vocab_size, rank]`. `mtp_vocab_size`
//! (the reduced draft-head lever) must be IGNORED here — a narrowed head would
//! mis-index the bigram bias.

use anyhow::Result;

use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

use super::{DeepseekV4DsparkHead, bf16_at};

impl DeepseekV4DsparkHead {
    /// Collapse the highway, run the shared LM head, then the greedy Markov
    /// chain. Returns the γ drafted token ids, newest last.
    ///
    /// `hc_streams` holds the γ block rows after stage 2. `anchor` is the last
    /// VERIFIED token — the chain's `output_ids[0]`.
    pub(super) fn head_and_chain(
        &self,
        anchor: u32,
        ctx: &ForwardContext,
        stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        let h = self.hidden as u32;
        let gamma = self.block as u32;
        let vocab = self.vocab_size as u32;
        let rank = self.markov_rank as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let hc = self.stage_hc(0)?;
        let hc_streams = ctx.buffers.hc_streams();

        // ── x = hc_head(h) — the module's OWN head mixer (`mtp.2.hc_head_*`) ──
        ops::hc_head_site(
            ctx.gpu,
            self.hc_head_k,
            hc_streams,
            &self.weights.hc_head,
            hc,
            self.scratch.x,
            ctx.buffers.hc_lowrank_scratch(),
            gamma,
            h,
            eps,
            stream,
        )?;

        // ── logits = lm_head(norm(x)) ──
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_vanilla_k,
            self.scratch.x,
            &self.weights.norm,
            self.scratch.normed,
            gamma,
            h,
            eps,
            stream,
        )?;
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            self.scratch.normed,
            &self.lm_head,
            self.scratch.logits,
            gamma,
            vocab,
            h,
            stream,
        )?;

        // ── greedy Markov chain ──
        // `draft_tokens_dev[0]` is the anchor; row i reads slot i and writes
        // slot i+1, so the whole chain stays on device unless a grammar mask
        // forces the per-row host round trip.
        ctx.gpu
            .copy_h2d_async(&anchor.to_le_bytes(), self.scratch.draft_tokens_dev, stream)?;
        for i in 0..self.block {
            let prev = self.scratch.draft_tokens_dev.offset(i * 4);
            let embed_i = self.scratch.markov_embed.offset(i * self.markov_rank * 2);
            let logits_i = self.scratch.logits.offset(i * self.vocab_size * 2);

            // embed = markov_w1[prev]  (a [vocab, rank] gather)
            ops::batched_embed(
                ctx.gpu,
                self.batched_embed_k,
                prev,
                self.weights.markov_w1.weight,
                embed_i,
                1,
                rank,
                stream,
            )?;
            // bias = markov_w2 @ embed  → [vocab]
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                embed_i,
                &self.weights.markov_w2,
                self.scratch.markov_bias,
                vocab,
                rank,
                stream,
            )?;
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                logits_i,
                self.scratch.markov_bias,
                vocab,
                stream,
            )?;

            let next = self.scratch.draft_tokens_dev.offset((i + 1) * 4);
            match grammar_bitmask {
                None => ops::argmax_bf16(ctx.gpu, self.argmax_k, logits_i, next, vocab, stream)?,
                Some(mask) => {
                    // The chain's `prev` must become the MASKED pick, or the
                    // next row's bigram bias is conditioned on a token the
                    // grammar rejected.
                    let tok = crate::layers::deepseek_v4_mtp::argmax_grammar_masked(
                        ctx.gpu,
                        logits_i,
                        self.vocab_size,
                        mask,
                        i,
                    )?;
                    ctx.gpu.copy_h2d_async(&tok.to_le_bytes(), next, stream)?;
                }
            }
        }

        // ── confidence — PRE-`norm` x, two slices of a bias-free [1, h+rank] ──
        // Only when the gate is armed: it costs two GEMVs per row plus a D2H.
        if self.conf_enabled {
            for i in 0..self.block {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    self.scratch.x.offset(i * self.hidden * 2),
                    &self.weights.confidence_w,
                    self.scratch.conf_x.offset(i * 2),
                    1,
                    h,
                    stream,
                )?;
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    self.scratch.markov_embed.offset(i * self.markov_rank * 2),
                    &DenseWeight {
                        weight: self.weights.confidence_w.weight.offset(self.hidden * 2),
                    },
                    self.scratch.conf_m.offset(i * 2),
                    1,
                    rank,
                    stream,
                )?;
            }
        }

        ctx.gpu.synchronize(stream)?;

        let mut ids = vec![0u8; (self.block + 1) * 4];
        ctx.gpu.copy_d2h(self.scratch.draft_tokens_dev, &mut ids)?;
        let drafts: Vec<u32> = (1..=self.block)
            .map(|i| {
                u32::from_le_bytes([ids[i * 4], ids[i * 4 + 1], ids[i * 4 + 2], ids[i * 4 + 3]])
            })
            .collect();

        if self.conf_enabled {
            let mut xs = vec![0u8; self.block * 2];
            let mut ms = vec![0u8; self.block * 2];
            ctx.gpu.copy_d2h(self.scratch.conf_x, &mut xs)?;
            ctx.gpu.copy_d2h(self.scratch.conf_m, &mut ms)?;
            // The two halves are summed in f32 on the host, matching the
            // reference's `dtype=torch.float32` confidence head; only the two
            // partial dots round through BF16 (a documented deviation).
            let conf: Vec<f32> = (0..self.block)
                .map(|i| bf16_at(&xs, i) + bf16_at(&ms, i))
                .collect();
            self.record_confidence(&conf);
        }

        Ok(drafts)
    }
}

/// Host f32 reference for the γ-row greedy chain and the two-slice confidence
/// dot — the piece Atlas re-implements around the stage bodies.
///
/// Fed the golden's OWN `x` and pre-Markov `logits`, it isolates the chain from
/// the 3-stage numeric path (which is verifiable only on a GPU).
#[derive(Debug, Clone)]
pub struct MarkovChainRef<'a> {
    /// `[gamma, vocab]` pre-Markov logits, row-major.
    pub logits: &'a [f32],
    /// `[vocab, rank]` bigram embedding table.
    pub markov_w1: &'a [f32],
    /// `[vocab, rank]` bigram head.
    pub markov_w2: &'a [f32],
    /// `[gamma, hidden]` PRE-`norm` hc_head output.
    pub x: &'a [f32],
    /// `[hidden + rank]` bias-free confidence projection.
    pub conf_w: &'a [f32],
    pub gamma: usize,
    pub vocab: usize,
    pub rank: usize,
    pub hidden: usize,
}

/// The chain's outputs: `output_ids` has `gamma + 1` entries (slot 0 is the
/// anchor), `confidence` has `gamma`.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainOut {
    pub output_ids: Vec<u32>,
    pub confidence: Vec<f32>,
    /// `[gamma, vocab]` logits AFTER the per-row bigram bias.
    pub biased_logits: Vec<f32>,
}

impl MarkovChainRef<'_> {
    pub fn run(&self, anchor: u32) -> ChainOut {
        let mut output_ids = Vec::with_capacity(self.gamma + 1);
        output_ids.push(anchor);
        let mut confidence = Vec::with_capacity(self.gamma);
        let mut biased_logits = vec![0.0f32; self.gamma * self.vocab];

        for i in 0..self.gamma {
            let prev = output_ids[i] as usize;
            let embed = &self.markov_w1[prev * self.rank..(prev + 1) * self.rank];
            let row = &mut biased_logits[i * self.vocab..(i + 1) * self.vocab];
            for (t, out) in row.iter_mut().enumerate() {
                let w2 = &self.markov_w2[t * self.rank..(t + 1) * self.rank];
                let bias: f32 = w2.iter().zip(embed).map(|(a, b)| a * b).sum();
                *out = self.logits[i * self.vocab + t] + bias;
            }
            let mut best = 0usize;
            for t in 1..self.vocab {
                if row[t] > row[best] {
                    best = t;
                }
            }
            output_ids.push(best as u32);

            let xi = &self.x[i * self.hidden..(i + 1) * self.hidden];
            let c: f32 = xi
                .iter()
                .zip(&self.conf_w[..self.hidden])
                .map(|(a, b)| a * b)
                .sum::<f32>()
                + embed
                    .iter()
                    .zip(&self.conf_w[self.hidden..self.hidden + self.rank])
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
            confidence.push(c);
        }

        ChainOut {
            output_ids,
            confidence,
            biased_logits,
        }
    }
}
