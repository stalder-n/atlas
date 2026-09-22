// SPDX-License-Identifier: AGPL-3.0-only

//! The `DraftProposer` impl: ring catch-up, the block forward, and the state
//! lifecycle.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::sync::atomic::Ordering;

use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_loader::deepseek_v4::DSPARK_STAGES;

use super::{DeepseekV4DsparkHead, DsparkProposerState, ring};

impl DeepseekV4DsparkHead {
    /// Project `count` capture rows into every stage's KV ring.
    ///
    /// `src` is the first row of a CONTIGUOUS `[count, n_targets * hidden]`
    /// BF16 band (the prefill ring or the model's capture band). `first_pos`
    /// is the absolute sequence position of row 0. Destination slots are
    /// `pos % window`, so a run that crosses the wrap is split by the caller
    /// of [`Self::ingest_run`].
    ///
    /// Reference: `forward_embed` (`main_x = main_norm(main_proj(main_hidden))`)
    /// followed by `DSparkAttention.forward`'s ring write
    /// (`kv_cache[:, pos % window] = rope(kv_norm(wkv(main_x)))`).
    #[allow(clippy::too_many_arguments)]
    fn ingest_run(
        &self,
        src: DevicePtr,
        count: usize,
        first_pos: usize,
        slot0: usize,
        rings: &[DevicePtr; DSPARK_STAGES],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let n = count as u32;
        let h = self.hidden as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let s = &self.scratch;

        // main_x = main_norm(main_proj(capture))
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            src,
            &self.weights.main_proj,
            s.main_x,
            n,
            h,
            (self.n_targets * self.hidden) as u32,
            stream,
        )?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_vanilla_k,
            s.main_x,
            &self.weights.main_norm,
            s.main_x,
            n,
            h,
            eps,
            stream,
        )?;

        let positions: Vec<u8> = (0..count)
            .flat_map(|i| ((first_pos + i) as u32).to_le_bytes())
            .collect();
        ctx.gpu.copy_h2d_async(&positions, s.positions, stream)?;

        for stage in 0..DSPARK_STAGES {
            let mla = self.stage_mla(stage)?;
            let kv_dim = self.kv_dim as u32;
            let nope = mla.nope as u32;
            let rope = mla.rope as u32;
            self.projection(
                ctx,
                mla.wkv_a_fp8.as_ref(),
                &mla.wkv_a,
                s.main_x,
                s.kv,
                n,
                kv_dim,
                h,
                stream,
            )?;
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_vanilla_k,
                s.kv,
                &mla.kv_a_norm,
                s.kv,
                n,
                kv_dim,
                eps,
                stream,
            )?;
            ops::mla_q_rope_extract_batched(
                ctx.gpu,
                self.mla_q_rope_extract_k,
                s.kv,
                s.kv_rope,
                n,
                1,
                kv_dim,
                nope,
                rope,
                kv_dim,
                stream,
            )?;
            // K only: `num_q_heads = 0`, so the Q pointer is never read.
            ops::rope_yarn(
                ctx.gpu,
                self.rope_interleaved_k,
                s.kv_rope,
                s.kv_rope,
                s.positions,
                n,
                0,
                1,
                rope,
                rope,
                mla.main_inv_freq,
                1.0f32,
                stream,
            )?;
            ops::mla_q_rope_writeback_batched(
                ctx.gpu,
                self.mla_q_rope_writeback_k,
                s.kv_rope,
                s.kv,
                n,
                1,
                kv_dim,
                nope,
                rope,
                kv_dim,
                stream,
            )?;
            ctx.gpu.copy_d2d_async(
                s.kv,
                rings[stage].offset(slot0 * self.kv_dim * 2),
                count * self.kv_dim * 2,
                stream,
            )?;
        }
        Ok(())
    }

    /// Split a contiguous run of capture rows at the ring wrap and ingest each
    /// piece. `src_row` is the first row's index in the source band.
    #[allow(clippy::too_many_arguments)]
    fn ingest(
        &self,
        base: DevicePtr,
        src_row: usize,
        count: usize,
        first_pos: usize,
        rings: &[DevicePtr; DSPARK_STAGES],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let row_bytes = self.n_targets * self.hidden * 2;
        let mut done = 0usize;
        while done < count {
            let pos = first_pos + done;
            let slot = ring::slot(pos, self.window);
            let run = (count - done).min(self.window - slot);
            self.ingest_run(
                base.offset((src_row + done) * row_bytes),
                run,
                pos,
                slot,
                rings,
                ctx,
                stream,
            )?;
            done += run;
        }
        Ok(())
    }

    /// Bring every stage ring up to `position`, from the prefill ring on the
    /// first propose and from the model's capture band afterwards.
    #[allow(clippy::too_many_arguments)]
    fn refresh_rings(
        &self,
        state: &mut DsparkProposerState,
        position: usize,
        committed: usize,
        capture_band: Option<DevicePtr>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let rings = state.stage_ring;
        if state.next_pos == 0 && state.prefill_filled > 0 {
            // Cold turn: the prompt's captures are already at their own
            // `pos % window` slots, so the source row IS the slot and the
            // run split lines up exactly.
            let drain = ring::prefill_drain(state.prefill_prompt_len, self.window);
            let first_pos = drain[0].0;
            self.ingest(
                state.prefill_capture_ring,
                drain[0].1,
                drain.len().min(self.window - drain[0].1),
                first_pos,
                &rings,
                ctx,
                stream,
            )?;
            // The wrapped remainder starts back at slot 0.
            let first_run = drain.len().min(self.window - drain[0].1);
            if drain.len() > first_run {
                self.ingest(
                    state.prefill_capture_ring,
                    0,
                    drain.len() - first_run,
                    first_pos + first_run,
                    &rings,
                    ctx,
                    stream,
                )?;
            }
            state.next_pos = state.prefill_prompt_len;
        }

        let plan = ring::catch_up(state.next_pos, position, committed);
        if plan.gap > 0 {
            state.gap_positions += plan.gap;
            tracing::debug!(
                "DSpark ring gap: {} position(s) owed with no capture row (next_pos={}, \
                 position={position}, committed={committed}, total gaps={}). Acceptance \
                 degrades; correctness cannot — every draft is re-verified.",
                plan.gap,
                state.next_pos,
                state.gap_positions,
            );
        }
        if plan.count > 0 {
            let band = capture_band.ok_or_else(|| {
                anyhow::anyhow!("DSpark propose has no capture band; the model allocated none")
            })?;
            self.ingest(
                band,
                plan.first_row,
                plan.count,
                plan.first_pos,
                &rings,
                ctx,
                stream,
            )?;
        }
        state.next_pos = state.next_pos.max(position + 1);
        state.ring_len = ring::ring_len(state.next_pos, self.window);
        Ok(())
    }
}

impl DraftProposer for DeepseekV4DsparkHead {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        let bf16 = 2usize;
        let capture_row = self.n_targets * self.hidden * bf16;
        let ring_bytes = ring::ring_slots(self.window, self.block) * self.kv_dim * bf16;
        let mut stage_ring = [DevicePtr::NULL; DSPARK_STAGES];
        for slot in stage_ring.iter_mut() {
            *slot = gpu.alloc(ring_bytes)?;
        }
        Ok(Box::new(DsparkProposerState {
            prefill_capture_ring: gpu.alloc(self.window * capture_row)?,
            prefill_filled: 0,
            prefill_prompt_len: 0,
            stage_ring,
            ring_len: 0,
            next_pos: 0,
            last_num_drafted: 0,
            gap_positions: 0,
        }))
    }

    /// γ, the checkpoint's trained block size. The serve layer derives
    /// `num_drafts` from it.
    fn block_gamma(&self) -> Option<usize> {
        Some(self.block)
    }

    /// 🔴 True under EP. The stages are EP-SHARDED (`force_all_experts =
    /// false`), so the block's routed MoE needs the all-reduce — and the
    /// WORKER rank must run this same propose, which the existing
    /// `EP_CMD_MTP_PROPOSE` lane already arranges.
    fn needs_comm(&self) -> bool {
        self.ep_comm
    }

    /// The ring drain consumes `ctx.buffers` (`hc_streams`, `norm_output`,
    /// `moe_output`), so it must run from the first `propose`, never from an
    /// eager end-of-prefill hook — the GLM `t61` failure.
    fn prefill_uses_shared_buffers(&self) -> bool {
        true
    }

    fn last_confidence(&self) -> Option<f32> {
        self.conf_enabled
            .then(|| f32::from_bits(self.last_conf_bits.load(Ordering::Relaxed)))
    }

    fn propose(
        &self,
        last_token: u32,
        _target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        _draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let st = state
            .as_any_mut()
            .downcast_mut::<DsparkProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid DSpark proposer state"))?;
        ensure!(
            num_drafts <= self.block,
            "DSpark was asked for {num_drafts} drafts but its trained block is {}",
            self.block
        );

        // ── the block's input ids: [last_token, noise × (γ-1)] ──
        // Every one is < vocab_size, so U2.3's `_vl` router selects `bias`
        // (not `bias_vl`) — the reference's `image_mask` is all-false here.
        let mut ids: Vec<u32> = vec![self.noise_token; self.block];
        ids[0] = last_token;
        let id_bytes: Vec<u8> = ids.iter().flat_map(|t| t.to_le_bytes()).collect();
        ctx.gpu
            .copy_h2d_async(&id_bytes, self.scratch.token_ids, stream)?;

        // Spelled out rather than `..*ctx`: `midchunk_capture` is a
        // non-`Copy` borrow, and every field here is a decision.
        let block_ctx = ForwardContext {
            buffers: ctx.buffers,
            hc_row_offset: 0,
            gpu: ctx.gpu,
            config: ctx.config,
            dispatch: ctx.dispatch,
            derived: ctx.derived,
            levers: ctx.levers,
            stats: ctx.stats,
            attn_metadata: None,
            profile: ctx.profile,
            // Carried through from `run_mtp_propose_inner`, which set it from
            // `needs_comm()`: the EP-sharded stage MoE needs the all-reduce.
            comm: ctx.comm,
            // Host-built ids are uploaded per propose; capture is illegal.
            graph_capture: false,
            decode_step: false,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: Some(self.scratch.token_ids),
            // The draft block is text by construction — sentinels are ≥
            // vocab_size and `lm_head` is `[vocab_size]`, so one can never be
            // sampled into the block — hence no image span and no bounds.
            image_window: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Skip,
        };

        // ── ring catch-up ──
        // `committed` is what the last verify left in the band: the accepted
        // prefix plus the bonus token, i.e. `num_accepted + 1`, which
        // `after_verify` recorded.
        let committed = st.last_num_drafted;
        self.refresh_rings(st, position, committed, target_hidden_stack, ctx, stream)?;
        let ring_len = st.ring_len;
        let rings = st.stage_ring;

        // ── block embed → highway seed ──
        ops::batched_embed(
            ctx.gpu,
            self.batched_embed_k,
            self.scratch.token_ids,
            self.embed_tokens.weight,
            self.scratch.x,
            self.block as u32,
            self.hidden as u32,
            stream,
        )?;
        ops::hc_expand(
            ctx.gpu,
            self.hc_expand_k,
            self.scratch.x,
            ctx.buffers.hc_streams(),
            self.block as u32,
            self.hidden as u32,
            ctx.config.hc_mult as u32,
            stream,
        )?;

        for stage in 0..DSPARK_STAGES {
            self.stage_forward(stage, position, ring_len, rings[stage], &block_ctx, stream)?;
        }

        let drafts = self.head_and_chain(last_token, &block_ctx, stream, grammar_bitmask)?;

        if self.conf_enabled {
            let conf = self.last_conf.lock();
            let min = conf
                .iter()
                .take(num_drafts)
                .copied()
                .fold(f32::INFINITY, f32::min);
            self.last_conf_bits.store(min.to_bits(), Ordering::Relaxed);
        }

        tracing::debug!(
            "DSpark propose: anchor={last_token} pos={position} ring_len={ring_len} \
             gaps={} → drafts={:?} (returning {num_drafts})",
            st.gap_positions,
            drafts,
        );
        Ok(drafts.into_iter().take(num_drafts).collect())
    }

    /// Nothing to trim: the ring is POSITION-indexed and every entry is
    /// rewritten from a committed capture on the next propose. What is
    /// recorded is how many capture rows that next propose may read.
    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let st = state
            .as_any_mut()
            .downcast_mut::<DsparkProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid DSpark proposer state"))?;
        st.last_num_drafted = num_accepted + 1;
        Ok(())
    }

    fn free_state(&self, gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        let st = state
            .as_any_mut()
            .downcast_mut::<DsparkProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid DSpark proposer state"))?;
        // `DevicePtr` has no `Drop`; these four leak unless released here.
        for ptr in std::iter::once(&mut st.prefill_capture_ring).chain(st.stage_ring.iter_mut()) {
            if *ptr != DevicePtr::NULL {
                gpu.free(*ptr)?;
                *ptr = DevicePtr::NULL;
            }
        }
        st.prefill_filled = 0;
        st.ring_len = 0;
        st.next_pos = 0;
        Ok(())
    }
}
