// SPDX-License-Identifier: AGPL-3.0-only

//! Host tests for the DeepSeek-V4-Flash-Vision-Exp prefill router dispatch:
//! the pure decision table (`bias_vl` × `tid2eid` × device token ids) and the
//! gated kernel lookup. No CUDA, no `ForwardContext` — `MockGpuBackend` only.

use super::{PrefillRouter, select_prefill_router, vl_router_kernels};
use spark_runtime::gpu::mock::MockGpuBackend;

/// All eight combinations, asserted literally. This table IS the contract; a
/// change here is a change to which kernel a Vision-Exp checkpoint routes
/// through, so it must be spelled out rather than derived.
#[test]
fn dispatch_table_is_exhaustive_over_bias_vl_tid2eid_and_token_ids() {
    // hash layer, VL checkpoint, ids present -> HashVl: sentinels have no
    // `tid2eid` row, so they need the learned `bias_vl` arm.
    assert_eq!(
        select_prefill_router(true, true, true),
        PrefillRouter::HashVl
    );
    // hash layer, VL checkpoint, no ids -> Hash: nothing to branch on.
    assert_eq!(
        select_prefill_router(true, true, false),
        PrefillRouter::Hash
    );
    // hash layer, text checkpoint -> Hash, ids or not (ids are required by the
    // text hash kernel too, but that is a separate error, not a dispatch bit).
    assert_eq!(
        select_prefill_router(true, false, true),
        PrefillRouter::Hash
    );
    assert_eq!(
        select_prefill_router(true, false, false),
        PrefillRouter::Hash
    );
    // non-hash layer, VL checkpoint, ids present -> SqrtSoftplusVl: per-token
    // `sel_bias = tok >= vocab_size ? bias_vl : bias`.
    assert_eq!(
        select_prefill_router(false, true, true),
        PrefillRouter::SqrtSoftplusVl
    );
    // non-hash layer, VL checkpoint, no ids -> text kernel (contract C7).
    assert_eq!(
        select_prefill_router(false, true, false),
        PrefillRouter::SqrtSoftplus
    );
    // non-hash layer, text checkpoint -> today's kernel, ids or not. Together
    // with the two `(true, false, _)` rows above this is the byte-for-byte
    // -unaffected invariant: no text checkpoint reaches a `_vl` kernel.
    assert_eq!(
        select_prefill_router(false, false, true),
        PrefillRouter::SqrtSoftplus
    );
    assert_eq!(
        select_prefill_router(false, false, false),
        PrefillRouter::SqrtSoftplus
    );
}

/// Regression test for the boot-audit hazard: `MoeLayer::new_with_hash` is the
/// constructor for EVERY MoE model, and `moe_router_vl` is registered only for
/// `deepseek-v4-flash/nvfp4` with no `[expected_absent]` declaration anywhere.
/// A lookup issued without a `bias_vl` would add a REQUIRED failed-lookup row
/// to every other MoE model's boot audit.
#[test]
fn no_vl_kernel_lookup_without_bias_vl() {
    let gpu = MockGpuBackend::new();
    let (topk, hash) = vl_router_kernels(&gpu, false).expect("disabled path never fails");
    assert_eq!(topk.0, 0);
    assert_eq!(hash.0, 0);
    assert!(
        !gpu.kernel_lookups_snapshot()
            .iter()
            .any(|(m, _)| m == "moe_router_vl"),
        "no moe_router_vl lookup may reach the backend without a bias_vl"
    );
}

/// When enabled, both handles are resolved by name — pinning the module name
/// and the two kernel names against `moe_router_vl.cu`.
#[test]
fn vl_kernels_are_looked_up_by_name_when_enabled() {
    let gpu = MockGpuBackend::new();
    let (topk, hash) = vl_router_kernels(&gpu, true).expect("mock resolves every kernel");
    assert_ne!(topk.0, 0);
    assert_ne!(hash.0, 0);
    assert_eq!(
        gpu.kernel_lookups_snapshot(),
        vec![
            (
                "moe_router_vl".to_string(),
                "moe_topk_sqrtsoftplus_batched_vl".to_string()
            ),
            (
                "moe_router_vl".to_string(),
                "moe_hash_route_batched_vl".to_string()
            ),
        ]
    );
}
