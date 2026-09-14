// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::HashMap;

use atlas_core::config::ModelConfig;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

use super::{discover_dspark_stages, expect_tensor, native_dspark_config};

fn store(entries: &[(&str, WeightDtype, &[usize])]) -> WeightStore {
    let weights = entries
        .iter()
        .map(|(name, dtype, shape)| {
            (
                (*name).to_string(),
                WeightTensor {
                    ptr: DevicePtr::NULL,
                    shape: shape.to_vec(),
                    dtype: *dtype,
                },
            )
        })
        .collect::<HashMap<_, _>>();
    WeightStore::from_map(weights)
}

#[test]
fn stage_discovery_is_exact_and_ordered() {
    let store = store(&[
        ("mtp.2.norm.weight", WeightDtype::BF16, &[1]),
        ("mtp.0.main_norm.weight", WeightDtype::BF16, &[1]),
        ("layers.0.attn_norm.weight", WeightDtype::BF16, &[1]),
        ("mtp.1.ffn_norm.weight", WeightDtype::BF16, &[1]),
        ("xmtp.9.ffn_norm.weight", WeightDtype::BF16, &[1]),
        ("mtp.bad.ffn_norm.weight", WeightDtype::BF16, &[1]),
        ("mtp.-1.ffn_norm.weight", WeightDtype::BF16, &[1]),
    ]);
    assert_eq!(
        discover_dspark_stages(&store),
        std::collections::BTreeSet::from([0, 1, 2])
    );
}

#[test]
fn tensor_contract_rejects_e8m0_where_fp8_weight_is_required() {
    let store = store(&[(
        "mtp.0.main_proj.weight",
        WeightDtype::FP8E8M0,
        &[4096, 12288],
    )]);
    let err = expect_tensor(
        &store,
        "mtp.0.main_proj.weight",
        WeightDtype::FP8E4M3,
        &[4096, 12288],
    )
    .expect_err("wrong dtype must fail closed");
    let msg = err.to_string();
    assert!(msg.contains("mtp.0.main_proj.weight"));
    assert!(msg.contains("FP8E8M0"));
    assert!(msg.contains("expected FP8E4M3"));
}

#[test]
fn tensor_contract_rejects_wrong_scale_shape() {
    let store = store(&[(
        "mtp.0.ffn.experts.0.w1.scale",
        WeightDtype::FP8E8M0,
        &[2048, 64],
    )]);
    let err = expect_tensor(
        &store,
        "mtp.0.ffn.experts.0.w1.scale",
        WeightDtype::FP8E8M0,
        &[2048, 128],
    )
    .expect_err("wrong scale geometry must fail closed");
    let msg = err.to_string();
    assert!(msg.contains("mtp.0.ffn.experts.0.w1.scale"));
    assert!(msg.contains("[2048, 64]"));
    assert!(msg.contains("expected [2048, 128]"));
}

#[test]
fn tensor_contract_rejects_missing_required_tensor() {
    let err = expect_tensor(
        &store(&[]),
        "mtp.2.confidence_head.proj.weight",
        WeightDtype::BF16,
        &[1, 4096],
    )
    .expect_err("missing tensor must fail closed");
    assert!(
        err.to_string()
            .contains("missing required tensor `mtp.2.confidence_head.proj.weight`")
    );
}

#[test]
fn pure_tp_draft_contract_restores_full_width_and_partitions_experts() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "deepseek_v4".to_string();
    config.moe_intermediate_size = 1024;
    config.shared_expert_intermediate_size = 1024;
    config.tp_rank = 1;
    config.tp_world_size = 2;
    config.ep_rank = 0;
    config.ep_world_size = 1;

    let draft = native_dspark_config(&config).unwrap();
    assert_eq!(draft.moe_intermediate_size, 2048);
    assert_eq!(draft.shared_expert_intermediate_size, 2048);
    assert_eq!((draft.tp_rank, draft.tp_world_size), (1, 2));
    assert_eq!((draft.ep_rank, draft.ep_world_size), (1, 2));
    assert_eq!(draft.local_expert_range(), (256, 512));
}

#[test]
fn pure_tp_draft_contract_rejects_invalid_or_overflowing_geometry() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.tp_world_size = 2;
    config.tp_rank = 2;
    config.ep_world_size = 1;
    let err = native_dspark_config(&config).expect_err("rank outside world must reject");
    assert!(
        err.to_string()
            .contains("TP rank 2 is outside world size 2")
    );

    config.tp_rank = 0;
    config.moe_intermediate_size = usize::MAX;
    let err = native_dspark_config(&config).expect_err("routed width overflow must reject");
    assert!(err.to_string().contains("routed width overflow"));
}

fn pinned_store() -> WeightStore {
    let headers: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/0731-tensor-headers.json")).unwrap();
    WeightStore::from_map(
        headers
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, header)| {
                let dtype = match header["dtype"].as_str().unwrap() {
                    "U8" | "I8" => WeightDtype::UInt8,
                    "F8_E4M3" => WeightDtype::FP8E4M3,
                    "F8_E8M0" => WeightDtype::FP8E8M0,
                    "F32" => WeightDtype::FP32,
                    other => panic!("unexpected fixture dtype {other}"),
                };
                let shape = serde_json::from_value(header["shape"].clone()).unwrap();
                (
                    name.clone(),
                    WeightTensor {
                        ptr: DevicePtr::NULL,
                        shape,
                        dtype,
                    },
                )
            })
            .collect(),
    )
}

#[test]
fn pinned_base_and_draft_have_independent_quant_contracts() {
    use crate::weight_map::v4_quant::{PackedExpertFormat, resolve_packed_expert};
    let store = pinned_store();
    for expert in [0, 128] {
        for projection in ["w1", "w2", "w3"] {
            let prefix = format!("layers.0.ffn.experts.{expert}.{projection}");
            assert_eq!(
                resolve_packed_expert(&store, &prefix).unwrap(),
                PackedExpertFormat::Nvfp4
            );
        }
    }
    for projection in ["w1", "w2", "w3"] {
        let prefix = format!("mtp.0.ffn.experts.0.{projection}");
        assert_eq!(
            resolve_packed_expert(&store, &prefix).unwrap(),
            PackedExpertFormat::Mxfp4
        );
    }
    super::check_fp8_block_scaled(&store, "layers.0.attn.wq_a").unwrap();
}

#[test]
fn pinned_config_ep_partition_is_128_experts_per_rank() {
    let mut config =
        atlas_core::config::parse_config(include_str!("fixtures/0731-config.json")).unwrap();
    assert_eq!(config.num_hidden_layers, 43);
    assert_eq!(config.dspark_target_layer_ids, [40, 41, 42]);
    assert_eq!(
        config.quantization_config.as_ref().unwrap().quant_method,
        "fp8"
    );
    config.ep_world_size = 2;
    config.ep_rank = 0;
    assert_eq!(config.local_expert_range(), (0, 128));
    config.ep_rank = 1;
    assert_eq!(config.local_expert_range(), (128, 256));
    super::check_shared_expert(&pinned_store(), "layers.0.ffn", &config).unwrap();
}

#[test]
fn malformed_nvfp4_contracts_fail_closed() {
    use crate::weight_map::v4_quant::resolve_packed_expert;
    for (dtype, shape) in [
        (WeightDtype::FP8E8M0, vec![128, 8]),
        (WeightDtype::FP8E4M3, vec![128, 4]),
    ] {
        let store = store(&[
            ("p.weight", WeightDtype::UInt8, &[128, 64]),
            ("p.weight_scale", dtype, &shape),
            ("p.weight_scale_2", WeightDtype::FP32, &[]),
            ("p.input_scale", WeightDtype::FP32, &[]),
        ]);
        assert!(resolve_packed_expert(&store, "p").is_err());
    }
    for scalar_shape in [vec![], vec![1], vec![2]] {
        let store = store(&[
            ("p.weight", WeightDtype::UInt8, &[128, 64]),
            ("p.weight_scale", WeightDtype::FP8E4M3, &[128, 8]),
            ("p.weight_scale_2", WeightDtype::FP32, &scalar_shape),
            ("p.input_scale", WeightDtype::FP32, &scalar_shape),
        ]);
        assert_eq!(
            resolve_packed_expert(&store, "p").is_ok(),
            scalar_shape != [2]
        );
    }
    assert!(
        resolve_packed_expert(&store(&[("p.weight", WeightDtype::UInt8, &[128, 64])]), "p")
            .is_err()
    );
}

#[test]
fn dspark_request_is_explicitly_unsupported_before_tensor_validation() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "deepseek_v4".into();
    config.dspark_block_size = 5;
    let err = super::check_native_dspark_checkpoint(&store(&[]), &config, true).unwrap_err();
    assert!(
        err.to_string()
            .contains("native DSpark proposer is not implemented")
    );
}

#[test]
fn target_preflight_does_not_require_draft_tensors() {
    let fixture = pinned_store();
    let mut entries = fixture
        .names()
        .filter(|name| name.starts_with("layers."))
        .map(|name| (name.to_string(), copy_header(fixture.get(name).unwrap())))
        .collect::<HashMap<_, _>>();
    for projection in ["wq_b", "wkv", "wo_a", "wo_b"] {
        for suffix in ["weight", "scale"] {
            let source = format!("layers.0.attn.wq_a.{suffix}");
            entries.insert(
                format!("layers.0.attn.{projection}.{suffix}"),
                copy_header(fixture.get(&source).unwrap()),
            );
        }
    }
    let store = WeightStore::from_map(entries);
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "deepseek_v4".into();
    config.dspark_block_size = 5;
    config.num_hidden_layers = 1;
    config.num_experts = 1;
    config.ep_world_size = 1;
    config.ep_rank = 0;
    config.hidden_size = 4096;
    config.moe_intermediate_size = 2048;
    super::check_native_dspark_checkpoint(&store, &config, false).unwrap();
}

#[test]
fn older_non_dspark_checkpoint_keeps_legacy_preflight() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "deepseek_v4".into();
    config.dspark_block_size = 0;
    super::check_native_dspark_checkpoint(&store(&[]), &config, false).unwrap();
}

fn copy_header(tensor: &WeightTensor) -> WeightTensor {
    WeightTensor {
        ptr: tensor.ptr,
        shape: tensor.shape.clone(),
        dtype: tensor.dtype,
    }
}
