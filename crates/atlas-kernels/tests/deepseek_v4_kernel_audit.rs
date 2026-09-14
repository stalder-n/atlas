// SPDX-License-Identifier: AGPL-3.0-only

//! Regression for the two optional probes that stopped the 0731 EP=2 boot.

use std::path::Path;

const OPTIONAL_PROBES: &[(&str, &str)] = &[
    ("inferspark_prefill_512tc", "inferspark_prefill_512tc"),
    ("moe_silu_mul", "silu_mul_quant_fp8"),
];

#[test]
fn v4_optional_probes_have_runtime_audit_declarations() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/gb10/deepseek-v4-flash/MODEL.toml");
    let doc: toml::Value = toml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for &(module, kernel) in OPTIONAL_PROBES {
        let reason = doc["expected_absent"]
            .get(module)
            .and_then(|entries| entries.get(kernel))
            .and_then(toml::Value::as_str);
        assert!(
            reason.is_some_and(|reason| !reason.trim().is_empty()),
            "missing runtime audit declaration for {module}::{kernel}"
        );
    }
}

#[test]
#[ignore = "requires nvcc and ATLAS_SKIP_BUILD unset with deepseek-v4-flash selected"]
fn compiled_v4_registry_carries_the_optional_probes() {
    let targets: Vec<_> = atlas_kernels::available_targets()
        .into_iter()
        .filter(|target| target.target.model == "deepseek-v4-flash")
        .collect();
    assert!(!targets.is_empty(), "build must include deepseek-v4-flash");
    for target in targets {
        for probe in OPTIONAL_PROBES {
            assert!(target.expected_absent.contains(probe), "missing {probe:?}");
        }
    }
}
