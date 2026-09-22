// SPDX-License-Identifier: AGPL-3.0-only

//! Regression for the two optional probes that stopped the 0731 EP=2 boot,
//! plus the source-level pins for the Vision-Exp kernel path.
//!
//! Why the vision pins live HERE and not only in `kernel_arity.rs`: that test
//! reads the PTX baked into the binary, so it is `#[ignore]` and vacuous on
//! every host without nvcc — which is CI. The `cuLaunchKernel` over-read it
//! guards against is exactly as fatal on a CUDA-less CI's watch, so the same
//! numbers are re-derived here from the `.cu` SOURCES, which are checked in
//! and readable anywhere.

use std::path::{Path, PathBuf};

const OPTIONAL_PROBES: &[(&str, &str)] = &[
    ("prefill_attn_compressed_tc", "prefill_attn_compressed_tc"),
    ("inferspark_prefill_512tc", "inferspark_prefill_512tc"),
    ("moe_silu_mul", "silu_mul_quant_fp8"),
];

/// (file, module) of every source the V4 vision path requires. No fallback
/// exists for any kernel in them, so absence is a boot failure, not a soft
/// probe — which is why `[expected_absent]` in MODEL.toml stays untouched by
/// the vision path. The entry points themselves are [`VISION_ARITY_PINS`].
const VISION_SOURCES: &[(&str, &str)] = &[
    ("vision_encoder.cu", "vision_encoder"),
    ("vision_deepseek.cu", "vision_deepseek"),
    (
        "prefill_attn_compressed_vis.cu",
        "prefill_attn_compressed_vis",
    ),
    ("moe_router_vl.cu", "moe_router_vl"),
];

/// (file, module) of every source the native DSpark proposer requires. Same
/// posture as [`VISION_SOURCES`]: no fallback exists, so absence is a boot
/// failure and MODEL.toml gets no `[expected_absent]` entry for it.
const DSPARK_SOURCES: &[(&str, &str)] = &[("dspark_capture.cu", "dspark_capture")];

/// (kernel, parameter count), re-derived from the `.cu` sources by
/// [`global_signatures`] and asserted against these numbers.
///
/// This is the HOST-RUNNABLE half of the arity pin. Its PTX-side twin is
/// `kernel_arity.rs::PINS` — the two tables state the same contract from the
/// two ends of nvcc, so a divergence between them is itself the signal.
/// Change one, change the other.
const VISION_ARITY_PINS: &[(&str, usize)] = &[
    ("vision_gemm_bias", 7),
    ("vision_gemm_bias_nn", 7),
    ("vision_add_bias", 4),
    ("vision_layer_norm", 6),
    ("vision_add_inplace", 3),
    ("vision_gelu", 2),
    ("vision_attention_rope", 7),
    ("vit_rope_deinterleave", 9),
    ("vit_softmax_rows", 4),
    ("vit_scatter_head", 5),
    ("vision_spatial_merge", 6),
    ("vision_f32_to_bf16", 3),
    ("vision_bf16_copy", 3),
    ("ds_vision_unshuffle3", 7),
    ("ds_gelu_erf_inplace", 2),
    ("ds_swiglu_split", 4),
    // 16, not 15: the `_vis` twin drops `sliding_window` and gains
    // `win_lo`/`win_hi`.
    ("prefill_attn_compressed_vis", 16),
    ("moe_topk_sqrtsoftplus_batched_vl", 11),
    ("moe_hash_route_batched_vl", 11),
    // The mHC-stream mean the DSpark capture band is built from.
    ("dspark_hc_mean_capture", 8),
];

fn kernels_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels")
}

fn v4_nvfp4_dir() -> PathBuf {
    kernels_dir().join("gb10/deepseek-v4-flash/nvfp4")
}

fn read_toml(path: &Path) -> toml::Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    toml::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Blank out `/* … */` and `// …` before the signature scan: a commented-out
/// declaration must not count, and a `,`, `<` or `>` in a parameter comment
/// must not skew the parameter split.
fn strip_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    loop {
        // A line comment keeps its `\n`; a block comment leaves a space, so
        // neither can weld the tokens around it together.
        let (at, close, keep) = match (rest.find("//"), rest.find("/*")) {
            (Some(l), Some(b)) if b < l => (b, "*/", 2),
            (Some(l), _) => (l, "\n", 0),
            (None, Some(b)) => (b, "*/", 2),
            (None, None) => break,
        };
        out.push_str(&rest[..at]);
        out.push(' ');
        match rest[at..].find(close) {
            Some(close_at) => rest = &rest[at + close_at + keep..],
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

/// Every `extern "C" __global__ [void] <name>(…)` in a `.cu`, as
/// `(name, parameter count)`. Balanced-paren scan for the parameter list, then
/// a TOP-LEVEL comma count, so a parameter whose type carries commas inside
/// `<>`/`()`/`[]` still counts once.
fn global_signatures(src: &str) -> Vec<(String, usize)> {
    let src = strip_comments(src);
    let mut out = Vec::new();

    for tail in src.split("extern \"C\" __global__").skip(1) {
        let Some((head, params)) = tail.split_once('(') else {
            continue;
        };
        // `void` may sit on the marker's line or (as in vision_encoder.cu) the
        // next one; either way the name is the last word before the `(`.
        let words: Vec<&str> = head.split_whitespace().collect();
        let name = match words[..] {
            [name] | ["void", name] => name,
            // Not a shape we understand — skip rather than guess at a count.
            _ => continue,
        };

        let mut nest = 0i32;
        let mut commas = 0usize;
        let mut end = None;
        for (i, ch) in params.char_indices() {
            match ch {
                '(' | '<' | '[' => nest += 1,
                '>' | ']' => nest -= 1,
                ')' if nest == 0 => {
                    end = Some(i);
                    break;
                }
                ')' => nest -= 1,
                ',' if nest == 0 => commas += 1,
                _ => {}
            }
        }
        let end = end.unwrap_or_else(|| panic!("unbalanced parameter list for {name}"));
        let count = if params[..end].trim().is_empty() {
            0
        } else {
            commas + 1
        };
        out.push((name.to_string(), count));
    }
    out
}

#[test]
fn v4_optional_probes_have_runtime_audit_declarations() {
    let doc = read_toml(&kernels_dir().join("gb10/deepseek-v4-flash/MODEL.toml"));
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
fn v4_vision_kernel_arities_match_the_pins() {
    let mut seen = Vec::new();
    for &(file, _) in VISION_SOURCES.iter().chain(DSPARK_SOURCES) {
        let path = v4_nvfp4_dir().join(file);
        let src =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        for (name, count) in global_signatures(&src) {
            if let Some(&(_, pin)) = VISION_ARITY_PINS.iter().find(|(k, _)| *k == name) {
                assert_eq!(
                    count, pin,
                    "{file}::{name} has {count} parameters, the launcher pin says {pin} — \
                     update the launcher, this pin AND kernel_arity.rs::PINS together"
                );
                seen.push(name);
            }
        }
    }
    // Doubles as the presence check: every required entry point is pinned, so
    // a kernel that is renamed or deleted fails here.
    for &(kernel, _) in VISION_ARITY_PINS {
        assert!(
            seen.iter().any(|s| s == kernel),
            "pin for {kernel} matched no source signature — did the kernel move or get renamed?"
        );
    }
}

#[test]
fn v4_vision_modules_are_registered() {
    let doc = read_toml(&v4_nvfp4_dir().join("KERNEL.toml"));
    let modules = doc
        .get("modules")
        .and_then(toml::Value::as_table)
        .expect("nvfp4/KERNEL.toml must carry a [modules] table");
    for &(file, module) in VISION_SOURCES.iter().chain(DSPARK_SOURCES) {
        let stem = file.strip_suffix(".cu").expect("source must be a .cu");
        let declared = modules
            .get(stem)
            .and_then(toml::Value::as_str)
            .unwrap_or_else(|| panic!("nvfp4/KERNEL.toml [modules] has no entry for {stem}"));
        assert_eq!(
            declared, module,
            "{stem} is registered as module `{declared}`, the vision path resolves `{module}`"
        );
    }
}

/// The DeepSeek ViT tower IS the Qwen3.6 one, reached by a same-hardware
/// symlink — the tree's idiom for this file (`holo-3.1-*`, `ornith-1.0-9b`,
/// `qwen3.5-27b`, `qwen3.6-27b`, `qwen3.8-flash-next` all link to the same
/// source; `qwen3.5-122b/397b` link to the qwen3-vl one).
///
/// It is NOT a regular byte-identical copy, and that is enforced, not
/// stylistic: `scripts/check_kernel_shadows.py` RULE2 fails CI on "identical
/// regular copies of vision_encoder.cu — keep one canonical file, symlink the
/// rest", and `.github/workflows/ci.yml` runs it as a required check.
///
/// The cost of the link is real — a Qwen-motivated ViT edit changes what
/// DeepSeek compiles — and it is the cost the repo has already chosen for nine
/// other targets. Divergence is therefore an EXPLICIT act: replace the symlink
/// with a real, genuinely different file (RULE2 keys on content, so a diverged
/// file does not trip it) and update this test in the same commit, saying why.
#[test]
fn vision_encoder_cu_links_to_the_qwen36_tower() {
    let ours = v4_nvfp4_dir().join("vision_encoder.cu");
    let theirs = kernels_dir().join("gb10/qwen3.6-35b-a3b/nvfp4/vision_encoder.cu");
    let meta =
        std::fs::symlink_metadata(&ours).unwrap_or_else(|e| panic!("{}: {e}", ours.display()));
    assert!(
        meta.file_type().is_symlink(),
        "{} must be a symlink into the canonical tower, not a regular file — \
         check_kernel_shadows.py RULE2 fails CI on an identical regular copy",
        ours.display()
    );
    assert_eq!(
        std::fs::canonicalize(&ours).unwrap_or_else(|e| panic!("{}: {e}", ours.display())),
        std::fs::canonicalize(&theirs).unwrap_or_else(|e| panic!("{}: {e}", theirs.display())),
        "{} resolves somewhere other than the qwen3.6 tower",
        ours.display()
    );
}

#[test]
#[ignore = "requires nvcc and AVAROK_SKIP_BUILD unset with deepseek-v4-flash selected"]
fn compiled_v4_registry_carries_the_optional_probes() {
    let targets: Vec<_> = avarok_kernels::available_targets()
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

#[test]
#[ignore = "requires nvcc and AVAROK_SKIP_BUILD unset with deepseek-v4-flash selected"]
fn compiled_v4_registry_carries_the_vision_modules() {
    let targets: Vec<_> = avarok_kernels::available_targets()
        .into_iter()
        .filter(|target| target.target.model == "deepseek-v4-flash")
        .collect();
    assert!(!targets.is_empty(), "build must include deepseek-v4-flash");
    for target in targets {
        for &(_, module) in VISION_SOURCES.iter().chain(DSPARK_SOURCES) {
            assert!(
                target.modules.iter().any(|(name, _)| *name == module),
                "compiled deepseek-v4-flash target is missing required module {module}"
            );
        }
    }
}
