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

// ★ U1.4 (the Vision-Exp kernel unit) added FOUR modules and NO new
// `[expected_absent]` declaration, so this table is deliberately unchanged —
// it has not fallen behind. `qwen3-vl-30b-a3b/MODEL.toml` declares
// `vision_add_bias` / `vit_rope_deinterleave` / `vit_softmax_rows` /
// `vit_scatter_head` absent only because ITS OWN `vision_encoder.cu` defines
// nine entry points; the `qwen3.6-35b-a3b` file this target symlinks to
// defines all thirteen, so all four RESOLVE here and declaring them absent
// would be a FALSE absence — it would let a genuinely missing kernel boot.
// `REQUIRED_VISION_KERNELS` below is the positive statement of that decision.

/// (file, module, entry points) that the V4 vision path REQUIRES. No fallback
/// exists for any of them, so absence is a boot failure, not a soft probe.
const REQUIRED_VISION_KERNELS: &[(&str, &str, &[&str])] = &[
    (
        "vision_encoder.cu",
        "vision_encoder",
        &[
            "vision_gemm_bias",
            "vision_gemm_bias_nn",
            "vision_add_bias",
            "vision_layer_norm",
            "vision_add_inplace",
            "vision_gelu",
            "vision_attention_rope",
            "vit_rope_deinterleave",
            "vit_softmax_rows",
            "vit_scatter_head",
            "vision_spatial_merge",
            "vision_f32_to_bf16",
            "vision_bf16_copy",
        ],
    ),
    (
        "vision_deepseek.cu",
        "vision_deepseek",
        &[
            "ds_vision_unshuffle3",
            "ds_gelu_erf_inplace",
            "ds_swiglu_split",
        ],
    ),
    (
        "prefill_attn_compressed_vis.cu",
        "prefill_attn_compressed_vis",
        &["prefill_attn_compressed_vis"],
    ),
    (
        "moe_router_vl.cu",
        "moe_router_vl",
        &[
            "moe_topk_sqrtsoftplus_batched_vl",
            "moe_hash_route_batched_vl",
        ],
    ),
];

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

fn v4_model_toml() -> toml::Value {
    read_toml(&kernels_dir().join("gb10/deepseek-v4-flash/MODEL.toml"))
}

/// Blank out `/* … */` and `// …` so a commented-out signature is never read
/// as a declaration. Replacing with spaces (rather than deleting) keeps byte
/// offsets stable, which is not required here but makes the scan easier to
/// reason about.
fn strip_comments(src: &str) -> String {
    let bytes: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == '/' && i + 1 < bytes.len() && bytes[i + 1] == '*' {
            out.push_str("  ");
            i += 2;
            while i < bytes.len()
                && !(bytes[i] == '*' && i + 1 < bytes.len() && bytes[i + 1] == '/')
            {
                out.push(if bytes[i] == '\n' { '\n' } else { ' ' });
                i += 1;
            }
            if i < bytes.len() {
                out.push_str("  ");
                i += 2;
            }
        } else if bytes[i] == '/' && i + 1 < bytes.len() && bytes[i + 1] == '/' {
            while i < bytes.len() && bytes[i] != '\n' {
                out.push(' ');
                i += 1;
            }
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    out
}

/// Every `extern "C" __global__ [void] <name>(…)` in a `.cu`, as
/// `(name, parameter count)`. Balanced-paren scan for the parameter list, then
/// a TOP-LEVEL comma split, so a parameter whose type carries commas inside
/// `<>`/`()`/`[]` still counts once.
fn global_signatures(src: &str) -> Vec<(String, usize)> {
    let chars: Vec<char> = strip_comments(src).chars().collect();
    let marker: Vec<char> = "extern \"C\" __global__".chars().collect();
    let mut out = Vec::new();
    let mut at = 0usize;

    while at + marker.len() <= chars.len() {
        if chars[at..at + marker.len()] != marker[..] {
            at += 1;
            continue;
        }
        let mut i = at + marker.len();
        at = i;

        // `extern "C" __global__` may be followed by `void <name>` on the same
        // line or, as in vision_encoder.cu, by a newline and then `void <name>`.
        let mut word = String::new();
        for _ in 0..2 {
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            word = chars[start..i].iter().collect();
            if word != "void" {
                break;
            }
        }
        while i < chars.len() && chars[i].is_whitespace() {
            i += 1;
        }
        // Not a shape we understand — skip rather than guess at a count.
        if word.is_empty() || word == "void" || i >= chars.len() || chars[i] != '(' {
            continue;
        }

        let open = i;
        let mut depth = 0usize;
        while i < chars.len() {
            match chars[i] {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
            i += 1;
        }
        assert!(i < chars.len(), "unbalanced parameter list for {word}");

        // Top-level comma split: a type carrying commas inside `<>`/`()`/`[]`
        // still counts as one parameter.
        let mut nest = 0i32;
        let mut count = 0usize;
        let mut current = String::new();
        for &ch in &chars[open + 1..i] {
            match ch {
                '(' | '<' | '[' => nest += 1,
                ')' | '>' | ']' => nest -= 1,
                ',' if nest == 0 => {
                    if !current.trim().is_empty() {
                        count += 1;
                    }
                    current.clear();
                    continue;
                }
                _ => {}
            }
            current.push(ch);
        }
        if !current.trim().is_empty() {
            count += 1;
        }
        out.push((word, count));
        at = i;
    }
    out
}

#[test]
fn v4_optional_probes_have_runtime_audit_declarations() {
    let doc = v4_model_toml();
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
fn v4_vision_sources_declare_their_entry_points() {
    for &(file, _module, entry_points) in REQUIRED_VISION_KERNELS {
        let path = v4_nvfp4_dir().join(file);
        let src =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(
            src.lines().next(),
            Some("// SPDX-License-Identifier: AGPL-3.0-only"),
            "{file}: SPDX must be line 1"
        );
        let declared: Vec<String> = global_signatures(&src)
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        for &entry in entry_points {
            assert!(
                declared.iter().any(|d| d == entry),
                "{file} does not declare `extern \"C\" __global__ … {entry}(`; found {declared:?}"
            );
        }
    }
}

#[test]
fn v4_vision_kernel_arities_match_the_pins() {
    let mut seen = Vec::new();
    for &(file, _module, _entry_points) in REQUIRED_VISION_KERNELS {
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
    for &(file, module, _) in REQUIRED_VISION_KERNELS {
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

#[test]
fn v4_vision_kernels_are_not_declared_expected_absent() {
    let doc = v4_model_toml();
    let Some(absent) = doc.get("expected_absent").and_then(toml::Value::as_table) else {
        return;
    };
    for &(_, module, entry_points) in REQUIRED_VISION_KERNELS {
        let Some(entries) = absent.get(module).and_then(toml::Value::as_table) else {
            continue;
        };
        for &entry in entry_points {
            assert!(
                !entries.contains_key(entry),
                "{module}::{entry} is declared [expected_absent], but the V4 vision path has NO \
                 fallback for it — a false absence lets a genuinely missing kernel boot"
            );
        }
    }
}

#[test]
fn model_toml_declares_vision() {
    let doc = v4_model_toml();
    assert_eq!(
        doc["model"].get("vision").and_then(toml::Value::as_bool),
        Some(true),
        "deepseek-v4-flash serves Vision-Exp checkpoints; [model] vision must be true"
    );
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
        for &(_, module, _) in REQUIRED_VISION_KERNELS {
            assert!(
                target.modules.iter().any(|(name, _)| *name == module),
                "compiled deepseek-v4-flash target is missing required module {module}"
            );
        }
    }
}
