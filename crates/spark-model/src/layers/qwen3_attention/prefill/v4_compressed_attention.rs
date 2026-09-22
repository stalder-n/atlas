// SPDX-License-Identifier: AGPL-3.0-only

//! Pre-launch selection keeps missing optional kernels and unsupported shapes
//! on the scalar ABI, without catching execution errors after dispatch.
use spark_runtime::gpu::KernelHandle;

pub(super) fn select_kernel(
    enabled: bool,
    scalar: KernelHandle,
    tensor_core: KernelHandle,
    head_dim: u32,
    query_heads: u32,
    kv_heads: u32,
    ratio: u32,
) -> (KernelHandle, u32) {
    if enabled
        && tensor_core.0 != 0
        && head_dim == 512
        && kv_heads > 0
        && query_heads > 0
        && query_heads.is_multiple_of(kv_heads)
        && matches!(ratio, 4 | 128)
    {
        (tensor_core, 32)
    } else {
        (scalar, 16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_kernel_and_supported_shape_are_both_required() {
        let scalar = KernelHandle(11);
        let tc = KernelHandle(22);
        for ratio in [4, 128] {
            let (handle, rows) = select_kernel(true, scalar, tc, 512, 8, 2, ratio);
            assert_eq!((handle.0, rows), (22, 32));
        }
        for (enabled, kernel, dim, q, kv, ratio) in [
            (false, tc, 512, 8, 2, 4),
            (true, KernelHandle(0), 512, 8, 2, 4),
            (true, tc, 576, 8, 2, 4),
            (true, tc, 512, 8, 0, 4),
            (true, tc, 512, 0, 2, 4),
            (true, tc, 512, 7, 2, 4),
            (true, tc, 512, 8, 2, 0),
            (true, tc, 512, 8, 2, 16),
        ] {
            let (handle, rows) = select_kernel(enabled, scalar, kernel, dim, q, kv, ratio);
            assert_eq!((handle.0, rows), (11, 16));
        }
    }
}
