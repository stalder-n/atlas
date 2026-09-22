# SPDX-License-Identifier: AGPL-3.0-only
"""CUDA correctness: pytest -q tests/gpu/test_v4_fp8_prefill.py.

Uses independently dequantized FP64 weights, BF16 inputs, and the existing
w8a16_microtest cosine gate. This does not measure or claim performance.
"""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(scope="module")
def kernels():
    from torch.utils.cpp_extension import load_inline

    common = Path(__file__).resolve().parents[2] / "kernels/gb10/common"
    cuda = '\n'.join((common / name).read_text() for name in (
        "w8a16_gemm_pipelined.cu", "w8a16_gemv_batch4.cu"))
    cuda += r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
void run(torch::Tensor a, torch::Tensor b, torch::Tensor scales,
         torch::Tensor out, int64_t groups) {
    const auto stream = c10::cuda::getCurrentCUDAStream();
    const unsigned m=a.size(0), k=b.size(1), n=b.size(0)/groups;
    auto ap=reinterpret_cast<__nv_bfloat16*>(a.data_ptr());
    auto bp=reinterpret_cast<unsigned char*>(b.data_ptr());
    auto cp=reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    if(groups==1) {
        w8a16_gemm_pipelined<<<dim3((n+31)/32,(m+127)/128),256,0,stream.stream()>>>(
            ap,bp,scales.data_ptr<float>(),cp,m,n,k);
    } else for(unsigned t=0;t<m;t+=16) for(unsigned g=0;g<groups;g++) {
        w8a16_gemv_batch16_strided<<<dim3((n+3)/4),256,0,stream.stream()>>>(
            ap+(t*groups+g)*k,bp+g*n*k,scales.data_ptr<float>()+g*(n/128)*(k/128),
            cp+(t*groups+g)*n,min(16u,m-t),n,k,groups*k,groups*n);
    }
}
'''
    return load_inline(
        name="atlas_v4_fp8_prefill_correctness",
        cpp_sources='''#include <torch/extension.h>
void run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);
''', cuda_sources=cuda, functions=["run"], extra_include_paths=[str(common)],
        extra_cuda_cflags=["--fmad=false"])


# Each contiguous projection, and the grouped wo_a projection. Real V4 widths;
# groups=8 corresponds to 64 heads of 512 elements split over eight groups.
@pytest.mark.parametrize("name,n,k,groups", [
    ("wq_a",1024,4096,1), ("wq_b",32768,1024,1),
    ("wkv",512,4096,1), ("wo_b",4096,8192,1), ("wo_a",1024,4096,8),
])
@pytest.mark.parametrize("m", [1,15,16,17,31,32,33,127,128,129])
def test_projection_reference_and_sentinels(kernels, name, n, k, groups, m):
    torch.manual_seed(514)
    a = torch.randn(m, groups*k, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(groups*n, k, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand(groups*n//128, k//128, device="cuda") * 0.125 + 0.001
    guarded = torch.full((m*groups*n+64,), 123, device="cuda", dtype=torch.bfloat16)
    out = guarded[32:-32].view(m, groups*n)
    kernels.run(a, weights, scales, out, groups)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all(), name
    assert (guarded[:32] == 123).all() and (guarded[-32:] == 123).all()
    for g in range(groups):
        # Independent E4M3 decode (PyTorch), expanded scales, FP64 dense GEMM.
        w = weights[g*n:(g+1)*n].double()
        s = scales[g*n//128:(g+1)*n//128].double()
        w *= s.repeat_interleave(128, 0).repeat_interleave(128, 1)
        expected = a[:, g*k:(g+1)*k].double() @ w.T
        actual = out[:, g*n:(g+1)*n].double()
        cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
        assert cosine >= 0.9995, (name, g, cosine.item())
        relative = torch.linalg.vector_norm(actual-expected) / torch.linalg.vector_norm(expected)
        assert relative < 0.01, (name, g, relative.item())
