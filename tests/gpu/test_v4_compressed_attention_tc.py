# SPDX-License-Identifier: AGPL-3.0-only
"""Run on a CUDA Spark: pytest -q tests/gpu/test_v4_compressed_attention_tc.py.

Compiles the production scalar and TC kernels; reference uses independent
float64 dense attention and checkpoint-shaped BF16 inputs. No timing claims.
"""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(scope="module")
def kernels():
    from torch.utils.cpp_extension import load_inline

    root = Path(__file__).resolve().parents[2]
    source = root / "kernels/gb10/deepseek-v4-flash/nvfp4"
    cuda = '\n'.join((source / name).read_text() for name in (
        "prefill_attn_compressed.cu", "prefill_attn_compressed_tc.cu"))
    cuda += r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
void run(torch::Tensor q, torch::Tensor k, torch::Tensor v,
         torch::Tensor kc, torch::Tensor vc, torch::Tensor sink,
         torch::Tensor output, int64_t ratio, int64_t window, bool tc) {
    const auto stream = c10::cuda::getCurrentCUDAStream();
    const unsigned s=q.size(0), h=q.size(1), kh=k.size(1), d=q.size(2);
    const dim3 grid(h, (s+(tc?31:15))/(tc?32:16));
    auto ptr=[](torch::Tensor t){return reinterpret_cast<__nv_bfloat16*>(t.data_ptr());};
    if(tc) prefill_attn_compressed_tc<<<grid,128,0,stream.stream()>>>(
        ptr(q),ptr(k),ptr(v),ptr(kc),ptr(vc),sink.data_ptr<float>(),ptr(output),
        s,h,kh,d,kc.size(0),ratio,window,1.0f/sqrtf(float(d)));
    else prefill_attn_compressed<<<grid,128,0,stream.stream()>>>(
        ptr(q),ptr(k),ptr(v),ptr(kc),ptr(vc),sink.data_ptr<float>(),ptr(output),
        s,h,kh,d,kc.size(0),ratio,window,1.0f/sqrtf(float(d)));
}
'''
    return load_inline(
        name="atlas_v4_compressed_tc_correctness", cpp_sources='''
#include <torch/extension.h>
void run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
         torch::Tensor,torch::Tensor,int64_t,int64_t,bool);
''', cuda_sources=cuda, functions=["run"], extra_cuda_cflags=["--fmad=false"])


def reference(q, k, v, kc, vc, sink, ratio, window):
    s, heads, d = q.shape
    # Repeat each KV head over its query-head group, independently of kernel indexing.
    k = k.double().repeat_interleave(heads // k.shape[1], dim=1)
    v = v.double().repeat_interleave(heads // v.shape[1], dim=1)
    keys = torch.cat((k, kc.double()[:, None, :].expand(-1, heads, -1)))
    vals = torch.cat((v, vc.double()[:, None, :].expand(-1, heads, -1)))
    scores = torch.einsum("qhd,khd->hqk", q.double(), keys) / d**0.5
    query = torch.arange(s, device=q.device)[:, None]
    raw = torch.arange(s, device=q.device)[None, :]
    visible = raw <= query
    if window:
        visible &= raw > query - window
    comp = (torch.arange(len(kc), device=q.device)[None, :] + 1) * ratio
    visible = torch.cat((visible, comp <= query + 1), dim=1)
    scores.masked_fill_(~visible[None], -torch.inf)
    scores = torch.cat((scores, sink.double()[:, None, None].expand(-1, s, 1)), dim=-1)
    probs = scores.softmax(-1)[..., :-1]
    return torch.einsum("hqk,khd->qhd", probs, vals)


@pytest.mark.parametrize("s,ratio,window", [
    (1,4,128),(15,4,16),(16,4,16),(17,4,16),(31,4,16),(32,4,16),
    (33,4,16),(127,128,128),(128,128,128),(129,128,128),
    (257,4,128),(33,4,0),
])
@pytest.mark.parametrize("sink_value", [-1000.0, 0.0, 1000.0])
def test_reference_scalar_and_sentinels(kernels, s, ratio, window, sink_value):
    torch.manual_seed(943)
    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    q, k, v = rand(s, 4, 512), rand(s, 2, 512), rand(s, 2, 512)
    kc, vc = rand(s // ratio, 512), rand(s // ratio, 512)
    sink = torch.full((4,), sink_value, device="cuda", dtype=torch.float32)
    expected = reference(q, k, v, kc, vc, sink, ratio, window)
    outputs = []
    for tc in (False, True):
        guarded = torch.full((q.numel() + 64,), 123, device="cuda", dtype=q.dtype)
        out = guarded[32:-32].view_as(q)
        kernels.run(q, k, v, kc, vc, sink, out, ratio, window, tc)
        torch.cuda.synchronize()
        assert torch.isfinite(out).all()
        assert (guarded[:32] == 123).all() and (guarded[-32:] == 123).all()
        torch.testing.assert_close(out.double(), expected, atol=0.016, rtol=0.016)
        outputs.append(out.clone())
    torch.testing.assert_close(outputs[0], outputs[1], atol=0.016, rtol=0.016)
    # Future raw and compressed rows cannot change the first half's result.
    cut = s // 2
    if cut:
        k[cut:] = 100
        v[cut:] = -100
        kc[cut // ratio:] = 100
        vc[cut // ratio:] = -100
        kernels.run(q, k, v, kc, vc, sink, out, ratio, window, True)
        torch.testing.assert_close(out[:cut], outputs[1][:cut], atol=0, rtol=0)
