"""Execute exact effective-gain equivalence and live candidate smoke; not a submission."""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.cpp_extension import load_inline
import compare_decode_public as common

common.torch = torch
CUDA = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
__global__ void gain_bits(const __nv_bfloat16* raw, const __nv_bfloat16* cached, unsigned* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float old = __bfloat162float(__float2bfloat16_rn(1.f + __bfloat162float(raw[i])));
    out[2*i] = __float_as_uint(old);
    out[2*i+1] = __float_as_uint(__bfloat162float(cached[i]));
  }
}
void check(torch::Tensor raw, torch::Tensor cached, torch::Tensor out) {
  int n = raw.numel();
  gain_bits<<<(n+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)raw.data_ptr(), (const __nv_bfloat16*)cached.data_ptr(),
      (unsigned*)out.data_ptr(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="decode_alygg77_bf16.py")
    parser.add_argument("--candidate", default="decode_gain_cached.py")
    parser.add_argument("--output", default="gain-cache-check.json")
    args = parser.parse_args()
    report = dict(status="running", sources={name:common.sha(path) for name,path in
                  (("baseline",args.baseline),("candidate",args.candidate))}, finalizer_cases=[])
    common.save(args.output, report)
    try:
        ext = load_inline(name="experts_v32_gain_equivalence", cpp_sources="#include <torch/extension.h>\nvoid check(torch::Tensor,torch::Tensor,torch::Tensor);",
                          cuda_sources=CUDA, functions=["check"],
                          extra_cuda_cflags=["-O3","-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
        raw = torch.arange(65536,device="cuda",dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        cached = (raw.float() + 1.0).to(torch.bfloat16)
        bits = torch.empty((65536,2),device="cuda",dtype=torch.int32)
        ext.check(raw,cached,bits)
        finite = torch.isfinite(raw.float())
        mismatches = int((bits[finite,0] != bits[finite,1]).sum().item())
        report["exhaustive_gain"] = dict(finite_bf16_patterns=int(finite.sum().item()), mismatches=mismatches,
                                         policy="all finite BF16 inputs, exact FP32 output bits; nonfinite weights outside qualification")
        if mismatches:
            raise RuntimeError("Cached gain differs from native audited scalar expression")
        common.save(args.output, report)
        reference = common.load("reference.py","gain_reference")
        common.reference = reference
        weights = reference.generate_weights(731283)
        before_norm = weights["post_expert_norm"].clone()
        base = common.load(args.baseline,"gain_base")
        candidate = common.load(args.candidate,"gain_candidate")
        started = time.monotonic()
        state = candidate.prepare(weights)
        torch.cuda.synchronize()
        report["candidate_prepare_seconds"] = time.monotonic() - started
        expected_gain = (weights["post_expert_norm"].float()+1.0).to(torch.bfloat16)
        report["prepared_gain"] = dict(
            exact=bool(torch.equal(state["g_postexp"],expected_gain)),
            native_alias=state["ma_w"][7].data_ptr()==state["g_postexp"].data_ptr(),
            raw_checkpoint_unchanged=bool(torch.equal(weights["post_expert_norm"],before_norm)),
            dtype=str(state["g_postexp"].dtype))
        if not all(report["prepared_gain"][key] for key in ("exact","native_alias","raw_checkpoint_unchanged")):
            raise RuntimeError("Prepared gain/state contract failed")
        generator = torch.Generator(device="cuda").manual_seed(772310)
        for tokens in (1,8,16,32,128,256):
            y = torch.randn((tokens,8,4096),device="cuda",generator=generator).to(torch.bfloat16)
            tw = torch.rand((tokens,8),device="cuda",generator=generator)
            ids = torch.arange(8,device="cuda",dtype=torch.int32).repeat(tokens,1)
            ids[:,::3] = -1
            a,b = (torch.empty((tokens,4096),device="cuda",dtype=torch.bfloat16) for _ in range(2))
            for module,gain,out in ((base,before_norm,a),(candidate,state["g_postexp"],b)):
                module._pexp_norm_sum[(tokens,)](y,gain,tw,out,ids,TOPK=8,H=4096,eps=1e-8,BLOCK_H=4096,num_warps=8,num_stages=2)
            equal = bool(torch.equal(a,b))
            report["finalizer_cases"].append(dict(tokens=tokens,bitwise_equal=equal))
            if not equal:
                raise RuntimeError("Marlin finalizer output changed")
        pristine = reference.generate_input(batch=32,q_len=1,ctx=4096,layer="sliding",page=1,spread="agentic",dist="soft",seed=781633)
        private = common.clone_data(pristine)
        cap = common.capture(candidate.kernel_fn,private,state)
        cap["graph"].replay(); torch.cuda.synchronize()
        report["native_graph_smoke"] = common.oracle(private,cap["output"],weights,pristine)
        if not report["native_graph_smoke"]["passed"]:
            raise RuntimeError("Native graph smoke failed public oracle")
        candidate._check_guards()
        report["scratch_guards"] = "passed"
        report["status"] = "complete"
    except BaseException as exc:
        import traceback
        report.update(status="failed",error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        common.save(args.output,report)
        print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
