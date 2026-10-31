#!/usr/bin/env python3
"""Diagnostic only: shared-resident BF16 WGMMA roof, NOT a projection implementation.

Run in /workspace:
  /opt/sglang/bin/python raw_wgmma_bound.py --result large_raw.json
Only this script is required. Cold compile/run estimate: 2--5 minutes (unmeasured).
The default bounded matrix is eight board shapes x seven tiles x two shared-memory
resource envelopes. No global split-K, TMA, persistent scheduler, or real weights.

Each CTA executes the exact full K loop on a repeated synthetic K64 operand tile.
The default orientation matches the approved swapped-operand GEMM: W rows are
WGMMA M, tokens are WGMMA N. Both unreserved (1 stage) and hypothetical three-stage
pipeline shared-memory occupancy are reported; no pipeline work is performed.
All accumulator registers contribute to stored checksums. A separate launch saves
EVERY FP32 accumulator and compares to an exact dyadic synthetic CPU oracle.
Stamped core cycles are not whole-kernel timings. Full timings are external CUDA
events INSIDE a CUDA graph, with a 256-MiB flush before the begin event, no replay
subtraction; a hot-graph control is also reported. Synthetic initialization and
checksum epilogue are included. Real operand traffic/producer instructions and
real BF16 output stores are omitted, so this is only a favorable diagnostic.
A slow result is NOT a proof that every possible production GEMM is slower.

Preregistered hypothesis: 61% -> 75% nominal tensor utilization would remove
18.7% of compute time, before missing traffic/epilogue/occupancy costs. Credit is
ZERO us on every board load until compared with the paired baseline. Root must
retain router and all small-T latencies when forming the six-load board ceiling.
Only build a production candidate after that optimistic ceiling reaches 3%.
Next implementation if gate passes: nonsplit-K, TMA-fed swapped-operand GEMM with
the winning tile and measured resource envelope; consider multicast only for
repeated activation tiles. Neither deep rings nor persistence earns credit here.

Attribution: WGMMA instruction generation, sw128 descriptors, and accumulator
layout adapted from approved work/prior-art/alygg77/kernels/muse_bf16_proj.py,
pinned alygg77 commit 6cae49a639ee6d14b929a5fd4a63c627f96d7dc0; portions Copyright
vjeux, https://github.com/vjeux/okc-2026 (f35a41e), Apache License 2.0; alygg77
contributions Apache License 2.0. https://www.apache.org/licenses/LICENSE-2.0
No Nathen Fernandes routes or external source imports are used. Graph timing
follows the approved local work/bench/d1_cost_ladder.py capture helper.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback

SHAPES = {"qkv_gate": (2304, 8192), "o": (8192, 1024),
          "pooler": (4096, 8192), "unpooler": (8192, 4096)}
CONFIGS = [(bn, wg) for bn in (64, 128, 256) for wg in (1, 2)] + [(32, 1)]

CPP = r"""
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdint.h>
#include <vector>
void raw_launch(int,int,int,int,int,int64_t,int64_t,int64_t,float,int64_t);
std::vector<int64_t> raw_info(int,int,int);
"""

CUDA = r"""
// Derived helpers: Copyright vjeux / alygg77, Apache-2.0; see script attribution.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <vector>
#include <stdexcept>
#include <string>
#define D __device__ __forceinline__
D uint64_t desc(uint32_t p) {
  return uint64_t((p & 0x3ffff) >> 4) | (uint64_t(1) << 16)
       | (uint64_t(1024 >> 4) << 32) | (uint64_t(1) << 62);
}
D uint64_t timer_ns() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory"); return t; }
D uint64_t cycles() { uint64_t t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory"); return t; }
D unsigned smid() { unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s; }
@MMA@

template<int BN, int WG, int MODE>
__global__ __launch_bounds__(WG*128) void raw_kernel(
    float* sums, float* full, int64_t* stamps, int K, float scale) {
  constexpr int BM = WG*64, NA = BN/2;
  extern __shared__ __align__(1024) unsigned char storage[];
  auto* values = reinterpret_cast<__nv_bfloat16*>(storage);
  const int tid = threadIdx.x, wg = tid/128, lane = tid%32;
  // Physical SW128 layout: XOR each 16-byte K segment with row modulo eight.
  // Constant dyadic fixture repeats every K64; no global matrix reads at all.
  for (int i=tid; i<(BM+BN)*64; i+=WG*128) {
    const int row = i/64, k = i%64;
    const int phys = row*64 + (k ^ ((row%8)*8));
    const float v = row < BM
      ? scale*(1.f + float((row%7)-3)/8.f + float(k%4)/16.f)
      : float(((row-BM)%5)-2)/8.f + float(k%8)/16.f;
    values[phys] = __float2bfloat16_rn(v);
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  const uint32_t base = uint32_t(__cvta_generic_to_shared(storage));
  uint64_t da[4], db[4];
#pragma unroll
  for (int j=0;j<4;++j) {
    da[j] = desc(base + wg*64*128 + j*32);
    db[j] = desc(base + BM*128 + j*32);
  }
  float acc[NA];
  uint64_t c0=0,t0=0,c1=0,t1=0;
  if constexpr(MODE==1) { t0=timer_ns(); c0=cycles(); }
  // Four K16 instructions per group, bounded at two committed groups in flight.
  // No ordinary accumulator accesses until wait_group 0 finishes all operations.
  @CALL@
  if constexpr(MODE==1) { c1=cycles(); t1=timer_ns(); }
  float sum=0.f;
#pragma unroll
  for (int q=0;q<NA;++q) {
    sum += acc[q];
    if constexpr(MODE==2) full[(size_t(blockIdx.x)*WG*128+tid)*NA+q]=acc[q];
  }
  sums[size_t(blockIdx.x)*WG*128+tid]=sum;
  if constexpr(MODE==1) {
    if (tid%128==0) {
      int64_t* p=stamps+(size_t(blockIdx.x)*WG+wg)*5;
      p[0]=c0; p[1]=c1; p[2]=t0; p[3]=t1; p[4]=smid();
    }
  }
}

void check(cudaError_t e) {
  if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template<int BN,int WG,int MODE>
void launch_t(int blocks,int K,int smem,int64_t sums,int64_t full,int64_t stamps,float scale,int64_t stream) {
  raw_kernel<BN,WG,MODE><<<blocks,WG*128,smem,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<float*>(sums),reinterpret_cast<float*>(full),reinterpret_cast<int64_t*>(stamps),K,scale);
  check(cudaGetLastError());
}
template<int BN,int WG,int MODE>
std::vector<int64_t> info_t(int smem) {
  check(cudaFuncSetAttribute(raw_kernel<BN,WG,MODE>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
  cudaFuncAttributes a; check(cudaFuncGetAttributes(&a,raw_kernel<BN,WG,MODE>));
  int active=0; check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,raw_kernel<BN,WG,MODE>,WG*128,smem));
  return {a.numRegs,int64_t(a.localSizeBytes),int64_t(a.sharedSizeBytes),
          a.maxThreadsPerBlock,a.binaryVersion,a.ptxVersion,active,a.maxDynamicSharedSizeBytes};
}
void raw_launch(int cfg,int mode,int blocks,int K,int smem,int64_t sums,int64_t full,int64_t stamps,float scale,int64_t stream) {
  @LAUNCH@
  throw std::runtime_error("unknown raw launch configuration");
}
std::vector<int64_t> raw_info(int cfg,int mode,int smem) {
  @INFO@
  throw std::runtime_error("unknown raw resource configuration");
}
"""


def source():
    helpers = []
    for bn in sorted({bn for bn, _ in CONFIGS}):
        na = bn // 2
        regs = ",".join(f"%{q}" for q in range(na))
        constraints = ",".join(f'"=f"(d[{q}])' for q in range(na))
        ops = ["{", ".reg .u32 it;", ".reg .pred again;"]
        ops += [f"mov.f32 %{q}, 0f00000000;" for q in range(na)]
        ops += ["wgmma.fence.sync.aligned;", "mov.u32 it, 0;", "raw_loop:"]
        ops += [f"wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 "
                f"{{{regs}}}, %{na+j}, %{na+4+j}, 1, 1, 1, 0, 0;" for j in range(4)]
        ops += ["wgmma.commit_group.sync.aligned;", "wgmma.wait_group.sync.aligned 1;",
                "add.u32 it, it, 1;", f"setp.lt.u32 again, it, %{na+8};",
                "@again bra raw_loop;", "wgmma.wait_group.sync.aligned 0;", "}"]
        text = "\\n".join(ops)
        inputs = ",".join([f'"l"(a[{j}])' for j in range(4)] +
                          [f'"l"(b[{j}])' for j in range(4)] + ['"r"(steps)'])
        helpers.append(f'''D void mma{bn}(float* d,const uint64_t* a,const uint64_t* b,int steps) {{
  asm volatile("{text}" : {constraints} : {inputs} : "memory");
}}''')
    launches, infos = [], []
    for cfg, (bn, wg) in enumerate(CONFIGS):
        for mode in range(3):
            cond = f"if(cfg=={cfg} && mode=={mode})"
            launches.append(f"{cond} {{ launch_t<{bn},{wg},{mode}>(blocks,K,smem,sums,full,stamps,scale,stream); return; }}")
            infos.append(f"{cond} return info_t<{bn},{wg},{mode}>(smem);")
    call = " ".join(f"if constexpr(BN=={bn}) mma{bn}(acc,da,db,K/64);" for bn in sorted({bn for bn, _ in CONFIGS}))
    return CUDA.replace("@MMA@", "\n".join(helpers)).replace("@CALL@", call).replace(
        "@LAUNCH@", "\n".join(launches)).replace("@INFO@", "\n".join(infos))


def smi():
    fields = "name,uuid,driver_version,clocks.sm,clocks.max.sm,clocks.mem,power.draw,power.limit,temperature.gpu"
    result = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=False)
    return {"fields": fields.split(","), "csv": result.stdout.strip(),
            "stderr": result.stderr.strip(), "returncode": result.returncode}


def save(path, report):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def describe(values):
    values = [float(v) for v in values]
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def capture(torch, flush, call):
    # Matched to parent's baseline external-event timing, with no host-gap cost.
    begin = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
        sink = flush.sum() if flush is not None else None
        begin.record()
        end.record()
    current.wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        if flush is not None:
            sink = flush.sum()
        begin.record()
        call()
        end.record()
    current.wait_stream(stream)
    for _ in range(3):
        graph.replay()
        end.synchronize()
        elapsed = begin.elapsed_time(end)
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError("invalid external CUDA graph timing")
    return graph, begin, end, sink


def samples(captured, count):
    graph, begin, end, _ = captured
    result = []
    for _ in range(count):
        graph.replay()
        end.synchronize()
        us = begin.elapsed_time(end) * 1000
        if not math.isfinite(us) or us <= 0:
            raise RuntimeError("invalid CUDA graph duration")
        result.append(us)
    return result


def expected_accumulators(torch, bn, wg, k, scale):
    # CPU double matmul over one K64 tile, then map WGMMA register coordinates.
    rows = torch.arange(wg*64, dtype=torch.float64)[:, None]
    cols = torch.arange(bn, dtype=torch.float64)[:, None]
    ks = torch.arange(64, dtype=torch.float64)[None, :]
    a = scale*(1 + (rows.remainder(7)-3)/8 + ks.remainder(4)/16)
    b = (cols.remainder(5)-2)/8 + ks.remainder(8)/16
    matrix = (a @ b.T) * (k//64)
    tid = torch.arange(wg*128)[:, None]
    q = torch.arange(bn//2)[None, :]
    r = (tid//128)*64 + ((tid%128)//32)*16 + (tid%32)//4 + ((q%4)//2)*8
    c = (q//4)*8 + (tid%4)*2 + q%2
    return matrix[r, c].float()


def run(args, report):
    import torch
    from torch.utils.cpp_extension import load_inline
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("sm_90a Hopper device required")
    props = torch.cuda.get_device_properties(0)
    report["device"] = {"name": props.name, "sm_count": props.multi_processor_count,
                        "total_memory": props.total_memory, "capability": [props.major, props.minor],
                        "torch": torch.__version__, "cuda": torch.version.cuda, "smi_before": smi()}
    report["arguments"] = vars(args)
    generated = source()
    source_file = Path(args.result).with_name(Path(args.result).stem + "_generated.cu")
    source_file.write_text(generated)
    report["artifact_files"].append(source_file.name)
    report["source_sha256"] = hashlib.sha256(generated.encode()).hexdigest()
    report["status"] = "compiling"
    save(args.result, report)
    print(json.dumps({"phase": "compile", "source_sha256": report["source_sha256"]}), flush=True)
    os.environ.setdefault("MAX_JOBS", "2")
    started = time.perf_counter()
    ext = load_inline(name="dense_raw_" + report["source_sha256"][:16], cpp_sources=CPP,
                      cuda_sources=generated, functions=["raw_launch", "raw_info"],
                      no_implicit_headers=True, with_pytorch_error_handling=False, verbose=True,
                      extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v"])
    report["compile_wall_seconds"] = time.perf_counter() - started
    report["extension_file"] = ext.__file__
    report["status"] = "measuring"
    save(args.result, report)
    flush = torch.ones(256*1024*1024//4, dtype=torch.float32, device="cuda")
    sm_count = props.multi_processor_count
    resource_keys = ("registers_per_thread", "local_bytes_per_thread", "static_shared_bytes",
                     "max_threads", "binary_version", "ptx_version", "resident_ctas_per_sm", "max_dynamic_shared_bytes")
    for m in (128, 256):
        for layer, (n, k) in SHAPES.items():
            for cfg, (bn, wg) in enumerate(CONFIGS):
                bm = 64*wg
                rows, cols = (n, m) if args.orientation == "weights_rows" else (m, n)
                grid_m, grid_n = math.ceil(rows/bm), math.ceil(cols/bn)
                blocks = grid_m*grid_n
                threads = wg*128
                useful_flops = 2*m*n*k
                issued_flops = 2*blocks*bm*bn*k
                checksums = torch.empty((blocks, threads), dtype=torch.float32, device="cuda")
                full = torch.empty((blocks, threads, bn//2), dtype=torch.float32, device="cuda")
                stamps = torch.empty((blocks, wg, 5), dtype=torch.int64, device="cuda")
                for stages in args.smem_stages:
                    smem = stages*(bm+bn)*128
                    row = {"layer": layer, "m": m, "n": n, "k": k, "cfg": cfg,
                           "tile_rows": bm, "tile_cols": bn, "warpgroups": wg,
                           "orientation": args.orientation, "cta_grid": [grid_m, grid_n],
                           "ctas": blocks, "k64_steps": k//64, "k16_instructions_per_wg": k//16,
                           "shared_memory_resource_stages": stages, "dynamic_shared_bytes": smem,
                           "useful_flops": useful_flops, "padded_issued_flops": issued_flops,
                           "padding_fraction": 1-useful_flops/issued_flops,
                           "nominal_989tf_useful_floor_us": useful_flops/989e6,
                           "nominal_989tf_issued_floor_us": issued_flops/989e6,
                           "nominal_peak_tflops": 989.0, "nominal_peak_clock_mhz": 1830.0}
                    report["rows"].append(row)
                    def launch(mode=0, scale=1.0):
                        ext.raw_launch(cfg, mode, blocks, k, smem, checksums.data_ptr(), full.data_ptr(),
                                       stamps.data_ptr(), scale, torch.cuda.current_stream().cuda_stream)
                    try:
                        row["resources"] = {str(mode): dict(zip(resource_keys, ext.raw_info(cfg, mode, smem)))
                                            for mode in range(3)}
                        occupancy = row["resources"]["0"]["resident_ctas_per_sm"]
                        if occupancy < 1:
                            raise RuntimeError("zero CTA occupancy")
                        row["residency_waves"] = math.ceil(blocks/(sm_count*occupancy))
                        row["one_cta_per_sm_waves"] = math.ceil(blocks/sm_count)
                        row["last_residency_wave_fill"] = ((blocks-1)%(sm_count*occupancy)+1)/(sm_count*occupancy)
                        validation = []
                        for scale in (1.0, 0.5):
                            expected = expected_accumulators(torch, bn, wg, k, scale)
                            launch(2, scale)
                            actual = full.cpu()
                            error = (actual - expected[None]).abs().max().item()
                            sum_error = (checksums.cpu()-expected.sum(-1)[None]).abs().max().item()
                            validation.append({"scale": scale, "max_abs_accumulator_error": error,
                                               "max_abs_checksum_error": sum_error,
                                               "checksum_sum": float(checksums.double().sum().item())})
                            if error != 0 or sum_error != 0:
                                raise RuntimeError(f"synthetic accumulator/checksum mismatch: {validation[-1]}")
                        row["validation"] = validation
                        cold = capture(torch, flush, launch)
                        cold_values = samples(cold, args.samples)
                        expected_sum = expected_accumulators(torch, bn, wg, k, 1.0).sum(-1)
                        if not torch.equal(checksums.cpu(), expected_sum[None].expand(blocks, -1)):
                            raise RuntimeError("cold graph replay checksum mismatch")
                        del cold
                        hot = capture(torch, None, launch)
                        hot_values = samples(hot, args.samples)
                        if not torch.equal(checksums.cpu(), expected_sum[None].expand(blocks, -1)):
                            raise RuntimeError("hot graph replay checksum mismatch")
                        del hot
                        row["cold_graph_us_samples"] = cold_values
                        row["hot_graph_us_samples"] = hot_values
                        row["cold_graph_us"] = describe(cold_values)
                        row["hot_graph_us"] = describe(hot_values)
                        cold_median = statistics.median(cold_values)
                        row["full_kernel_useful_tflops"] = useful_flops/cold_median/1e6
                        row["full_kernel_issued_tflops"] = issued_flops/cold_median/1e6
                        instrumented = capture(torch, flush, lambda: launch(1))
                        stamp_timings, stamp_raw = [], []
                        for _ in range(args.stamp_samples):
                            stamp_timings.extend(samples(instrumented, 1))
                            stamp_raw.append(stamps.cpu().tolist())
                        if not torch.equal(checksums.cpu(), expected_sum[None].expand(blocks, -1)):
                            raise RuntimeError("stamped graph replay checksum mismatch")
                        del instrumented
                        row["stamped_graph_us_samples"] = stamp_timings
                        row["stamps_raw"] = stamp_raw
                        row["stamp_columns"] = ["clock64_begin", "clock64_end", "globaltimer_begin_ns", "globaltimer_end_ns", "smid"]
                        durations, ticks, clocks, rates, spans, active_sms = [], [], [], [], [], []
                        cta_ticks, cta_us, cta_rates = [], [], []
                        for sample in stamp_raw:
                            flat = [wgstamp for cta in sample for wgstamp in cta]
                            spans.append((max(s[3] for s in flat)-min(s[2] for s in flat))/1000)
                            active_sms.append(len({s[4] for s in flat}))
                            for cta in sample:
                                core_cycles = max(s[1] for s in cta)-min(s[0] for s in cta)
                                core_ns = max(s[3] for s in cta)-min(s[2] for s in cta)
                                if core_cycles <= 0 or core_ns <= 0:
                                    raise RuntimeError("invalid CTA core stamp span")
                                cta_ticks.append(core_cycles)
                                cta_us.append(core_ns/1000)
                                cta_rates.append((2*bm*bn*k)/core_cycles)
                            for s in flat:
                                cyc, ns = s[1]-s[0], s[3]-s[2]
                                if cyc <= 0 or ns <= 0:
                                    raise RuntimeError("invalid clock/globaltimer stamp delta")
                                ticks.append(cyc)
                                durations.append(ns/1000)
                                clocks.append(cyc/ns*1000)
                                rates.append((2*64*bn*k)/cyc)
                        row["per_wg_core_cycles"] = describe(ticks)
                        row["per_wg_core_us"] = describe(durations)
                        row["per_wg_issued_flops_per_cycle"] = describe(rates)
                        row["per_cta_core_cycles"] = describe(cta_ticks)
                        row["per_cta_core_us"] = describe(cta_us)
                        row["per_cta_issued_flops_per_cycle"] = describe(cta_rates)
                        row["in_kernel_clock_mhz"] = describe(clocks)
                        row["full_grid_core_span_us"] = describe(spans)
                        row["observed_active_sms_samples"] = active_sms
                        row["clock_corrected_nominal_peak_tflops"] = 989*statistics.median(clocks)/1830
                        row["issued_fraction_of_clock_corrected_nominal"] = row["full_kernel_issued_tflops"]/row["clock_corrected_nominal_peak_tflops"]
                        row["smi_after"] = smi()
                        row["status"] = "ok"
                    except Exception:
                        row["status"] = "error"
                        row["error"] = traceback.format_exc()
                        save(args.result, report)
                        raise
                    save(args.result, report)
                    print(json.dumps({key: row[key] for key in ("layer", "m", "cfg", "shared_memory_resource_stages",
                          "cold_graph_us", "full_kernel_useful_tflops", "in_kernel_clock_mhz", "status")}), flush=True)
                del full, checksums, stamps
    best = []
    for m in (128, 256):
        for layer in SHAPES:
            eligible = [r for r in report["rows"] if r["m"]==m and r["layer"]==layer and r["status"]=="ok"]
            winner = min(eligible, key=lambda r: r["cold_graph_us"]["median"])
            best.append({"m": m, "layer": layer, "raw_favorable_us": winner["cold_graph_us"]["median"],
                         "cfg": winner["cfg"], "shared_memory_resource_stages": winner["shared_memory_resource_stages"],
                         "useful_tflops": winner["full_kernel_useful_tflops"],
                         "production_savings_credit_us": 0})
    report["best_raw_per_shape"] = best
    report["device"]["smi_after"] = smi()
    report["status"] = "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result", default="large_raw.json")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--stamp-samples", type=int, default=3)
    parser.add_argument("--smem-stages", type=int, nargs="+", choices=(1, 3), default=[1, 3])
    parser.add_argument("--orientation", choices=("weights_rows", "tokens_rows"), default="weights_rows")
    args = parser.parse_args()
    if args.samples < 3 or args.stamp_samples < 1:
        parser.error("require samples>=3 and stamp-samples>=1")
    report = {"schema_version": 1, "diagnostic_only": True, "status": "starting", "rows": [],
              "artifact_files": [], "board_savings_credit_us": {str(m): 0 for m in (1,16,32,8,128,256)},
              "limits": ["synthetic repeated K64 operands, not generated board weights",
                         "no TMA/HBM operand traffic, producer, persistent or multicast protocol",
                         "checksum epilogue instead of real BF16 matrix output",
                         "shared reservation emulates capacity only, not pipeline instructions",
                         "clock correction assumes 989 TF/s at 1830 MHz; stamps report measured clocks separately",
                         "per-WG rates are not additive when warpgroups/CTAs share an SM",
                         "favorable raw timing is a gate, not a production speedup or universal lower bound"]}
    started = time.perf_counter()
    try:
        run(args, report)
    except Exception:
        report["status"] = "error"
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["total_wall_seconds"] = time.perf_counter()-started
        save(args.result, report)
        print(json.dumps({"status": report["status"], "result": args.result,
                          "completed_rows": sum(r.get("status")=="ok" for r in report["rows"])}), flush=True)


if __name__ == "__main__":
    main()
