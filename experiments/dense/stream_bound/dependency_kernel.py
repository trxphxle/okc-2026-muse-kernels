# SPDX-License-Identifier: Apache-2.0
"""Dependency-aware D1/V1 byte ceiling; deliberately NOT a production kernel.

The adjacent CUDA source retains the approved vjeux/alygg77 TMA attribution.
Only the shared ABI enum is generated; dependency_abi.FIELDS and FLAG are the
single source of positional indices. All heavy GEMM/attention/expert arithmetic
is omitted, but real frozen producer rows are copied only after their matching
full-K weight chunks are consumed. The actual new-KV stores, B1's two sequential
BF16-boundary RMS reductions, pooler normalization, top8, active expert mask and
repaired sliced B2 formula execute inside the measured kernel.

Raw API
-------
build() -> extension
extension.dep_info() -> int64 resource vector, names in INFO_FIELDS
extension.dep_launch(params.data_ptr(), mode, cuda_stream)

Call build()/dep_info() on the launch thread and current CUDA device BEFORE
capture. dep_info sets the opt-in shared-memory attribute, checks cooperative
launch support and exactly one resident CTA/SM, then caches host launch geometry
for that thread/device. dep_launch does not allocate, synchronize, inspect GPU
values or modify module globals. It uses cudaLaunchCooperativeKernel with one
256-thread CTA per SM (132 on H200), guaranteeing all CTA waiters can progress.
Do not share a mutable params/flags/live-buffer plan between concurrent streams.
The planner initially zeros FLAGS/ARRIVALS. Producers release distinct
cache-line-padded epochs and one warp collects them; no contended counter RMW
or external reset is needed between graph replays.

Modes and bounded prefetch
--------------------------
0: strict dependencies, four 32-KiB current-phase TMA slots.
1: same dependency chain, with at most TWO constant-weight chunks per CTA held
   in slots0/1: O after local QKV, router/pooler after local O, unpool after local
   router/pooler. Intervening work uses only slots2/3. Held chunks are awaited
   and consumed at the proper gate, then rejoin the four-slot ring; never
   discarded/reloaded. No KV issue precedes KV_WRITTEN, and no dynamic expert
   bank descriptor/payload is selected before ROUTES_READY.
2/3: exactly modes0/1 plus per-word uint32 XOR, timestamps and optional delayed
     ready publication. These timings are validation overhead, not the ceiling.
4: feature-off free-oracle FLAT_CHUNKS/FLAT_COUNT reader in the SAME runtime
   kernel, with the same compiled registers/shared/local memory. No flags or
   dependent live outputs advance. All nine sink rows are zeroed; phase0 records
   the full flat byte stream, XOR=0. Compare this with old stream_flat's 32-thread
   control to expose edit/resource overhead, not as a dependency-aware result.

Validation outputs
------------------
SINK[blocks,9,2] = payload bytes, uint32 XOR (XOR is zero in timed modes).
Per-phase fixed chunks and the selected expert bank are one concatenated list,
striped at indices b::blocks. Phase6's appended list enumerates active local
experts in ascending bit order; each bank expert contributes EXP_CHUNKS_PER
chunks. Each physical payload chunk is counted once; oracle activation reads,
new-KV writes and descriptor/flag/output traffic are additional real traffic,
not counted as model-weight/KV payload. The planner discloses metadata bytes.

STAMPS[blocks,9,3] = first issue, last completed TMA, consumer enter, globaltimer
nanoseconds. Empty local phases have zero issue/completion but a consumer enter.
A prefetched phase's first issue can precede the producer flag, but its consumer
enter and copies cannot. Last completion brackets the final TMA wait, before the
final matching oracle-row copy; READY_TIMES[10] follows all producer stores and
precedes their release-publication instruction. Delay stage is a ReadyIndex,
not a phase: DELAY_STAGE >= 0 delays that release by 20us in modes2/3 ONLY.
READY_TIMES/STAMPS are untouched in modes0/1/4. Inspect them after completion.

ERROR is cleared inside startup and ORs ERROR_BITS on invalid input/protocol.
Descriptor address/size/offset guards run in modes2/3 only; timed modes trust
the planner's validated immutable descriptors, just like the old flat control.
Early descriptor errors are CTA-local until after QKV's startup join, avoiding
a race against ERROR initialization without adding a grid startup barrier.
Valid repeated graphs require ERROR==0, exact per-phase byte sums/XOR and IDs/
mask/new-cache bytes; compare O/POOL/SCORES/UN copies exactly, and H/HR/CARRY/
TAIL/XM/OUT/SOUT to actual baseline oracles with the planner's FP32/BF16 envelope.
B1 preserves repaired FMA order and the original 1024-logical-thread reduction
mapping within one CTA/token. B2 owns one 1024-output tile/CTA, repeats its two
1024-wide head reductions and uses B1's >=1024 carry squared tail. B1/B2 and
pooler norm retain BF16 round boundaries; no FP32-router activation rounding.

Measured resource qualification belongs to the root H200 harness: inspect
num_regs/local_bytes_per_thread (spills), cooperative residency, mode4 edit
overhead, delayed-publication ordering, repeated graphs and cold-L2 NCU bytes.
No GPU build or verification is performed by importing this module.
"""

from functools import lru_cache
import hashlib
from pathlib import Path

from dependency_abi import FIELDS, cpp_enum

INFO_FIELDS = (
    "num_regs",
    "static_smem_bytes",
    "local_bytes_per_thread",
    "max_threads_per_block",
    "dynamic_smem_bytes",
    "threads_per_block",
    "active_blocks_per_sm",
    "device_sm_count",
    "stage_bytes",
    "ring_slots",
    "cooperative_launch_supported",
    "launch_blocks",
    "param_count",
    "validation_delay_ns",
)
SINK_FIELDS = ("completed_payload_bytes", "uint32_xor")
STAMP_FIELDS = ("first_tma_issue_ns", "last_tma_completion_ns", "consumer_enter_ns")
ERROR_BITS = {
    1: "invalid scalar parameters or BLOCKS differs from cooperative grid",
    2: "invalid TMA chunk address, byte count or nonzero offset (validation modes)",
    4: "dense chunk does not cover aligned complete rows of the expected weight",
    8: "duplicate producer readiness publication in one invocation",
    16: "ready epoch advanced past the invocation epoch",
    32: "nonfinite route score (validation modes)",
    64: "local_start outside global expert IDs 0..255",
}
CPP = r"""
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <vector>
void dep_launch(int64_t param_ptr, int mode, int64_t stream);
std::vector<int64_t> dep_info();
"""
CPP_FLAGS = ("-O3", "-std=c++20")
CUDA_FLAGS = (
    "-O3",
    "-std=c++20",
    "-gencode=arch=compute_90a,code=sm_90a",
    "-lineinfo",
    "--ptxas-options=-v,--warn-on-spills",
)
EXPORTS = ("dep_launch", "dep_info")


@lru_cache(maxsize=1)
def build():
    """Build and initialize on the target current CUDA device before capture."""
    from torch.utils.cpp_extension import load_inline

    abi = cpp_enum() + f"static_assert(FLAT_COUNT + 1 == {len(FIELDS)});\n"
    source = abi + Path(__file__).with_suffix(".cu").read_text()
    identity = "\0".join((CPP, source, *CPP_FLAGS, *CUDA_FLAGS, *EXPORTS))
    digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
    extension = load_inline(
        name="dense_dependency_stream_" + digest,
        cpp_sources=CPP,
        cuda_sources=source,
        functions=list(EXPORTS),
        extra_cflags=list(CPP_FLAGS),
        extra_cuda_cflags=list(CUDA_FLAGS),
        no_implicit_headers=True,
        with_pytorch_error_handling=False,
        verbose=True,
    )
    extension.dep_info()
    return extension
