# SPDX-License-Identifier: Apache-2.0
"""Build the single-grid decode-byte ceiling diagnostic, not a submission.

TMA helpers derive from the approved d1_stream_modes.cu and vjeux/alygg77
Apache-2.0 patterns; attribution is retained in the adjacent CUDA source.
No CUDA is generated: stream_kernel.cu is the full compiled CUDA artifact.

Immutable descriptor ABI (all tensors contiguous CUDA int64):
  chunks[nchunk,3]: source address, byte count, destination offset in stage.
  stages[nstage,3]: first chunk, number of chunks, total physical stage bytes.
Every chunk address/size/offset is 16-byte aligned, each size is positive and
at most 32768, and offsets exactly pack the stage with no undeclared holes.
Each stage is at most 32768 bytes. The planner owns descriptor/coverage checks
and must retain every source allocation plus both descriptor tensors.

Call build() on the target current CUDA device before capture. It initializes
both kernels' dynamic-shared-memory attributes without launching a kernel.
On a different device call ext.stream_info(0) and (1) before capturing there.
The caller supplies the current device and CUDA stream; launch does not infer
them from raw pointers, allocate, synchronize, or preprocess any input.

ext.stream_launch(chunks_ptr, stages_ptr, sink_ptr, stamps_ptr,
                  nstages, blocks, mode, cuda_stream)
  blocks=132 is the H200 nominal one-CTA-per-SM persistent grid.
  mode=0: volatile TMA plus completion waits, no payload checksum drain.
  mode=1: same descriptors/ring plus every loaded uint32 word XORed exactly.
  sink: int64[blocks,4], columns SINK_FIELDS below. Empty CTAs write all zero.
  stamps: int64[blocks,4], columns STAMP_FIELDS; pointer 0 disables stamps.
  Stamps bracket each CTA's initialization, stream and sink stores, not launch.

Validation (untimed host work, never inside the captured callable): CTA b owns
stages[b::blocks]. Its bytes, stage count and chunk count must match that slice;
mode1 XOR must equal the XOR of every physical uint32 in those chunks. Global
byte/stage/chunk sums and XOR must match the manifest too. Declared padding is
included in physical-byte/XOR checks. A completely empty plan is legal with
nstages=0 and null descriptor pointers; sink storage is still required.
Mode1 timing is validation overhead, NOT the bandwidth bound. Mode0 additionally
requires cold-L2 NCU DRAM-byte validation, with descriptor reads and output
writes accounted for separately from payload. Frozen routes/pages are a free
oracle idealization, not legal submission preparation or a deployable kernel.

Descriptor payload read once is 24*(nchunks+nstages) bytes, separate from the
streamed model/KV bytes. That is 48/32768 ~= 0.1465% for full one-chunk weight
stages and (256*24+24)/32768 ~= 18.8232% for full 128-byte KV stages. These are
logical metadata bytes, not measured DRAM traffic: cache-line/sector overfetch,
metadata reuse and physical-address locality must be measured with counters.
Each CTA also writes 32 sink bytes and, when enabled, 32 timestamp bytes.
"""

from functools import lru_cache
import hashlib
from pathlib import Path

SINK_FIELDS = (
    "completed_payload_bytes",
    "uint32_xor",
    "completed_stages",
    "completed_chunks",
)
STAMP_FIELDS = (
    "clock64_begin",
    "clock64_end",
    "globaltimer_begin_ns",
    "globaltimer_end_ns",
)
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
)
CPP = r"""
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <vector>
void stream_launch(int64_t chunks, int64_t stages, int64_t sink, int64_t stamps,
                   int nstages, int blocks, int mode, int64_t cuda_stream);
std::vector<int64_t> stream_info(int mode);
"""
CUDA_FLAGS = (
    "-O3",
    "-std=c++20",
    "-gencode=arch=compute_90a,code=sm_90a",
    "-lineinfo",
    "--ptxas-options=-v,--warn-on-spills",
)
CPP_FLAGS = ("-O3", "-std=c++20")
EXPORTS = ("stream_launch", "stream_info")


@lru_cache(maxsize=1)
def build():
    """Return the raw pybind extension after untimed resource initialization."""
    from torch.utils.cpp_extension import load_inline

    source = Path(__file__).with_suffix(".cu").read_text()
    identity = "\0".join((CPP, source, *CPP_FLAGS, *CUDA_FLAGS, *EXPORTS))
    digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
    extension = load_inline(
        name="dense_layer_stream_bound_" + digest,
        cpp_sources=CPP,
        cuda_sources=source,
        functions=list(EXPORTS),
        extra_cflags=list(CPP_FLAGS),
        extra_cuda_cflags=list(CUDA_FLAGS),
        no_implicit_headers=True,
        with_pytorch_error_handling=False,
        verbose=True,
    )
    extension.stream_info(0)
    extension.stream_info(1)
    return extension
