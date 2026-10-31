# Three-kernel counter diagnostic

**All three native records were collected and both output comparisons passed.** The GPU session was then deliberately terminated.

| Counter | Original, balanced | Wider, balanced | Original, skewed |
|---|---:|---:|---:|
| DRAM read (MB) | 685.59 | 684.07 | 130.04 |
| DRAM throughput (% peak sustained) | 76.49 | 57.98 | 26.86 |
| L2 hit rate (%) | 12.65 | 19.69 | 57.62 |
| Tensor-pipe activity (%) | 16.94 | 12.38 | 37.52 |
| Issue activity (%) | 62.57 | 46.94 | 33.48 |
| Long-scoreboard stall (%) | 14.66 | 28.75 | 20.39 |
| Active warps (% peak) | 27.74 | 15.42 | 27.31 |

## Decision

Retain the original kernel. Both balanced runs read approximately 685 MB, close to the planned 679.5 MB packed-weight stages. The wider version saves no material DRAM traffic on this fixture; its lower active-warp and issue activity and higher long-scoreboard stalls are consistent with weaker latency hiding. These aggregate counters do not identify the particular stalled instruction or prove a single cause.

Active-warp percentages track the deliberate reduction from 18 to 10 warps per block. Both are about 98.6% of their respective one-block limits; this is not an unexplained residency collapse.

The original skewed fixture touches 3 expert banks instead of 24 and reads 130 MB instead of 686 MB, while processing 351 local route rows instead of 183. This supports weight-traffic reduction as a major reason concentrated routing is cheaper. Its higher tensor activity and different padding also matter; this is not a controlled isolation of traffic alone.

The retained balanced path already reports 76.49% peak sustained DRAM throughput. A new design must explain how it reduces required traffic or improves useful overlap. Smaller job counts or zero spills alone are insufficient evidence.

## Measurement limits

One fixed fresh fixture per distribution, kernel replay with cache flushing, and unmodified GPU clocks. The profiler warned that uncontrolled clocks can affect measurements. These are diagnostic counters, not official scores or end-to-end performance. Stall percentages use active-warp denominators and must not be added to elapsed-time percentages. Global counter aggregation does not localize phase A.

All five cached binaries remained byte-identical. Preparation took 13.89 seconds, versus 285.96 seconds for the earlier fresh compile.

[Raw counters](modal-lab/counter-okc-h200-04/profile.csv) · [Validated analysis](modal-lab/counter-okc-h200-04/analysis.json) · [Inputs and correctness](modal-lab/counter-okc-h200-04/counter-probe.json) · [NVIDIA metric interpretation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
