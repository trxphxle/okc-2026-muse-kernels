# Modal H200 lab

## Whole-layer continuation

**Routing/local-ID fusion is the only measured whole-layer improvement in this continuation.** It retains SGLang's warp top-k arithmetic, tie-breaking, FP32 weights and global expert IDs, while writing `(global_id - local_start) % 256` in the same kernel. The FP32 router-input boundary at eight or fewer tokens is unchanged. Normalization remains separate.

The full experiment completed **72 fresh-input timing pairs across 12 decoder shapes**. It includes a separately captured, unchanged baseline control. Outputs requiring adjudication, and the first pair of each shape, were checked against the independent public reference: **45 reference comparisons, zero failures**. Same-logit routing checks matched stock IDs, local IDs and FP32 weights exactly in all 36 experimental comparisons.

| Load | Baseline / routing-fusion time | Baseline / unchanged-control time |
|---|---:|---:|
| Decode 1 | 1.0412× | 1.0077× |
| Decode 16 | 1.0212× | 1.0010× |
| Decode 32 | 1.0140× | 0.9997× |
| Verify 1 | 1.0182× | 0.9981× |
| Verify 16 | 1.0064× | 1.0031× |
| Verify 32 | 1.0099× | 1.0032× |
| Geometric mean | **1.0184×** | **1.0021×** |

Each load combines sliding/full times with the published weights 46/16; the final aggregate is the geometric mean over six loads. These are **synthetic comparisons against our retained implementation, not official speedups against SGLang or leaderboard results**. One fresh weight set was used; the official judge remains the promotion gate.

The cleaned production candidate subsequently passed all 25 official cases and 18 exact routing comparisons in check `36e468b9084e4c8c`: **2.035848× stock SGLang**, CI [2.005311,2.066386], preparation 211.34 seconds. Its one ranked run, `5d788c4ff2fe4413`, also passed 25 cases at **2.024927×**, CI [1.994553,2.055301], preparation 252.71 seconds. The retained best remains **2.035572×**; the synthetic improvement is not presented as an established leaderboard gain.

### Rejected experiments

- **Routing plus pre-expert normalization:** passed same-input correctness, but slowed the first three measured whole-layer cases by approximately 2–4%. It removes useful overlap and is not promoted.
- **Fused pooler/router, with repaired lossless escape weights:** passed shape-boundary/random/zero checks at 9, 15, 16, 17, 31 and 32 tokens and six whole-layer cases. At 16/32 tokens it lost to routing-only fusion. Adding normalization to that path was worse again. Neither variant is promoted.
- **Two resident expert CTAs:** the previous agent's completed experiment admitted two resident blocks, 96 registers, zero local memory and 114,432 shared bytes. All 48 synthetic comparisons passed, but every affected measured shape regressed: baseline/new paired ratios were 0.958 at M32 normal, 0.911/0.947 at M128 normal/skewed, and 0.887/0.930 at M256. No official attempt was spent.
- **Register redistribution, 128-token expert jobs:** a complete producer warpgroup donates registers to four consumer warpgroups, preserving their parallelism. WGMMA serialization disappeared and spill stores/loads fell to four bytes each; local storage was eight bytes. Fourteen eager and four graph correctness probes passed, followed by 56 paired comparisons. Normal M128/M256 ratios were 0.974/0.982; skewed ratios were 1.016/1.020, with 1.060 at M384. It is not a general replacement.
- **Register redistribution, 96-token jobs and restored three-stage weight lookahead:** zero spills/local storage, 230,656 shared bytes and one resident 640-thread block. Sixteen eager and four graph probes passed, then 56 pairs. Normal M128/M256 ratios were 0.950/0.977; skewed M128/M256/M384 ratios were 0.979/1.024/1.071. It also failed to improve the general path and was not submitted officially.

### Correctness and timing protocol

Whole-layer reductions are not bitwise deterministic. The unchanged baseline control itself differed by up to 1.33% in hidden-output relative L2. Two earlier diagnostics stopped on overly strict inter-execution agreement assertions; their failure logs remain saved. The completed experiment uses the public reference to check actual selected routes, residual updates, per-row errors and cache writes whenever baseline agreement is insufficient. It does not simply widen the reference's tolerances. One routing-fusion execution chose different near-tie IDs and passed that independent check.

Timing uses serialized CUDA graph replays, alternating variant order and a **read-only 256 MiB cache sweep** before each measured graph. An unchanged control quantifies noise. Fixed replay overhead is not subtracted in these synthetic results. Hardware: H200, PyTorch 2.14.1+cu130, CUDA 13.0. No clocks or judge behavior are modified.

The production candidate removes the rejected normalization fusion: [submission_fused_routing.py](../../archive/outputs/openkernel-challenge/decode-layer/submission_fused_routing.py). It reuses the existing boundary extension rather than adding a separate compiler invocation.

## Dense serial-path screen

The existing-family feasibility screen is **not promoted**. Its hypothesis, all five Modal executions, partial paired result, and numerical coverage are recorded in [RESULTS.md](RESULTS.md) and [the experiment record](modal-lab/dense-existing-family-01/experiment.json). The frozen map changes only scalar QKV/O/unpooler; the other public serial winners were already retained.

Use `archive/work/modal/run_dense_groups.py` (archived) for the complete gate. It resumes complete weight groups and gives each remaining weight a separate profiler process; `dense_family_screen.py --weight-index N` measures one group, not the full gate. Event timings run outside profiling. CUDA-graph capture streams are not reliable stage identities, and CPU/GPU timestamp containment misattributed one replay. The repaired parser joins GPU activities to CPU graph launches by CUPTI correlation ID, retaining compressed raw traces and child reports on failure. Clock-skew/failure-retention smoke passed locally; this latest repair has **not yet been rerun on H200**.

## Warm-runtime process and spending

- The retained complete layer ran successfully on Modal and passed an independent reference smoke check. Its first preparation took 127.59 seconds with the prior expert binaries cached.
- `work/modal/run_layer.py` keeps one bounded sandbox warm between experiments and snapshots compiler artifacts before termination, including after a numerical rejection. Subsequent runs still validate outputs; a cached binary is not correctness evidence.
- Every allocation retains the existing 1,800-second maximum lifetime, 300-second idle timeout and hard CPU/memory limits. Root alone launches paid work.
- The cap is **$60 of real spend**. `lab.py` now checks monthly billing across month boundaries and the hourly resource report before allocating, adding bounded unreported usage. Historical $5 reservations are not the gate. Latest billing evidence is in `modal-budget.json`; `python3 work/modal/lab.py budget` refreshes it without allocating.
- All sixteen GPU sessions and the CPU-only artifact-recovery sandbox are finished. Sessions12–16 tested the dense portfolio; no official check or ranked submission was spent. Further execution is paused at the user's five-check boundary, interpreted conservatively to include these Modal attempts.

## Earlier expert investigation

The earlier wider two-warpgroup kernel passed 40 synthetic comparisons but regressed substantially at normal M128/M256. Hardware counters showed essentially unchanged DRAM traffic but lower delivered throughput; zero spills alone was not an improvement. See [counter findings](MODAL-COUNTERS.md).

## Evidence

- [Complete routing comparison](modal-lab/layer-okc-h200-08/layer-comparison.json), [weighted analysis](modal-lab/layer-okc-h200-08/analysis.json), [runtime log](modal-lab/layer-okc-h200-08/layer_compare.log).
- [Fused projection comparison](modal-lab/layer-okc-h200-08/fpr-comparison.json), [runtime log](modal-lab/layer-okc-h200-08/fpr_compare.log).
- [Completed two-CTA comparison](modal-lab/two-cta-okc-h200-05/two-cta-comparison.json), [residency gate](modal-lab/two-cta-okc-h200-05/two-cta-resource.json).
- [Earlier two-warpgroup comparison](modal-lab/synthetic-compare-okc-h200-03.json).
- [128-token register redistribution](modal-lab/layer-okc-h200-09/register-comparison.json), [compiler and runtime log](modal-lab/layer-okc-h200-09/register_compare.log).
- [96-token register redistribution](modal-lab/layer-okc-h200-10/register96-comparison.json), [compiler and runtime log](modal-lab/layer-okc-h200-10/register96_compare.log).
- [Unchanged-best stage profiles and actual routes](modal-lab/floors-okc-h200-11/baseline-floors.json), [floor math and prospective glue plan](modal-lab/floors-okc-h200-11/analysis.json).
- [Official results](RESULTS.md), [public workload/scoring specification](https://openkernelchallenge.dev/problems/muse_decode_layer).
