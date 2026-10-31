# Muse Spark Open Kernel Challenge 2026: H200 decode-layer kernels

Kernels, measurements and negative results from the [Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev): GPU kernels for Meta's **Muse Spark Open** (816B parameters, 256-expert NVFP4 MoE) that must beat SGLang's kernels on one **NVIDIA H200 SXM**, scored against SGLang's own implementation on fresh inputs.

## Results

| Board | Speedup vs SGLang | Rank |
|---|---:|---:|
| **Whole decoder layer** (`muse_decode_layer`, determines overall rank) | **2.1756×** ranked (2.1916× check, 95% CI 2.159–2.225) | #191 / 1,093 |
| Routed experts (`muse_moe_w4a16`) | 2.3249× | #260 / 1,228 |
| Gated attention decode (`muse_attention_decode`) | 2.5673× | #241 / 1,176 |
| Dense projections (`muse_bf16_proj`) | 1.4423× | #404 / 1,526 |
| Prefill layer (`muse_prefill_layer`, not part of overall rank) | 1.4774× | #386 / 737 |

**Overall: #191 of 1,743 teams** (peak #185). Speedup is the judge's geometric mean, over six loads, of SGLang time ÷ submission time. Loads: D1/D16/D32 are 1/16/32-token decode steps; V1/V16/V32 are 8/128/256-token speculative-verify blocks. Each load is weighted 46 sliding-window : 16 full-attention layers. All correctness checks pass; ranks are provisional pending organizer reruns.

## Workload

One decoder layer as one H200 runs it under TP8/EP8 in SGLang:

```
x → QKV+gate proj (BF16 2304×8192) → QK-norm → RoPE → FP8 KV-cache write
  → gated attention (2048-token sliding window, or full attention at ~80K context; paged FP8 cache, page size 1)
  → O proj (8192×1024) → residual-gate boundary (RMSNorm)
  → FP32 router (256×8192, top-8 of sigmoid+bias) ┐
  → pooler (4096×8192) → RMSNorm                  ┴→ 32 local NVFP4 SwiGLU experts (~28.3 MB each)
  → post-expert norm + weighted sum → unpooler (8192×4096) → second boundary
```

**Constraints:** lossless weight repacking only (NVFP4 stays NVFP4, BF16 stays BF16 or a lossless encoding); BF16 activations wherever SGLang keeps BF16 (FP8-E4M3 queries and an FP32 router input at T ≤ 8 are the stated exceptions); FP8 KV cache; every key attended; CUDA-graph capturable (no host sync or value-dependent host control flow); no caching across calls; untimed `prepare()` ≤ 300 s. The judge flushes L2 before every replay.

## Final decode-layer implementation

Built on [vjeux/okc-2026](https://github.com/vjeux/okc-2026) and [alygg77/okc-muse-spark-kernels](https://github.com/alygg77/okc-muse-spark-kernels) (both Apache-2.0). About 14 kernels, PDL-chained in one CUDA graph:

| Stage | Technique |
|---|---|
| Dense projections | Lossless 12-bit BF16 weight codec; skinny split-K GEMV ("sk2") at small T; WGMMA kernels for the O projection at large T; per-shape backend selection timed in `prepare()` |
| Attention | Fused QK-norm / RoPE / FP8 cache write; split-KV FP8 attention with FP8-E4M3 queries; fused merge, per-head RMSNorm and sigmoid gate |
| Router | Exact FP32 router matvec fused into the pooler kernel at T ≤ 8; BF16 hi+lo pair with FP32 accumulation above 8 tokens |
| Experts | Persistent, warp-specialized NVFP4 kernel (producer warps on TMA, four consumer warpgroups on BF16 WGMMA), NVFP4→BF16 dequant in registers, 3-stage weight ring, KPERM3 lossless K-permutation so one scale pair serves four K16 steps (no spills), token-count-specialized builds |
| Glue | Fused residual-gate / RMSNorm boundaries; PDL weight prefetch across kernel boundaries |
| Precision | Restored the BF16 rounding points SGLang performs (QK-norm, RoPE, cache scaling, attention output, expert GEMM1/GEMM2 outputs and gains, pooler input/output) that the upstream file omitted. Cost: ~0.6% |

## Optimization history (decode layer)

| Step | Score |
|---|---:|
| SGLang-equivalent control | 1.00× |
| Small-batch router + Marlin expert tiles | ~1.08× |
| Native attention + persistent NVFP4 experts (vjeux) | ~1.70× |
| Broad static BF16 projection routes, compact FFN boundary, two expert layouts, PDL prefetch | 2.0356× |
| alygg77 base (KPERM3, fused FP32 router, WGMMA O-proj) with BF16 boundaries restored | **2.1756×** |

## Measured negative results

| Change | Result | Cause |
|---|---|---|
| Deeper expert pipelines (4–6 ring slots, half-K stages, batched WGMMA commits) | −1 to −3% each | Weight pipeline saturates at 3 slots; stalls come from unit boundaries, not depth |
| Wider tiles, two resident CTAs, register donation | Slower | 576 threads × 96 registers per CTA; shared memory ~226 KiB per CTA |
| Router GEMM + top-k fusion | −4.2% layer | Removed existing inter-kernel overlap |
| Pre-expert RMSNorm fusion | −2 to −4% | Same: lost overlap |
| From-scratch small-M register-dequant expert kernel (7 variants) | 1.5–2× slower | Weight read capped near 3 TB/s; math serialized after loads |
| Hand-written TMA/WGMMA dense GEMMs (with and without multicast) | 0.45–0.89× | WGMMA serialization; incumbent kernels are better tuned |
| ~10.9-bit lossless BF16 codec (measured entropy 10.46 bits) | Decode 6–11% slower | Byte savings don't cover decode cost |
| Hopper inline (generic) memory compression on dense weights | ~0 | Weights behave like random draws |
| VMM-backed, granularity-aligned expert arenas | +0.1–0.2% | Banks already 2-MiB aligned and adjacent |

## Ceiling (deletion) tests

Each bounds the best case by removing or idealizing one cost while keeping real routing, real weight streams and end-to-end paired cold-L2 timing.

| Idealization | Upper bound |
|---|---:|
| Delete nearly all FP4 unpack and half the scale multiplies | +0.5–0.9% expert board: experts are transfer/sync-bound |
| Perfect finish-time balancing (heaviest-expert-first) | +1.1–1.3% expert board |
| Remove re-reads across 64-token jobs of the same expert (multicast / L2 reuse) | ~1.5% expert board; V32 duplicate weight bytes 8.6% on average (0% at D32) |
| Exact KV gather-read in place of full attention | +0.63% decode |
| Overlap recovery between attention and GEMMs | 0: verify attention never co-runs with a GEMM in this chain |
| Pre-warmed code and constants for every D1/V1 kernel | +0.16% decode |
| Every deletable dense-projection epilogue and consumer cost | +1.0% decode at D16/D32 (+2.75% across all four affected loads) |
| Expert-weight interleaving across DRAM channels | Negative (slower) |

## Engineering findings

1. **The layer is a serial, PDL-chained path with <1.1 µs of idle gaps at T ≤ 8.** Deleting a whole glue launch saves only 1.2–2.0 µs, so launch removal alone can't close the gap to the byte floor (d1-sliding ≈ 73 µs vs ≈ 45 µs).
2. **Achieved DRAM bandwidth depends on transfer size.** ~4.6 TB/s at 2 GiB, ~4.2 TB/s at 128 MiB, ~3 TB/s for D1's ~64 MB expert read across 13 access patterns. Small-batch loads are bounded by finite-transfer bandwidth, not instruction count.
3. **Expert-stage DRAM bytes equal the planned weight stream, including per-job re-reads**, at 53–75% of peak bandwidth with no register spills. The remaining expert headroom is in how the stream is scheduled and completed, not in arithmetic.
4. **The per-stage board leaders imply where the decode leader's ~11% comes from.** MoE (~15% ahead of our base), attention (~21%) and dense (~6.5%), weighted by each stage's share of the layer, add up to roughly the decode leader's margin. That points to a different expert-kernel design, not one missing trick.
5. **Judge noise (±1.5–2% run to run, plus ~1% between judge clusters) is the size of most remaining wins.** Paired local A/B screening with unchanged controls (A/A within ±0.3%) was necessary to resolve anything below ~2%.
6. **Robustness:** the final file passes 192 nearby-shape and dispatch-boundary cases (31/32/33, 63/64/65 and 127/128/129 routes per expert; three routing seeds). Crossing a 64-token job boundary costs the expected 1–3% step, with no cliffs.

## Repository layout

| Path | Contents |
|---|---|
| [`kernels/`](kernels/) | Ranked submission files, one self-contained Python file per board with CUDA compiled via `load_inline`; [`BEST.json`](kernels/BEST.json) lists SHA-256, judge job ID and score |
| [`results/`](results/) | Official check/ranked-run log (`RESULTS.md`), paired synthetic benchmarks (`MODAL-LAB.md`), Nsight Compute counter analysis (`MODAL-COUNTERS.md`), raw ranked job records |
| [`experiments/`](experiments/) | Experiment logs, ceiling-test scripts and summary data by area (experts, scheduling, attention, dense, public prior-art survey) |
| [`tools/`](tools/) | H200 sandbox launcher (`lab.py`, `run_layer.py`, built on Modal) and paired A/B screening benches (`screen_layer.py`, `screen_moe.py`) |
| [`licenses/`](licenses/) | Upstream license and notice files |

## Running a kernel

Each file in `kernels/` is a complete submission for the challenge's judge image (`lmsysorg/sglang:dev-muse-spark-open`, one H200):

```bash
curl -fsSL https://openkernelchallenge.dev/install.sh | sh
okc init muse_decode_layer
okc check kernels/decode_layer.py
```

## Credits and licenses

- [vjeux/okc-2026](https://github.com/vjeux/okc-2026) (Apache-2.0): attention, persistent MoE and 12-bit dense codec kernels.
- [alygg77/okc-muse-spark-kernels](https://github.com/alygg77/okc-muse-spark-kernels) (Apache-2.0): decode base, KPERM3, fused FP32 router, WGMMA O-projection, and published H200 measurements and negative results.
- [NathenFernandes/muse-bf16-proj](https://github.com/NathenFernandes/muse-bf16-proj) (MIT): small-M projection routes, included via the alygg77 base.
- SGLang (Apache-2.0).

Licensed under **Apache-2.0** ([`LICENSE`](LICENSE)), as the challenge rules require. Upstream notices are kept in the kernel sources and in [`licenses/`](licenses/). Not affiliated with or endorsed by Meta or the challenge organizers.
