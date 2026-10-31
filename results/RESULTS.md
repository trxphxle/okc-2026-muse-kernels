# Official H200 test results

These are saved official checks and ranked evaluations on the challenge service. Each candidate is compared with stock SGLang on the same GPU and fresh inputs. A speed ratio above 1 is faster. Projections, attention, and experts are separate component workloads; whole-layer results cover one GPU’s layer computation and exclude communication between GPUs. Neither is a measurement of the eight-GPU serving prize workload.

| Experiment | Mode | Speed ratio | Reported 95% interval | Correctness |
|---|---|---:|---:|---|
| [Original projections (control)](runs/control.json) | Unranked check | 0.9926× | 0.9777–1.0074× | 39/39 cases passed |
| [Small-batch router](runs/router-v1.json) | Unranked check | 1.0651× | 1.0491–1.0811× | 39/39 cases passed |
| [All-batch fusion experiment](runs/router-v2.json) | Unranked check | 1.0033× | 0.9883–1.0184× | 39/39 cases passed |
| [Small/medium hybrid](runs/router-hybrid.json) | Unranked check | 1.0931× | 1.0767–1.1095× | 39/39 cases passed |
| [Fixed-split router experiment](runs/router-split16.json) | Unranked check | 1.0829× | 1.0666–1.0991× | 39/39 cases passed |
| [Original whole layer (control)](runs/layer-control.json) | Unranked check | 0.9998× | 0.9848–1.0148× | 25/25 cases passed |
| [Whole layer with hybrid router](runs/layer-hybrid.json) | Unranked check | 1.0231× | 1.0078–1.0385× | 25/25 cases passed |
| [Isolated expert tile-16 experiment](runs/layer-marlin16.json) | Unranked check | 1.0437× | 1.0281–1.0594× | 25/25 cases passed |
| [Router plus expert tile-16](runs/layer-router-marlin16.json) | Unranked check | 1.0759× | 1.0598–1.0921× | 25/25 cases passed |
| [Bounded router plus expert tile-16](runs/layer-router-marlin16-bounded.json) | Unranked check | 1.0671× | 1.0511–1.0831× | 25/25 cases passed |
| [Isolated expert tile-8 experiment](runs/layer-marlin8.json) | Unranked check | 0.9680× | 0.9535–0.9826× | 25/25 cases passed |
| [Fused conversion and cache store](runs/layer-qkv-cache-fused.json) | Unranked check | — | — | Failed or incomplete |
| [Corrected conversion/cache fusion](runs/layer-qkv-cache-fused-v2.json) | Unranked check | 1.0058× | 0.9907–1.0208× | 25/25 cases passed |
| [Local-only expert scheduling](runs/layer-local-expert-alignment.json) | Unranked check | 1.0027× | 0.9877–1.0178× | 25/25 cases passed |
| [Attributed persistent MoE reproduction](runs/moe-persistent-reproduction.json) | Unranked check | 2.3225× | 2.2876–2.3573× | 11/11 cases passed |
| [Attributed attention reproduction](runs/attention-reproduction.json) | Unranked check | 2.5852× | 2.5464–2.6396× | 28/28 cases passed |
| [One-layout persistent MoE](runs/moe-single-layout.json) | Unranked check | 2.0720× | 2.0409–2.1031× | 11/11 cases passed |
| [Native attention plus router/expert tuning](runs/layer-native-attention.json) | Unranked check | 1.2040× | 1.1859–1.2220× | 25/25 cases passed |
| [Native attention plus one-layout MoE](runs/layer-native-attention-single-layout-moe.json) | Unranked check | 1.6962× | 1.6707–1.7216× | 25/25 cases passed |
| [Same implementation on long prompts](runs/prefill-native-single-layout.json) | Unranked check | 1.2620× | 1.2430–1.2809× | 13/13 cases passed |
| [Unit-layout fallback validation](runs/moe-unit-fallback-validation.json) | Unranked check | 1.2212× | 1.2029–1.2395× | 11/11 cases passed |
| [Bounded lossless QKV compression](runs/projections-qkv-sk2.json) | Unranked check | 1.1396× | 1.1225–1.1567× | 39/39 cases passed |
| [Lossless QKV plus native attention and MoE](runs/layer-native-qkv-attention-moe.json) | Unranked check | 1.7384× | 1.7123–1.7644× | 25/25 cases passed |
| [Parameter-shaped expert storage and byte proofs](runs/moe-split-storage-validated.json) | Unranked check | 2.0696× | 2.0386–2.1007× | 11/11 cases passed |
| [Serialized scratch and graph-replay validation](runs/moe-shared-arena-validation.json) | Unranked check | 2.0485× | 2.0177–2.0792× | 11/11 cases passed |
| [Parameter-shaped expert storage on long prompts](runs/prefill-native-split-storage.json) | Unranked check | 1.2466× | 1.2279–1.2653× | 13/13 cases passed |
| [Specialized pre-expert normalization](runs/layer-pre-expert-rmsnorm.json) | Unranked check | 1.7448× | 1.7186–1.7710× | 25/25 cases passed |
| [Combined split storage, QKV, and normalization](runs/layer-native-split-qkv-rmsnorm.json) | Unranked check | — | — | Failed or incomplete |
| [Four-layout experts in the complete layer](runs/layer-native-multilayout-qkv-rmsnorm.json) | Unranked check | 1.8627× | 1.8348–1.8907× | 25/25 cases passed |
| [Extended split-skinny dispatcher through 32 tokens](runs/moe-multilayout-mid32.json) | Unranked check | 2.3022× | 2.2677–2.3368× | 11/11 cases passed |
| [Power-of-two resident expert grid](runs/moe-multilayout-grid128.json) | Unranked check | 2.3071× | 2.2725–2.3417× | 11/11 cases passed |
| [Larger-chunk grouped expert consumer](runs/moe-grouped-prefill.json) | Unranked check | 0.9486× | 0.9343–0.9628× | 11/11 cases passed |
| [Grouped experts in complete long-prompt layers](runs/prefill-grouped-experts.json) | Unranked check | 1.4902× | 1.4678–1.5125× | 13/13 cases passed |
| [Two layouts with unchanged faster expert scheduling](runs/moe-two-layout.json) | Unranked check | 2.3227× | 2.2879–2.3576× | 11/11 cases passed |
| [Request-aligned attention tile scheduling](runs/attention-request-tiles.json) | Unranked check | 2.5006× | 2.4631–2.5381× | 28/28 cases passed |
| [Guarded KV cache-line lookahead](runs/attention-kv-lookahead.json) | Unranked check | 1.9414× | 1.9122–1.9705× | 28/28 cases passed |
| [Existing per-request attention family on batched verification](runs/attention-request-q8.json) | Unranked check | 2.4422× | 2.4055–2.4788× | 28/28 cases passed |
| [Compiled attention resource diagnostic; unchanged timed computation](runs/attention-resource-diagnostic.json) | Unranked check | 2.5859× | 2.5471–2.6414× | 28/28 cases passed |
| [Broader static BF16 projection kernels](runs/projections-broad-v1.json) | Unranked check | 1.3538× | 1.3335–1.3741× | 39/39 cases passed |
| [Broader projections with larger-batch CUTLASS pooler](runs/projections-broad-v2.json) | Unranked check | 1.3584× | 1.3380–1.3787× | 39/39 cases passed |
| [Share pooler weight loads across two token tiles](runs/projections-pooler-cluster2.json) | Unranked check | 1.4452× | 1.4235–1.4669× | 39/39 cases passed |
| [Smaller QKV output-row tiles at larger batches](runs/projections-qkv-tile64.json) | Unranked check | 1.4161× | 1.3949–1.4374× | 39/39 cases passed |
| [Broader projections in the complete layer](runs/layer-two-layout-broad-projections.json) | Unranked check | 1.9128× | 1.8841–1.9415× | 25/25 cases passed |
| [Compact unpooler output fused into the FFN boundary](runs/layer-compact-ffn-boundary.json) | Unranked check | — | — | Failed or incomplete |
| [Compact boundary with corrected architecture setting](runs/layer-compact-ffn-boundary-v2.json) | Unranked check | 1.8699× | 1.8419–1.8980× | 25/25 cases passed |
| [Native router and compact boundary in the complete layer](runs/layer-rk-compact.json) | Unranked check | 1.9844× | 1.9546–2.0141× | 25/25 cases passed |
| [Finish independent expert-input chains before stream join](runs/layer-rk-compact-overlap.json) | Unranked check | 2.0071× | 1.9770–2.0372× | 25/25 cases passed |
| [Prefetch unpooler weights while native experts finish](runs/layer-unpooler-pdl.json) | Unranked check | 2.0531× | 2.0223–2.0839× | 25/25 cases passed |
| [Prefetch output-projection weights while native attention finishes](runs/layer-attention-o-pdl.json) | Unranked check | 2.0194× | 1.9891–2.0497× | 25/25 cases passed |
| [Ranked native expert-to-unpooler weight prefetch](runs/layer-unpooler-pdl-ranked.json) | Ranked | 2.0356× | 2.0050–2.0661× | 25/25 cases passed |
| [Fuse exact top-k routing with local expert-ID generation](runs/layer-fused-routing.json) | Unranked check | 2.0358× | 2.0053–2.0664× | 25/25 cases passed |
| [Ranked exact routing/local-ID fusion](runs/layer-fused-routing-ranked.json) | Ranked | 2.0249× | 1.9946–2.0553× | 25/25 cases passed |
| [Ranked layer with independent preparation before join](runs/layer-rk-compact-overlap-ranked.json) | Ranked | 2.0141× | 1.9839–2.0443× | 25/25 cases passed |
| [Ranked complete layer with native router and compact boundary](runs/layer-rk-compact-ranked.json) | Ranked | 1.9901× | 1.9602–2.0199× | 25/25 cases passed |
| [BF16 intermediate storage with exact norm reconstruction](runs/moe-bf16-ybuf.json) | Unranked check | 2.2927× | 2.2583–2.3270× | 11/11 cases passed |
| [Precomputed BF16 effective expert norm weights](runs/moe-cached-norm.json) | Unranked check | 2.3190× | 2.2842–2.3538× | 11/11 cases passed |
| [Expert phase tracing; unchanged production computation](runs/moe-wgs-phase-trace.json) | Unranked check | 2.3257× | 2.2908–2.3606× | 11/11 cases passed |
| [Larger expert jobs with fewer repeated weight/decode passes](runs/moe-wgs-wide128.json) | Unranked check | — | — | Failed or incomplete |
| [Wider expert kernel with device-safe constant access](runs/moe-wgs-wide128-v2.json) | Unranked check | 2.2478× | 2.2141–2.2815× | 11/11 cases passed |
| [Wider expert jobs with fewer consumer warpgroups](runs/moe-wgs-two-wg.json) | Unranked check | 2.1246× | 2.0927–2.1564× | 11/11 cases passed |
| [Designated finisher for split gate/up work](runs/moe-phase-a-finisher.json) | Unranked check | 2.2626× | 2.2287–2.2965× | 11/11 cases passed |
| [Five-slot small-M TMA weight pipeline](runs/moe-tma5.json) | Unranked check | 2.2733× | 2.2392–2.3074× | 11/11 cases passed |
| [Combine retained router with faster large-batch pooler](runs/projections-rk-pooler-cluster1.json) | Unranked check | 1.4536× | 1.4318–1.4754× | 39/39 cases passed |
| [Ranked router plus faster large-batch pooler](runs/projections-rk-pooler-cluster1-ranked.json) | Ranked | 1.4423× | 1.4207–1.4639× | 39/39 cases passed |
| [Integrate the large-batch pooler into the retained complete layer](runs/layer-unpooler-pooler-cluster1.json) | Unranked check | 2.0277× | 1.9973–2.0581× | 25/25 cases passed |
| [Counter-free native BF16 hi/lo router](runs/projections-rk-v1.json) | Unranked check | 1.4452× | 1.4236–1.4669× | 39/39 cases passed |
| [Larger grouped-prefill chunk](runs/prefill-grouped-16k.json) | Unranked check | 1.4944× | 1.4720–1.5169× | 13/13 cases passed |
| [Ranked attention entry](runs/attention-ranked.json) | Ranked | 2.5673× | 2.5288–2.6058× | 28/28 cases passed |
| [Ranked one-layout MoE entry](runs/moe-single-layout-ranked.json) | Ranked | 2.0579× | 2.0270–2.0888× | 11/11 cases passed |
| [Ranked parameter-shaped expert storage](runs/moe-split-storage-ranked.json) | Ranked | 2.0732× | 2.0421–2.1043× | 11/11 cases passed |
| [Ranked four-layout expert kernel](runs/moe-multilayout-ranked.json) | Ranked | 2.3249× | 2.2900–2.3597× | 11/11 cases passed |
| [Ranked projection entry](runs/projections-ranked.json) | Ranked | 1.0955× | 1.0791–1.1119× | 39/39 cases passed |
| [Ranked lossless QKV projection entry](runs/projections-qkv-sk2-ranked.json) | Ranked | 1.1407× | 1.1236–1.1578× | 39/39 cases passed |
| [Ranked broader BF16 projections](runs/projections-broad-v1-ranked.json) | Ranked | 1.3505× | 1.3303–1.3708× | 39/39 cases passed |
| [Ranked native hi/lo router and broad projections](runs/projections-rk-ranked.json) | Ranked | 1.4273× | 1.4059–1.4487× | 39/39 cases passed |
| [Ranked complete-layer entry](runs/layer-native-ranked.json) | Ranked | 1.7207× | 1.6949–1.7465× | 25/25 cases passed |
| [Ranked combined layer with FP64 diagnostic](runs/layer-native-combined-v2-ranked.json) | Ranked | 1.7751× | 1.7485–1.8017× | 25/25 cases passed |
| [Ranked layer with four-layout experts](runs/layer-multilayout-ranked.json) | Ranked | 1.8451× | 1.8174–1.8728× | 25/25 cases passed |
| [Ranked layer with broad projections and two expert layouts](runs/layer-broad-projections-ranked.json) | Ranked | 1.9143× | 1.8856–1.9430× | 25/25 cases passed |
| [Ranked prefill entry](runs/prefill-native-ranked.json) | Ranked | 1.2512× | 1.2324–1.2699× | 13/13 cases passed |
| [Ranked prefill with grouped experts](runs/prefill-grouped-ranked.json) | Ranked | 1.4774× | 1.4552–1.4996× | 13/13 cases passed |

## Dense serial-path time box

Experiment `dense-existing-family-01`. Deadline: **2026-10-31T23:59:59-07:00**. Status: **Stopped by user; no promotion or further dense work**. Official checks used: 0.

**Hypothesis, recorded before implementation:** The retained fixed projection dispatch omits existing legal family/layout choices at some shapes. Selecting approved existing QKV/O/unpooler families by metadata-only buckets can reduce exposed serial time by at least10% without changing any other stage.

**Keep/kill gate:** geometric mean of paired candidate/base exposed-serial latency ratios across six46:16-weighted loads <=0.90, correctness passing; no credit for pooler or overlapped unpooler work.

**Metric:** QKV span + O span + max(0,unpooler_end-max(unpooler_start,MoE_end)); also report whole-layer latency and routing counts to detect false savings caused by longer MoE.

**Scope:** Existing approved families and library/layout routes only; no new GEMM, precision change, pooler/router/attention/expert change.

| Load | Initial expected saving (µs) at the 10% serial target |
|---|---:|
| decode 1 | 1.847 |
| decode 16 | 3.163 |
| decode 32 | 3.211 |
| verify 1 | 1.878 |
| verify 16 | 4.364 |
| verify 32 | 5.570 |

Use discovery inputs to select fixed per-shape-bucket choices; use separate weight/input seeds for the gate. Include unchanged controls and nearby shapes.

These numbers are hypotheses, not measured gains. Every failed or rejected screen will be retained with this experiment; a serial-path win is not automatically a whole-layer or official leaderboard win.

- **Existing-family portfolio, first screen:** FAIL_DIAGNOSTIC after85.60s: retained CUDA graph anchors used different execution streams. Discovery completed;63 dense checks and12 nearby-layer checks passed for first weight; no serial result. Sandbox terminated.
- **Frozen-map held-out completion:** FAIL_DIAGNOSTIC after26.89s: CUPTI stopped returning GPU events after27 complete profile replays. Eight D1-sliding pairs and oneD1-full pair completed correctly; no aggregate result. Sandbox terminated.
- **Batched-profile held-out completion:** FAIL_DIAGNOSTIC after128.41s: duplicate graph annotation labels while parsing the completed continuous trace. All96 first-weight input pairs,63 dense cases and12 nearby-layer checks passed; timing summary unavailable. Peak GPU allocation12.81GB. Sandbox terminated.
- **CPU-marker qualified held-out completion:** FAIL_DIAGNOSTIC after270.70s: first weight complete, second profiler lifecycle returned no graph events.192 pairs passed correctness;96 pairs have complete timings. Restarting CUPTI in one process is unreliable. GPU terminated.
- **Isolated profiler-process completion:** FAIL_DIAGNOSTIC: two weight groups completed; third-weight timing parser selected post-graph collection operations at D32/full/pair7/control because CPU/GPU timestamp containment was unreliable. All288 input pairs passed numerical qualification. Recovered the third report from the saved filesystem using a bounded CPU-only sandbox; no additional GPU check. All resources terminated.

**Measured decision:** Two complete weight groups (192 paired inputs): exposed-serial reduction0.685%, whole-layer reduction0.519%; both inside the user’s ±1.5% noise band and nowhere near the10% gate. All288 three-weight paired inputs,189 dense cases and36 nearby-layer replays passed numerical qualification. The third weight has incomplete timing attribution, so no complete three-weight95% interval is claimed. No official check or submission. Recommendation: kill this family-selection portfolio rather than spend another run confirming a noise-sized gain.

### Partial paired measurements — not a qualified three-weight result

Two complete weight groups; third-weight timing attribution failed. No three-weight 95% interval or leaderboard gain is claimed.

| Load | Exposed serial reduction | Mean serial saving (µs) | Whole-layer reduction |
|---|---:|---:|---:|
| D1 | +5.87% | +0.970 | +1.57% |
| D16 | -0.21% | +0.093 | +0.20% |
| D32 | -0.61% | -0.262 | +0.03% |
| V1 | +0.11% | +0.007 | +0.24% |
| V16 | -0.74% | -0.300 | +0.35% |
| V32 | -0.48% | -0.017 | +0.72% |

**Proposed next experiment:** None scheduled: user stopped dense work and redirected the campaign to the expert kernel. The QKV geometry proposal was not implemented or tested.


## Expert-kernel bandwidth campaign

Deadline: **2026-11-01T23:59:59-08:00**. Status: **Launching the one authorized baseline-only Modal diagnostic; preflight passed**.

**Hypothesis before the one authorized Modal run:** Actual whole-layer-generated scored-like routing leaves material expert-stage headroom below85% of H200 HBM bandwidth. Low DRAM utilization may instead reflect cache reuse, route/job padding, decode/compute work or phase tails; the existing kernel already splits weight units across the SM grid. Measure those distinctions before choosing a change.

**Change:** None to production kernels. One Modal diagnostic allocation: genuine unprofiled graph timing, then a single NsightCompute counter process on exactly the same extracted expert input fixtures.

| Load | Historical active experts | Historical expert µs | Logical minimum MB | Logical TB/s | Ideal µs at 85% HBM |
|---|---:|---:|---:|---:|---:|
| decode 1 | 2.25 | 23.44 | 63.62 | 2.71 | 15.59 |
| decode 16 | 8.41 | 69.48 | 238.06 | 3.43 | 58.35 |
| decode 32 | 9.33 | 86.97 | 264.24 | 3.04 | 64.76 |
| verify 1 | 8.84 | 68.47 | 250.24 | 3.65 | 61.33 |
| verify 16 | 18.76 | 157.52 | 531.22 | 3.37 | 130.20 |
| verify 32 | 23.51 | 207.90 | 665.47 | 3.20 | 163.11 |

These are predictions from retained historical profiles, not hardware-byte measurements. Actual DRAM traffic, cache reuse and re-streaming must be measured.

**Scope:** All36 previously profiled public-reference layer fixtures:3weights ×6loads ×sliding/full. Actual observed routes retained; historical near-tie differences logged, not treated as kernel failures. This is not the hidden official workload.

**Cache/clocks:** Read-only256MiB sweep before isolated timing; NCU cache-control all for counter replays; native clocks unchanged. Report profiler time separately from unprofiled event time (fixed replay cost not subtracted).

**Gate:** By deadline, >=10% reduction in paired expert-only latency over six equally weighted loads, correctness passing; otherwise stop. Diagnostic baseline numbers alone cannot pass this gate.

**Official validation:** muse_moe_w4a16 own checks first; port qualifying improvement to decode afterward


## Stage floors and proposed glue fusion

The unchanged retained kernel was measured on three public-reference weight sets: 144 fresh inputs and 36 CUDA-graph profiles. The following times and actual expert counts belong to that local dataset, not to recovered official-judge inputs. Each load weights sliding/full profiles 46:16 across three weight sets.

[Raw inputs, routing histograms and kernel timestamps](modal-lab/floors-okc-h200-11/baseline-floors.json) · [Calculation data and assumptions](modal-lab/floors-okc-h200-11/analysis.json).

### Floor model

[H200 SXM peaks](https://www.nvidia.com/en-us/data-center/h200): 4.8 TB/s HBM, approximately 989 TFLOP/s dense BF16, and 67 TFLOP/s FP32. NVIDIA’s 1,979 BF16 figure includes sparsity and is not used here. In microseconds, divide bytes by 4.8e6, BF16 FLOPs by 989e6, and FP32 FLOPs by 67e6.

- Dense: sum over QKV/O/pooler/unpooler of `max(weight_payload_bytes / 4.8e6, 2*M*N*K / 989e6)`, also charging the cold QKV input. Current weight payload is 1.5 bytes/element through M32, otherwise 2; escape-table overhead is omitted. Matrix shapes are 2304×8192, 8192×1024, 4096×8192 and (1024 if M≤8 else 8192)×4096.
- Attention at page1: `max(260 * old_visible_keys / 4.8e6, 4 * 16 * 64 * query_key_pairs / 989e6)`. The 260 bytes are 256 bytes of FP8 K/V plus one 4-byte page index. Actual sequence lengths are recorded.
- Experts: each touched bank contains `3*4096²*(0.5+1/16) = 28,311,552` packed bytes. With actual distinct experts E and local token/expert routes R, `floor_us = max(5.89824*E, 0.101782908*R)`. Compute is `2*R*(8192*4096 + 4096*4096) = 100,663,296*R` FLOPs.
- Norms/residuals: `max((188416 + 32768*M)/4.8e6, 188416*M/67e6)`. The first term charges observed cached norm/gate weights and the cold FP32 residual input; the second is an approximate scalar-operation count. Intermediate tensors may remain in L2.
- Router/indexing: `max(8389632/4.8e6, 2*M*256*8192/67e6)` for M≤8; above M8 use `4*M*256*8192/989e6` for the hi/lo GEMMs. The cold weight/bias payload is about 8.39 MB.

**These are optimistic roofline estimates, not achievable latency guarantees.** They omit launch, reduction, indexing, conversion, dequantization and SFU costs. HBM is not the right denominator for every intermediate access: many hit L2. Kernel spans overlap, so their gaps must not be added as removable wall time.

### Dense projections

| Load | Measured span (µs) | Floor (µs) | Gap (µs) |
|---|---:|---:|---:|
| decode 1 | 33.63 | 20.32 | 13.31 |
| decode 16 | 48.74 | 29.55 | 19.20 |
| decode 32 | 52.79 | 29.60 | 23.19 |
| verify 1 | 41.59 | 20.34 | 21.24 |
| verify 16 | 66.21 | 39.76 | 26.45 |
| verify 32 | 89.19 | 48.86 | 40.33 |

### Attention

| Load | Measured span (µs) | Floor (µs) | Gap (µs) |
|---|---:|---:|---:|
| decode 1 | 6.23 | 1.20 | 5.03 |
| decode 16 | 28.08 | 19.35 | 8.73 |
| decode 32 | 50.69 | 38.38 | 12.32 |
| verify 1 | 8.69 | 1.20 | 7.49 |
| verify 16 | 34.76 | 19.14 | 15.62 |
| verify 32 | 58.63 | 38.37 | 20.25 |

### Experts

| Load | Measured span (µs) | Floor (µs) | Gap (µs) |
|---|---:|---:|---:|
| decode 1 | 23.44 | 13.26 | 10.18 |
| decode 16 | 69.48 | 49.60 | 19.89 |
| decode 32 | 86.97 | 55.05 | 31.91 |
| verify 1 | 68.47 | 52.13 | 16.34 |
| verify 16 | 157.52 | 110.67 | 46.85 |
| verify 32 | 207.90 | 138.64 | 69.26 |

### Norms/residuals

| Load | Measured span (µs) | Floor (µs) | Gap (µs) |
|---|---:|---:|---:|
| decode 1 | 11.43 | 0.05 | 11.38 |
| decode 16 | 12.12 | 0.15 | 11.97 |
| decode 32 | 11.92 | 0.26 | 11.66 |
| verify 1 | 12.58 | 0.09 | 12.48 |
| verify 16 | 13.77 | 0.91 | 12.86 |
| verify 32 | 21.21 | 1.79 | 19.43 |

### Router/indexing

| Load | Measured span (µs) | Floor (µs) | Gap (µs) |
|---|---:|---:|---:|
| decode 1 | 10.83 | 1.75 | 9.08 |
| decode 16 | 23.24 | 1.75 | 21.49 |
| decode 32 | 24.29 | 1.75 | 22.55 |
| verify 1 | 28.88 | 1.75 | 27.13 |
| verify 16 | 31.24 | 1.75 | 29.49 |
| verify 32 | 43.72 | 2.17 | 41.55 |

### Actual routed-expert counts

Weighted means and observed ranges below refer to the six profiled inputs per load. The floor is averaged **after** taking each input’s max(memory, compute), not estimated from token count or an assumed expert distribution.

| Load | Distinct E: mean (range) | Local routes R: mean (range) | Weight-read floor (µs) | FLOP floor (µs) |
|---|---:|---:|---:|---:|
| decode 1 | 2.25 (2–3) | 2.25 (2–3) | 13.26 | 0.23 |
| decode 16 | 8.41 (7–9) | 25.81 (21–29) | 49.60 | 2.63 |
| decode 32 | 9.33 (9–10) | 53.41 (42–60) | 55.05 | 5.44 |
| verify 1 | 8.84 (7–10) | 13.51 (11–16) | 52.13 | 1.37 |
| verify 16 | 18.76 (12–24) | 171.85 (148–193) | 110.67 | 17.49 |
| verify 32 | 23.51 (21–27) | 358.25 (304–390) | 138.64 | 36.46 |

### Glue hypothesis: conditional, before implementation

The roughly 2 µs ideal glue floor is plausible only with cache-resident intermediates; the model rises from about 1.8 µs at small M to 3.5 µs at M256. New timestamps expose approximately 18–29 µs in the two boundaries and the post-pooler join, not the sum of every overlapping router/norm span. A 10% geomean improvement is an optimistic possibility, not established by that floor.

**Plan: three separately screened changes, never one untraceable combined rewrite.** First fuse router GEMM finalization with exact top-8/global-local IDs; keep the parallel pooler/pre-expert norm chain. Then consider O→attention-boundary and unpooler→FFN-boundary finalizers, preserving whole-row reductions and every required BF16 round trip. Do not repeat the regressing pre-expert-norm fusion.

| Load | Router fusion target saved (µs) | O-boundary target (µs) | Unpooler-boundary target (µs) | Combined target (µs) |
|---|---:|---:|---:|---:|
| decode 1 | 4.7 | 2.8 | 2.4 | 9.9 |
| decode 16 | 5.1 | 2.0 | 2.4 | 9.5 |
| decode 32 | 5.5 | 2.0 | 2.2 | 9.7 |
| verify 1 | 4.8 | 2.9 | 3.1 | 10.8 |
| verify 16 | 5.3 | 2.9 | 2.6 | 10.8 |
| verify 32 | 5.4 | 6.7 | 5.3 | 17.5 |

Targets assume no GEMM regression and incremental fused-finalizer costs no greater than 1.5 µs for routing, 2.0 µs for O-boundary, and 2.5 µs for unpooler-boundary. These budgets have not been achieved by an implementation. They project about **3.1–3.3%** for the first experiment and **6.9–7.2%** for all three, approximately **2.18×** versus the current **2.2248×** top-50 snapshot. The full plan does not yet justify a promised 10% gain or top-50 placement.

**Kill criteria:** reject the isolated router experiment if paired whole-layer gain is below approximately 3%, a clear nearby-shape regression appears, correctness/graph replay fails, or finalizer overhead consumes the saved tail. No ranked submission without a passing official check clearly beyond the retained best’s 95% interval. Ask before a rewrite beyond half a day or more than five checks on the idea.

**Rejected before a candidate run:** the earlier cache-retention proposal targeted about +5.1% overall by assuming 25% less expert time on both large verify loads. Actual profiled routing has zero repeated 64-row-job weight stages at Verify16 and only about 5.04% at Verify32. That invalidates the proposed reuse-based gain model for this dataset. The three-expert synthetic counter fixture was not representative enough. No cache-policy candidate was compiled, timed or submitted; no official check was spent.

**Status:** no fusion implementation or official check has started. Cache-policy, attention-metadata and large compressed-GEMM drafts created before the new protocol are parked, untested and unpromoted. This session ran the unchanged baseline only; the GPU was stopped.


## Latest decisions

The [exact routing/local-ID fusion check](runs/layer-fused-routing.json) passed 25/25 cases at **2.035848×**, CI [2.005311,2.066386], with preparation in 211.34 seconds. Its one [ranked run](runs/layer-fused-routing-ranked.json) also passed 25/25 at **2.024927×**, CI [1.994553,2.055301], preparing in 252.71 seconds. Neither establishes a gain over the retained **2.035572×** ranked layer. The separate controlled Modal gain of 1.84% is not an official score.

The [ranked router/pooler composition](runs/projections-rk-pooler-cluster1-ranked.json) passed 39/39 cases at **1.4423×**, improving the saved dense best from 1.4273×. The targeted M256 pooler was faster in the preceding check; aggregate confidence intervals overlap.

The [phase-A designated-finisher check](runs/moe-phase-a-finisher.json) passed 11/11 cases at 2.2626× but showed no gain over the retained 2.3249× expert result. It was rejected.

The [complete-layer pooler integration](runs/layer-unpooler-pooler-cluster1.json) passed 25/25 cases at 2.0277×, with preparation completed in 253.74 seconds. It did not improve the retained 2.0356× layer result and was not promoted.

The [attention-to-output prefetch check](runs/layer-attention-o-pdl.json) passed 25/25 cases at 2.0194× (95% interval 1.9891–2.0497×), but did not establish a gain over the retained 2.0356× ranked layer. It was not promoted.

The [two-warpgroup expert check](runs/moe-wgs-two-wg.json) passed 11/11 cases at 2.1246× (95% interval 2.0927–2.1564×). It removed register spills but regressed against the retained expert implementation, so it was not promoted. The ranked expert best remains 2.3249×.

Paid Modal runs are separate synthetic diagnostics. Their compiler reports or timings do not enter this official-results table, the leaderboard, or ranked personal bests.

## Leaderboard snapshots

Positions below are snapshots of official ranked results, not estimates. They may change and remain provisional until challenge review.

| Challenge | Our rank | Ranked speed ratio | Evidence timestamp (Pacific) |
|---|---:|---:|---|
| [Projections](https://openkernelchallenge.dev/problems/muse_bf16_proj) | 345 | 1.4423× | 2026-10-30 05:50:26 PDT |
| [Attention](https://openkernelchallenge.dev/problems/muse_attention_decode) | 183 | 2.5673× | 2026-10-30 05:50:26 PDT |
| [Experts](https://openkernelchallenge.dev/problems/muse_moe_w4a16) | 201 | 2.3249× | 2026-10-30 05:50:26 PDT |
| [Complete layer](https://openkernelchallenge.dev/problems/muse_decode_layer) | 260 | 2.0356× | 2026-10-30 05:50:25 PDT |
| [Prefill](https://openkernelchallenge.dev/problems/muse_prefill_layer) | 333 | 1.4774× | 2026-10-30 05:50:26 PDT |

The original-code controls should measure approximately 1×. Their intervals include 1×, which is consistent with a sound comparison. All-batch fusion is retained as a negative result: its large-batch slowdown cancelled its small/medium gains. The hybrid keeps the original implementation above 32 tokens.

## Router timing by batch

Times below are the official paired benchmark estimates, not profiler totals. A benchmark has its own simultaneous stock baseline, so do not compare absolute times across separate jobs as if machine conditions were identical.

| Candidate | Tokens | Stock SGLang | Candidate | Ratio |
|---|---:|---:|---:|---:|
| Small-batch router | 1 | 10.624 μs | 5.975 μs | 1.778× |
| Small-batch router | 16 | 14.864 μs | 14.792 μs | 1.005× |
| Small-batch router | 32 | 15.280 μs | 15.175 μs | 1.007× |
| Small-batch router | 8 | 26.144 μs | 6.394 μs | 4.089× |
| Small-batch router | 128 | 16.080 μs | 16.233 μs | 0.991× |
| Small-batch router | 256 | 17.696 μs | 17.529 μs | 1.010× |
| All-batch fusion experiment | 1 | 10.640 μs | 5.875 μs | 1.811× |
| All-batch fusion experiment | 16 | 14.976 μs | 10.603 μs | 1.412× |
| All-batch fusion experiment | 32 | 15.424 μs | 10.748 μs | 1.435× |
| All-batch fusion experiment | 8 | 26.128 μs | 6.149 μs | 4.249× |
| All-batch fusion experiment | 128 | 15.984 μs | 30.294 μs | 0.528× |
| All-batch fusion experiment | 256 | 17.632 μs | 57.708 μs | 0.306× |
| Small/medium hybrid | 1 | 10.464 μs | 5.646 μs | 1.853× |
| Small/medium hybrid | 16 | 14.896 μs | 10.607 μs | 1.404× |
| Small/medium hybrid | 32 | 15.472 μs | 10.958 μs | 1.412× |
| Small/medium hybrid | 8 | 26.272 μs | 6.033 μs | 4.355× |
| Small/medium hybrid | 128 | 16.416 μs | 16.214 μs | 1.012× |
| Small/medium hybrid | 256 | 17.600 μs | 17.495 μs | 1.006× |
| Fixed-split router experiment | 1 | 10.400 μs | 5.791 μs | 1.796× |
| Fixed-split router experiment | 16 | 14.560 μs | 10.156 μs | 1.434× |
| Fixed-split router experiment | 32 | 15.232 μs | 10.605 μs | 1.436× |
| Fixed-split router experiment | 8 | 26.272 μs | 6.152 μs | 4.270× |
| Fixed-split router experiment | 128 | 16.400 μs | 14.316 μs | 1.146× |
| Fixed-split router experiment | 256 | 17.632 μs | 23.533 μs | 0.749× |

## Attribution and interpretation

The persistent expert and native attention implementations are derived from [vjeux/okc-2026](https://github.com/vjeux/okc-2026/tree/f35a41ee1c13e316dabbf772830d47b23b453393), under Apache 2.0. The source files retain full licensing and attribution. Our changes restore stock BF16 activation boundaries, repair empty-expert handling, reduce retained expert layouts, and compose the components with the layer. These results do not claim invention of the upstream kernels.

The four-layout expert reproduction is a research comparison: copying that storage across the full model would exceed H200 memory. The one-layout variant reduces prepared weight storage by 75%, but actual serving still needs original-parameter storage ownership, scratch sharing, full-model prefill validation, and a measured memory peak. The same one-layout implementation already passes the isolated long-prompt layer workload.

## Reproducibility

- Original projections (control): job `1459478fff7c4685`, server content hash `f737e43bbce88003f58bdf6adf9d36049c095574f8d81e2ad7b70ed79f04fb2d`.
- Small-batch router: job `7e6e67ff1e144b05`, server content hash `54b34c13a5fc8c3b80ffbae706aa8bc6570b6974e6d00cbe8434d1810288fcce`.
- All-batch fusion experiment: job `7f93d4c2a8484880`, server content hash `71aedd91090f49a1348433f99190139c83cd8f57c3879cc7429ab523360d3eda`.
- Small/medium hybrid: job `6095f087a94a4dc8`, server content hash `8010fc5d5b15231917bc86b675ae2d8be74265cc101d7ec94997cab5f79cd15e`.
- Fixed-split router experiment: job `1bd65c6619e24918`, server content hash `03702eb21a9b3354b7463cb3ebe242d68d67b63cf710c8133865b48acf52ce65`.
- Original whole layer (control): job `cd161496d2734d5e`, server content hash `e6d38e6461aadb63a8303c624238b387d26a115e78d56accd57f19f7b2760396`.
- Whole layer with hybrid router: job `d954fc35930f4464`, server content hash `c8d7757074cd1e5773fc7ce71c7f8f644c770f8cc43bec1ae7ced94478781df1`.
- Isolated expert tile-16 experiment: job `0996005048cd4ea7`, server content hash `44a0c79414e96249e21e9a2654827f6ab71478bf4626fd6a52402361dc236905`.
- Router plus expert tile-16: job `f51410cec72b4a8b`, server content hash `caf7c71586ace492565548a62b827eaed0af841811444e47c3a3c0d24a6d46b4`.
- Bounded router plus expert tile-16: job `042e1064044b4de1`, server content hash `bd2e429ca22092363b002b7c8f81f0a301d838fdb0d59703e4b21782cf818ced`.
- Isolated expert tile-8 experiment: job `9b86d9ee3e7d4047`, server content hash `808e576aa35843e137eae8704f56a9098628ed65473bb07145da1cd64c57c19c`.
- Fused conversion and cache store: job `349b0a7ec65b4461`, server content hash `ed728d777ba4f070ed039e0644dc3244734ba0f6cd41cd87708ae59340459fb0`.
- Corrected conversion/cache fusion: job `cc3ba210079e4cc1`, server content hash `216c8f08315138fa97fc2ea6039c4359a55050fb4fb37d06891abfda8874c1c7`.
- Local-only expert scheduling: job `dd6854a1ff0549c0`, server content hash `fa990a8b777714d64ec75cc95eb532c2f9a991a6731ee837d42b5bccbbbc0068`.
- Attributed persistent MoE reproduction: job `ce1196fa534c466c`, server content hash `d23cf29a0305ae9d58583d891de6ea2877ad27984080c332fddaa917ca55ed8d`.
- Attributed attention reproduction: job `26cc941db6ca4a42`, server content hash `3de1c724d155c0819e97fb14e078a75607b783188cf984e45e37d39aa8c62628`.
- One-layout persistent MoE: job `870d03fb5edd4fb7`, server content hash `856427c5631f9c9afa1cbff1c8c6313c4a5bb09bc53a80f8582a3c91cc168cea`.
- Native attention plus router/expert tuning: job `35f03769f18045e6`, server content hash `90d26e45634630d47997046532b566c26ff0a4af3a5f0655774ab514129ba717`.
- Native attention plus one-layout MoE: job `9b21da6a3f4a4441`, server content hash `fa2afcf8303246c67c6e2e0dee110b5cc0f04860a51144ead5c11e633529f21a`.
- Same implementation on long prompts: job `7bdc35ef747d4f3f`, server content hash `cf1d40f985f74f2a5aafb5ec100181b034802912634d5f7a5a5cc9b269b13a47`.
- Unit-layout fallback validation: job `47a9c433bb7f48b8`, server content hash `e90d9e1c0b629e0b946ab14561a02c0c144c7202ac4f138e546c364fc1f2e5d4`.
- Bounded lossless QKV compression: job `0e2d5eebb29c488a`, server content hash `e877a8877a0b3d49354db10cedbe7f50cf22a7b108d8af076a127fa44028e986`.
- Lossless QKV plus native attention and MoE: job `a14f210fea2b438a`, server content hash `a9e11012a549ebcc74b677612b1c8f77dd78f022e9f661887ae3c95c3fee925d`.
- Parameter-shaped expert storage and byte proofs: job `7b7115d1405c498c`, server content hash `71374a5286e3c80996a6f1d4eabb43b61fa478694fa5c8ca36b6e2f3ed35b2a7`.
- Serialized scratch and graph-replay validation: job `675676b2a5864409`, server content hash `3f8efa0a945403dd3529f99c85a62c07459e034697e0e136db898933bd0d8e52`.
- Parameter-shaped expert storage on long prompts: job `eed26b9fec3942e6`, server content hash `786352f66a27344c82675e7c7489e019c1a5e92c1b44971be3e6dee72d306f30`.
- Specialized pre-expert normalization: job `aef42190a8bc4572`, server content hash `d73a2a25136de1ca65b229000c53d851573019a06563d7e73bf58ef51ad96230`.
- Combined split storage, QKV, and normalization: job `99d6e4c4e1d84dc9`, server content hash `19f567c1d3ae5e89f5d97047158a45996f2bd2bd79c8e0b0cdc3dfa0c51534dd`.
- Four-layout experts in the complete layer: job `d8e8469e72c54ca5`, server content hash `b76d931a80d3ade94ea6dd0a46fd9d63618e415e71a2923d2d1182bd6afea767`.
- Extended split-skinny dispatcher through 32 tokens: job `d71f466d2a594eb8`, server content hash `54530219243cbc140b716f5148ead6f3fed6f6bd5917b53c5c15082e06f3f6ea`.
- Power-of-two resident expert grid: job `ca0b74f6dc984e2e`, server content hash `88ee2e428c3bc48e6b7e65ce4107ef2c9fbaf5570394c38cf53abddaca516a03`.
- Larger-chunk grouped expert consumer: job `cf0cca423d434166`, server content hash `099692f07cf2a25aad75575e627a57c82189321e54e1d9464f26b3c46ad25b77`.
- Grouped experts in complete long-prompt layers: job `dc2a76a503f44566`, server content hash `0bd535c1e43d5323ea03aee06d7ab7103b7a66b1e31b9fd28bd5ec45a98a3951`.
- Two layouts with unchanged faster expert scheduling: job `79e2cbfcccd24863`, server content hash `1efef3d1cd6364739547c6bf8728f324a98a963a72c5850788fb8a09811360ba`.
- Request-aligned attention tile scheduling: job `e71da810280f4d25`, server content hash `465c56b8eb3546a0b381eec6b5dd1ee2d165a5fa046b27f9360d12e2872e3053`.
- Guarded KV cache-line lookahead: job `a61fb656c64c4830`, server content hash `1b3861410b54c00c87f67ab6a0cf35cddec1a7f3753c130157e960cbc7184138`.
- Existing per-request attention family on batched verification: job `ef9a9c3e389e4792`, server content hash `6e1bb776a15743440835a9ac14b01a2d3a7339561f2bc387c5615c9ec274b61f`.
- Compiled attention resource diagnostic; unchanged timed computation: job `4ed8f2f34af24dbc`, server content hash `6d40a84fccd4c0e70ca4d9fc83b17df2322545ef52ad0ecd54291e874e33c79c`.
- Broader static BF16 projection kernels: job `59e0691cb3334f71`, server content hash `bd34ab768f28240bfe6d21e4a3db41e8ddcc4cc00b51ff6fc06f8672f34a67bc`.
- Broader projections with larger-batch CUTLASS pooler: job `6d3fb8a300d644be`, server content hash `96e2013be56deab07b2b439303c9a9b2fc625f16b9f8c1cb33cdec5cba2c7f0c`.
- Share pooler weight loads across two token tiles: job `e1b89989e7e14bd1`, server content hash `f5f88af6c77a7b300dadf4691f5e37cec66f98def45a504f1c851ee03b2d61f6`.
- Smaller QKV output-row tiles at larger batches: job `6e960dd9a7b941de`, server content hash `6a87703dfa82e607dd674e011e2501df94829d0c351faf7988f93b43e8686f77`.
- Broader projections in the complete layer: job `7924a226013944ed`, server content hash `8e8bb4867aaa0db990f782335564f586f9929195cf984aec572d2d746f0880f3`.
- Compact unpooler output fused into the FFN boundary: job `fcaa6b1846374454`, server content hash `68aa5e6a37ba12fd3a0d356c45d7cbb90e81c6d91135a09f8d3fcd92ca1ea382`.
- Compact boundary with corrected architecture setting: job `b320bfcc86fd466c`, server content hash `6dbe763470c938cc702e96c565e20e08948c2984e85f69b1bd4f4be3310bea18`.
- Native router and compact boundary in the complete layer: job `5024a85338b64990`, server content hash `3c33f1ffd009ff99348770d4af7d1c1fca9a956c41169efd06ad3045197d7540`.
- Finish independent expert-input chains before stream join: job `aa7988bfdb4747b4`, server content hash `098a4cf74cb6642589d786dcbeddf88c0c852dfdae31fa43b707dff632d27bb5`.
- Prefetch unpooler weights while native experts finish: job `d0b7fed99b784b03`, server content hash `c472d96be565c71ca77a676737e92d0d539be0f61260962a7744e581a639123e`.
- Prefetch output-projection weights while native attention finishes: job `b391dad5e0e64534`, server content hash `d02824691fbcbdff2121e48a99ea0911e3e6c1490a61d92288337c869843f081`.
- Ranked native expert-to-unpooler weight prefetch: job `f2bca5e0fa9c474f`, server content hash `c472d96be565c71ca77a676737e92d0d539be0f61260962a7744e581a639123e`.
- Fuse exact top-k routing with local expert-ID generation: job `36e468b9084e4c8c`, server content hash `bfea6fb8d4497edd21dca47e169608079e8cd79c503875478ec1d6d39b14a027`.
- Ranked exact routing/local-ID fusion: job `5d788c4ff2fe4413`, server content hash `bfea6fb8d4497edd21dca47e169608079e8cd79c503875478ec1d6d39b14a027`.
- Ranked layer with independent preparation before join: job `cbd853d69af04976`, server content hash `098a4cf74cb6642589d786dcbeddf88c0c852dfdae31fa43b707dff632d27bb5`.
- Ranked complete layer with native router and compact boundary: job `a4e3de73531f416d`, server content hash `3c33f1ffd009ff99348770d4af7d1c1fca9a956c41169efd06ad3045197d7540`.
- BF16 intermediate storage with exact norm reconstruction: job `040b959db2bf47f0`, server content hash `81e4a96601a22562b912aedfde90c001506c85189be9c293d2d446e853985daf`.
- Precomputed BF16 effective expert norm weights: job `801dc49123c44466`, server content hash `1d967504e03967abf4638c5238556bb48a50780cb1feebc4c4c9978370fa20d0`.
- Expert phase tracing; unchanged production computation: job `48d2792ded2f4c8f`, server content hash `8982ab251dc19094eddd47a8502ab8af96f6f30bff8f2dc976a34918fa47fead`.
- Larger expert jobs with fewer repeated weight/decode passes: job `e9150db1c3384ab4`, server content hash `c4eddcafd39a263753901c3229af3aa7acf2f235a84169c950dc62b30b2bd72f`.
- Wider expert kernel with device-safe constant access: job `68721c16fa314c08`, server content hash `ae95f688b00ebc681e39c968108615af778b3f994669649e52e61d06b8ca40c2`.
- Wider expert jobs with fewer consumer warpgroups: job `83c1530718fc43d1`, server content hash `82ec6a98de779d9b12dfbcd2cfb7f9fc41848d472224e7634c1a80df3f67eb19`.
- Designated finisher for split gate/up work: job `60d8214a514d4ec2`, server content hash `917c7f8857f0bb9d5ee430be14a4e55b665f2374099f94a1a1e6320c1c66193a`.
- Five-slot small-M TMA weight pipeline: job `6d0f8ead3dd54993`, server content hash `c2affd7d7be6129c5fa56dfbd89e4a2c21cea1bcbbe216a54391d4cb66697d6a`.
- Combine retained router with faster large-batch pooler: job `f492bc568662491f`, server content hash `818dbb0e467ba941390ba02fc752ec5626d97cbe5fe786b3a4d29822dba5fcdf`.
- Ranked router plus faster large-batch pooler: job `adb94f9d829a44e3`, server content hash `818dbb0e467ba941390ba02fc752ec5626d97cbe5fe786b3a4d29822dba5fcdf`.
- Integrate the large-batch pooler into the retained complete layer: job `cb1c3fe8e3a44ed0`, server content hash `6f7ac02a683238e6ead430e0afd9fee5d16d454701967b77ee492affea3bc308`.
- Counter-free native BF16 hi/lo router: job `6bac6602a33548c6`, server content hash `bde5199b51dfcb913d85d819f815b7396ed2bd9d49860f40cabda0cdc68fb298`.
- Larger grouped-prefill chunk: job `69828662748d4577`, server content hash `0cac03eb7fa16192d1f88d9bed24b4228719a3fd71492429de927e0c63729c16`.
- Ranked attention entry: job `7401cdbfccba4357`, server content hash `3de1c724d155c0819e97fb14e078a75607b783188cf984e45e37d39aa8c62628`.
- Ranked one-layout MoE entry: job `8615b88a11ae425b`, server content hash `856427c5631f9c9afa1cbff1c8c6313c4a5bb09bc53a80f8582a3c91cc168cea`.
- Ranked parameter-shaped expert storage: job `ceca74492e4249e5`, server content hash `71374a5286e3c80996a6f1d4eabb43b61fa478694fa5c8ca36b6e2f3ed35b2a7`.
- Ranked four-layout expert kernel: job `c159271f4be34bc6`, server content hash `d23cf29a0305ae9d58583d891de6ea2877ad27984080c332fddaa917ca55ed8d`.
- Ranked projection entry: job `d678085cee1e43ea`, server content hash `8010fc5d5b15231917bc86b675ae2d8be74265cc101d7ec94997cab5f79cd15e`.
- Ranked lossless QKV projection entry: job `4b9369b70f744fb0`, server content hash `e877a8877a0b3d49354db10cedbe7f50cf22a7b108d8af076a127fa44028e986`.
- Ranked broader BF16 projections: job `c2727ab2ccc145ea`, server content hash `bd34ab768f28240bfe6d21e4a3db41e8ddcc4cc00b51ff6fc06f8672f34a67bc`.
- Ranked native hi/lo router and broad projections: job `465f6f96e0e24a6b`, server content hash `bde5199b51dfcb913d85d819f815b7396ed2bd9d49860f40cabda0cdc68fb298`.
- Ranked complete-layer entry: job `dc620e6578424e08`, server content hash `fa2afcf8303246c67c6e2e0dee110b5cc0f04860a51144ead5c11e633529f21a`.
- Ranked combined layer with FP64 diagnostic: job `5f78520eb2d14876`, server content hash `3f13951a1d3e85cd3ad4d79652b1909409a4aff3056b9b30a008ddb47692ae07`.
- Ranked layer with four-layout experts: job `64206e2a6f034fc3`, server content hash `b76d931a80d3ade94ea6dd0a46fd9d63618e415e71a2923d2d1182bd6afea767`.
- Ranked layer with broad projections and two expert layouts: job `ff7b257c9afa45fd`, server content hash `8e8bb4867aaa0db990f782335564f586f9929195cf984aec572d2d746f0880f3`.
- Ranked prefill entry: job `a85170504acb41c9`, server content hash `cf1d40f985f74f2a5aafb5ec100181b034802912634d5f7a5a5cc9b269b13a47`.
- Ranked prefill with grouped experts: job `23badfd3441943cc`, server content hash `0bd535c1e43d5323ea03aee06d7ab7103b7a66b1e31b9fd28bd5ec45a98a3951`.

The `runs/` JSON files retain the official correctness diagnostics, fresh test inputs’ seeds, evaluator hashes, machine metadata, paired timings, and uncertainty estimates. Original source contracts and generated profiles remain in the corresponding problem directories. No results are fabricated or inferred from local syntax checks.

Remaining limits: the official hidden workload determines correctness and timing; passing it does not prove all possible inputs. The custom code must still be validated in actual eight-GPU serving before a prize-track performance claim.
