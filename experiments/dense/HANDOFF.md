# Dense lane handoff

## Round2 — small-T whole-layer stream ceiling

**The ≥5% decode ceiling gate passes twice. This is a ceiling measurement, not a production speedup. No production megakernel was built.**

Replacing only D1/V1 with the measured zero-math stream gives **+7.024–7.151%** six-load decode geomean on matched captured-event timings, or **+7.918–8.044%** on independently measured native spans. D16/D32/V16/V32 stay unchanged in this calculation.

### Current layer versus one continuous read kernel

Control: frozen copy of MAIN's repaired `work/candidates/decode_alygg77_bf16.py`, SHA `053dbacc8f0de4e6111ccb89cb91f629840852cf482a6ca423fe0eaff419c086`; current source still matched at completion. This control has9 native kernels per tested call, rather than the earlier approximate14-kernel chain.

Values below are native spans, arithmetic means of two runs, each using the same three fixed fresh-input fixture seeds. Per-load rows weight sliding/full46:16.

| Case/load | Mean payload MB | Current layer µs | Continuous stream µs | Gap µs |
|---|---:|---:|---:|---:|
| D1 sliding |214.533|71.994|60.711|11.283|
| D1 full |225.052|75.742|50.631|25.111|
| **D1 weighted** |—|**72.961**|**58.110**|**14.851**|
| V1 sliding |346.656|107.206|88.416|18.790|
| V1 full |376.049|116.487|82.195|34.292|
| **V1 weighted** |—|**109.601**|**86.811**|**22.790**|

The captured-event weighted means, with no overhead subtraction, were D1 76.866→62.686µs and V1 113.735→92.477µs. These are separate observations, not corrected native estimates.

For each timer:

`decode_ratio = ((D1_layer / D1_stream) * (V1_layer / V1_stream)) ** (1/6)`

Both runs clear5% on both timers. The remaining margin is narrow: adding the same **4.42–4.68µs of exposed work to each improved load's weighted event time** would reduce the result to5%. Native-span accounting permits6.00–6.22µs. These are arithmetic budgets, not evidence that actual math/synchronization fits.

### What was read, and what was idealized

- Original BF16 qkv, o, pooler and unpooler weights; unpooler first1024 rows at T≤8.
- Original unrounded FP32 router, all norm/gate/bias vectors.
- Every actual unique local routed expert's NVFP4 W13/W2 codes, block scales and global scales, once per layer. Actual local-expert counts: D1 sliding2/4/2, D1 full2/2/3, V1 sliding7/9/6, V1 full8/6/9.
- Exact causal union of FP8 K/V rows through the actual page table, including new block tokens and both heads. Per-fixture KV bytes: D1 sliding524,288; D1 full20,480,256; V1 sliding526,080; V1 full20,482,048. Only verified physically adjacent covered rows were coalesced; no holes or omitted keys.
- Original weight/cache allocations are read directly. Only descriptors and a16-byte packet holding the two exact FP32 KV scales plus8 padding bytes are prepared. Full32-entry global-scale arrays add a few disclosed conservative extra bytes.

**This is the native public weight format, not the retained decoder's compressed physical traffic.** No unverified lossless-compression credit was used. Selection of actual routes/pages is a free frozen-fixture oracle, and the baseline has already generated new KV. The bound omits real arithmetic, attention, normalization, activation/output traffic, KV writes and inter-stage dependencies/barriers. A real submission cannot cache this input-derived plan outside its timed call.

### Reader correction and verification

The first grouped reader was byte-correct but not a useful roof: its extra dependent descriptor lookup and concentrated terminal KV-copy work gave only+0.939% native D1/V1 headroom. Flattening **the identical physical chunks** into individually striped ring stages removed that measurement artifact. It did not repack model data or change precision/routing.

`stream_flat.cu` uses one132-CTA persistent kernel, one warp/CTA, four32KiB TMA slots,25 registers and zero local bytes in pure mode. Every transfer completes before slot reuse. A separate exact-XOR mode checks every payload word; completed-byte/stage/chunk counters are checked per CTA. Empty,16-byte and multi-ring/mixed-copy smoke cases pass.

- A/A controls within±1% for all relevant case/load aggregates, both runs.
- Current decoder checked against the public oracle before/after timed fixtures.
- One native kernel per stream trace; native layer span is first-start to last-end, not the sum of stage times.
- Independent cold-L2 NCU proof:18/18 original-reader observations,18/18 first-flat observations,12/12 repeat-flat observations pass. Timed and profiled payload/route/page coverage match. Extra DRAM traffic is explained by descriptors plus small code/sector overhead.
- No official checks or ranked submissions.

### Optional D16

No D16 improvement credited. The grouped reader is the better of the two tested readers, but its matched weighted native span is158.821µs versus144.863µs for the current layer (**−13.958µs gap**). Flat individual copies are worse on batched fragmented KV. Keep D16 unchanged; this is not a proof against every possible D16 reader.

### Evidence and next move

- Combined result: `work/lanes/dense/stream_bound/CEILING_REPORT.json`.
- Audited first/repeat aggregations: `stream_bound/SUMMARY.json`, `stream_bound/REPEAT_SUMMARY.json`.
- Baseline and failed compile evidence: `outputs/openkernel-challenge/modal-lab/layer-okc-h200-dense-03/`.
- Original grouped timing: `.../layer-okc-h200-dense-04/layer-stream-bound.json`.
- Flat first/repeat timings, native traces, all NCU CSVs/child manifests: `.../layer-okc-h200-dense-05/{layer-stream-flat.json,layer-stream-flat-repeat.json,stream-counter-proof.json,stream-flat-counter-proof.json,stream-flat-repeat-counter.json}`.
- Reproduction sources and offline aggregation script: `work/lanes/dense/stream_bound/`. GPU invocations use `OKC_LANE=dense`, workspace-aware `work/modal/run_layer.py`, and the default `secondary-workspace` profile.

**Next:** PM can scope a D1/V1-only implementation experiment against the exposed-work budget above. No production implementation or promotion is part of this ceiling-only task. Shutdown/accounting receipt is in the final LOG entry and combined result.

## Round1 decision — dense projections

**Done via measured reasons; no winner to integrate.** Keep the unchanged alygg77 BF16 projection routes. No candidate was promoted, no official check was justified, and no ranked submission was made.

- Large-T: exact-shape raw WGMMA had sufficient idealized headroom to justify a real implementation. The correct full-K TMA implementation, including two-CTA weight multicast, lost on every tested shape after compiler serialization was repaired.
- Small-T: **≤11-bit lossless BF16 storage is achievable**: measured10.957234–10.968246bits/value including metadata. Its exact GPU prefix decoder was2.219–3.021× slower than the matched fixed12-bit core, so it did not justify a GEMV integration.
- Summary: `work/lanes/dense/RESULTS.json`. Chronology, preregistrations, failures and findings: `work/lanes/dense/LOG.md`.

## Baseline and measurement contract

Baseline: `work/prior-art/alygg77/kernels/muse_bf16_proj.py`, unchanged. The source SHA is recorded in the A/A JSON. The quoted1.5502× is the upstream reported official result, not a new lane score.

`screen_dense.py` adapts the existing paired benchmark and captured-event diagnostic:256MiB cold flush; external CUDA events inside each graph; independent captured inputs/outputs; fresh public inputs; FP64 public reference, per-row and probe-column checks; no replay-cost subtraction. Sum five projection latencies per load, then geometric-mean six loads. The unpooler uses1024 rows at T≤8. Router and all unaffected loads stay unchanged.

Calibration:36 balanced triplets for each of30 shapes,3240 reference/probe checks passed. A/A candidate1.000046×, baseline-repeat0.997709×; **every load within±1%**. Initial12-pair calibration was inconclusive and is retained, not discarded. Warm baseline prepare8.842s; first cold prepare204.406s, within300s.

Evidence root below: `outputs/openkernel-challenge/modal-lab/layer-okc-h200-dense-02/`.

## (a) Large-T tensor efficiency

### Bound

`large/raw_wgmma_bound.py` measured112 exact-shape/resource combinations: four BF16 matrices, T128/T256, seven tile geometries, one-/three-stage shared-capacity envelopes. Two exact dyadic fixtures and all observed accumulator/checksum checks passed. Full K was computed without split-K.

Long-K per-CTA rates reached approximately4053–4081 BF16 FLOP/clock against4096 architectural FLOP/clock. Grid occupancy and waves still matter. The raw full-event spans include synthetic initialization/checksum work; the stamped core spans omit transfers, pipeline control and production epilogues. Neither is an official latency.

An explicitly optimistic model using measured core spans,4.6TB/s byte transfer, free launch/pipeline/epilogue, and upstream official unchanged loads gave **7.554% board headroom**. That passed the3% implementation gate; it was never claimed as a real speedup. See `large/ceiling_gate.json` and `large_raw_fixed.json`.

The brief's19.3GFLOP pooler-router number belongs to decode. This board's standalone pooler T256 is17.180GFLOP; the separate router is retained in board aggregation.

### Real implementation

Sources: `large/tma_dense.py`, `large/screen_tma_dense.py`. Full-K independent output tiles, BF16 TMA staging, FP32 WGMMA, BF16 epilogue; matched plain and two-CTA weight-multicast variants. No global partials, split-K, precision change or input-dependent host work.

The first build hit PTXAS C7520. A per-stage WGMMA fence after the ready wait removed that diagnostic without changing geometry or transfers. The repaired screen passed **6912** public reference/probe checks. Best measured arm per shape:

| Projection | T128 base µs | T128 prototype µs | T128 base/prototype | T256 base µs | T256 prototype µs | T256 base/prototype |
|---|---:|---:|---:|---:|---:|---:|
| qkv_gate |18.592|41.696|0.4465|22.656|42.160|0.5359|
| o |11.424|13.120|0.8689|13.392|16.448|0.8202|
| pooler |24.576|41.984|0.5846|28.368|42.784|0.6629|
| unpooler |23.696|26.352|0.8889|28.352|36.080|0.7872|

Ratios are median paired ratios, not necessarily quotients of the displayed medians. Every fastest arm was plain TMA; no multicast arm won. Some individual variant controls exceeded1%, so no precise small-gain inference is valid. All losses were much larger, and there was no positive arm to repeat/promote. This rejects these implementations, **not every possible custom GEMM**. Raw compute throughput did not cover required staging/control costs.

Evidence: `dense-tma-screen.json` (first build), `dense-tma-fence.json` (repaired), `large_raw_fixed.json`. Runner logs/generated CUDA can be overwritten by later runs of the same script; the separately named JSON files preserve each experiment.

## (b) Small-T weight bytes and decode cost

`codec/codec_bound.py` and `codec/codec_probe.cu` measured generator seeds5151/6161/7171: four matrices plus the1024-row unpooler prefix,15 cases. Six concrete encodings per case were packed and bit-roundtripped, including exhaustive65536-code CPU checks. GPU decoder/checksum paths passed. Peak CUDA allocation659.05MB.

Per-matrix entropy is approximately10.464bits. The pooled histogram is10.586bits because matrices have different scales; it is not the per-matrix coding limit. Fixed11-bit payloads require escapes. Compact tile512 side tables and dictionary11 still exceed11bits/value. Odd full-exponent window alignment helps the unpooler, but it cannot be substituted for exact reconstruction using unchecked FP32 additive corrections: seed5151/unpooler had3 nonexact corrections.

The original fixed-window11 decoder, even omitting all escape execution, was5.88–10.82% slower than12 in its cold decode diagnostic. That diagnostic's checksum-dominated register loop did not isolate decode cost, and its slow scalar-read loop is not an HBM roof. No conclusion depends on mislabeling those numbers.

### Actual sub-11-bit format

`codec/huffman_bound.py` uses sign+mantissa8, Huffman-coded full exponents, independently addressable128-value blocks,32-bit word offsets and aligned bitstreams. Storage includes offsets, padding, lookup tables and header. GPU validation reconstructs **every original BF16 bit**, not a corrected floating-point approximation.

Across14 matrix/seed cases, including two small-T unpooler prefixes:

- Storage: **10.957234–10.968246bits/value**.
- Full GPU bit comparisons:14/14 passed.
- Captured checksum replays:1008/1008 passed.
- Paired exact Huffman/fixed12 decoder slowdown: **2.219–3.021×**.

Initial seed5151 timings illustrate the cost:

| Projection | Fixed12 core µs | Exact Huffman decode µs | Stored bits/value |
|---|---:|---:|---:|
| qkv_gate |29.168|70.624|10.96814|
| o |22.352|66.768|10.96012|
| pooler |38.512|85.664|10.96711|
| unpooler |38.640|89.312|10.95792|

Both arms use the same low-byte layout, output consumption and launch geometry. The fixed12 arm clips rare tails and omits correction, deliberately favoring the control; this is a decoder-cost experiment, **not an integrated GEMV comparison**. The exact decoder already loses before multiply/reduction work. Seeds6161/7171 repeated the loss.

Byte-only optimistic board headroom is5.286% for Huffman, but that assumes free decoding and scales overheads that do not actually shrink. It does not survive measured prefix-decode cost. No production codec/GEMV candidate was built or submitted. See `codec/byte_ceilings.json`, `codec-bound.json`, `huffman-bound.json`, `huffman-6161.json`, `huffman-7171.json`.

## Integration and next move

**Do not port these diagnostics into decode.** They import the public reference only as external measurement tools; they are not submission files. This lane did not modify upstream kernels, production callers or ranked files. New diagnostic helpers retain approved-source attribution; no new external implementation was imported.

MAIN should retain alygg77 dense projections and integrate only independently qualified expert/attention winners. These results leave the wider custom-GEMM/variable-code design space open, but provide no dense winner or official score claim.

Official checks used:0. Ranked submits:0. Sandbox shutdown and observed billing are recorded in the final LOG entry.
