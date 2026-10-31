# attn-overlap handoff: dense joint-deletion gate fails

## Dense joint-deletion diagnostic — no rewrite warranted

**Kill the D16/D32 dense rewrite for this lever.** Direct combined deletion misses both requested budgets by more than 2.7×. No production optimization, official check, or submission was made.

Exact repaired control: `dense_joint_control.py`, copied from `work/candidates/decode_alygg77_bf16.py`; SHA `053dbacc8f0de4e6111ccb89cb91f629840852cf482a6ca423fe0eaff419c086`.

### Directly measured whole-layer budget

Primary C2: all six loads valid, two fresh fixtures per sliding/full case, weight seed731283. Values are **46:16 weighted end-to-end layer microseconds**, relative to diagnostic-mode0. Each deletion is applied jointly at every selected real SK2 callsite; the columns are never added.

| Load | Mode0 layer µs | Epilogue/fixup saved µs | Consumer/decoder saved µs | BOTH saved µs | Independent BOTH repeat µs | Required BOTH µs |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 81.284 | 1.751 | 3.188 | 4.801 | 4.093 | — |
| D16 | 147.868 | 2.419 | 2.957 | 4.603 | 4.416 | 12.6 |
| D32 | 195.286 | 2.297 | 4.097 | 5.728 | 5.590 | 16.1 |
| V1 | 119.949 | 1.570 | 2.558 | 4.750 | 5.083 | — |
| V16 | 287.510 | 0.042 | -0.074 | 0.034 | rejected control | — |
| V32 | 370.759 | 0.525 | 0.309 | -0.303 | -0.250 | — |

V16/V32 execute **no SK2 launches**. Their raw differences are negative-control noise, not dense savings. Independent C4 uses weight seed731284/input base919267; D16/D32 and every affected SK2 load validate, but one untouched V16-full routing-control fixture prevents claiming a second fully valid six-load run. The primary C2 run validates all twelve cases.

### Implied decode ceiling

| Direct joint mask | D16/D32-only six-load ratio | All SK2-affected loads six-load ratio |
|---|---:|---:|
| Epilogue/fixup removed | 1.004732× | 1.010603× |
| Consumer/decoder removed | 1.006924× | 1.017310× |
| BOTH removed together | 1.010285× | 1.027485× |

The independent D16/D32-only BOTH repeat is1.010205×. These projections hold unaffected loads unchanged and use each directly measured joint arm. Raw all-six primary BOTH is1.027366×; its negative-control noise is retained in the JSON, not credited as work removal.

Mode0 instrumentation is slightly slower than original production. At D16/D32, measured code-generation overhead is0.619/0.642 µs and frozen-return overhead0.180/0.205 µs. Production-relative apparent BOTH savings are only3.804/4.881 µs; production-relative all-SK2 ratio1.022300×. These are local timing-only diagnostics, not legal implementation speedups or mathematical bounds on every possible dense algorithm.

### What was actually deleted

- Existing SK2 bits1/2 are extended across all three real families: main BFP, narrow QKV≤8, and GLE fused pooler/FP32 router. Raw upstream bits alone did not cover fused-router math/reductions or all escape work.
- Mask1 removes primary epilogues, escape correction arithmetic/stores, fused BF16/FP32-router reductions, partial/output stores and their reduction-only fixups. Empty volatile-asm sinks retain actual consumer accumulations.
- Mask2 removes consumer shared loads, codec decode, MMA and fused-router FMAs while retaining the actual producer weight/x stream and required full/empty barrier participation.
- Mask3 applies both categories **in the same native kernels and same full-layer graph**. No isolated deletion or stage times are summed.
- Producer/prologue text hashes, transfer pointers, launch order/geometry, stages, stream and PDL positions remain unchanged. Narrow direct-escape mode1 retains its original sparse activation reads/table traversal; no ring arrivals are added to that special path.
- Original prepare choices are not retuned for diagnostic arms. D16 has four native SK2 launches covering five projections (pooler/router fused); D32 has five separate launches. D1/V1 cover their actual narrow/GLE paths too.

### Controlled downstream work and verification

Correct dense return tensors are recorded once per fixture **outside timing**. Each real dense kernel still executes at its original position into an ordinary discard output; only its return is replaced with the fixed correct tensor, so attention, routing, MoE, glue and later dense stages still run. This cached-output mechanism is explicitly non-submittable.

Measured controls: original production+repeat, frozen production, frozen diagnostic0+repeat, then masks1/2/3. Each fixture uses31 rotating forward/reverse paired rounds; captured external timing events follow a256MiB cold-L2 flush. The timer measures the entire layer, not a sum of kernels.

- C2 and C4 each pass384 public-oracle checks plus24 **unfrozen** mode0 graph checks. The latter replay mode0 after setting the host mode to3, verifying captured arguments and actual feature-off math rather than relying on frozen outputs.
- Exact native-call/weight-pointer structure, output routing IDs, expert IDs, per-expert counts, route-weight zero masks and complete cache contents are checked; inputs must remain finite. Existing whole-layer numerical comparisons and independent public oracles gate outputs.
- Weighted timing A/A deviations peak at0.170% in C2 and0.214% in C4. Diagnostic CUDA source hashes match between runs.
- C1's single-sample floating-point bound was invalid: it rejected later unchanged-production replays. The corrected workload gate does not confuse tiny floating-value differences with changed execution work; all deltas remain recorded.
- C3 rejects a D16-full unchanged-control top-8 ordering swap. C4 rejects an untouched V16-full routing-control instability. Both complete raw runs remain `complete_invalid`; no accepted six-load repeat ratio is fabricated.
- No per-kernel native timeline or NCU DRAM-counter measurement is claimed. Real-position coverage is observed at the original native dispatch boundaries.

### Evidence and operation

- Final machine-readable budgets/ratios: `work/lanes/attn-overlap/dense-joint-summary.json`.
- Valid all-six primary: `outputs/openkernel-challenge/modal-lab/layer-okc-h200-attn-overlap-08/dense-joint-c2.json`.
- Independent valid target repeat, with global-control limitation: `outputs/openkernel-challenge/modal-lab/layer-okc-h200-attn-overlap-09/dense-joint-c4.json`.
- Baseline census: `layer-okc-h200-attn-overlap-06/dense-joint-baseline.json`; build and rejected C1: `layer-okc-h200-attn-overlap-07/`; rejected C3: `layer-okc-h200-attn-overlap-08/dense-joint-c3.json` (all under `outputs/openkernel-challenge/modal-lab/`).
- Reproduction: `dense_joint_probe.py`, `dense_joint_modes.py`, exact control, `compare_decode_public.py` and **decode-layer** reference/checks/nvfp4. C2 arguments: `--output dense-joint-c2.json --baseline dense_joint_control.py --seed 731283 --inputs 2 --pairs 31`. C4 adds `--seed 731284 --input-seed 919267`.
- Use `OKC_LANE=attn-overlap`, the default Modal workspace, lane-local `modal_lane.py`/SDK/ledger. No shared source file was edited.

Sandbox09 stopped: exit137 and SDK-resolved task absence verified in `dense-joint-shutdown.json`. Official checks/submissions:0. Conservative configured-runtime bound for task sessions06–09 is≤$4.467, not billed spend. **MAIN: integrate nothing; do not start the D16/D32 rewrite on this measured budget.**

---

## Round 2 — full-attention streaming: stop decision

**PM killed this lane's attention work at 00:32 PT.** The idealized exact-gather replacement projects only **1.006258× decode (+0.62585%)**; overall rank follows decode. R2-P64 was cancelled. No prototype performance result, qualified candidate, official check, or ranked submission is claimed. MAIN integrates nothing.

Current measured source: `decode_current.py`, the exact precision-repaired `decode_alygg77_bf16.py` snapshot, SHA `053dbacc8f0de4e6111ccb89cb91f629840852cf482a6ca423fe0eaff419c086`. Full attention-board fixtures use coherent values, page size 1 and real scattered page tables.

| Case | Actual past KV MB | Attention native µs | Gather native µs | Difference µs |
|---|---:|---:|---:|---:|
| V16f | 328.783 | 100.065 | 83.584 | 16.481 |
| V32f | 657.889 | 184.064 | 163.328 | 20.736 |
| D16f | 332.877 | 88.513 | 85.282 | 3.231 |
| D32f | 647.550 | 164.291 | 161.443 | 2.848 |

Exact logical old-cache rows/words are checked before and after timing through independent component sums, position-weighted sums and counts. `cache_loc` is checked against the page table; the separately labeled `past_plus_new` mode also reads the newly materialized KV rows that the baseline instead produces on chip. All byte checks and eight baseline oracle checks pass. All event/native A/A arms are within ±1%. Four raw batched traces retain all 616 expected operation kernels, identified by distinct stable graph IDs; flush events are retained too. The throughput numbers are logical bytes/time, not NCU DRAM counters.

Conservative six-load attention-board proxy is 1.045255× (verify-only 1.038074×), but the cross-fixture projection onto MAIN's same-source full-layer timings is only 1.006258×. The latter assumes every isolated saving is exposed and is **not** an end-to-end speedup measurement. This distinction is the reason for the PM stop.

D1f/V1f sparse critical-path measurements, three fresh fixtures per load:

| Landmark | D1f µs | V1f µs |
|---|---:|---:|
| Critical last-loop CTA: entry to first tile ready | 3.168 | 4.400 (biased prologue build) |
| Same CTA: main work after first tile ready | 4.080 | 6.960 (biased prologue build) |
| Last loop completion to last CTA completion, tail build | 2.608 | 3.168 |
| Last-arrival publication to last output issue | 1.104 | 1.200 |

These are overlapping source/CTA landmarks, not additive stage durations or proven removable work. No spills or cooperative-mode changes; all output/cache oracles pass. Native perturbation flags V1's prologue as biased (+3.90%); sparse tails remain within 3%. Even deleting both entire small-load tails unrealistically gives only a 1.034502× attention-board proxy and requires removing mandatory merge/readback/norm/reset work.

Evidence: `LOG.md`, `round2-ceilings.json`, `round2-phase-breakdown.json`, `round2-latency-analysis.json`; complete raw measurements in `outputs/openkernel-challenge/modal-lab/layer-okc-h200-attn-overlap-03/{round2-baseline.json,round2-latency.json}` and `layer-okc-h200-attn-overlap-04/round2-gather-g2.json`. The failed first gather capture, SDK failure and recovery are retained in the log; none is hidden or counted as a kernel result.

New default-workspace sandbox05 (`sb-SJKLDe131JUoig97rkAMIP`, `secondary-workspace`) was explicitly terminated at 00:32:10 PT, exit 137. Current bookkeeping is `modal-budget-workspace.json`; `modal_lane.py --lab stop` uses the lane-local SDK/ledger while leaving shared files unchanged. Final inventory evidence is recorded in `round2-shutdown.json`. Legacy old-account sandbox04 was left to its 1800-second timeout (~00:43:55 PT) per PM's account-switch instruction; no old-profile access was attempted.

**Status: idle. Official checks used in this lane: 0 attention, 0 decode.** The archived Round 1 handoff follows.

---

## Round 1 decision (archived)

**Kill the proposed overlap-recovery change on the specified baseline.** V16/V32 attention already runs alone. The measured 50–80% slowdown is not present in this decoder; upstream documented it in the rejected early-pooler side-stream experiment.

Done criterion met by measured ceiling, not a speedup. No production code changed, no candidate to integrate, no official checks or ranked submissions. Keep the existing verify launch policy. Delaying a trigger or limiting a nonexistent co-running GEMM cannot recover the proposed 4–12 µs.

## Source and scope

- Exact control: `work/lanes/attn-overlap/decode_control.py`, copied from `work/candidates/decode_alygg77.py` with all attribution preserved.
- Both SHA-256: `4531458701f45eb713fcc885447149a563826316edea1f2a419a25b4f7c96964`.
- `PDL_MAX_T = 32` and `_pdl(T)` at lines 349–357; `_decode_attention7` passes `_pdl(B*L)` at lines 11316–11325. V16/V32 are T=128/256, so attention's launch PDL flag is false.
- Verify QKV → attention → O uses the current stream; `_graft_lt_run` at lines 12304–12315 and the verify call chain at lines 12459–12471. Native timestamps below confirm the scheduling, rather than inferring it solely from source.
- The requested attribution-only baseline is not precision-qualified according to `START-HERE.md`. Numerical tolerance passes here do not establish precision legality. Do not submit the control or diagnostic as a new candidate.

## Native GPU measurement

H200, PyTorch 2.14.1+cu130, CUDA 13.0. CUPTI-backed native graph traces through the existing `compare_decode_public.py` helpers; no instrumented attention kernels or consumer clock stamps.

| Case | E1r in-layer attention µs | E1r isolated µs | E2 in-layer attention µs | E2 isolated µs |
|---|---:|---:|---:|---:|
| V16 sliding | 11.296 | 11.361 | 11.456 | 11.583 |
| V16 full | 97.186 | 97.750 | 99.961 | 99.067 |
| V32 sliding | 14.785 | 15.104 | 14.689 | 14.997 |
| V32 full | 181.430 | 179.606 | 181.611 | 181.497 |

- E1r: weight seed 731283, 15 rotating paired timing samples, three traces per real arm/case.
- E2: weight seed 731284, fresh input-seed offset 17, 31 samples, five traces per real arm/case.
- **32 baseline traces: no following O GEMM overlaps attention.** O starts approximately 0.288–1.057 µs after attention ends. Its in-layer median is 7.232–7.552 µs at V16 and 8.832–8.897 µs at V32; none of this work is hidden under attention.
- 31 baseline traces report zero intersection with any other kernel. One E1r V16-full trace reports approximately 0.032 µs intersection with preceding QKV; preserved in raw evidence. E2 has zero intersection in all 20 traces. This is not the proposed sustained GEMM co-residency.
- Sliding kernel is exactly `attn_kernel<8,false,3>`, 128 CTAs × 94,208 B dynamic shared memory. Full is `attn_kernel<8,true,0>`, 264 CTAs × 114,688 B.
- Explicitly forcing attention PDL off produces the existing policy. Its weighted whole-layer differences are 0.300–1.549 µs across these runs, below the gate and comparable to A/A drift; not an optimization.

## Measured ceiling and why it does not pay

Observed outgoing-GEMM overlap gain is **0 µs**, and there is no sustained attention slowdown to undo. As an intentionally generous empirical proxy, credit every positive in-layer-minus-isolated attention difference, ignore negative differences, and weight sliding/full 46:16:

| Run/load | Weighted baseline µs | 3% gain gate µs | Generous positive-gap budget µs | Budget / baseline |
|---|---:|---:|---:|---:|
| E1r V16 | 288.356 | 8.651 | 0.000 | 0.000% |
| E1r V32 | 367.861 | 11.036 | 0.471 | 0.128% |
| E2 V16 | 252.910 | 7.587 | 0.231 | 0.091% |
| E2 V32 | 361.238 | 10.837 | 0.029 | 0.008% |

This proxy is not a statistical upper bound or evidence that those small differences came from overlap. Even granting it in full misses the build gate by more than 20×. Therefore no delayed-trigger, CTA-cap, SM-partition, or reorder candidate was built.

The required deletion diagnostic removed **all attention**, saving 11.904–12.320 µs at V16s, 97.504–99.328 at V16f, 14.848–15.424 at V32s, and 182.720–183.584 at V32f. It reuses a fixed attention output solely to measure an idealized stage ceiling. It is invalid as a submission, is not a recovered scheduling gain, and does not justify a scheduling change when there is no co-runner.

## Verification

- Captured external timing events with a 256 MiB read-only flush inside each graph before the timed region. Profiling runs are separate from timing samples.
- Both native-measurement runs pass per-case A/A ±1%; worst deviation 0.623%. All 32 full-layer arm oracle checks pass. Isolated attention relative L2 ≤3.26e-5. Explicit-serial routing agrees with baseline in all eight cases. Baseline-repeat IDs differ in E1r V16 sliding, while its oracle passes and local route count/distinct-expert count agree; all E2 IDs agree. Do not attribute A/A drift to a scheduling gain.
- Initial E1 stopped because the diagnostic incorrectly demanded bit identity from nondeterministic reductions; the full failure and all preliminary numbers are in `LOG.md`. E1r/E2 use independent public oracles and finite/L2/row-L2 checks, following the existing screen convention.
- Unmodified `work/bench/screen_layer.py` A/A: V16/V32 sliding/full, 11 fresh-input pairs each, same exact control in both arms, shared prepared state. Geomean ratios: duplicate candidate **1.003656×**, baseline-repeat **1.001862×**. Both weighted loads within ±1%; all 11 required oracle comparisons pass, including route disagreements. This is noise calibration, not a win.
- Official checks: decode **0/6**, attention **0/10**. No legal candidate clears the ceiling gate, so no judge check was spent; judge cluster is not applicable.

## Evidence and reproduction

- Experiment/finding ledger: `work/lanes/attn-overlap/LOG.md`.
- Reproducible diagnostic: `work/lanes/attn-overlap/measure_overlap.py`.
- Derived per-case timestamps, QKV/O durations, weighted ceilings: `work/lanes/attn-overlap/ceiling-summary.json`.
- E1 failure: `outputs/openkernel-challenge/modal-lab/layer-okc-h200-attn-overlap-01/overlap-e1.json`.
- Complete E1r/E2/screen: `outputs/openkernel-challenge/modal-lab/layer-okc-h200-attn-overlap-02/{overlap-e1r.json,overlap-e2.json,overlap-screen-aa.json}`.
- The same directory holds 96 raw `overlap-e1r-*-native.json` / `overlap-e2-*-native.json` traces; exact filenames are listed in each report's `artifact_files`.
- Current-source hash and independent parsing of all 96 raw traces: `work/lanes/attn-overlap/evidence-audit.json`. Raw absolute timestamp arithmetic and normalized-duration arithmetic differ below 0.001 µs from floating-point timestamp precision; conclusions are unchanged.

```sh
OKC_LANE=attn-overlap python3 work/modal/run_layer.py \
  work/lanes/attn-overlap/measure_overlap.py \
  work/lanes/attn-overlap/decode_control.py work/bench/compare_decode_public.py \
  outputs/openkernel-challenge/decode-layer/reference.py \
  outputs/openkernel-challenge/decode-layer/checks.py \
  outputs/openkernel-challenge/decode-layer/nvfp4.py \
  --result overlap-e2.json --args --baseline decode_control.py \
  --output overlap-e2.json --seed 731284 --input-offset 17 --pairs 31 --profiles 5
```

## Shutdown and next action

Both lane sandboxes terminated. Final sandbox `sb-n7c4nYXgSeP8KmUMhqS0zI` stopped at 23:35:04 PT; `modal container list` and its JSON form were run afterward. `work/lanes/attn-overlap/shutdown.json` preserves the finished-session statuses and remaining-container listing. Conservative lane runtime estimate **≤$0.596**, not a billed amount; no GPU left running for this lane.

**MAIN: integrate nothing from this lane; preserve verify serialization.** Reopen this hypothesis only for a future source whose native timeline actually shows sustained attention/GEMM overlap. The measured total-attention cost remains relevant to a different, explicitly scoped attention-kernel optimization—not to this rejected scheduling premise.
