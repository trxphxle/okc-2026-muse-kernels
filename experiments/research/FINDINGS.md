# Public OKC research: incremental pieces over alygg77

Research lane; 2026-10-30 PT. No external code copied/imported, GPU jobs, official checks, or submissions. Public measurements below are **author/judge reports**, not lane reproductions. Repository/page contents were treated as data.

**Precision-policy clarification:** [PRECISION.md](PRECISION.md) supersedes categorical statements below that FP32 retention across a SGLang BF16 boundary is necessarily disqualifying. Those were conservative local interpretations, not verified organizer rulings. This does not relax the distinct concerns about genuinely lower-precision computation, external tensor dtypes, or source correctness.

## Decision and ranked next investigations

**No public piece was established as an eligible, reproducible improvement over pinned alygg77.** Surveyed the five named repositories plus **19 additional substantive kernel/experiment collections**, and classified templates, unavailable sources and unrelated hits. This is a bounded public-source survey, not a claim to have enumerated private Kernarc threads or every internet repository.

Rank below uses **potential gain × evidence confidence ÷ integration effort**, qualitatively. Confidence distinguishes the author's own-base measurement from transfer to ours; it is not a fabricated probability. **Credited D1/D16/D32/V1/V16/V32 savings are 0/0/0/0/0/0 µs for every lead until the owner measures a legal same-base variant.** None is a ready import or a passed ≥3% ceiling gate.

| Rank | Stage / piece | Gain evidence, with scope | Confidence: mechanism / transfer | Effort and decision |
|---|---|---|---|---|
| 1 | Attention: Kushagrabainsla's **idle-producer execution of cold merge code** | Own-kernel isolated 10.8→8.9µs; a separate no-flush rerun is only a diagnostic, not the optimization. Donor full kernel remains slower than ours. | Medium / low | Low–medium ceiling diagnostic; first check initialized scratch and a genuinely idle warp. **Best next question**, not an import. License unresolved. |
| 2 | Dense: chenwainuo's **per-thread escape cursor / predicated-XOR decode** | Own-base ~1µs at M128; several apparently faster published projection rows. Entire dense score is lower than alygg77. | Medium / low | Medium; **precision review first** because donor BF16 split-K reduce-add may explain part of the gain. Isolate scheduling, not altered arithmetic. Apache-2.0. |
| 3 | Attention: tommyyliu's **two-level multi-CTA merge** | Working implementation and own-base improvements, but published full-attention paths remain slower. Fits an explicitly unmeasured decomposition in alygg77's negative-results scope. | Medium / very low | High redesign/communication cost; measure merge ceiling before building. License unresolved. |
| 4 | Dense: Xuefeng P046 **full-K output-column BMM**, especially M32 pooler | Own-base paired ~10% pooler reductions; not paired with alygg77, and raw replay times are not judge-adjusted timings. | High for own-base result / low | Low–medium isolated route; current small-M routes already stronger. Do not reopen the failed broad M128 composition. Apache-2.0. |
| 5 | Dense: selective Nathen O m1/m32 and router m16/m32 | Only sub-µs cross-run signals in the named-donor table; no new paired win. | Low / very low | Medium once final active route and absence in baseline are established. Watchlist only; adopted/closed Nathen grafts excluded. MIT. |
| 6 | Decode integration: standalone alygg77 O m128/m256 and selected dense paths | A source-path difference, **no measured decoder transfer**; not “faster than alygg77 standalone.” | High for source inventory / very low for gain | Medium; preserve fused router/PDL and allocation layout. Lower priority than a measured missing mechanism. Existing Apache/MIT provenance. |

**Not ranked as gains:** Zhebrak D32-full attention's normalized-time anomaly; Rohan's benchmark timing-data anomalies; ruwwww's same-plan sub-µs attention rows; inherited KPERM3/pins/context/deepcopy; closed Nathen QKV grafts. Details and exact sources below.

| Stage | Research outcome / owner-facing decision |
|---|---|
| Attention | Investigate safe in-call cold-code warming first; retain tree-merge idea as a separate high-cost option. No full-file replacement. |
| Dense | Chen's escape handling is the strongest independent code lead, subject to FP32-reduction legality. Avoid attributing all of its gain to packing already in our base. |
| MoE | No public expert implementation demonstrated superiority over KPERM3. Lee's heaviest-first queue and other scheduling/warp pipelines are older, slower-base evidence, not new expert-lane wins. Cold-code warming on MoE is **[INFERENCE]**, not a measured extension. |
| Router/glue | Preserve FP32 small-T input, BF16 boundaries, context binding and state ownership. They are qualification work, not incremental speedups. |
| Prefill | No reliable faster eligible public piece established. Rohan's higher official row is provisional/under review and cannot validate its anomalous local benchmark artifacts. Prefill does not raise decode rank. |


## Comparison contract

- Baseline: [alygg77/okc-muse-spark-kernels](https://github.com/alygg77/okc-muse-spark-kernels/tree/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0), revision `6cae49a639ee6d14b929a5fd4a63c627f96d7dc0`. GitHub HEAD still matched during this survey. [Per-shape results](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/RESULTS.md) and [machine-readable results](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/results.json) identify jobs and source hashes.
- The reported 2.2058× decode is **not a qualified result of our repaired decoder**. `START-HERE.md` and `work/prior-art/alygg77-precision-audit.json` record attention/expert/glue precision repairs and pending runtime qualification. Integration must use MAIN's compliant base, not restore upstream's missing BF16 boundaries.
- D1/D16/D32 mean 1/16/32 decode tokens; V1/V16/V32 mean 8/128/256 verification tokens. Sliding/full layer weights are 46:16. A standalone board, one shape, or an isolated kernel gain is not a measured whole-layer gain.
- The saved decode #50 was 2.2413×. Against reported 2.2058× this is **1.6094% more speedup / 1.5839% less latency**; future rank movement, judge noise, and precision-repair costs are additional requirements.
- Required before any adoption: PM approval for new external code; preserve attribution; ceiling ≥3% on the relevant board; unchanged-control A/A within ±1%; repeat ≥2–3% screen win; correctness/nearby shapes/capture/prepare≤300s; official qualification by the owning lane. Research recommendations are not permission to bypass these gates.

## Saved leaderboard identity map

These are **snapshot ranks**, not live ranks or proof that repository HEAD generated every row. Decode/overall: `outputs/openkernel-challenge/runs/{muse_decode_layer,overall}-leaderboard-20261031T011105Z.json` (2026-10-31 01:11:05 UTC). Components: `outputs/openkernel-challenge/runs/muse_{moe_w4a16,bf16_proj,attention_decode,prefill_layer}-leaderboard.json` (2026-10-30 ~12:50:26 UTC). Cell = rank / speedup; absent means no exact team-name row in that snapshot. Public README/team attribution establishes the named-repository mappings; additional aliases require separate evidence.

| Team / repository | Decode | MoE | Dense | Attention | Prefill | Overall rank |
|---|---:|---:|---:|---:|---:|---:|
| [alygg77](https://github.com/alygg77/okc-muse-spark-kernels) | 120 / 2.2058× | 64 / 2.4015× | 110 / 1.5502× | 63 / 2.6530× | 42 / 1.8167× | 120 |
| [vjeux](https://github.com/vjeux/okc-2026) | 246 / 2.1019× | 193 / 2.3267× | 282 / 1.5014× | 122 / 2.6104× | 79 / 1.7982× | 246 |
| [ruwwww](https://github.com/ruwwww/open-kernel-challenge) | 272 / 2.0751× | 171 / 2.3370× | 277 / 1.5056× | 135 / 2.6032× | 94 / 1.7950× | 272 |
| [zhebrak](https://github.com/zhebrak/muse-spark-kernels) | 306 / 2.0146× | 391 / 2.0182× | 463 / 1.3403× | 275 / 2.2418× | 652 / 1.0039× | 306 |
| [NathenFernandes](https://github.com/NathenFernandes/muse-bf16-proj) | absent | absent | 224 / 1.5186× | absent | absent | 1261 |

Alygg77's README #40 is a 2026-10-29 rank, not this snapshot's rank. Empty `screening` metadata does not mean screened/final. Some rows have distinct `screened_speedup` values; ranks above follow the saved official ordering, not a recomputed screening filter. All positions remain provisional under [official review rules](https://openkernelchallenge.dev/llms.txt).

## Named donor inventory and apparent per-shape wins

| Repository / revision | Available stages | Source/result evidence | License and disposition |
|---|---|---|---|
| [ruwwww `3b15eb8a6a16e0519378e28ea6c9b44d04e0a613`](https://github.com/ruwwww/open-kernel-challenge/tree/3b15eb8a6a16e0519378e28ea6c9b44d04e0a613) (`main`) | All five kernel boards; serving patch empty | Each `muse_*/submission.py` has `profiles/submission.<job>.txt`/JSON. Latest jobs match the saved board rows above. README explicitly identifies team ruwwww and committer kobeniw. | Metadata license null; complete tree has no LICENSE/NOTICE and README no grant. Unique edits need provenance/permission. Dense is Nathen's exact MIT upstream, below. No ready donor. |
| [zhebrak `ff4d55ab669977f15874943044d20312d98a44d1`](https://github.com/zhebrak/muse-spark-kernels/tree/ff4d55ab669977f15874943044d20312d98a44d1) (`main`) | Decode, attention, MoE, dense; **no published prefill** despite a historical board row | `winners/<problem>/submission.py` plus adjacent `result.json`; job IDs match saved rows. Author says publication edits only comments/docstrings. | Apache-2.0; retained SGLang attention-helper notice. Distinct attention architecture, but no clean faster eligible piece established. |
| [NathenFernandes `86aff75e6aca72511bf9b41b6dc190bf1e763373`](https://github.com/NathenFernandes/muse-bf16-proj/tree/86aff75e6aca72511bf9b41b6dc190bf1e763373) (**`solution`**, not main) | Dense/router only | [v18 result](https://github.com/NathenFernandes/muse-bf16-proj/blob/86aff75e6aca72511bf9b41b6dc190bf1e763373/results/v18.json): job66d8b007f6c641ee, 1.518570×, CI1.495792–1.541349, prepare53.799s; [all30 shapes](https://github.com/NathenFernandes/muse-bf16-proj/blob/86aff75e6aca72511bf9b41b6dc190bf1e763373/results/v18_shapes.csv). Historical “#1” is not a current rank. | MIT, full copyright/permission notice required. Best known selective routes already adopted; v19 investigations explicitly found no qualifying improvement. |

### Deduplicate ruwwww dense

GitHub contents API independently returned **the same blob `3999820b14433482a18a9c0f3f5672dd47dcee48`, 633375 bytes**, for [ruwwww dense](https://github.com/ruwwww/open-kernel-challenge/blob/3b15eb8a6a16e0519378e28ea6c9b44d04e0a613/muse_bf16_proj/submission.py) and [Nathen v18](https://github.com/NathenFernandes/muse-bf16-proj/blob/86aff75e6aca72511bf9b41b6dc190bf1e763373/submission.py). Their score difference is not independent algorithmic evidence. Nathen publishes source SHA256 `111630ff363e5988ca57f28434a2e1a7e9c1d524079c64727b52c6aa287dcb44`; this SHA256 is author-provided, not recomputed here. If reuse is approved, take provenance from the MIT upstream.

### Shape-level signals: not paired wins

| Piece / source | Donor versus alygg77, µs | What changes / compatibility | Confidence and rule concerns |
|---|---|---|---|
| Nathen O m1/m32; router m16/m32 | O m1 **5.101 vs5.24**, O m32 **5.807 vs6.09**, router m16 **4.230 vs4.50**, router m32 **5.043 vs5.24**. Identical ruwwww rerun:4.90/5.76/4.10/5.13. | Shape-specialized small-M routes, lossless codec/escape layouts. A single route may be isolated; whole layered/compressed-literal source is a high-cost graft. | **Low**: 0.14–0.40µs unpaired signals, final selected route must be proven absent. M16/32 BF16 router inputs do not authorize BF16 T≤8 decoder router inputs. No ≥3% board evidence. |
| Ruwwww small sliding attention | D1 **4.60 vs5.19**, V1 **6.54 vs6.77**, V16 **11.52 vs11.77** | Inspected `_plan`/`_TUNED` tables match alygg77 and use the same fused kernel family. [Exact profile](https://github.com/ruwwww/open-kernel-challenge/blob/3b15eb8a6a16e0519378e28ea6c9b44d04e0a613/muse_attention_decode/profiles/submission.2827c6e61a784868.txt). | **Very low**: no isolated new mechanism; layout/noise/codegen not separated. Missing repository license for unique edits. Not a recommendation. |
| Zhebrak D32 full decode attention, transposed WSD2 | Scored **150.555 vs159.93** appears favorable; **raw event174.896**, outside171.432, profiler170.4 contradict interpreting150.555 as execution time. D16full loses91.553vs83.65. | [q_len1 transposed QK/PV, three CTAs/SM](https://github.com/zhebrak/muse-spark-kernels/blob/ff4d55ab669977f15874943044d20312d98a44d1/winners/muse_attention_decode/submission.py#L7348-L8199), three stages, four producer/four consumer warps,32/128 registers. Different from the closed q_len8 V-transpose experiment. | **Very low / blocked as direct transplant**: [benchmark and profile](https://github.com/zhebrak/muse-spark-kernels/blob/ff4d55ab669977f15874943044d20312d98a44d1/winners/muse_attention_decode/result.json) show a normalized-statistic artifact; new-K and pre-output-norm paths omit BF16 boundaries. FP16 staging needs range analysis. High layout/PDL/counter/precision integration cost. |
| Zhebrak unpooler m128 | Scored **18.473 vs19.07**; donor raw event23.008 | cuBLASLt orientation2/default heuristic rank0, not a new custom GEMM. [Dense result](https://github.com/zhebrak/muse-spark-kernels/blob/ff4d55ab669977f15874943044d20312d98a44d1/winners/muse_bf16_proj/result.json). | **Very low**: no incremental algorithm beyond already-searched/pinned library routes; not a new sweep recommendation. |
| Ruwwww prefill favorable rows | T1sliding **579.13 vs585.11**, T4full **3112.47 vs3134.27**, first-turn ctx0full **9237.78 vs9426.50**; aggregate still lower | Advertises cuBLASLt pins/direct residual boundary; [profile](https://github.com/ruwwww/open-kernel-challenge/blob/3b15eb8a6a16e0519378e28ea6c9b44d04e0a613/muse_prefill_layer/profiles/submission.0ceef6beb3f24141.txt). | **Low**: no paired same-base evidence, pins inherited, not a decode-ranking lever. The1.8180× check is only a commit-message claim; scored row1.7950×. |

All six MoE loads in both ruwwww and zhebrak are slower than alygg77's corresponding published rows. Zhebrak's claimed +7–14% MoE / +2.4–5.9% large-token whole layer compares against its **own older implementation**, not alygg77. Its small-T eight-CTA cluster residual idea is already represented in alygg77's cluster boundary path. Ruwwww's “zero-DRAM intermediate activations” description is contradicted by its BF16 `inter` buffer and FP32 `ybuf`/`ss`; do not budget imaginary removed traffic.

Focused source review is not a full conformance audit: very large donor decoder/Nathen source responses were clipped by the reader even after alternate retrieval attempts. Complete recursive trees were available, but missing final source suffixes must not be treated as absent code. No new source was downloaded to this workspace.

## Baseline inventory: already present, not new wins

| Piece | Provenance / license | Measured claim and disposition |
|---|---|---|
| Five native CUDA board entries; lossless BF16 12-bit format; fused attention; expert pipelines; PDL decode chain | [vjeux `f35a41ee1c13e316dabbf772830d47b23b453393`](https://github.com/vjeux/okc-2026/tree/f35a41ee1c13e316dabbf772830d47b23b453393), Apache-2.0; current HEAD unchanged | Alygg77 already derives all five files from this revision. Attention and prefill are explicitly unchanged except attribution. Different public scores for identical code are not algorithmic gains. |
| KPERM3 NVFP4 scale-sharing permutation | [Alygg77 facts §7](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/docs/H200_MEASURED_FACTS.md#7-lossless-weight-re-layout-kperm-nvfp4-scale-sharing), Apache-2.0 | Reported same-cluster MoE +1.25%; D32 +2.6%, V16 +3.9%, V32 +3.1%. Decoder already contains it; no incremental credit. |
| O m8 and pooler m8/m16 routes | [Nathen credit and measurements](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/docs/H200_MEASURED_FACTS.md#8-muse_bf16_proj), MIT portions inside Apache-2.0 file | Saves 0.379/0.601/0.468 µs respectively in 8/8 local comparisons; +0.60% dense board. Already in alygg77's standalone dense entry. This alone does not prove the same path is active in the fused decoder. |
| Exact small-T router activation; per-thread CUDA primary-context binding; cuBLASLt plan lifetime/pins; deepcopy ownership | [Alygg77 NOTICE](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/NOTICE), credits Matsvei / kernarc `matvey`; Apache-2.0 as inherited challenge submissions | Correctness/robustness obligations, not new speedups. Context bind reported +0.054% with CI spanning zero; dense deepcopy fix +0.02% ±0.06%. Must preserve MAIN's stronger precision repairs. |

### Standalone-to-decoder gap is not yet a winning transplant

The standalone dense entry dispatches custom O-projection WGMMA paths at m128/m256 (`oc_ol`/`oc_rl`; [source lines 4555–4586](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/kernels/muse_bf16_proj.py#L4555-L4586)) and explicit Nathen route families. Searches found no corresponding `_oc_`, `_VO_`, `_NP_`, `_NDD_`, `nathen_pool16`, or decodediet symbols in the decoder. **[INFERENCE]** These may be unported opportunities, but different names/equivalent implementations are possible; absence of symbols is not a benchmark.

Reported standalone O latency is 6.96/9.26 µs at m128/m256. No paired decoder-graft gain was located. Integration must retain BF16 output, packed-weight identity and state ownership; small-T O must preserve PDL, while a standalone pooler replacement cannot simply discard the fused router. Larger prepare allocations can regress unchanged paths. The dense owner should establish the in-layer ceiling before any transplant; the documented failed Nathen QKV graft is not reopened by this observation.

## New mechanism details and legal integration boundaries

### R1 — Cold-code warming, not cross-call cache reuse

Source: [A13/A14 experiment log](https://github.com/Kushagrabainsla/muse-spark-open/blob/0778ecd8293e375ffe33faa466db0231929aaac3/muse_attention_decode/EXPERIMENTS.md), [implementation](https://github.com/Kushagrabainsla/muse-spark-open/blob/0778ecd8293e375ffe33faa466db0231929aaac3/muse_attention_decode/submission.py). No explicit repository grant found.

The idle producer executes the same `__noinline__ verify_merge(..., store=false)` after its final copy, **inside the current timed call**, before the later real merge. Reported merge cost fell6928→2524cycles. The no-flush repeat fell further to2298cycles but is not a permitted replacement for the cold-L2 workload. Bundled A13 judge sliding-verify B1/B16 changed12.3/21.6→9.9/17.2µs; board1.473→1.499×. Other A13 changes mean those board deltas cannot all be assigned to warming.

Compatibility: preserve the existing ATT7/KPERM layouts and all precision boundaries; isolate only an equivalent once-per-CTA epilogue. Alygg77 already compacts cold code, so generic “reduce code size” is not new. The donor's early call reads merge storage even when stores are disabled: initialized scratch, race freedom, complete key coverage, instruction issue and register pressure must be established before a candidate. **Next owner action:** source-check safe inputs/idle warp, then measure current-base exposed epilogue cost. No warm state may be carried between timed calls.

### R2 — Independent dense escape scheduling, with a precision blocker

Source: [chenwainuo README and30-shape table](https://github.com/chenwainuo/okc/blob/3d418e1625f6de2075b5684a27ddbc3d1442422d/README.md), [CUDA source](https://github.com/chenwainuo/okc/blob/3d418e1625f6de2075b5684a27ddbc3d1442422d/muse_bf16_proj/submission.py), Apache-2.0. The novel portion is per-thread stage-sorted escape cursors plus warp-vote skipping, not the already-inherited12-bit format. Other mechanisms include direct register GEMV/predicated XOR, two math warpgroups,12KB stages and a five-slot ring; prepare is reported~200s.

The most favorable unpaired published rows versus alygg77 are:

| Load / projection | Chen µs | Alygg77 µs |
|---|---:|---:|
| D32 QKV | 10.14 | 10.78 |
| V1 QKV | 9.17 | 9.78 |
| V1 O | 5.18 | 5.52 |
| V16 QKV | 12.31 | 13.79 |
| V16 pooler | 17.11 | 19.94 |
| V16 unpooler | 17.38 | 19.07 |
| V32 O | 8.57 | 9.26 |

These rows are **not qualifying current-base wins**. The RS128 source explicitly rounds split-K partials to BF16 and uses BF16 `cp.reduce.async.bulk.tensor...add`; `_rqval` allows KS-dependent tolerance. That is not automatically equivalent to required FP32 partial accumulation. A legal FP32 reduction can consume the apparent saving. Larger-M codec work also conflicts with alygg77's measured integer/MMA issue-contention result; only a materially different legal schedule warrants reconsideration. Preserve lossless escapes and small-T FP32 router activation; do not transplant donor arithmetic wholesale.

### R3/R4 — Other measured implementations, weaker transfer evidence

- [Tommy attention design](https://github.com/tommyyliu/okc/blob/f134725e27f368588bd9ef27cb5e97d023b415a5/muse_attention_decode/README.md): last arriver merges each split group, then last group merges final output. New tokens own a split; decode packs both KV heads into128-byte rows. D16/D32full~109/199µs and V1full36.6→34.1µs are not competitive with alygg77's published rows. The useful distinction is multi-CTA decomposition, not the donor's whole implementation. Final last-arriver serialization, release/acquire/reset correctness, BF16 boundaries and all-key coverage remain gates; no repository license found.
- [Xuefeng P048 paired record](https://github.com/Xuefeng-Zhu/open-kernel-challenge/blob/5372a758c7d4c03431ed4887f75dfb7d5e53aa1b/experiments/048-p046-paired-confirmation.md), Apache-2.0: P039→P046 pooler M8/M16/M32 medians24.432/25.392/25.328→21.888/22.544/22.848µs;12 reversed cold-graph pairs,11/12 wins per changed shape, controls<0.7%,480 author-reported oracle comparisons. Full-K output-column BMM avoids K-partial reduction on those shapes. These include replay overhead and are not directly comparable to alygg77's adjusted judge rows. M128 composition lost; current baseline already retains better-selected M8/M16 paths. Only an independently bounded M32 comparison remains plausible.

### Quarantine: Rohan benchmark measurements

The [benchmark tree](https://github.com/rohanarun/Auto-research-agent/tree/018b7a102348ea3b73701e8be52d24f82d8c91ee/benchmarks) **does contain real candidate source files**, correcting the initial guide-only overview assessment recorded in LOG.md. App/benchmark MIT and upstream vjeux Apache/Nathen MIT notices apply; no blanket relicensing of those portions.

Parent fetched [dense evaluation.json](https://github.com/rohanarun/Auto-research-agent/blob/018b7a102348ea3b73701e8be52d24f82d8c91ee/benchmarks/data/candidates/gemini-3.8-flash-opencode/kernel-muse_bf16_proj/evaluation.json) and checked all30 first-bank cases: baseline times are exactly100+15×indexµs, each case has five identical timing pairs, and every baseline/candidate ratio is1.5601 within2.22e-16. [Prefill evaluation](https://github.com/rohanarun/Auto-research-agent/blob/018b7a102348ea3b73701e8be52d24f82d8c91ee/benchmarks/data/candidates/gemini-3.8-flash-opencode/kernel-muse_prefill_layer/evaluation.json) shows analogous exact1.9562 ratios in inspected samples. These regularities and conflicting surrounding results prose require a provenance explanation; they are **not credible raw hardware evidence as presented**, and do not establish author intent.

The independent saved official rows—dense1.56039485×, attention2.65372913×, prefill1.95549648× (`under_review`)—exist, but no artifact linkage proves these benchmark source bytes produced them. Source ideas such as `rq_*` candidates/Triton TMA routes can be audited independently; generic pins and Nathen composition are inherited, and a long compressed payload prevented a complete source audit here. No timing-based import recommendation.


## Kernarc: public evidence and access limit

The [public homepage](https://agents.kernarc.com/), [JavaScript client](https://agents.kernarc.com/assets/app.js), and [OpenAPI schema](https://agents.kernarc.com/openapi.json) were read. The homepage explicitly requires a key for every API call; the client gates content behind that key. Unauthenticated thread-list and author=`matvey` searches, dated feed, tags, files, and members all returned **HTTP401**; GET `/mcp` returned405. No real thread/reply ID, original posting date, or attachment version was recoverable. No registration, authentication, or posting was attempted. The example “split-K gives1.08× at M16” in the guide is a placeholder, **not experimental evidence**.

Domain/matvey/Matsvei/router/cuBLAS searches and GitHub issue/README searches found no public thread mirror. Thus this report covers the accessible public surface and alygg77's explicit attribution, **not the authenticated forum catalogue**. Arbitrary unseen snippets have unknown licenses. The five recoverable findings are the inherited FP32 router, context bind, plan pins, deepcopy isolation, and the closed stock-Marlin occupancy result listed above; none earns new speedup credit.

Additional dated public discussions were read, including replies:
- [Anvith Shetty, Oct23](https://x.com/anvithbuilds/status/2113628560674193893): attention0.914× aggregate despite decode up to2.57× and verify down to0.34×; later replies1.19→1.51×. KV-walk and buffer-latency explanations were hypotheses, not isolated measurements. No competitive source/license established.
- [Rohan Arun, Oct26](https://x.com/RohanArun/status/2114830856393027974): decode1.736→1.785×, MoE2.013→2.033×, dense1.443→1.453×, attention1.788→1.801×. Later tree inspection found public benchmark candidates beyond the guide catalogue; see the provenance quarantine above. Neither the social numbers nor later official rows establish those files' same-base transfer performance.

## Community-reported failures and non-transferable wins

Primary evidence for this section: [NEGATIVE_RESULTS.md](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/docs/NEGATIVE_RESULTS.md) and [H200_MEASURED_FACTS.md](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/docs/H200_MEASURED_FACTS.md). These are scoped author measurements, not universal impossibility claims. Do not repeat a closed configuration merely because another base reports a win.

| Stage / experiment | Reported result | Decision / scope |
|---|---|---|
| Pooler/router stream before attention; more side streams | D32 sliding 2.1–7.7 µs slower; V32 sliding attention alone 14.9 µs becomes 22–27 µs next to GEMM; second graph root starts 21–33 µs late | Do not recommend naive overlap. Isolation/partitioning must first demonstrate an in-layer ceiling; concurrency inside one kernel was not covered. |
| More PDL pre-wait stages / glue-launch removal | Two stages optimal; four/six slow attention by 1.7 µs and boundary by 0.9 µs. Removing one glue launch saves 1.2–2.0 µs, all three 4.0–4.8 µs at D1/V1 sliding | Already overlapped serial chain. Synthetic stand-in overstated real early-trigger gain ~6.6×. |
| L2 prefetch / launch-table search | Prefetch from sliding attention loses 2.6–5.1 µs; unpooler lines evicted. Best 20-point launch search +0.15 µs versus A/A 1–1.5 µs | Closed tested prefetch/sweep designs. No speculative credit. |
| cuBLASLt exhaustive verify search / removing split-K | 1,789–2,309 valid configurations per shape yield ~2 µs/layer; standalone savings shrink ~35% in-layer. Split-K combine exposed time zero on 6/6 tested D16/D32 rows except QKV reduction | Do not sell a routine algorithm search or hidden combine removal as a ≥3% board lever. |
| Dense QKV stream-K and m256 split-K | Decode QKV ceiling ≤0.6 µs versus 1.5–4 µs fix-up; m256 split-K loses 3.3–4.6 µs despite 94% clock-corrected tensor peak | L2 writes only 40–50 GB/s/SM; tensor efficiency alone is not a win. |
| BF16 codec rewrites and large-m register decode | Lane-major repack loses 0.78–1.02 µs; producer warp loses 6.28–6.50 µs. m128 codec: O/unpooler +2.4–4.1 µs, pooler ~32.6 vs 20.0–21.4 µs | Keep lossless codec for small m. Tighter format remains only a bounded hypothesis: entropy 10.46 bits vs 12-bit code; realistic ≤0.5–1% board, free-decode bound +2.4%. |
| Nathen blanket graft / QKV graft into decode | Unpooler m8/router m1/O m16 lose 0.175/0.163/0.098 µs. T128 QKV decoder graft: whole-board −0.091%/−0.160% despite V16 +0.30–1.51 µs; older T256 graft loses 5.5–7.6 µs | Keep already selected winning standalone routes. Do not repeat wholesale or QKV decoder transplant; layout regressions affected unchanged paths. |
| More expert ring slots / WGS32 | Fourth slot within −1.0 to +0.9 µs; other x-ring changes lose; WGS32 gains 1.3–3.8 µs on another base but loses 2.2–3.0 µs on alygg77 D32 | Pipeline is unit-boundary/schedule sensitive, not simply depth limited. |
| Dynamic whole-unit tail / cost-table refit | Unit costs 18.5–19.4 µs versus removable tail spread 2.7–8.2 µs; refit halves phase-A spread yet loses 0.8–4.6 µs | Judge end-to-end phase-B completion, not an attractive phase-A statistic. |
| Fence relocation / one-fence deletion | Per-issue producer fence loses 6.8–19.5 µs; NOFENCE MoE +0.234% but no decode gain (D32 full −1.35 µs across four pads) | Memory-safe is not enough; do not spend a new decode campaign on this measured sub-noise edit. |
| D16 second weight read / expert compression | Removing duplicated expert read speeds other CTAs only ~1%; E2M1 entropy exactly 4 bits, scales 3.71 bits, practical savings ~5% bytes | D16 is co-bound; lower byte count alone does not establish latency gain. |
| SIMT tiny expert groups / Marlin occupancy transplant | SIMT loses 52.8–149.2 µs; extra 132 Marlin CTAs cost ~1 µs D32. Matvey's +2.4–3.0% M8 occupancy gain was against stock Marlin and did not transfer to alygg77's per-load tuning | Already explored on this family. |
| Full-verify attention balance / producer V transpose | Equalizing pair end times leaves SM completion unchanged; one-CTA pipeline 116.4 vs 100.5 µs. Producer transpose loses 0.55–0.87 µs V16 full | Consumer throughput, not apparent inter-CTA slack. Register instrumentation itself spills and distorts ~100 µs to ~186 µs. |
| Decode attention single-CTA/cluster merge; page sorting; Q-prep folding | Large split merge loses ~60/96 µs D1/V1 full; final clustered D1 design loses 1.12–1.18 µs. Sorting loses 161.9–275.7 µs; queue 3.5–70 µs. Q-prep folding only ~0.09% board ceiling | Do not revive these exact designs. A parallel multi-CTA tree merge was not measured, but no public win is established. |
| Prefill boundary overlap / deeper/narrower expert pipelines / exponent tricks | GEMM+glue overlap only 4.8% pair gain, GEMM slows 8–15%; persistent boundary often +3–32%. Narrow WGMMA triggers C7511/C7512. Removing all exp retains 87–91% attention time, board ceiling +1.2% | Prefill is not decode ranking. Keep scoped negative results; no claim that a different byte-removing fusion is closed. |
| Prefill KPERM/cluster merge | Local stack ~+0.35–0.41%; same-hour judge +0.31% with pairs disagreeing; later 1.8040× fails to beat older 1.8167× high run | Sub-1% board changes cannot be ranked from one lucky run. |
| Layout, process and judge artifacts | +3 MB scratch slows unchanged SASS 0.5–1.3 µs; pad can move V32 1.5–2.5 µs; same dense file 1.5259–1.5502×; cross-cluster apparent +4.19% was ~+2.4% | Use paired same-process pads/seeds and same-cluster controls. Keep raw times, no subtraction of profiler/graph overhead to manufacture wins. |

## Rule and integration gates

[Official rules](https://openkernelchallenge.dev/llms.txt) permit lossless BF16/NVFP4 repacking, dequantization inside kernels, BF16 hi+lo router **weights** with FP32 accumulation at any batch size, and FP8 E4M3 attention queries. They do not permit rounding away the small-T router's FP32 activation, dropping required BF16 materialization boundaries, reducing cache precision, skipping keys/values, output caching, or input-dependent host work during graph capture.

Source review must check all effective late overrides and fallback paths, not just a file's introductory comments. Preserve private state/deepcopy ownership and per-thread context binding. A successful numerical tolerance test is not proof of internal precision compliance. New source requires the PM's import decision; public visibility alone is not a license. Retain vjeux Apache-2.0 attribution, alygg77 NOTICE, and Nathen's full MIT notice where applicable.

## Additional public repository inventory

All19 substantive collections below were inspected beyond metadata (README/results/source sections as relevant). Pins are observed public tips, **not assertions that the tip generated every published score**. Abbreviations: D decode, M MoE, P dense, A attention, F prefill. “Check” is off-board; “local” is not official. Saved rows use the snapshot dates above. **† = owner used only as an exact-name lookup key; not an independently established repository→team/source→job binding.** Explicit mappings include operatorstack→bigboateng, Madhavan113, rohanarun, iyerramji, ScrashOff, xharshvardhan and Ranoobaba.

| Source / pinned revision | License found | Published claim / available code | Saved alias evidence / disposition |
|---|---|---|---|
| [chenwainuo/okc `3d418e1625f6de2075b5684a27ddbc3d1442422d`](https://github.com/chenwainuo/okc/tree/3d418e1625f6de2075b5684a27ddbc3d1442422d) | Apache-2.0 | P1.542×; independent register-GEMV/RS-WGMMA,30 shapes | †P#136/1.5419×; mechanism R2, BF16 partial-reduction gate |
| [Madhavan113/ParadigmKernelChallenge `721c62729fa0b407799692f311790643706f292d`](https://github.com/Madhavan113/ParadigmKernelChallenge/tree/721c62729fa0b407799692f311790643706f292d) | Apache-2.0; vjeux NOTICE | Historical D2.1388×, A2.5883, M2.3318, P1.4933, F1.7960; mature derived kernels/handoff | D/overall#134, D2.2000×; public notes lag row. No same-base win |
| [rohanarun/Auto-research-agent `018b7a102348ea3b73701e8be52d24f82d8c91ee`](https://github.com/rohanarun/Auto-research-agent/tree/018b7a102348ea3b73701e8be52d24f82d8c91ee/benchmarks) | MIT plus Apache-vjeux/MIT-Nathen portions | Substantive benchmark candidates; claimed local A2.6537/P1.5601/D2.1724/M2.3768/F1.9562 | D/overall#173; P#71, A#62, F#13 under_review. Artifact anomalies above; no source→official-row proof |
| [operatorstack/open-kernel-challenge-solutions `6d9b77656b6c295e421be9e1424b0f8f99299b0a`](https://github.com/operatorstack/open-kernel-challenge-solutions/tree/6d9b77656b6c295e421be9e1424b0f8f99299b0a) | MIT + SGLang Apache-2.0 | Team bigboateng; D1.7352/M1.8320/A1.8228/P1.3066; register-weight streaming, DSMEM split-K, PDL | bigboateng D#477/1.7352×, overall#481; not a replacement |
| [lee101/openkernels `c565e373d34575c5f7df862b05f8be7619078b0f`](https://github.com/lee101/openkernels/tree/c565e373d34575c5f7df862b05f8be7619078b0f) | Apache-2.0 | Documented best **checks** A1.599/M1.418/D1.348/P1.149/F1.034; persistent/heaviest-first MoE queue | †D#425/1.7883×, overall#428; author Lee Penkman, exact board identity not proven; notes/files and row differ |
| [Xuefeng-Zhu/open-kernel-challenge `5372a758c7d4c03431ed4887f75dfb7d5e53aa1b`](https://github.com/Xuefeng-Zhu/open-kernel-challenge/tree/5372a758c7d4c03431ed4887f75dfb7d5e53aa1b) | Apache-2.0 | P046 ranked P1.189713/A1.242925; strong own-base pairing. Root submission remains starter; selected paths documented | †P#971/A#980, overall#1343; R4 only, no wholesale graft |
| [iyerramji/open-kernel-challenge `1d57ff7749433d0d18b294744e607cd82e33057f`](https://github.com/iyerramji/open-kernel-challenge/tree/1d57ff7749433d0d18b294744e607cd82e33057f) | Apache-2.0 + NOTICE | P v21 1.158264×, v34 1.156021×; native GEMV, TinyGEMM, layouts, Lt | P#1146/1.1583×, overall#1567; useful failed-codec controls |
| [ozgureyilmaz/okc-muse-kernels `74daa3a6990168b52359c80955014a945fc7aa4e`](https://github.com/ozgureyilmaz/okc-muse-kernels/tree/74daa3a6990168b52359c80955014a945fc7aa4e) | Apache-2.0 | P v0271.1718× scored/1.1804× check; native FP32 router, TinyGEMM, DeepGEMM | †P#1077/1.1718×, overall#1545; small-router5.5µs not competitive with baseline |
| [MadanChaollaPark/open-kernel-experiments `40154c27d4b694c7d641c7959b00d7ff24a7c2e4`](https://github.com/MadanChaollaPark/open-kernel-experiments/tree/40154c27d4b694c7d641c7959b00d7ff24a7c2e4) | Original modifications Apache-2.0; upstream notices | Old scored A1.4807/P1.2195/D1.1852/M1.0971/F1.1206; later checks explicitly separate;627 snapshots/115 local records | †D#940/1.2001×, overall#1068; negative archive. Unfinished mixed-input prototype not a speed result |
| [infraax/okc-kernel-challenge `896b816dc5fd418e1f167578eb35680b116b68f3`](https://github.com/infraax/okc-kernel-challenge/tree/896b816dc5fd418e1f167578eb35680b116b68f3) | Apache-2.0 | A1.770/D1.433/M1.486/P1.170;86-run archive | †D#699/1.4334×, overall#719; failure evidence, not donor win |
| [ScrashOff/okc-muse-kernels `d870912e7c5a030c52050bb2445418224c28d9de`](https://github.com/ScrashOff/okc-muse-kernels/tree/d870912e7c5a030c52050bb2445418224c28d9de) | No repository grant found | Claimed P1.214/M1.270/D1.300 scored; A1.414/F1.167 **checks under rules review**; Marlin/glue/split-KV | D#822/1.2997×, overall#889; component snapshot differs; no A/F qualification credit |
| [xharshvardhan/open-kernel-challenge `4958d3141aa977c4b6b26df7a8d06c90dd094ee9`](https://github.com/xharshvardhan/open-kernel-challenge/tree/4958d3141aa977c4b6b26df7a8d06c90dd094ee9) | Upstream Apache portions; original-modification grant unclear | P1.1991/M1.3749/A1.2990/D1.2252; four kernels, route alignment/aggregation | D#910/1.2252×, overall#959; no stronger stage established |
| [Kushagrabainsla/muse-spark-open `0778ecd8293e375ffe33faa466db0231929aaac3`](https://github.com/Kushagrabainsla/muse-spark-open/tree/0778ecd8293e375ffe33faa466db0231929aaac3) | No repository grant found | Later A1.525/M1.704/D1.425/P1.190; independent native/Triton source | †D#705/1.4247×, overall#724; saved earlier A#860/1.3911×. R1 mechanism, not full kernel |
| [Aravind-11/muse-moe-w4a16 `399305738cfb9effc85555e05b8691ba6c769813`](https://github.com/Aravind-11/muse-moe-w4a16/tree/399305738cfb9effc85555e05b8691ba6c769813) | No repository grant found | README M1.629×; **only early v5/v6 public, expressly not current best**; fused swap-AB G1/SiLU, Marlin G2 | M#510/1.8541×, overall#904; do not assign stronger row to early source |
| [GiulianoIntelligence/Muse-FlashSpeed-Kernel `d47f9fc825fd2ae6c05b1af02e51f8d570a71437`](https://github.com/GiulianoIntelligence/Muse-FlashSpeed-Kernel/tree/d47f9fc825fd2ae6c05b1af02e51f8d570a71437) | No repository grant found | Logged MoE0.2376×; substantive slow Triton/decode experiments | No exact alias in saved snapshots; failure archive only |
| [Ranoobaba/okc-campaign `3831ab3c3538a9878031b667624240bd24bb7082`](https://github.com/Ranoobaba/okc-campaign/tree/3831ab3c3538a9878031b667624240bd24bb7082) | No repository grant found | Old campaign P1.107/A1.282/M1.263/D1.231;57 local experiments/10 scored submissions | D/overall#359, D1.8951×; published campaign is older than row. Also read earlier learnings pin7eb366f |
| [lilaclilac09/open-kernel-challenge `d9bdcc9f01dff1018f2f256a0a1a63911a16557e`](https://github.com/lilaclilac09/open-kernel-challenge/tree/d9bdcc9f01dff1018f2f256a0a1a63911a16557e) | README submission-Apache statement; no root LICENSE | Real starter-derived trials, not untouched template: naive Triton0.585, M1-only0.870, hi/lo1.033 scored/1.042 check, hybrid router1.050 check | †P#1388/1.0707×, overall#1673; no competitive lead |
| [tommyyliu/okc `f134725e27f368588bd9ef27cb5e97d023b415a5`](https://github.com/tommyyliu/okc/tree/f134725e27f368588bd9ef27cb5e97d023b415a5) | No repository grant found | A1.559 scored/1.717 newer check; P1.283122 check; D1.187664 check; source modules/generated submissions | †D#396/1.8293×, overall#397; no HEAD→row assertion. R3 decomposition only |
| [KIROKOWA/QWEN-ASRN `5fea9906a8a66e594ac3008fcd992fc3ee834fb0`](https://github.com/KIROKOWA/QWEN-ASRN/tree/5fea9906a8a66e594ac3008fcd992fc3ee834fb0/openkernel) | Apache-2.0 | Real Muse attention under openkernel/: v14 A1.325, v15 unmeasured; fixed-reference softmax/quorum finalize | †A#736/1.5125×, overall#1546; published version evidence weaker than row |

Full exact-alias lookup data and snapshot hashes are in [EVIDENCE.json](EVIDENCE.json). Repository scores, saved board rows, and final source revisions are intentionally not conflated.

### Additional scoped community failures

| Source / evidence | Reported outcome | Consequence |
|---|---|---|
| [Kushagrabainsla A12/A15/A16](https://github.com/Kushagrabainsla/muse-spark-open/blob/0778ecd8293e375ffe33faa466db0231929aaac3/muse_attention_decode/EXPERIMENTS.md) | Dual-head full verify226.8/439.7µs vs FA3199/382;128-key stages325/618; full-D1 last-CTA merge27.6→37.4. Later FA3 split tuning~1% only; page-table staging fullB16/B32116/215→128/234. | R1 warming is narrower than these failed whole-kernel rewrites; no general “more overlap/stages” recommendation. |
| [Madhavan handoff](https://github.com/Madhavan113/ParadigmKernelChallenge/blob/721c62729fa0b407799692f311790643706f292d/notes/claude_to_codex_handoff.md) | Dynamic claiming19–60% slower; >112 registers at20warps hung checks; dense fixups4–5µs; single-CTA boundary O8.8→46µs. PDL cached/`const __restrict__` loads hoisted before dependency wait failed first replay. | Preserve coherent producer-consumer ordering. Identical-code expert72.1vs56.5µs also corroborates timing noise; prepare trimmed~300→122s, not a kernel gain. |
| [Iyerramji](https://github.com/iyerramji/open-kernel-challenge/blob/1d57ff7749433d0d18b294744e607cd82e33057f/README.md) | Lossless native pooler codec32.048/32.064µs vs raw24.704/24.720; Triton39.120vs32.160. Native router M8~2× slower; Modalv34+0.577% failed official aggregate improvement. | Compression/one-shape wins do not generalize; retain raw controls and official/local distinction. |
| [Xuefeng failure index](https://github.com/Xuefeng-Zhu/open-kernel-challenge/blob/5372a758c7d4c03431ed4887f75dfb7d5e53aa1b/FAILED_EXPERIMENTS.md) | Full-K QKV M128, broad column-BMM M128, codec and weight-first WGMMA M8 lost. Attention M64N8 producer1.09955vs1.23363× despite correctness; changed paths21.1–79.6% slower. | R4 is specifically small-M pooler on its own base, not a general library-replacement claim. |
| [Madan archive](https://github.com/MadanChaollaPark/open-kernel-experiments/blob/40154c27d4b694c7d641c7959b00d7ff24a7c2e4/README.md) | Custom Marlin stripes76/76 passed but1.066–1.077× vs1.167 control; exact FP4 register-byte lookup also lost. Mixed-input prototype had no full-MoE timing. | Correctness-only or incomplete prototype cannot qualify a performance lead. |
| [Infraax](https://github.com/infraax/okc-kernel-challenge/blob/896b816dc5fd418e1f167578eb35680b116b68f3/README.md) | Exp polynomial12–21% slower; exp ceiling~7%; early PDL light+2µs/full−3.9µs; WSTREAM0.2–2.5TB/s vs existing2.6–3.7. | Do not infer portable gain from reduced instruction count or bandwidth-only intuition. |
| [Lee ideas](https://github.com/lee101/openkernels/blob/c565e373d34575c5f7df862b05f8be7619078b0f/docs/ideas.md) | Shared-B WGMMA MoE lost; dense deeper ring0.94→0.83×; full-attention lookahead2–12 lost; reports FP8 mma.sync lowered to conversion/HMMA. | Inspect generated instruction path; older-base ablations are not current headroom. |
| [ScrashOff](https://github.com/ScrashOff/okc-muse-kernels/blob/d870912e7c5a030c52050bb2445418224c28d9de/README.md) | Generic Triton dense0.58–0.89×; inline grouped NVFP4 decode4–10× slower than Marlin. A/F checks explicitly under rules review. | Numerical passes are not performance or eligibility. |
| [Giuliano results](https://github.com/GiulianoIntelligence/Muse-FlashSpeed-Kernel/blob/d47f9fc825fd2ae6c05b1af02e51f8d570a71437/RESULTS.md) | E2M1 code3 incorrectly decoded as2.0 instead of1.5; extra shared-memory/barrier rounds; some configurations zeroed output. | Validate exact formats first; “silent miscompile” is author's hypothesis, not our finding. |
| [Ranoobaba learnings](https://github.com/Ranoobaba/okc-campaign/blob/3831ab3c3538a9878031b667624240bd24bb7082/LEARNINGS.md) | Skinny QKV M1633.6vs22µs cuBLAS; one-launch attention12.8µs equals three-launch; query-prep fusion slowed verify. Triton MoE0.022→0.40× still failed; CUDA D1159→78µs lost to Marlin+finalize62. | Dependent memory trips/consumer work matter more than launch count alone. |
| [Tommy dense round8](https://github.com/tommyyliu/okc/blob/f134725e27f368588bd9ef27cb5e97d023b415a5/muse_bf16_proj/results/round8_experiment.md) | Five variants1.2280–1.2713× failed to beat1.2831221; M1 router32warps4.058vs3.805µs. Decode Marlin cap8 1.1028vs cap16/finalize1.1877×. | No automatic benefit from rings/geometry/warp changes. |
| [KIROKOWA version log](https://github.com/KIROKOWA/QWEN-ASRN/blob/5fea9906a8a66e594ac3008fcd992fc3ee834fb0/openkernel/README.md) | Slot merge full-verifyB1190µs/board0.91×; C++17 failed where C++20 needed; v12 CUDA full decode186/428vs Triton156/306µs at219regs; v13 80regs only tied. | Toolchain and registers alone do not establish a faster algorithm. |

Ruwwww's historical numerically passing profiles also include MoE1.836–1.893× and prefill0.919–1.780×; causes were not documented, so none are invented here. Nathen reports v19 no qualifying improvement and disclaims all-finite-BF16 correctness outside the generator. Zhebrak reports FP8 hi-only query error~7e-2 can flip near-tie routing; its hi+lo query path lowers~1e-2 but adds work, and its alternate WSV2 full-verify path is disabled without measured reason.

## Excluded, unavailable and non-kernel search hits

These were retained in coverage rather than silently dropped. No license inference from public visibility; an unspecified license below remains unconfirmed.

| Repository / public source | Disposition |
|---|---|
| [sid19991/open-kernel-challenge](https://github.com/sid19991/open-kernel-challenge) | Official CLI/spec/reference/SGLang/leaderboard mirror, not optimized submissions |
| [arudomanenko/open_kernel_challenge](https://github.com/arudomanenko/open_kernel_challenge) | MIT; one-commit README/.gitignore/LICENSE only; same-name P#752/1.2322× does not expose code |
| [nibague/open-kernel-challenge](https://github.com/nibague/open-kernel-challenge) | Apache-2.0 educational tree; inspected dense entry is PyTorch starter |
| [KingCobra52/muse_bf16_optimization](https://github.com/KingCobra52/muse_bf16_optimization) | PyTorch starter, no score/grant found |
| [Saikat-too/muse_spark_kernel_challenge](https://github.com/Saikat-too/muse_spark_kernel_challenge) | MIT starter, no performance claim |
| [Tetraslam/muse-kernel-challenge](https://github.com/Tetraslam/muse-kernel-challenge) | Package skeleton, no substantive kernel/score |
| [harikanthl/muse](https://github.com/harikanthl/muse) | STATE explicitly “No kernel written yet”; starter/research notes |
| [Reedo3/Muse_Attention_Decode](https://github.com/Reedo3/Muse_Attention_Decode) | README-only challenge pointer |
| [sadra-etaei/muse_spark_speedup](https://github.com/sadra-etaei/muse_spark_speedup) | Empty repository, no revision/code to pin |
| [theluckystrike/muse-spark-open-kernels](https://github.com/theluckystrike/muse-spark-open-kernels) | Post-Nov4 publication promise/intended Apache; no implementation now. Explicit team, saved D/overall#337/1.9536× |
| [danyloooah-web/okc-claims](https://github.com/danyloooah-web/okc-claims), [Cacheon-28ef5-devstar/okc-claims](https://github.com/Cacheon-28ef5-devstar/okc-claims) | Sealed/encrypted archives, no plaintext kernel inspected; danyloooah-web same-name A#570/1.7432× is not accessible source |
| [3xC1ibR/open-kernel-challenge](https://github.com/3xC1ibR/open-kernel-challenge) | Root/status/raw404; API fallback403. Indexed19/45/42% M1 claims lack recoverable baseline/revision/license; excluded |
| [chinesepowered/hack-argonspeed](https://github.com/chinesepowered/hack-argonspeed) | Campaign tooling; kernels expressly withheld. Same-name saved D#265/2.0850× is not downloadable winning code |
| [lachydotmcg/muse-sputter](https://github.com/lachydotmcg/muse-sputter) | MIT parody143.8× **slower** than PyTorch on RTX3060, not H200/SGLang |
| [msaroufim/openkernelchallenge-audit](https://github.com/msaroufim/openkernelchallenge-audit) | Static challenge/KernelBot audit, not entrant kernels |
| [Lurrobert/guild](https://github.com/Lurrobert/guild), [Lurrobert profile](https://github.com/Lurrobert/Lurrobert), [meet447/guild-keys](https://github.com/meet447/guild-keys) | Apache forum infrastructure/profile/key record, no GPU submission |
| [abozten profile](https://github.com/abozten/abozten), [itsryanlenk profile](https://github.com/itsryanlenk/itsryanlenk) | Private challenge work/team participation statements; no public code or proven alias |
| [randombullshitgo12/messy-qwen](https://github.com/randombullshitgo12/messy-qwen) | Qwen steering hooks, not Muse kernels |
| [chins3959/okc-project](https://github.com/chins3959/okc-project) | README-only ambiguous-name repository |
| [chillqoder/Muse-Spark-Open](https://github.com/chillqoder/Muse-Spark-Open) | Model-generated HTML games/design examples, not inference kernels |
| [ranjanj1/muse-spark-moe-experiments](https://github.com/ranjanj1/muse-spark-moe-experiments) | MIT routing research/model-data licensing;4.5×chance expert overlap on five texts, **no latency claim**; cannot assume challenge routing/cache outputs |
| [Kevin19980730/muse-spark-inferencex](https://github.com/Kevin19980730/muse-spark-inferencex) | Static serving dashboard/CSV archive |
| [Studio1-OSS/muse-spark-open-bench](https://github.com/Studio1-OSS/muse-spark-open-bench) | Metadata-only exclusion: code-review quality benchmark, not GPU timing |
| [shi3z/muse-spark-a100x1](https://github.com/shi3z/muse-spark-a100x1) | A100 llama.cpp serving project, not same H200 OKC kernels |
| [mpeng19/auto-inference](https://github.com/mpeng19/auto-inference) | Qwen/H100 research harness, not public Muse leaderboard kernel |
| [colinlikescode/Open-Inference](https://github.com/colinlikescode/Open-Inference) | Apache generic inference optimizer/ServePilot, no Muse candidate found |
| [naklecha/simple-llm](https://github.com/naklecha/simple-llm) | GPT-OSS MXFP4/H100, not the Muse BF16-board leader's implementation |

## Search coverage, evidence and verification limits

Used `gh search repos`, `gh api`, public REST/static GitHub/Atom feeds, official rules, saved leaderboard JSONs, and primary-source web searches. No authenticated code search, cloned repository, downloaded source file, GPU command, check or submit. Public GitHub metadata/API reads used the existing CLI access; no account registration or service mutation.

| Exact discovery query / family | Returned count | Coverage |
|---|---:|---|
| `"Open Kernel Challenge" in:name,description,readme` via public REST | 40 | All returned metadata; relevant non-excluded hits classified |
| `"muse_decode_layer" in:readme` | 15 | Complete returned set; parent gh API reproduced15/15 |
| `"muse_moe_w4a16" in:readme` | 18 | Complete returned set |
| `"muse_bf16_proj" in:readme fork:true` | 19 | Complete returned set |
| `"muse_attention_decode" in:readme fork:true` | 16 | Complete returned set |
| `"muse_prefill_layer" in:readme fork:true` | 10 | Complete returned set |
| `"openkernelchallenge.dev" in:readme fork:true` | 10 | Complete set; parent also retrieved10 without fork qualifier |
| `open-kernel-challenge in:name fork:true` | 9 | Complete set; parent gh search matched9 |
| `openkernelchallenge in:name fork:true` | 1 | Static audit repository only |
| `okc kernel in:name,description fork:true` | 7 | Complete set |
| `okc in:name language:cuda fork:true` | 0 | Broadened: CUDA-in-Python makes language filter unsuitable |
| `muse-spark in:name fork:true` | 149 | First100 plus overlapping50-row pages2/3; metadata coverage, ordering changed between requests |
| `okc in:name created:>2026-10-01 fork:true` | 98 | First100 plus page2 at50 rows to recover rendered tail; metadata triage |
| `muse_moe_w4a16 in:readme` unquoted | 59 | Full names recovered; tokenized false positives excluded |
| Broad `"Muse Spark" kernel`, unquoted decode, and `"Muse Spark" "kernels"` | 255 / 1587 / 1661 | First100 only, not exhaustive; exact identifiers replaced noisy searches |
| Parent broad gh phrase query | 100 returned | Noisy result set, explicitly not used as an exhaustive census |
| Kernarc README/issue references | 2 / 24 / 0 | `kernarc` README2, `matvey`+kernel24, domain/broader issue queries0; only alygg77 relevant |

Three additional web-search families covered exact challenge/Muse GitHub names, ten top-board aliases, and3xC1ibR;10 displayed results each, no pagination. The ten aliases were mpeng19, pjmullin3, dolanzhao, kwanUm, tonybolivar, colinlikescode, olety, tjiang-fair, naklecha, zzz0906. Public profile fallbacks replaced failed/empty API reads. No public Muse entry located; pjmullin3/tjiang-fair profiles reported no public repositories. Other profiles exposed unrelated projects or older upstream forks; those forks were not fully diff-audited. Alias resemblance alone is not identity evidence.

All five named repositories' fork APIs returned403 in the unauthenticated scout; public fork-page fallbacks explicitly reported no forks, consistent with search metadata. Independent copies still exist, as the dense blob identity proves. Named-donor successor searches covered each owner with muse/kernel and ruwwww's kobeniw alias; no successor located, but kobeniw API access was limited. Some API403s and large-source rendering limits were recovered via static pages/feeds; no completeness claim for hidden suffixes, gists, unindexed/arbitrarily named repositories, nondefault branches, sealed sources or private forum content.

**Research checks actually exercised:** live public-source/metadata retrieval; exact Git blob identity comparison; target-gap arithmetic; Zhebrak scored/raw/profile comparison; Rohan first-bank30-case timing-regularity check. The final CPU evidence-consumer run passed: **6 snapshot SHA256 matches,264 stored alias/board cells checked (198 unique pairs across33 aliases),24 pinned repository identifiers,30 benchmark cases**. These are evidence checks, **not kernel correctness/performance tests**. [EVIDENCE.json](EVIDENCE.json) preserves data, source hashes and provenance; [LOG.md](LOG.md) records each experiment/finding and corrections.

**Next move:** PM route R1 to attention owner for safe-source/ceiling analysis and R2 to dense owner for precision-preserving escape-schedule inspection. Do not import new code or spend qualification checks based on this report alone.
