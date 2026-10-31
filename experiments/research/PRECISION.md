# FP32 intermediates versus SGLang's BF16 rounding

Read-only research, 2026-10-31 PT. No GPU jobs, official checks, submissions, code imports, or account registration. Question: may a fused kernel keep an intermediate in FP32 where SGLang materializes BF16, then cast later?

## Verdict

**Not authoritatively settled by the public evidence located. Do not label the omission either organizer-approved or an established automatic disqualification.**

- The specific rule says activations **“stay BF16 wherever SGLang's kernel keeps them in BF16.”** It does not say **“at least BF16”** or expressly exempt higher-precision fused temporaries. The separate prohibition on *lower* precision therefore does not establish permission.
- Conversely, no located organizer statement says **“reproduce every intermediate BF16 rounding even inside a fused kernel.”** Our previous categorical “mandatory boundary / noncompliant” language went beyond what was verified as organizer policy. The source differences are real; their adjudication is not established by our audit.
- The official attention numerical reference itself retains FP32 through several of the questioned boundaries. Thus matching the reference within tolerance is **not** a test of SGLang's exact rounding sequence. This supports distinguishing implementation choices from numerical correctness, but does **not** waive a separately stated rule.
- A **screened** badge means a daily **automatic** review passed, not final manual approval. It cannot settle whether organizers knowingly approve these particular skipped casts.

**Operational recommendation, not an organizer ruling:** keep existing BF16-rounding-preserving variants as the conservative qualification path; treat a higher-precision variant as policy-unconfirmed until there is an attributable answer. Do not undo repairs solely because “FP32 is higher,” and do not report FP32 retention alone as proven cheating. This conclusion does not relax output/cache dtypes, all-key attention, lossless weight storage, or a genuinely lower-precision router path.

### Confidence

| Claim | Confidence |
|---|---|
| Exact current rules/task text and automatic-screening meaning quoted below | High; primary sources retrieved and six rule/UI quotations checked verbatim after whitespace normalization |
| Official numerical reference does not reproduce every SGLang BF16 boundary | High; live API source parsed and inspected without executing GPU code |
| No explicit organizer clarification found in the searched public material | Medium; bounded search, inaccessible Kernarc bodies and unpublished screening document |
| “Higher FP32 retention is definitely allowed” or “definitely disqualifying” | Insufficient evidence for either categorical claim |

## 1. Controlling text: what it says, and what it does not

Current [rules-version API](https://openkernelchallenge.dev/api/v1/rules) returns **`2026-10-22.1`**, linking [technical rules](https://openkernelchallenge.dev/rules) and [official terms](https://openkernelchallenge.dev/terms.pdf). The rules page was rendered in a browser because its static response is only an application shell. Its wording matches [llms.txt](https://openkernelchallenge.dev/llms.txt).

**Technical rules / llms.txt, Kernels → Restrictions:**

> “Activations stay BF16 wherever SGLang's kernel keeps them in BF16, and the FP8 KV cache stays FP8. The one exception is attention's queries: they can be rounded to FP8 E4M3 (as FA3 does with an FP8 cache), but nothing coarser.”

**Validation:**

> “This includes submissions that tamper with the timing or harness, do work outside the timed calls, cache or replay outputs, and compute at lower precision than the rules allow.”

> “All leaderboard positions are provisional until that review is done.”

The first sentence is a specific activation constraint; the second is part of a nonexclusive list of removal grounds. Reading only “lower precision” would omit the first constraint. Reading “stay BF16” as an explicit instruction to duplicate *every rounding event* would add an interpretation the text does not spell out. FP32 accumulation and FP32 evaluation *within* an operation are also not the same question as bypassing a BF16-valued activation between operations.

**Terms.pdf, §§1, 5, 6, 9:** §1 incorporates the Technical Rules; §5 makes sponsor decisions binding and gives Official Rules precedence over inconsistent interpretations. §9 states:

> “Sponsor reserves the right, in its discretion, to interpret and apply the rules, criteria, and mechanics of the Contest and to resolve any ties, disputes, or ambiguities.”

The terms contain no technical exception or explicit mandate about these FP32/BF16 intermediate casts. They leave this ambiguity to the sponsor, not to contestant consensus.

## 2. Official task and oracle: stronger evidence than contestant assumptions

Both live endpoints were read:

- [GET /problems/muse_attention_decode](https://openkernelchallenge.dev/api/v1/problems/muse_attention_decode)
- [GET /problems/muse_decode_layer](https://openkernelchallenge.dev/api/v1/problems/muse_decode_layer)

Their `task_md` fields say:

> “`reference.py`, the computation as the judge checks it: `generate_input`, the reference and the correctness checks, with `checks.py` beside it: the spec your output must meet.”

Both also repeat:

> “**Activations.** BF16 wherever SGLang's kernel keeps BF16. The KV cache stays FP8; queries may be rounded to FP8 E4M3, as FA3 does, and no coarser.”

Attention `task_md` correctness describes the output bound:

> “The output: within 2.5× the reference's own error when its queries are rounded to FP8 E4M3, plus 1.5%, kept between 3% and 15%, against an FP32 reference.”

The same endpoint's `description` explicitly acknowledges implementation-dependent cache rounding:

> “The new cache entries must be within 8e-2 of the reference's, and since FP8 entries round differently by implementation, the output is checked against the reference attending your own new entries.”

### Concrete discrepancy in the organizer-provided source

Source locations below are **fields/functions in the live attention problem API**, not a new implementation or code copied into this lane.

| Boundary | Official `files.reference.py` behavior | Contrast / implication |
|---|---|---|
| QK norm and RoPE | `_rms` begins with `x = x.float()`; `ref_kernel` sends those FP32 results through `_rope`; the BF16 query branch rounds after RoPE/scaling | Reference does not duplicate all SGLang materializations. Do not assume every baseline cast is a checker requirement |
| New K and cache scaling | `_quant` computes FP32 division then converts directly to FP8; `k_new` comes from FP32 norm/RoPE | No intermediate BF16 key/scaling round in this oracle |
| Attention result before output norm/gate | `ref_kernel` allocates `out` without a dtype override (FP32 in the normal environment), fills it with the attention weighted sum, then calls `_rms(out)` directly | In contrast `files.baseline.py` explicitly passes `out...to(torch.bfloat16)` into `rms_norm_gate` |
| Externally collected result | `collect` requires BF16 output and FP8 new cache entries; `check_implementation` applies tolerances and uses the candidate's checked new cache entries | External representation is enforced; intermediate exact-rounding equivalence is not established by this output check |

**Inference:** the official checker is intentionally a tolerance-based numerical oracle, not a bitwise emulator of SGLang. Retaining more precision can change downstream rounding and values; it is not numerically identical simply because it is higher precision. Neither fact decides the separate policy question.

## 3. Screening is automatic, not a precision-policy precedent

The llms.txt API documentation references `docs/SCREENING.md`. [That URL](https://openkernelchallenge.dev/docs/SCREENING.md) returns the SPA shell, not Markdown. The site's route dispatcher falls back to its homepage for unrecognized routes. No contents of SCREENING.md were recovered. The discovered [sid19991 mirror manifest](https://github.com/sid19991/open-kernel-challenge/blob/main/MANIFEST.md) corroborates the fallback, but is a third-party observation, not policy.

The **organizer-hosted client bundle**, [index-VBg5pvCv.js](https://openkernelchallenge.dev/assets/index-VBg5pvCv.js), functions `Tn`/`kn`, supplies exact user-facing explanations:

> “S: screened, passed the daily auto-review (at 0:00 UTC).”

> “P: pending, the auto-review was unsure.”

> “F: flagged.”

> “Note: final decisions will come from manual review after submissions close.”

The parent independently fetched the bundle and verified these strings. The client maps `under_review` to the Flagged display, and can show a team's alternate `screened_speedup` when its best row is not screened. Consequently, **a team's screened score is not necessarily the status of the exact public file/job under discussion**. The UI does not describe the automatic review's precision coverage or identify a manual ruling about omitted casts.

## 4. What identifiable public entries actually do

Source: live [attention leaderboard API](https://openkernelchallenge.dev/api/v1/leaderboard/muse_attention_decode) and [decode leaderboard API](https://openkernelchallenge.dev/api/v1/leaderboard/muse_decode_layer), captured during this investigation. Ranks are one-based positions in the ordered response, not the older snapshot ranks in FINDINGS.md. **Absent status is neither a rejection nor an approval.**

| Board / team | Rank / exact job | API screening evidence | Public-code linkage / implication |
|---|---|---|---|
| Attention / tjiang-fair | #1 / `7d1f783733cf49c6` | `screened`, 3.222074× | No exact submitted source identified; public owner-repository list was empty. Precision handling **unknown** |
| Decode / mpeng19 | #1 / `f91b449a02764f8f` | `screened`, 2.417806× | No job-to-source link found. Its public auto-inference harness is not this submission. Precision handling **unknown** |
| Attention / alygg77 | #81 / `1c8d81a0dbd94ee0` | `screening` absent, 2.653000× | Publisher explicitly links job/path/hash; public source omits questioned BF16 rounds. **Not a screened precedent** |
| Decode / alygg77 | #135 / `a61afb9687f14eff` | `screening` absent, 2.205807× | Publisher links exact job/path/hash; reachable embedded ATT7 path has the same omissions. **Not a screened precedent** |
| Attention / vjeux | #153 / `aff08d29ada347bb` | Best row status absent; 2.610411×; alternate `screened_speedup` **2.589572324545661** | Public file known, exact current-job linkage unproven. The other screened run's job/code are not supplied |
| Decode / vjeux | #259 / `2b144f55b8f64e0f` | `screening` absent, 2.101878× | Public file known, exact current-job linkage unproven; no screened-policy precedent |

**No screened top-50 job could be linked to public submitted code** using the existing24-source map and focused leader searches. Other screened top-50 rows found: attention colinlikescode#7, gauravjain14#9, DTennant#11, schrepa#21; decode dolanzhao#4, tonybolivar#6, colinlikescode#7, ntovas#13, Abarbir12#16, zzz0906#18. Their intermediate precision remains unknown here. The API exposes no source URL/hash; unauthenticated job lookup returned401 and documented authenticated job access is own-jobs-only. No access bypass attempted.

### Concrete source evidence, without treating it as permission

- **SGLang before RoPE:** [pinned FusedQKNorm](https://github.com/sgl-project/sglang/blob/a55ab7d5/python/sglang/kernels/ops/muse_spark_v12/fused_qk_norm.py#L80-L126) says `# Rope reads the bf16-rounded row, as the separate rope kernel did.` It performs `x = x.to(QKV.dtype.element_ty).to(tl.float32)`. That is a real baseline rounding point, irrespective of policy interpretation.
- **Alygg77 attention:** [QK/RoPE/cache path](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/kernels/muse_attention_decode.py#L1405-L1510) computes normalized/rotated values in float before later half/FP8 conversion. The [output helper](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/kernels/muse_attention_decode.py#L279-L283) is `float pr8(float x) { return x; }`; its [epilogue](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/kernels/muse_attention_decode.py#L896-L914) forms output-norm statistics and applies the gate before final BF16 conversion.
- **Alygg77 decoder:** [embedded QK/cache](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/kernels/muse_decode_layer.py#L1695-L1800) and [epilogue](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/kernels/muse_decode_layer.py#L1096-L1120) show the same omissions. The local dispatch trace establishes a reachable normal ATT7 route, not merely an unused source string; no actual judge-run execution trace was inspected here.
- **Vjeux:** [attention QK/cache](https://github.com/vjeux/okc-2026/blob/f35a41ee1c13e316dabbf772830d47b23b453393/muse_attention_decode.py#L1400-L1510), [attention epilogue](https://github.com/vjeux/okc-2026/blob/f35a41ee1c13e316dabbf772830d47b23b453393/muse_attention_decode.py#L890-L915), and [decoder epilogue](https://github.com/vjeux/okc-2026/blob/f35a41ee1c13e316dabbf772830d47b23b453393/muse_decode_layer.py#L1091-L1117) contain equivalent retained-float behavior. Its README has no exact judge-job manifest.

Do **not** describe either whole implementation as “FP32 everywhere”: ordinary query paths use FP16 staging and some variants use E4M3; these are separate precision questions. Likewise, the standalone cache-scaling boundary alone does not prove a numerical difference with the supplied power-of-two scales1 and1/64.

Alygg77's [published results.json](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/results.json) links attention job to reported SHA256 `e6d929d40e6cff8aa82270a423fca24dfccad64b6a82428db9c3377b661bf124` and decoder job to `8f17613bafb2369152f2739638bf8f6dfb193f6a79dbad3e6d25795cf1ebaf4a`. These are publisher provenance, not organizer-authenticated submitted-byte downloads or manual approval.

## 5. Organizer statements versus public interpretations

- [Official Paradigm launch article](https://www.paradigm.xyz/writing/muse-spark-open-kernel-challenge): no higher-precision/fused-rounding clarification located.
- [Kernarc public guide](https://agents.kernarc.com/) says “Every call needs a key” and describes a board where agents share ideas. Public guide/schema are accessible; discussion/list endpoints previously returned401. No existing `KERNARC_AGENTS_KEY` was available. No registration, posting, or inaccessible thread body was used. A peer's forum statement would still need organizer attribution.
- [Alygg77's router discussion](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/docs/H200_MEASURED_FACTS.md#5-router-precision-at-t-of-8-or-less-muse_decode_layer) says “Matsvei (kernarc user `matvey`, agents.kernarc.com) found this and we checked it against the task baseline code.” This is peer baseline analysis, not an organizer quotation, and concerns reducing an FP32 router input to BF16—the opposite direction.
- [Nikolas Sapalidis's contestant account](https://www.nikolassapalidis.tech/writing/gpu-kernel-challenge) directly claims: “The prefill one kept intermediate values in FP32 where the reference rounds to BF16” and “Each of those is a disqualification at final review.” The article describes a separate reviewer-agent audit but supplies no named organizer ruling or correspondence supporting that assertion. It is evidence of a conservative contestant interpretation, **not adjudication**.
- Vjeux/alygg77 source comments, passing scores, and our prior local precision audit demonstrate implementation differences and engineering judgments. They are not additional independent authorities on the rules.

Focused searches covered higher/higher-precision, FP32/BF16 rounding, organizer/clarification/permission, exact activation wording, screening filename, challenge/model/site names, relevant public repositories, and launch-author names. No verifiable organizer answer to this exact question was found. The homepage exposes no separate FAQ; the discovered `/faq` route also falls back to the application shell. This is not a claim that no private or unindexed clarification exists.

## 6. Exact clarification needed

> May a fused attention/decode implementation omit BF16 rounding after QK normalization/RoPE/cache scaling and before output RMSNorm, retain FP32 until a later cast, preserve all required external BF16/FP8 tensors and all keys, and pass the published tolerances? Or must it reproduce those BF16 rounding boundaries even when the intermediate is never materialized?

A written organizer answer to that question would resolve the gap. Until then, report **“policy-unconfirmed higher-precision fusion”**, not **“organizer-approved”** and not **“proven lower-precision violation.”** No kernel changes were made for this investigation.

## Verification and limits

Live rules/version/problem GETs and rendered site text were exercised; six rule/screening quotations matched the fetched primary text; the live numerical-reference source parsed successfully as Python AST. Public source/status/provenance checks were read-only. This investigation ran **zero GPU jobs and zero official checks**; it makes no new runtime correctness claim. Both browser tabs were closed. The unavailable original SCREENING.md, authenticated Kernarc discussions, and unexposed top-50 submitted sources prevent a stronger policy ruling.

This document supersedes the **certainty** of this lane's earlier BF16-boundary policy language in FINDINGS.md/LOG.md. It does not erase the observed arithmetic differences, certify any complete kernel, or change files owned by MAIN.
