"""Diagnostic precision attribution for the repaired decode glue (stdlib only).

Input: the complete work/candidates/decode_alygg77_bf16.py source. Output:
exactly one repair boundary disabled, NOT a submission. No torch/CUDA import,
execution, timing, compilation, policy selection, or extension renaming occurs
here. The caller must isolate modules, freeze the repaired composition/dense/
expert policy, and measure both stage and complete decode. Compilation may erase
a round-trip when the producer already stores BF16; this is not evidence that
an FP32-producing policy has zero repair cost.

Source authority: work/prior-art/alygg77-precision-audit.json, glue_audit 1..7.
Mappings below were read against the current repaired and attributed raw files,
not inferred from the audit's historical line numbers. Numerical adapters undo
NEW repair sites only: pre-existing compliant CUDA/graft rounds remain intact.
Structural/host records deliberately reject transform rather than return an
unchanged source or resurrect several numerical defects as one toggle.

No measurements or runtime qualification are claimed. All original source
notices survive because edits are confined to AST-selected function bodies.
"""

import ast


REPAIRED_SOURCE = "work/candidates/decode_alygg77_bf16.py"
RAW_SOURCE = "work/candidates/decode_alygg77.py"
LOADS = [name + suffix for name in ("D1", "D16", "D32", "V1", "V16", "V32")
         for suffix in ("s", "f")]

# Dispatch is resolved through aliases captured BEFORE later definitions.
DISPATCH = {
    "kernel": "final kernel_fn binds context -> _glue_context_parent_kernel -> late QKV/Lt kernel_fn -> original attention -> final _after_attention",
    "prepare": "base tuning -> graft/Lt setup -> static projection pinning -> composition tuning -> cluster/direct-QKV/native-router/FRT warmups -> final pin of compose_graft_enabled[1,8]=True",
    "D1_V1": "_after_attention -> _sl_tile_after_attention for T=1/8 with graft enabled; non-BF16 or split O output falls back to _graft_after_attention",
    "small_boundary": "_sl_cluster_module().b1 from _SL_GLUE_CUDA writes BF16 h, FP32 hr, alpha2*s1 carry and true tail squares; _sl_cluster_finish consumes BF16 sliced unpooler",
    "cluster_alternate": "_sl_tile_prior_after -> cluster wrapper -> _sl_cluster_after_attention -> _sl_graft_route; shadowed for both default SL_TILE_CONFIG keys, retained if tile configuration is changed",
    "other_loads": "_sl_previous_after_attention -> graft wrapper -> _graft_after_attention when compose_graft_enabled[T], else _vj_after_attention (the base row chain)",
    "graft_boundary": "_graft_boundary: S!=1 -> base; BF16 T<=16 -> _graft_glue_module().b1/b2c or _graft_b2_module().b2s; 16<T<=32 -> _graft_row_boundary_pre; T>32 -> _graft_row_boundary; unsupported cases -> base",
    "graft_route": "S=-1 -> split tensors; S=1 and FP32 4608 columns -> combined tensor; T=16/32 uses _sl_tile_route, other supported loads _graft_route; remaining partials -> _route",
    "FRT": "final _pool_router_fp32 -> _gle_frt only for GLE_FRT,T<=8,contiguous h/hr,ppick bfp/bfpf,router_fp32 present and both enabled SK2 entries; else _gle_prior_pool_router_fp32",
    "separate_router": "T<=8 fallback produces BF16 pooler via _bfp_gemm or F.linear, FP32 logits via _router_fp32_gemv(hr); above 8 uses BF16 hi/lo router and FP32 accumulation",
    "removed": "No GLE_T, _gle_after_attention, _gle_route, _gle_carry, _gle_b1x_source, _GleB1x, _GLE_W_CUR, _gemm_o, _bnd1_apply, _route_a/_route_b, _bnd2_sums/_bnd2_apply, or gluev2 policy in repaired dispatch",
}

ISOLATION = {
    "python_modules": ["_graft_glue_module.cache_clear", "_graft_b2_module.cache_clear",
                       "_sl_cluster_module.cache_clear", "_GLE_EXT", "_ROUTER_FP32_EXT",
                       "_DE_EXT", "bfp__get_ext", "_WS7"],
    "native_names": ["graft_glue_<source-md5>", "graft_b2_<source-md5>",
                     "sl_early_tail_<source-md5>", "gle_frt_<source-md5>"],
    "native_mutable_globals": ["g_sk2_pdl", "g_gle"],
    "per_state": ["graft", "graft_lt (including Lt workspace)", "router_fp32_ext",
                  "bfp.sk2", "bfp.fused_part", "bfp.fused_cnt", "ma_scratch", "ma_ext"],
    "policy_to_freeze": ["mm_pick", "moe_pick", "compose_graft_enabled", "compose_boundary_warps",
                         "compose_base_mm", "compose_base_graft_lt", "graft_lt", "bfp.choice",
                         "bfp.disabled", "bfp.sk2", "ma_sfx", "att_ok", "ma_ok"],
    "warning": "Independent source module and state per arm; parent owns collision-safe extension names. Clearing a Python lru_cache alone does not unload/isolate a native module. _WS7 and native launch globals remain concurrency hazards after explicit-gain repair. Serialize paired captures; do not infer thread safety from serialized measurements.",
}

# Each tuple is (unique top-level function, exact old anchor, replacement,
# expected occurrences). Scope is never selected by source line number.
_GAIN = "(1.0 + tl.load(GPRE + {index}).to(tl.float32)).to(tl.bfloat16).to(tl.float32)"
_LOAD = "_gl.load(Branch + row * F + c, c < F, 0)"
_B2 = "f = (f.to(tl.bfloat16).to(tl.float32) * gp).to(tl.bfloat16).to(tl.float32)"
_GLUON = ("_graft_row_boundary", "_graft_row_boundary_pre")
_EDITS = {
    "glue_pooler_output_bf16": [
        ("_route", "    pooled = pooled.to(tl.bfloat16).to(tl.float32)\n", "", 1),
        ("_graft_route", "tl.load(ALL + t * 4608 + c).to(tl.bfloat16).to(tl.float32)",
         "tl.load(ALL + t * 4608 + c).to(tl.float32)", 1),
        ("_sl_graft_route", "tl.load(ALL + t * 4608 + c).to(tl.bfloat16).to(tl.float32)",
         "tl.load(ALL + t * 4608 + c).to(tl.float32)", 1),
        ("_sl_tile_route", "tl.load(ALL+t*4608+c).to(tl.bfloat16).to(tl.float32)",
         "tl.load(ALL+t*4608+c).to(tl.float32)", 1),
        ("_sl_tile_route", "tl.load(ALL+t*4608+oc).to(tl.bfloat16).to(tl.float32)",
         "tl.load(ALL+t*4608+oc).to(tl.float32)", 1),
    ],
    "glue_pre_expert_effective_gain": [
        (name, _GAIN.format(index=index),
         "(1.0 + tl.load(GPRE + " + index + ").to(tl.float32))", count)
        for name, index, count in (("_route", "c", 1), ("_route_split", "c", 1),
                                   ("_graft_route", "c", 1), ("_sl_graft_route", "c", 1),
                                   ("_sl_tile_route", "oc", 2))
    ],
    "glue_b1_projection_bf16": [
        ("_boundary1", "    a = a.to(tl.bfloat16).to(tl.float32)\n", "", 1),
        *[(name, "b = " + _LOAD + ".to(_gl.bfloat16).to(_gl.float32)",
           "b = " + _LOAD + ".to(_gl.float32)\n    if POST:\n        b = b.to(_gl.bfloat16).to(_gl.float32)", 1)
          for name in _GLUON],
    ],
    "glue_base_b1_branch_bf16": [
        ("_boundary1", "    n = n.to(tl.bfloat16).to(tl.float32)\n", "", 1),
    ],
    "glue_base_b1_effective_gain": [
        ("_boundary1", "(1.0 + tl.load(G + offs).to(tl.float32)).to(tl.bfloat16).to(tl.float32)",
         "(1.0 + tl.load(G + offs).to(tl.float32))", 1),
    ],
    "glue_b2_projection_bf16": [
        ("_boundary2", _B2, "f = (f * gp).to(tl.bfloat16).to(tl.float32)", 1),
        *[(name, "b = " + _LOAD + ".to(_gl.bfloat16).to(_gl.float32)",
           "b = " + _LOAD + ".to(_gl.float32)\n    if not POST:\n        b = b.to(_gl.bfloat16).to(_gl.float32)", 1)
          for name in _GLUON],
    ],
    "glue_base_b2_weighted_bf16": [
        ("_boundary2", _B2, "f = f.to(tl.bfloat16).to(tl.float32) * gp", 1),
    ],
    "glue_base_b2_branch_bf16": [
        ("_boundary2", "    n = n.to(tl.bfloat16).to(tl.float32)\n", "", 1),
    ],
    "glue_base_b2_post_effective_gain": [
        ("_boundary2", "(1.0 + tl.load(GPOST + offs).to(tl.float32)).to(tl.bfloat16).to(tl.float32)",
         "(1.0 + tl.load(GPOST + offs).to(tl.float32))", 1),
    ],
    "glue_base_b2_next_effective_gain": [
        ("_boundary2", "(1.0 + tl.load(GNEXT + offs).to(tl.float32)).to(tl.bfloat16).to(tl.float32)",
         "(1.0 + tl.load(GNEXT + offs).to(tl.float32))", 1),
    ],
}

_BASE_CONDITION = "Only when _vj_after_attention or a _graft_boundary fallback invokes the base row kernel; final D1/V1 normally take the cluster/tile chain. Nearby shapes may select base. Record actual policy, not just load name."
_PROJECTION_LIMIT = "BF16 producer storage is deliberately untouched: an already-BF16 dense/SK2/Lt output cannot be unrounded by changing its consumer. This toggle attributes newly inserted FP32-consumer conversion only; physical producer output is a separate stage contrast."


def _numerical(group_id, requirement, stage, description, reasons, ideas, **extra):
    return dict(id=group_id, audit_requirement=requirement, stage=stage,
                description=description, source_symbols=list(dict.fromkeys(e[0] for e in _EDITS[group_id])),
                inactive_reasons=reasons, compliant_recovery_ideas=ideas,
                attribution_kind="single_boundary", ablatable=True, **extra)


GROUPS = [
    _numerical("glue_pooler_output_bf16", "glue_audit.2 (also output portion of .1)",
               "pooler projection -> pre-expert RMSNorm",
               "Remove only the fully-reduced pooler slice BF16 conversion, including both tile numerator and statistics loads. Router hi/lo columns and logits stay FP32; all gains stay repaired.",
               ["T<=8 separate/FRT producers already store BF16 pooler; _route_split has no new round to remove.",
                "_sl_graft_route is shadowed by tile dispatch at default T=1/8 but is covered as a selectable alternate.",
                "Deleted _route_a and _gle_route are mapped by structural records, not silently omitted."],
               ["For an FP32 combined projection, round only pooler columns in its producer epilogue before publishing norm partials; keep both router slices FP32. Round after complete split-K accumulation, not per partial.",
                "For D1 FRT, pooled output is already BF16: use those exact rounded values in any fused norm partials; never reconstruct projected surrogate rows."],
               conditional_loads=LOADS, runtime_predicate="S!=-1 and FP32 pooler accumulation; V16/V32 default Lt pr plans have fp32=True, but prepare can choose another policy"),
    _numerical("glue_pre_expert_effective_gain", "glue_audit.3", "pre-expert RMSNorm",
               "Undo BF16(1+checkpoint_gain) in all five surviving route consumers; leave normalization/product FP32 and final activation BF16.",
               ["_route_b and _gle_route were removed with their policies; see structural records."],
               ["Prepare the immutable BF16 effective pre-expert gain once per state, then load it without add/cast in all route consumers, matching existing graft pre1/post2/pre2 preparation. Do not round normalized activation before gain multiplication."],
               active_loads=LOADS),
    _numerical("glue_b1_projection_bf16", "glue_audit.5", "O projection -> B1 statistics",
               "Remove the new projection-output conversion in base B1 and POST=False Gluon boundaries; POST=True remains repaired.",
               [_PROJECTION_LIMIT, "CUDA cluster B1 requires a BF16 branch and already complied; no new cast exists there."],
               ["Have a non-split O producer round its complete output to BF16 before producing sum-of-squares partials. B1 still needs a global first norm and a second reduction of the actual rounded-branch residual; the second cannot use unrounded polynomial moments."],
               conditional_loads=LOADS),
    _numerical("glue_base_b1_branch_bf16", "glue_audit.5 (native-router consistency in .4)",
               "base B1 normalized branch -> FP32 residual",
               "Remove only base B1's post-attention normalized-branch BF16 round. Subsequent residual statistics use that diagnostic branch, not stale moments; hr remains FP32.",
               [_BASE_CONDITION, "Graft/cluster/Gluon normalized-branch rounds predate this repair and remain intact."],
               ["Keep round-to-BF16 in registers before residual FMA and accumulate true updated-stream squares in that same row/cluster pass; do not add a global BF16 branch buffer."],
               conditional_loads=LOADS),
    _numerical("glue_base_b1_effective_gain", "glue_audit.4 and .5", "base B1 pre-FFN gain",
               "Remove only base B1's effective gain round; router output still uses the unrounded-final hh in FP32 from the same updated stream.",
               [_BASE_CONDITION, "Grafted pre1 is prepared in BF16 already and is intentionally unchanged."],
               ["Use a state-local cached BF16 effective pre-FFN gain in the base path, as graft pre1 already does; keep raw checkpoint data immutable."],
               conditional_loads=LOADS),
    _numerical("glue_b2_projection_bf16", "glue_audit.5", "unpooler projection -> B2 weighted input",
               "Remove only the fully-reduced unpooler conversion in base B2 and POST=True Gluon boundaries; preserve weighted-input/normalized-branch rounds, gains and POST=False B1 conversion.",
               [_PROJECTION_LIMIT, "_GRAFT_GLUE_CUDA b2c, _GRAFT_B2_CUDA b2s, and _sl_cluster_finish already receive BF16 tensors. Their rounds are unchanged."],
               ["Fuse rounding of a complete unpooler result, multiplication by cached BF16 post2, the second BF16 round, and local weighted squares into a producer epilogue where the complete result is owned. Do not fold post2 through the projection or round individual split partials."],
               conditional_loads=LOADS),
    _numerical("glue_base_b2_weighted_bf16", "glue_audit.5", "base B2 weighted input -> RMSNorm",
               "Remove only BF16(unpooler_bf16 * effective_post_gain), preserving the unpooler and normalized-branch boundaries.",
               [_BASE_CONDITION, "Small-load _sl_cluster_finish and all graft post-weight rounds were already present in raw source."],
               ["Round weighted values in registers, then derive the first RMS statistics from exactly those values; producer fusion can avoid a separate intermediate write without removing this conversion."],
               conditional_loads=LOADS),
    _numerical("glue_base_b2_branch_bf16", "glue_audit.5", "base B2 normalized branch -> FP32 residual",
               "Remove only base B2's normalized-branch conversion; FP32 s2 and its actual squares remain the next norm input.",
               [_BASE_CONDITION, "Small-load finish and graft normalized-branch rounds already complied and remain unchanged."],
               ["Fuse conversion, residual FMA, and true residual-square accumulation in one existing boundary pass. In D1 the early tail is legal because it is computed from the actual FP32 carry, independent of the later 1024-column branch."],
               conditional_loads=LOADS),
    _numerical("glue_base_b2_post_effective_gain", "glue_audit.5", "base B2 effective post-FFN gain",
               "Undo only the effective GPOST gain conversion, retaining all three activation rounds.",
               [_BASE_CONDITION, "Graft post2 already has BF16 storage."],
               ["Precompute BF16(1+post_ffn_norm) once per state and reuse the existing graft effective weight across base and graft consumers."],
               conditional_loads=LOADS),
    _numerical("glue_base_b2_next_effective_gain", "glue_audit.5", "base B2 effective next-pre-attention gain",
               "Undo only the effective GNEXT gain conversion; retain FP32 residual and BF16 output.",
               [_BASE_CONDITION, "Graft pre2 already has BF16 storage."],
               ["Precompute the immutable BF16 effective next-layer gain once per state; preserve FP32 normalization and multiplication until final BF16 output."],
               conditional_loads=LOADS),
]


def _structural(group_id, requirement, stage, description, symbols, reasons, raw, repaired, ideas, kind="structural"):
    return dict(id=group_id, audit_requirement=requirement, stage=stage, description=description,
                source_symbols=symbols, inactive_reasons=reasons, compliant_recovery_ideas=ideas,
                attribution_kind=kind, ablatable=False,
                raw_contrast=dict(source=RAW_SOURCE, entrypoints=raw),
                repaired_contrast=dict(source=REPAIRED_SOURCE, entrypoints=repaired))


GROUPS += [
    _structural("glue_deferred_gle_removed", "glue_audit.1", "O + B1 + pooler/router + route + finish tail",
                "Removal of the two-surrogate-row deferred projection is a multi-stage architecture contrast, not a single cast toggle. Its pooler input/output and carry-tail statistics all differ.",
                ["_gle_source", "_gle_after_attention", "_gle_route", "_gle_carry", "_sl_tile_after_attention", "_SL_GLUE_CUDA", "_sl_cluster_finish"],
                ["GLE_T, GLE_RC and deferred entrypoints are absent from repaired source. Raw defaults enable T=1 only, additionally requiring gate=None, PDL, codec O eligibility, graft, agent experts and router state.",
                 "Casting r2*(Wu+r1*Wv) cannot recover W*BF16(normalized_updated_stream*effective_pre). Raw tail q4+2*r1*q5+r1*r1*q6 does not describe the BF16-branch carry."],
                ["final _after_attention -> _gle_after_attention -> _gle_ext().gle_o -> _gle_frt(uv,uvf) on 2*T rows -> _gle_route/_gle_carry -> _sl_cluster_finish"],
                ["final _after_attention -> _sl_tile_after_attention -> dense(o) -> _sl_cluster_module().b1 -> _pool_router_fp32/_gle_frt(h,hr) on T rows -> _sl_tile_route -> _sl_cluster_finish"],
                ["D1: preserve the current materialized h/hr boundary. An O producer may BF16-round each completed output in registers and emit first-norm partials without another global pass. Completing B1 in that producer requires a real cross-CTA reduction/synchronization and a second reduction of actual BF16-branch residual values, not algebraic moments; measure this structural cost before implementing.",
                 "Current _SL_GLUE_CUDA already rounds normalized branch in registers, writes h/hr once, and derives carry/tail from the same stream. Current FRT interleaves legal BF16-pooler and FP32-router stages; retain it rather than paying a separate FP32-router launch.",
                 "A pooler producer epilogue can form partial norm squares after BF16 output rounding; final normalization requires all 4096 rounded elements. Reuse an existing complete-result/reduction owner only after measuring cost; do not assume launch deletion wins under PDL."]),
    _structural("glue_native_router_b1x_removed", "glue_audit.4", "B1 native FP32 router activation",
                "Raw B1X had a separate unrounded-branch stream and FP32(1+w), plus an extra cluster reduction. Repaired B1 shares the actual residual and BF16 effective gain while retaining the final router activation in FP32.",
                ["_gle_b1x_source", "_gle_b1x_module", "_GleB1x.b1", "_SL_GLUE_CUDA", "_boundary1"],
                ["The alternate source/wrapper were deleted, not switched off. Reinstating B1X disables two semantic repairs and changes reduction structure; it cannot be labeled a single-boundary ablation.",
                 "Raw D1 usually bypasses B1X via deferred GLE. Compare raw V1 B1X directly; for D1 record GLE-disabled diagnostic policy separately. Neither is a submission."],
                ["_sl_tile_after_attention -> _GleB1x.b1 -> _gle_b1x_module().b1 generated by _gle_b1x_source; downstream _gle_frt or _router_fp32_gemv", "D1 deferred router contrast: _gle_after_attention/gle_o -> _gle_frt(uv,uvf) -> _gle_route"],
                ["_sl_tile_after_attention -> _sl_cluster_module().b1 generated from _SL_GLUE_CUDA; _graft_glue_module().b1 or _boundary1 on fallback; downstream _gle_frt or _router_fp32_gemv"],
                ["The cheapest source-visible legal construction is the current shared hh register value: store it to BF16 h and FP32 hr, using the same actual-stream reduction and BF16 pre1. Do not upcast h into hr and do not add a second mathematical-oracle residual stream.",
                 "Possible producer integration must publish this same pre-final-round FP32 hh to the FP32 router; keeping FRT's interleaving avoids the documented separate-router launch penalty."]),
    _structural("glue_moment_policy_removed", "glue_audit.2, .3 and .5", "base parallel gluev2 B1/route/B2 policy",
                "The v2 atomic moment pipeline, tuner choices and router-input helper were structurally retired in favor of true row statistics after BF16 boundaries.",
                ["_gemm_o", "_bnd1_apply", "_route_a", "_route_b", "_bnd2_sums", "_bnd2_apply", "_router_input_norm", "_vj_after_attention"],
                ["All six v2 stage functions, _router_input_norm, FORCE_GLUEV2 and state.gluev2 are absent. Late casts cannot fix polynomial moments computed before normalized-branch rounding.",
                 "Raw v2 is reached only through base composition (compose_graft_enabled=False) and gluev2/ FORCE_GLUEV2; do not assume it ran for any measured load without mode-state evidence."],
                ["_vj_after_attention with v2=True: _gemm_o + _bnd1_apply (+_router_input_norm at T<=8)", "S!=-1: _route_a + _route_b; S=-1: _route_split", "_bnd2_sums + _bnd2_apply"],
                ["_vj_after_attention: dense/dense_ng + _boundary1; _route/_route_split; dense(un) + _boundary2", "graft/tile/cluster policy alternatives are separately identified in DISPATCH"],
                ["A legal parallel boundary can reduce rounded projection squares first, then form BF16 normalized branches and FP32 residuals, then reduce those actual residual squares. It needs real phases; no unrounded cross-moment shortcut is equivalent.",
                 "Keep the existing legal small-load early-tail optimization: columns outside the sliced unpooler depend only on actual carry, so their squared sum can be produced in B1. This does not justify raw GLE's unrounded moment tail."]),
    _structural("glue_explicit_state_gain", "glue_audit.6", "host B1 state ownership",
                "Deleted the module-global checkpoint gain handoff; native boundary receives the calling state's immutable effective pre1 explicitly. This is safety attribution, not a GPU arithmetic toggle.",
                ["_GLE_W_CUR", "_GleB1x.b1", "_after_attention", "_sl_cluster_module().b1", "_graft_boundary"],
                ["No old handoff remains to disable. Restoring the raw wrapper would also restore B1X arithmetic and its extra reduction.",
                 "Global attention workspace _WS7 and native g_sk2_pdl/g_gle still require isolation; explicit gain fixes only the audited gain race."],
                ["final _after_attention assigns _GLE_W_CUR -> _GleB1x.b1 later reads it"],
                ["_sl_tile_after_attention/_sl_cluster_after_attention pass g['pre1'] directly to _sl_cluster_module().b1", "_graft_boundary passes g['pre1']; base _boundary1 receives state['g_preffn'] directly"],
                ["Retain explicit per-state arguments. Benchmark host dispatch separately if desired; graph replay timing does not measure this Python ownership change. Do not claim zero cost from absence of a new device kernel."],
                kind="host_only"),
    _structural("glue_codec_fallback", "glue_audit.7", "host FRT/combined-SK2 eligibility and fallback",
                "Require actual present and enabled pooler/router SK2 entries. On rejection retain supported BF16 pooler plus FP32 router at T<=8, or ordinary dense selector fallback above 8.",
                ["_fused_pool_router_enabled", "_pool_router_fp32", "bfp_fused_pooler_router", "_gle_frt", "bfp_sk2_compress"],
                ["Guard decisions depend on prepare-time codec acceptance/configuration, not runtime tensor values. Disabling a guard can produce KeyError or launch a rejected configuration, not a numerical cost sample.",
                 "Raw direct _gle_after_attention -> _gle_frt bypass was removed with deferred GLE; no corresponding repaired direct caller exists."],
                ["final _pool_router_fp32 checks only bfp state presence before _gle_frt", "_gle_after_attention directly invokes _gle_frt when GLE_FRT", "bfp_fused_pooler_router unconditionally indexes SK2 entries"],
                ["final _pool_router_fp32 -> _fused_pool_router_enabled -> _gle_frt OR _gle_prior_pool_router_fp32 -> _bfp_gemm/F.linear + _router_fp32_gemv", "bfp_fused_pooler_router -> _fused_pool_router_enabled -> SK2 fusion OR separate bfp_kernel_fn(pooler/router)"],
                ["Preserve the guard. Record bfp.sm90/sk2/disabled/choice and benchmark accepted-codec FRT and rejected-codec fallback as distinct supported paths; a graph replay cannot price Python guard overhead.",
                 "Do not restore an invalid fused launch to save host checks. Any prepare-time cached eligibility must remain per-state and consistent with disabled routes."],
                kind="host_only"),
]

# Coverage of embedded sources matters even though none need text edits for these
# newly added numerical repair sites. These existing rounds must NOT be disabled
# by a blanket gb/pr8 replacement.
EMBEDDED_SOURCE_COVERAGE = {
    "_GRAFT_GLUE_CUDA": "b1/b2c load BF16 branches; POST weights and normalized branches round in registers; router stores FP32 h; effective gains are BF16 state tensors. Already present in raw source.",
    "_GRAFT_B2_CUDA": "b2s loads BF16 sliced unpooler, rounds weighted input and normalized branch, uses actual stream statistics. Dormant SBF=false partial-reduction branch already rounds its complete sum. Already present in raw source.",
    "_SL_GLUE_CUDA": "Final default D1/V1 B1 source, no B1X override. Normalized branch is BF16-rounded; ns/carry/tail and hr are FP32 with BF16 pre1. Its unused POST/SLICED templates retain existing rounds.",
    "_gle_source": "Derives non-deferred FRT from bfp_SK2_SRC. Inputs xb BF16 and x32 FP32; pooler output BF16 and router output FP32. No gle_o export, surrogate rows or deferred moments remain.",
    "_sl_cluster_finish": "Reads BF16 sliced unpooler and BF16 post2/pre2, rounds weighted input and normalized branch, combines actual head squares with B1's actual carry tail. No newly added round to toggle.",
}


def transform(source: str, group_id: str) -> str:
    """Return one diagnostic repair-off copy; reject structural/host entries.

    Exact anchors and counts are checked in unique AST-selected functions.
    Unknown IDs, missing/duplicate functions, syntax drift, or a no-op fail
    loudly. This intentionally does not import or execute the candidate and
    leaves native CUDA strings, dispatch, extension names and notices intact.
    """
    matches = [group for group in GROUPS if group["id"] == group_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous glue group: {group_id!r}")
    group = matches[0]
    if not group["ablatable"]:
        raise ValueError(f"{group_id}: {group['attribution_kind']} contrast, not a single-boundary ablation; "
                         "use raw_contrast/repaired_contrast entrypoints and preserve separate attribution")
    tree = ast.parse(source)
    functions = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.setdefault(node.name, []).append(node)
    edits_by_function = {}
    for name, old, new, expected in _EDITS[group_id]:
        edits_by_function.setdefault(name, []).append((old, new, expected))
    lines = source.splitlines(keepends=True)
    changes = []
    for name, edits in edits_by_function.items():
        nodes = functions.get(name, [])
        if len(nodes) != 1:
            raise AssertionError(f"{group_id}: expected one top-level {name}, found {len(nodes)}")
        node = nodes[0]
        original = "".join(lines[node.lineno - 1:node.end_lineno])
        changed = original
        for old, new, expected in edits:
            found = changed.count(old)
            if found != expected:
                raise AssertionError(f"{group_id}/{name}: expected {expected} anchors, found {found}: {old!r}")
            changed = changed.replace(old, new)
        if changed == original:
            raise AssertionError(f"{group_id}/{name}: ablation unexpectedly left source unchanged")
        changes.append((node.lineno - 1, node.end_lineno, changed))
    for start, end, replacement in sorted(changes, reverse=True):
        lines[start:end] = [replacement]
    result = "".join(lines)
    if result == source:
        raise AssertionError(f"{group_id}: empty transformation")
    return result
