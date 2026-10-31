"""Diagnostic attention-repair attribution; NEVER use transformed code as a submission.

Authority: work/prior-art/alygg77-precision-audit.json, core_audit findings 1-2.
Mapping was inspected in decode_alygg77_bf16.py, not inferred from old line numbers.
No CUDA imports, builds, timings, or optimization candidates are part of this module.
All edits preserve the input source's notices and unrelated arithmetic/dispatch.

Nine independent switches separate backend and semantic boundary. Native changes
include the dormant QDIRECT/QP16 sites, but do not enable them. Fallback switches
do not force fallback dispatch: a zero whole-layer delta on native scored cases is
not a measurement of fallback arithmetic. For fallback stage contrasts, use the
same fallback route in both arms and report those separately from default decode.
The parent owns collision-safe extension names, private module/state/cache objects,
and measurement. This adapter deliberately does not rename extensions or allocate.

Permanent mapping/usage documentation lives here because this assignment owns only
this Python module. All recovery ideas below are hypotheses, not measured wins.
"""

import ast


_LOADS = [name + suffix for name in ("D1", "D16", "D32", "V1", "V16", "V32")
          for suffix in ("s", "f")]
_NATIVE_CONDITION = (
    "prepare sets att_ok=True; kernel_fn requires _supported7(cfg) and "
    "_att_fits7(data,cfg). The scored head_dim=64, Hq=16, Hkv=2, "
    "q_len=1/8 shapes fit. A caught native RuntimeError re-enters with att_ok=False. "
    "Load labels mean source-proven eligible dispatch, not observed GPU execution."
)
_FALLBACK_REASON = (
    "No default scored-load launch: prepare sets att_ok=True and supported/fitting "
    "D1/D16/D32/V1/V16/V32 sliding/full use ATT7. Fallback is reached when att_ok "
    "is false, support/32-bit size guards fail, or the native try raises RuntimeError. "
    "Old comments claiming selected verify shapes use FA3 are stale; the actual "
    "agent_attn predicate has no such verify exclusion."
)
_NATIVE_CACHE = ["_ATT7", "_WS7", "_CL7", "_NUM_SMS7", "_INV_FREQ_CACHE",
                 "_GUARDS", "_GUARDED_IDS", "data.k_cache", "data.v_cache"]
_NATIVE_COMPILER = {
    "source": "ATT7_CUDA_SRC (one direct raw-string assignment; no late alternate)",
    "builder": "_att7 -> torch.utils.cpp_extension.load_inline",
    "extension_name": "muse_att_v298w1_bf16_{arch}",
    "flags": ["-DOKC_QDIRECT=0", "-DOKC_PK2=0", "-DOKC_W0ENABLE=1", "-O3",
              "-DOKC_SM90={1 if major >= 9 else 0}", "--expt-relaxed-constexpr"],
    "isolation": (
        "Parent must rename each diagnostic extension before import/build, use a "
        "fresh Python candidate module, and not share _ATT7 or mutable attention "
        "workspaces/counters/cache tensors across arms. prepare compiles _att7 "
        "and invokes _attn_selftest; source edits after that are too late."
    ),
}
_DISPATCH = (
    "Final kernel_fn binds CUDA context then calls _glue_context_parent_kernel; "
    "the preceding graft wrapper may use _graft_lt_run for T=128/256 QKV, then "
    "calls the same _decode_attention7 -> _launch7 -> _att7().run. Other shapes "
    "go through _vj_kernel_fn. Native ordinary torch.mm, BFP, and split-K-reduced "
    "QKV paths already materialize BF16; the late direct-escape BFP path retains "
    "BF16 output. No extra native QKV repair launch exists."
)


def _native(group_id, requirement, stage, description, symbols, ideas, caveats=()):
    return {
        "id": group_id,
        "audit_requirement": requirement,
        "stage": stage,
        "description": description,
        "source_symbols": ["ATT7_CUDA_SRC"] + list(symbols),
        "active_loads": list(_LOADS),
        "active_condition": _NATIVE_CONDITION,
        "inactive_reasons": list(caveats),
        "compliant_recovery_ideas": list(ideas),
        "runtime_entrypoints": ["kernel_fn", "_decode_attention7", "_launch7", "_att7().run"],
        "cache_handles": list(_NATIVE_CACHE),
        "compiler": dict(_NATIVE_COMPILER),
        "dispatch_caveat": _DISPATCH,
        "measurement_scope": (
            "Rounds execute inside existing ATT7 kernels, not separate launches. "
            "Pair repaired/diagnostic on the same plan and private KV contents; "
            "time the actual ATT7 launch chain and whole decode. Repeated >=0.5% "
            "whole-decode recovery is a ceiling to investigate legally, not a "
            "license to submit this repair-disabled source."
        ),
    }


def _fallback(group_id, requirement, stage, description, symbols, ideas, caveats=()):
    return {
        "id": group_id,
        "audit_requirement": requirement,
        "stage": stage,
        "description": description,
        "source_symbols": list(symbols),
        "inactive_reasons": [_FALLBACK_REASON] + list(caveats),
        "compliant_recovery_ideas": list(ideas),
        "runtime_entrypoints": ["_vj_kernel_fn", "_kernel_fn_fa3", "dense_ng"],
        "cache_handles": ["candidate-module Triton JIT function objects", "state.fa3",
                          "state.invf", "state.mm_pick", "data.k_cache", "data.v_cache"],
        "measurement_scope": (
            "No active_loads field: fallback execution is conditional, not established "
            "for any default scored load. Compare the same fallback stage and "
            "dispatch in repaired and diagnostic arms. Do not charge its separate "
            "_qk_prep/FA3/Triton launches to the native ATT7 repair."
        ),
    }


GROUPS = [
    _native(
        "attn_native_qk_norm", "core_audit finding 1: BF16 QK norm before RoPE",
        "attention/native-qk-preparation",
        "Remove only x*r BF16 rounds in the live 8-value Q/K loops and dormant "
        "16-value QP16 query loop. Keep projection, RoPE, cache-scale and output rounds.",
        ["segment", "round_bf16", "QDIR/OKC_QP16"],
        ["Use __floats2bfloat162_rn plus exact __bfloat1622float2 expansion for pairs "
         "already resident in each thread; retain FP32 reduction and the BF16 norm "
         "boundary before RoPE. Inspect generated instructions before claiming a gain.",
         "Keep the round inside the current Q/K preparation epilogue; do not add "
         "a standalone norm tensor or launch."],
        ["_att7 passes -DOKC_QDIRECT=0, so QDIR and QDIR&&OKC_QP16 are compile-time "
         "inactive. The transform covers their source but attributes no runtime cost."]),
    _native(
        "attn_native_rope", "core_audit finding 1: BF16 final Q/K after RoPE",
        "attention/native-rope",
        "Remove only the rotated even/odd BF16 rounds at all three Q/K sites; "
        "leave norm and cache-scaling boundaries intact.",
        ["segment", "round_bf16", "QDIR/OKC_QP16"],
        ["Pack the two rotated components with a BF16x2 RN conversion at the existing "
         "RoPE epilogue, then expand exactly for existing FP8/f16 staging. Preserve "
         "the original FP32 rotation expressions and rounding order.",
         "Reuse the rounded pair for all consumers rather than separately rounding "
         "each consumer; no additional BF16 boundary between norm and sigmoid."],
        ["No RoPE work when p.rope is false; a full/sliding label alone does not "
         "prove the configured rope_theta. QDIRECT/QP16 remain disabled by builder."]),
    _native(
        "attn_native_cache_scale", "core_audit finding 1: BF16 K/V division before FP8",
        "attention/native-cache-write",
        "Remove only BF16 rounds after K/V reciprocal-scale multiplication, before "
        "to_e4m3. New cache rows and their staged copies still traverse all keys.",
        ["segment", "to_e4m3", "round_bf16"],
        ["Combine adjacent scaled K/V values with BF16x2 RN conversions before "
         "existing FP8 packing; do not convert the original FP32 products directly "
         "to FP8 (double-rounding boundaries are required).",
         "Reuse the same rounded/packed word for S.kn/S.vn and global cache stores, "
         "as the existing code already does; avoid a new scale/materialization launch."],
        ["K/V preparation is guarded by has_new/live. Unit or power-of-two scales "
         "may make some rounds numerically redundant, but that is not evidence "
         "the compiler removed them or permission to remove general scale rounding."]),
    _native(
        "attn_native_output", "core_audit finding 2: BF16 attention output before norm/gate",
        "attention/native-finalize",
        "Make pr8 identity only; every native finalizer continues FP32 norm/sigmoid "
        "and its original final BF16 store. Do not disable the shared round_bf16 helper.",
        ["pr8", "cluster_finish", "fin_nc1", "segment", "flat_finalize"],
        ["Replace scalar pr8 sequences with packed BF16x2 RN conversion/expansion "
         "for existing float4/8 accumulator groups, then compute RMS statistics "
         "from those expanded BF16 values in the original FP32 reduction order.",
         "Fuse materialization into the already-existing finalizer epilogue; retain "
         "full FP32 norm-times-sigmoid until the single final BF16 output store. "
         "Do not repeat closed finalize-row-cap or wave-balancing experiments."],
        ["cluster_finish, fin_nc1, the two segment finalize branches, and "
         "flat_finalize are alternatives selected by _plan7/_TUNED7 and native "
         "residency/coop logic, not five extra launches per decode."]),
    _fallback(
        "attn_fallback_qkv_projection", "core_audit finding 1: fully reduced fallback Q/K/V/gate BF16",
        "attention/fallback-qkv-preparation",
        "Remove six BF16 projection rounds after the S-partial reduction only; "
        "keep subsequent norm/RoPE/cache rounds and the native BF16 QKV routes.",
        ["_qk_prep", "dense", "_gemm_splitk"],
        ["For S=1, have the selected projection's existing epilogue store BF16 "
         "rather than FP32 if the consumer contract is changed consistently; "
         "for S>1 keep FP32 partials and round only after their final reduction.",
         "Keep the existing fused _qk_prep reduction/rounding for split-K rather "
         "than inserting a separate tensor-conversion launch."]),
    _fallback(
        "attn_fallback_qk_norm", "core_audit finding 1: fallback BF16 QK norm before RoPE",
        "attention/fallback-qk-preparation",
        "Remove four normalized qe/qo/ke/ko BF16 rounds, preserving all other boundaries.",
        ["_qk_prep"],
        ["Keep paired even/odd normalization casts adjacent for Triton vector "
         "lowering; inspect PTX/SASS for packed conversion opportunities without "
         "changing FP32 norm statistics or the cast-before-RoPE boundary."]),
    _fallback(
        "attn_fallback_rope", "core_audit finding 1: fallback BF16 Q/K after RoPE and before q8",
        "attention/fallback-rope",
        "Remove four BF16 casts inside the ROPE branch only; retain BF16 norm, "
        "QOUT storage, score-scale placement, and cache-scaling rounds.",
        ["_qk_prep", "_f32_to_e4m3"],
        ["Retain one rounded Q value for both BF16 QOUT and FP8 Q8OUT, as already "
         "done. Explore packed pair conversion in this epilogue only after a "
         "measured cost; do not pre-scale the unit-RMS FA3 q8 query."],
        ["ROPE=False compiles out these sites. The downstream QOUT BF16 store "
         "and K cache-scale round remain; q8 is the direct unrounded consumer "
         "in this diagnostic, not a change to FP8-query policy."]),
    _fallback(
        "attn_fallback_cache_scale", "core_audit finding 1: fallback BF16 K/V scale before FP8",
        "attention/fallback-cache-write",
        "Remove only three BF16 round trips around ke*kinv, ko*kinv, v*vinv.",
        ["_qk_prep", "_f32_to_e4m3"],
        ["Keep scale, BF16 RN, and FP8 encoding fused in _qk_prep. Explore packed "
         "BF16 conversions while retaining the exact BF16-then-FP8 semantics; "
         "no independent cast/cache-write kernel is necessary."]),
    _fallback(
        "attn_fallback_output", "core_audit finding 2 and manifest: FA3/Triton pre-norm BF16 output",
        "attention/fallback-output-and-normgate",
        "Disable the same semantic boundary at all added fallback enforcement "
        "sites: FA3 reshape cast, _normgate input cast, _gemm_splitk_ng input "
        "cast, and _attn_combine's pre-norm cast. Leaving any FA3 consumer cast "
        "in place would mask the diagnostic. Original BF16 output stores remain.",
        ["kernel_fn (base FA3 branch)", "_normgate", "_gemm_splitk_ng", "_attn_combine", "dense_ng"],
        ["For non-BF16 FA3 output, enforce the BF16 RN boundary in the existing "
         "_normgate or fused o-projection consumer rather than a separate .to "
         "allocation/launch, provided every consumer keeps the cast before RMS "
         "statistics and output bits match. Already-BF16 output needs no copy.",
         "For _attn_combine(NORMGATE=False), the original BF16 output store already "
         "enforces the boundary; an extra round-trip should be compiler-redundant. "
         "Do not count a launch eliminated unless a trace actually contains it."],
        ["The only production _attn_combine call passes NORMGATE=False and writes "
         "a BF16 o_in; its new in-register cast is redundant there. NORMGATE=True "
         "is a dormant alternate, covered by this transform without enabling it.",
         "If FA3 already returns BF16, removing the host .to and defensive consumer "
         "casts does not remove an actual materialization or launch. Report dtype "
         "and trace, not an assumed cast cost. The unchanged Triton BF16 output "
         "store also means its active NORMGATE=False route remains rounded."]),
]


def _replace(text, old, new, expected=1):
    count = text.count(old)
    if count != expected:
        raise ValueError(f"precision attention source drift: expected {expected} "
                         f"occurrences of {old!r}, found {count}")
    return text.replace(old, new)


def _region(source, symbol, *, assignment=False, occurrence=None):
    """Find exact AST-owned source bytes; repeated wrappers require explicit index."""
    tree = ast.parse(source)
    if assignment:
        nodes = [node for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == symbol
                         for target in node.targets)]
    else:
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == symbol]
    expected = 3 if symbol == "kernel_fn" else 1
    if len(nodes) != expected:
        raise ValueError(f"precision attention source drift: {symbol} has "
                         f"{len(nodes)} definitions, expected {expected}")
    node = nodes[0 if occurrence is None else occurrence]
    lines = source.splitlines(keepends=True)
    # AST columns are UTF-8 byte offsets, not character indexes.
    start = sum(len(line) for line in lines[:node.lineno - 1])
    start += len(lines[node.lineno - 1].encode("utf-8")[:node.col_offset].decode("utf-8"))
    end = sum(len(line) for line in lines[:node.end_lineno - 1])
    end += len(lines[node.end_lineno - 1].encode("utf-8")[:node.end_col_offset].decode("utf-8"))
    return start, end, source[start:end]


def _patch_region(source, symbol, replacements, *, assignment=False, occurrence=None):
    start, end, text = _region(source, symbol, assignment=assignment, occurrence=occurrence)
    for old, new, expected in replacements:
        text = _replace(text, old, new, expected)
    return source[:start] + text + source[end:]


def transform(source: str, group_id: str) -> str:
    """Return one repair-disabled diagnostic; unknown groups/drift fail loudly.

    Each call starts with the full repaired source (or a deliberately composed
    source whose selected repair is still intact). No code from that source is
    imported/evaluated. The parent must isolate the compiled extension identities.
    """
    if group_id not in {group["id"] for group in GROUPS}:
        raise ValueError(f"unknown attention precision group: {group_id}")
    if group_id.startswith("attn_native_"):
        # Guard the actual compiler source and dormant-alternate assumption.
        _, _, builder = _region(source, "_att7")
        for anchor in ("cuda_sources=ATT7_CUDA_SRC", '"-DOKC_QDIRECT=0"'):
            if builder.count(anchor) != 1:
                raise ValueError(f"precision attention builder drift: {anchor}")
        replacements = {
            "attn_native_qk_norm": [
                ("round_bf16(x[e] * r)", "(x[e] * r)", 3)],
            "attn_native_rope": [
                ("round_bf16(a0 * c - a1 * sn)", "(a0 * c - a1 * sn)", 3),
                ("round_bf16(a0 * sn + a1 * c)", "(a0 * sn + a1 * c)", 3)],
            "attn_native_cache_scale": [
                ("to_e4m3(round_bf16(x[e] * sc))", "to_e4m3(x[e] * sc)", 1),
                ("to_e4m3(round_bf16(x[4 + e] * sc))", "to_e4m3(x[4 + e] * sc)", 1)],
            "attn_native_output": [
                ("float pr8(float x) {\n  return round_bf16(x);\n}",
                 "float pr8(float x) {\n  return x;\n}", 1)],
        }[group_id]
        return _patch_region(source, "ATT7_CUDA_SRC", replacements, assignment=True)

    if group_id == "attn_fallback_output":
        source = _patch_region(source, "kernel_fn", [
            ("o_in = out.reshape(T, 1024).to(torch.bfloat16)",
             "o_in = out.reshape(T, 1024)", 1)], occurrence=0)
        source = _patch_region(source, "_normgate", [
            ("tl.load(X + t * 1024 + h * 64 + d).to(tl.bfloat16).to(tl.float32)",
             "tl.load(X + t * 1024 + h * 64 + d).to(tl.float32)", 1)])
        source = _patch_region(source, "_gemm_splitk_ng", [
            ("tl.load(x_ptr, mask=tmask, other=0.0).to(tl.bfloat16).to(tl.float32)",
             "tl.load(x_ptr, mask=tmask, other=0.0).to(tl.float32)", 1)])
        return _patch_region(source, "_attn_combine", [
            ("    o = o.to(tl.bfloat16).to(tl.float32)\n", "", 1)])

    if group_id == "attn_fallback_qkv_projection":
        # Include indentation/newline so the nested RoPE casts cannot match.
        old = "    # Materialize the BF16 projection boundary only after every split is reduced.\n"
        old += "".join(f"    {name} = {name}.to(tl.bfloat16).to(tl.float32)\n"
                       for name in ("qe", "qo", "ke", "ko", "v", "g"))
        replacements = [(old, "", 1)]
    elif group_id == "attn_fallback_qk_norm":
        replacements = [(f"{name} = ({name} * {scale}).to(tl.bfloat16).to(tl.float32)",
                         f"{name} = {name} * {scale}", 1)
                        for name, scale in (("qe", "rq"), ("qo", "rq"),
                                            ("ke", "rk"), ("ko", "rk"))]
    elif group_id == "attn_fallback_rope":
        old = "".join(f"        {name} = {name}.to(tl.bfloat16).to(tl.float32)\n"
                      for name in ("qe", "qo", "ke", "ko"))
        replacements = [(old, "", 1)]
    else:
        replacements = [(f"_f32_to_e4m3(({name} * {scale}).to(tl.bfloat16).to(tl.float32))",
                         f"_f32_to_e4m3({name} * {scale})", 1)
                        for name, scale in (("ke", "kinv"), ("ko", "kinv"), ("v", "vinv"))]
    return _patch_region(source, "_qk_prep", replacements)
