"""Expert-only precision-cost adapters; every transformed source is DIAGNOSTIC ONLY.

Authority: work/prior-art/alygg77-precision-audit.json, core findings 3--5.
Inspected repaired source: work/candidates/decode_alygg77_bf16.py; contrast:
work/candidates/decode_alygg77.py. No CUDA/torch imports or runtime measurements.

transform edits the two Python CUDA-string assignments through their AST spans,
not physical line numbers (moe_helper0 is one escaped physical line). Only the
selected semantic rounding boundary is removed. In particular, disabling GEMM2
rounding does NOT restore raw WGS's folded global-scale/gain association: that
would conflate two arithmetic changes. B4FIN continues to derive its statistics
from exactly the products written to ybuf. All dispatch, tiles, weight packing,
SwiGLU BF16 stores, FP32 products/reductions and upstream notices are retained.

The metadata below is also the measurement/recovery handoff. Parent owns GPU
execution, policy pinning, extension isolation, and any compliant candidate.
"""

import ast


NATIVE_LOADS = ["D1s", "D1f", "D16s", "D16f", "D32s", "D32f",
                "V1s", "V1f", "V16s", "V16f", "V32s", "V32f"]

RUNTIME = {
    "entrypoints": [
        "kernel_fn -> _after_attention / _sl_tile_after_attention",
        "_moe(T, state).okc_run(ma_w, ma_scratch, xm, lids, wts, m, 0, 32, nsm)",
        "_marlin_experts(xm, wts, lids, state) -> _pexp_norm_sum (explicit fixture only)",
    ],
    "source_dispatch": {
        "D1 (T=1)": "tiny: MOE_TINY_DEFINES + @moe_helper0; WSBLK=2",
        "V1 (T=8)": "tiny2: MOE_TINY2_DEFINES + @moe_helper0; WSBLK=1",
        "D16 (T=16)": "mid: MOE_MID_DEFINES + @moe_helper0; split jobs, WSBLK=2",
        "D32 (T=32), PRMT-eligible": "b4: MOE_CUDA_SRC, MOE_DEFINES + OKC_B4FIN=1",
        "V16/V32 (T=128/256), PRMT-eligible": "main: MOE_CUDA_SRC, MOE_DEFINES/WGS",
        "T>16, PRMT-ineligible": "main: MOE_CUDA_SRC, MAIN_FALLBACK_DEFINES (mma.sync skinny/fat)",
        "layer_type": "Expert source selection depends on T/scales/backend, not sliding/full.",
    },
    "module_handles_to_isolate": [
        "_MA_EXTS (keyed by defines, not source)",
        "_MOE", "_MOE_TINY", "_MOE_TINY2", "_MOE_MID", "_MOE_B4",
        "_moe_global_cache (alias of original dispatcher)",
        "_MA_SFX", "_MA_PDL", "_MA_ALT_SRCS", "_pexp_norm_sum Triton JIT object",
        "g_okc_pdl in each native extension DSO; set via okc_set_pdl",
    ],
    "state_handles_to_isolate": [
        "ma_ext[tiny|tiny2|mid|b4|main] (_RouterExt wrappers)",
        "ma_scratch[0:3]", "bank.workspace (only if explicit Marlin fixture)",
    ],
    "fixed_policy_fields": [
        "moe_pick", "ma_ok", "ma_sfx", "nsm", "marlin_cfg", "bank availability",
        "mm_pick", "pbmax", "compose_graft_enabled", "_pdl(T)",
    ],
    "measurement_inputs": {
        "xm": "Identical BF16 [T,4096] normalized expert activations in every arm.",
        "lids": "Identical contiguous int32 [T,8], local 0..31 or -1; aligned 16 B for WGS.",
        "wts": "Identical FP32 [T,8] route weights; hold route skew/job count fixed.",
        "m": "Private BF16 [T,4096] output in each arm.",
        "ma_w": "Keep all packed weights/scales identical; ma_w[7] is raw BF16 post-expert norm weight.",
        "g_postexp": "Raw BF16 checkpoint weight consumed only by Marlin's _pexp_norm_sum.",
    },
    "scratch_layout": {
        "ma_scratch[0]": "int32 counters/doneA; ss1 is float view at 8+1024; WGS xready at 8+512",
        "ma_scratch[1]": "FP32 ss then ybuf at EMAX; WGS fixups after ybuf",
        "B4FIN": "sspart aliases last 64 ybuf rows, offset EMAX+(EMAX-64)*4096 floats in scratch[1]",
        "ma_scratch[2]": "BF16 inter then WGS xA/xB; read actual compiled EMAX/WGS_XROWS",
        "caution": "Host _MA_EMAX is 384*8; compiled EMAX differs by defines. Do not hard-code host EMAX into scratch probes.",
    },
    "protocol": [
        "Use separate imported modules and independent prepared states/caches for repaired and ablated arms. "
        "_RouterExt.__deepcopy__ returns self, so deepcopy(state) does NOT isolate native extensions.",
        "Copy baseline policy values, not extension handles or scratch. Do not retune each arm: "
        "_tune changes moe_pick, marlin_cfg, mm_pick and temporarily calibrates bias.",
        "Native extension name already hashes MOE_CPP_SRC + selected CUDA source + defines/NS; "
        "parent still owns unique module identities and cache bindings. Keep PDL settings equal.",
        "Native GEMM1, GEMM2 and finalize are phases of one persistent okc_run, not separately "
        "launchable kernels. Attribute per-repair cost by paired full expert-kernel timings on "
        "identical retained inputs; whole-layer repeats establish decode recovery. "
        "Do not call an expert-kernel delta a standalone GEMM duration.",
        "Cold-L2 paired ABBA/BAAB and an unchanged A/A control, both layer types, multiple layouts "
        "and repeated >=0.5% whole-decode gain are required before handing a legal candidate to MAIN.",
        "Repair-disabled outputs are diagnostic only and may fail reference tolerance. "
        "No ablated source is eligible for submission or a compliant speedup claim.",
    ],
}

FIXTURES = [
    "All six scored loads, sliding/full, identical retained xm/lids/wts across arms; "
    "multiple weight/input seeds and allocation pads.",
    "T=1,2,4,5,8,9,16 exercises every small-T boundary/helper and WSBLK=1/2 tail. "
    "T=17,32 exercises B4FIN; T=33,64,128,256 exercises WGS/full-row statistics. "
    "Include >64 routes on one expert to exercise split jobs without changing job tiles.",
    "All-remote/no-local routes, mixed -1 slots, one heavily loaded expert, multiple local experts, "
    "partial last token tiles, split-unit finishers and ordinary unsplit epilogues.",
    "Both PRMT-eligible scale bytes and finite scale bytes outside [0x08,0x7e], so "
    "_main_defines selects the real LUT fallback. Keep global scales FP32 and NVFP4 bits unchanged.",
    "Exact boundary oracle: fully reduce then globally scale in FP32, RN-even BF16 gate/up, "
    "existing SiLU/multiply then BF16 intermediate; RN-even BF16 scaled GEMM2, multiply by "
    "RN-even BF16(1+raw gain) in FP32; FP32 square sums, norm, route contributions, "
    "and only the final route sum store BF16. Never round the post-gain product.",
    "GEMM conversion microfixtures at BF16 half-ULPs (ties and adjacent FP32 values), both signs, "
    "zero/subnormals, overflow and finite extreme scales; classify infinities/NaNs separately. "
    "Check packed conversion lane order and exact finite BF16 bits against the scalar RN-even path.",
    "Gain precomputation fixture enumerates all 65536 BF16 bit patterns; finite cases must agree "
    "bit-for-bit with FP32(1+BF16(raw)) -> BF16 -> FP32, with explicit NaN/infinity policy. "
    "Include weights near -1 and BF16 addition ties, not just typical positive gains.",
    "B4FIN: compare each stored sspart against the existing fixed shuffle-tree sum of the "
    "very same v0/v1 stored in ybuf, including masked tokens. Do not replace that tree with "
    "a differently ordered full-row sum as a bit-exact oracle. WGSFINSS must read those products too.",
    "Repeated CUDA-graph replay with private scratch, sentinel checks, fresh prepared states, "
    "and alternating different weight states; prove no cached outputs or leaked PDL/counter state.",
    "Explicit Marlin diagnostic only: construct bank with _marlin_bank, hold marlin_cfg fixed, "
    "compare BF16 cache1/cache2/cache3 boundaries and _pexp_norm_sum against an identical "
    "Marlin control. Current prepare deliberately leaves bank=None; do not add Marlin packing "
    "to the main timing arms merely to measure an inactive repair.",
]

_COMMON_INACTIVE = [
    "The helper source's WGS code is not instantiated by current tiny/tiny2/mid defines; "
    "it is still transformed for complete source coverage.",
    "Base skinny/fat and both skinny_ring branches are retained for scale-ineligible fallback; "
    "their source presence does not imply that they run under WGS defines.",
    "Current repaired prepare sets bank=None even when sgl_kernel imports successfully. "
    "Marlin has no active scored-load attribution unless an explicit backend fixture installs a bank.",
]

GROUPS = [
    {
        "id": "experts_gemm1_bf16",
        "audit_requirement": "core_audit finding 3: BF16 gate and up after complete GEMM1 reduction/global scale, before SiLU",
        "stage": "experts: GEMM1/SwiGLU epilogues inside persistent okc_run",
        "description": "Remove only native gate/up BF16 round trips (10 scalar boundaries per source); "
                       "keep BF16 SwiGLU output, GEMM2 rounding and effective-gain rounding.",
        "source_symbols": ["MOE_CUDA_SRC::epi_A", "MOE_CUDA_SRC::skinny_ring (both branches)",
                           "MOE_CUDA_SRC::ws_helper", "MOE_CUDA_SRC::wgs_epi_tile<false>",
                           "_MA_ALT_SRCS['moe_helper0']::same five epilogue sites"],
        "active_loads": NATIVE_LOADS,
        "inactive_reasons": _COMMON_INACTIVE + [
            "Marlin already writes BF16 cache1 before silu_and_mul in both raw and repaired "
            "sources. There is no GEMM1 repair cost there; removing cache1's dtype changes the "
            "external backend contract and is not a single-toggle ablation."
        ],
        "compliant_recovery_ideas": [
            "[INFERENCE] Pair scaled gate/up (or two tokens) with __floats2bfloat162_rn and "
            "__bfloat1622float2 to express two exact scalar RN-even round trips with one packed "
            "conversion. Retain each original FP32 scale multiplication and SiLU expression. "
            "Inspect generated code: nvcc may already pack these; no measured saving is claimed.",
            "[INFERENCE] Keep converted values in registers and store only the existing BF16 "
            "SwiGLU output; do not introduce a GEMM1 cache or move rounding before split-K reduction."
        ],
        "runtime": RUNTIME,
        "required_fixtures": FIXTURES,
        "measurements": None,
    },
    {
        "id": "experts_gemm2_bf16",
        "audit_requirement": "core_audit finding 4: BF16 scaled GEMM2 output before effective gain, with consistent statistics including B4FIN",
        "stage": "experts: GEMM2/post-gain/statistics epilogues inside persistent okc_run",
        "description": "Remove only native GEMM2-output round trips (12 scalar boundaries per source). "
                       "Keep effective gain BF16 and products/statistics FP32. B4FIN reuses the altered "
                       "v0/v1, never raw d4 times a differently folded gain.",
        "source_symbols": ["MOE_CUDA_SRC::epi_B", "MOE_CUDA_SRC::skinny_ring (both branches)",
                           "MOE_CUDA_SRC::skinny_ring_ws last-unit epilogues", "MOE_CUDA_SRC::ws_helper",
                           "MOE_CUDA_SRC::wgs_epi_tile<true>/tile_sq", "MOE_CUDA_SRC::OKC_B4FIN final pass",
                           "MOE_CUDA_SRC::OKC_WGSFINSS final pass",
                           "_MA_ALT_SRCS['moe_helper0']::all corresponding GEMM2 epilogues"],
        "active_loads": NATIVE_LOADS,
        "inactive_reasons": _COMMON_INACTIVE + [
            "Marlin cache3 is BF16 in both raw and repaired sources. Removing its store boundary "
            "requires an external GEMM API/output-storage change, not an attribution toggle.",
            "B4FIN statistics are coupled to repaired GEMM2 and gain products, not a fourth "
            "independently optional semantic repair. There is no B4-only inconsistent-statistics toggle."
        ],
        "compliant_recovery_ideas": [
            "[INFERENCE] Pack the two already-scaled GEMM2 values into an RN-even bfloat162 "
            "conversion and immediately unpack to FP32 before multiplying separate BF16 effective "
            "gains. Prove scalar/packed identity and retain the current FP32 reduction tree.",
            "Repaired B4FIN already computes squares from v0/v1 used for ybuf stores. Retain "
            "that reuse; restoring raw's duplicated expressions is not a new optimization.",
            "Do not fold gsc into the gain, round an accumulator partial, store ybuf in BF16, "
            "or round the normalized route contribution: none implements the required boundary."
        ],
        "runtime": RUNTIME,
        "required_fixtures": FIXTURES,
        "measurements": None,
    },
    {
        "id": "experts_effective_gain_bf16",
        "audit_requirement": "core_audit findings 4--5: effective post-expert gain BF16(1+raw checkpoint weight), native and Marlin",
        "stage": "experts: native gain preloads/epilogues and Marlin _pexp_norm_sum",
        "description": "Remove only the effective-gain round trip (16 native scalar sites per source "
                       "plus one Triton expression); retain BF16 GEMM1/GEMM2 and FP32 post-gain "
                       "products, statistics, normalized route contributions and route sum.",
        "source_symbols": ["MOE_CUDA_SRC::epi_B", "MOE_CUDA_SRC::skinny_ring nwc/nwn preloads (both branches)",
                           "MOE_CUDA_SRC::skinny_ring_ws lnw0/lnw1 tail", "MOE_CUDA_SRC::ws_helper",
                           "MOE_CUDA_SRC::wgs_epi_tile<true>",
                           "_MA_ALT_SRCS['moe_helper0']::all corresponding gain sites", "_pexp_norm_sum"],
        "active_loads": NATIVE_LOADS,
        "inactive_reasons": _COMMON_INACTIVE,
        "compliant_recovery_ideas": [
            "[INFERENCE] Precompute weight-only effective = (raw.float()+1.0).to(torch.bfloat16) "
            "during prepare, reuse that BF16 tensor as ma_w[7] and g_postexp, and change every "
            "native/helper preload/tail plus _pexp_norm_sum to decode it without adding one or "
            "round-tripping again. This removes repeated FP32 adds and conversions without "
            "moving activation-dependent work outside the timed call.",
            "Use one existing-size 4096-element BF16 effective-gain allocation shared by the "
            "state's native/Marlin references, not an extra FP32 mirror or larger scratch. "
            "Never mutate the input checkpoint or retain a global effective gain shared by weights.",
            "The skinny_ring_ws tail already preloads raw BF16 bits into lnw0/lnw1. With an "
            "effective-gain buffer those bits decode directly; do not accidentally leave its +1 "
            "or apply +1 twice. Keep gsc separate from the gain and leave B4FIN product reuse intact."
        ],
        "runtime": RUNTIME,
        "required_fixtures": FIXTURES,
        "measurements": None,
    },
]


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _round(expression):
    return "__bfloat162float(__float2bfloat16(" + expression + "))"


def _replace(text, old, new, count, label):
    actual = text.count(old)
    _require(actual == count, f"{label}: expected {count} anchors, found {actual}: {old}")
    return text.replace(old, new)


def _native_transform(text, group_id, label):
    # A complete inventory prevents silent success if a new unclassified boundary
    # is added. Each source has 10 GEMM1 + 12 GEMM2 + 16 gain round trips.
    prefix = "__bfloat162float(__float2bfloat16("
    _require(text.count(prefix) == 38, f"{label}: native rounding inventory drift")
    for anchor, count in (("void epi_A(", 1), ("void epi_B(", 1),
                          ("void skinny_ring(", 2), ("void skinny_ring_ws(", 1),
                          ("void ws_helper(", 1), ("void wgs_epi_tile(", 1)):
        _require(text.count(anchor) == count, f"{label}: helper inventory drift: {anchor}")
    if label == "MOE_CUDA_SRC":
        for anchor in ("float tile_sq[2] = {0.f, 0.f};",
                       "tile_sq[h] = v0 * v0 + v1 * v1;",
                       "float q0 = tile_sq[0], q1 = tile_sq[1];"):
            _require(text.count(anchor) == 1, f"{label}: B4FIN statistics drift: {anchor}")
    else:
        _require("tile_sq" not in text, f"{label}: unexpected B4FIN helper variant")

    if group_id == "experts_gemm1_bf16":
        pairs = (("r[h] * gsc", "r[2 + h] * gsc", 1),
                 ("s0 * gsc", "s1 * gsc", 2),
                 ("(h ? sm.y : sm.x) * gsc", "(h ? sm.w : sm.z) * gsc", 1),
                 ("d4[h] * gsc", "d4[2 + h] * gsc", 1))
        for gate, up, count in pairs:
            old = f"gt = {_round(gate)}, up = {_round(up)};"
            new = f"gt = ({gate}), up = ({up});"
            text = _replace(text, old, new, count, label)
        removed = 10
    elif group_id == "experts_gemm2_bf16":
        products = (("r[h] * gsc", "w0", 1), ("r[2 + h] * gsc", "w1", 1),
                    ("s0 * gsc", "nwc0", 2), ("s1 * gsc", "nwc1", 2),
                    ("(h ? sm.y : sm.x) * gsc", "nw0", 2),
                    ("(h ? sm.w : sm.z) * gsc", "nw1", 2),
                    ("d4[h] * gsc", "w0", 1), ("d4[2 + h] * gsc", "w1", 1))
        for value, gain, count in products:
            text = _replace(text, f"{_round(value)} * {gain}",
                            f"({value}) * {gain}", count, label)
        removed = 12
    else:
        _require(group_id == "experts_effective_gain_bf16", "unknown native group")
        gains = (("1.f + __bfloat162float(p.normw[n0])", 2),
                 ("1.f + __bfloat162float(p.normw[n0 + 8])", 2),
                 ("1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg))", 4),
                 ("1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8))", 4),
                 ("1.f + __uint_as_float(lnw0 << 16)", 1),
                 ("1.f + __uint_as_float(lnw1 << 16)", 1),
                 ("1.f + __bfloat162float(__ldg(p.normw + n0))", 1),
                 ("1.f + __bfloat162float(__ldg(p.normw + n0 + 8))", 1))
        for value, count in gains:
            text = _replace(text, _round(value), f"({value})", count, label)
        removed = 16
    _require(text.count(prefix) == 38 - removed, f"{label}: unexpected round-trip removal count")
    return text


def _target_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and node.value.id == "_MA_ALT_SRCS"):
        _require(isinstance(node.slice, ast.Constant) and node.slice.value == "moe_helper0",
                 "unexpected alternate expert source; inventory must be extended")
        return "_MA_ALT_SRCS['moe_helper0']"
    return None


def transform(source: str, group_id: str) -> str:
    """Return a single-repair-disabled diagnostic, never import/execute the source.

    Input must be the repaired source, not a previously ablated expert source.
    Unknown groups raise ValueError; anchor/source drift raises AssertionError.
    UTF-8 byte spans honor Python AST column semantics, preserving all content
    outside the two literal values and (gain group only) the Triton gain RHS.
    """
    if group_id not in {group["id"] for group in GROUPS}:
        raise ValueError(f"unknown expert precision group: {group_id}")
    tree = ast.parse(source)
    targets = {}
    finalizers = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                name = _target_name(target)
                if name in ("MOE_CUDA_SRC", "_MA_ALT_SRCS['moe_helper0']"):
                    _require(name not in targets, f"duplicate source assignment: {name}")
                    _require(isinstance(node.value, ast.Constant) and isinstance(node.value.value, str),
                             f"{name}: expected a literal CUDA source")
                    targets[name] = node.value
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_pexp_norm_sum":
            finalizers.append(node)
    _require(set(targets) == {"MOE_CUDA_SRC", "_MA_ALT_SRCS['moe_helper0']"},
             "missing expert source assignment")
    _require(len(finalizers) == 1, "expected exactly one Marlin finalizer")

    edits = []
    for name, node in targets.items():
        changed = _native_transform(node.value, group_id, name)
        # repr serializes exactly the decoded CUDA string, including comments and
        # notices. Only Python literal spelling changes, never CUDA escaping.
        edits.append((node, repr(changed)))

    if group_id == "experts_effective_gain_bf16":
        gain_nodes = [node.value for node in finalizers[0].body
                      if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "w" for t in node.targets)]
        _require(len(gain_nodes) == 1, "expected one Marlin effective-gain assignment")
        node = gain_nodes[0]
        old = "(1.0 + tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)"
        new = "(1.0 + tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32))"
        _require(ast.get_source_segment(source, node) == old, "Marlin effective-gain RHS drift")
        edits.append((node, new))

    encoded = source.encode("utf-8")
    offsets = [0]
    for line in encoded.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    spans = [(offsets[node.lineno - 1] + node.col_offset,
              offsets[node.end_lineno - 1] + node.end_col_offset,
              replacement.encode("utf-8")) for node, replacement in edits]
    spans.sort(reverse=True)
    previous_start = len(encoded)
    for start, end, replacement in spans:
        _require(0 <= start < end <= previous_start, "overlapping/invalid AST replacement spans")
        encoded = encoded[:start] + replacement + encoded[end:]
        previous_start = start
    return encoded.decode("utf-8")
