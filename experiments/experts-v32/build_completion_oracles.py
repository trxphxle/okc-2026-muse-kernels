"""Timing-only completion oracles; generated candidates are NEVER submissions.

Python AST identifies the single literal to edit. Exact baseline SHA and unique
CUDA anchors make this a deliberately closed transformation, not a fuzzy patch.
No import of the candidate, GPU work, compilation, or execution occurs here.
"""

import ast
import hashlib
import re


BASELINE_SHA256 = "053dbacc8f0de4e6111ccb89cb91f629840852cf482a6ca423fe0eaff419c086"
WARNING = "# DIAGNOSTIC ONLY: timing oracle; intentionally invalid outputs; NEVER SUBMIT"


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _at(text, anchor):
    _require(text.count(anchor) == 1, "Expected exactly one anchor: " + repr(anchor))
    return text.index(anchor)


def _between(text, start, end):
    a, b = _at(text, start), _at(text, end)
    _require(a < b, "Reversed section anchors")
    return a, b


def _literal(source):
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "MOE_CUDA_SRC" for t in n.targets)]
    _require(len(nodes) == 1, "Expected one top-level MOE_CUDA_SRC assignment")
    value = nodes[0].value
    _require(isinstance(value, ast.Constant) and isinstance(value.value, str),
             "MOE_CUDA_SRC must be one string literal")
    lines = source.splitlines(keepends=True)
    # AST columns are UTF-8 byte offsets, not Unicode character offsets.
    def offset(line, column):
        return sum(len(s) for s in lines[:line - 1]) + len(lines[line - 1].encode("utf-8")[:column].decode("utf-8"))
    a = offset(value.lineno, value.col_offset)
    b = offset(value.end_lineno, value.end_col_offset)
    token = source[a:b]
    _require(token.startswith('r"""') and token.endswith('"""'),
             "Expected original raw triple-double-quoted literal style")
    raw = token[4:-3]
    _require(raw == value.value, "Literal payload differs from AST value")
    return a + 4, b - 3, raw


def build(source: str, mode: str):
    """Return (diagnostic_source, manifest) for exact repaired baseline and A/B.

    A selects one segment per GPU-routed unit (A: s1 == 16; B: s0 == 0),
    writes that segment's unsummed accumulator through the ordinary epilogue,
    and retains ordinary deferred readiness. B removes that epilogue/readiness
    too, but not producers, gather, tensor work, slot lifetime or final combine.
    Source invariants do NOT prove native register or instruction invariants;
    the separate native resource verifier is mandatory before interpretation.
    """
    _require(mode in ("A", "B"), "mode must be A or B")
    _require(_sha(source) == BASELINE_SHA256, "Unsupported baseline SHA256")
    lo, hi, cuda = _literal(source)
    patches = []

    def replace_span(name, a, b, replacement):
        _require(0 <= a < b <= len(cuda), "Invalid edit span: " + name)
        _require(cuda[a:b] != replacement, "No-op transformation: " + name)
        patches.append((a, b, replacement, name))

    def replace(name, old, new):
        a = _at(cuda, old)
        replace_span(name, a, a + len(old), new)

    epi_start = "template <bool PB>\n__device__ __forceinline__ void wgs_epi_tile"
    flush_start = "template <bool PB>\n__device__ __forceinline__ void wgs_flush"
    seg_comment = "// one segment (stages [s0, s1) of unit u)"
    seg_start = "template <bool PB, int NTT>\n__device__ __forceinline__ void wgs_seg"
    phase_start = "template <bool PB>\n__device__ __forceinline__ void wgs_phase"
    split_publish = "  if (slot >= 0) {  // a share of a unit split between CTAs: add it now, count it later (deferred)\n"
    ordinary_epi = "  const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;\n"
    deferred = ('        // the previous segment\'s deferred work, behind group OKC_WGSFCI of the first stage (scout: CI 3 lets this WG\'s\n'
                '        // own red.adds / epilogue stores drain a little longer before the count / release atomics: +0.5-1%)\n'
                '        if (FIRST && ci == OKC_WGSFCI && pd.kind) wgs_flush<PB>(p, S, pd);\n')
    done_wait = ('      // phase B stage st reads inter columns 256 st .. +255: the 2 phase-A units (x 4 warpgroups) that wrote them\n'
                 '      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.doneAs + j * 16 + st) < 8) __nanosleep(32); }\n')
    done_reset = '    for (int i = i0; i < JMAX * 16; i += G * NT) p.doneAs[i] = 0;\n'
    a, b = _between(cuda, split_publish, ordinary_epi)
    replace_span("split_publish_and_contributor_protocol", a, b,
                 "  if (!(PB ? s0 == 0 : s1 == 16)) return;  // diagnostic selected owner, never sum partials\n" if mode == "A" else "")
    replace("split_fixup_mapping",
            "    // a unit split between CTAs: its fixup slot is the CTA holding its first stage\n"
            "    const int slot = (s1 - s0 == 16) ? -1 : (s0 == 0 ? (int)blockIdx.x : wgs_cta_of(S, PB, u << 4, gridDim.x));\n",
            "    const int slot = (s1 - s0 == 16) ? -1 : 0;  // diagnostic split marker, no fixup mapping\n")
    replace("pending_protocol_comment",
            "// Work left from the previous segment, done after the next segment's first wgmma group (so the stores and atomics it\n"
            "// waits for have long completed): kind 1 = release doneA for job j (phase A), kind 2 = add the stage count of a split\n"
            "// unit's share and, if that completes the unit, read the summed partials back and run its epilogue.\n",
            "// Diagnostic pending state: only ordinary kind-1 readiness is used by A; B leaves this state idle.\n")
    if mode == "A":
        a, b = _between(cuda, "  int* fc = wgs_cnt(p, PB) + pd.slot * 4 + wg;\n", seg_comment)
        replace_span("split_flush_fixup_readback", a, b, "}\n")
    else:
        a, b = _between(cuda, epi_start, seg_comment)
        replace_span("all_unit_epilogue_and_flush_definitions", a, b, "")
        a, b = _between(cuda, ordinary_epi, phase_start)
        _require(cuda[b - 2:b] == "}\n", "Missing segment closing brace")
        replace_span("ordinary_unit_epilogue_and_readiness", a, b, "}\n")
        replace("deferred_result_flush", deferred, "")
        replace("phase_tail_result_flush", "  wgs_flush<PB>(p, S, pd);\n", "")
        replace("phase_b_doneAs_acquire", done_wait, "")
        replace("doneAs_reset", done_reset, "")

    patches.sort()
    for left, right in zip(patches, patches[1:]):
        _require(left[1] <= right[0], "Overlapping oracle edits")
    changed = cuda
    for a, b, replacement, _ in reversed(patches):
        changed = changed[:a] + replacement + changed[b:]

    # Hash broad immutable sections, including all layouts and routing in the
    # prefix. Narrow masked sections explicitly name their only permitted edit.
    sections = {}
    def invariant(name, start=None, end=None, masks=()):
        a = 0 if start is None else _at(cuda, start)
        b = len(cuda) if end is None else _at(cuda, end)
        c = 0 if start is None else _at(changed, start)
        d = len(changed) if end is None else _at(changed, end)
        before, after = cuda[a:b], changed[c:d]
        for old in masks:
            _require(before.count(old) == 1 and old not in after, "Invalid invariant mask: " + name)
            before = before.replace(old, "", 1)
        _require(before == after, "Preservation invariant failed: " + name)
        sections[name] = {"baseline_sha256": _sha(before), "oracle_sha256": _sha(after),
                          "bytes": len(before.encode("utf-8")),
                          "excluded_removed_blocks": list(masks)}

    wp = "__device__ void wgs_wproducer"
    xp = "__device__ void wgs_xproducer"
    gather = "__device__ void wgs_gather"
    invariant("routing_layouts_scratch_cost_tables_order", end=wp)
    invariant("weight_producer", wp, xp)
    invariant("x_producer", xp, gather, (done_wait,) if mode == "B" else ())
    # Pending commentary is intentionally changed, not part of gather body.
    gather_end = "}\n// Diagnostic pending state:"
    original_gather_end = "}\n// Work left from the previous segment,"
    ga, gb = _between(cuda, gather, original_gather_end)
    ca, cb = _between(changed, gather, gather_end)
    _require(cuda[ga:gb + 2] == changed[ca:cb + 2], "Gather changed")
    sections["gather"] = {"baseline_sha256": _sha(cuda[ga:gb + 2]),
                          "oracle_sha256": _sha(changed[ca:cb + 2])}
    drain = "  wg_wait<0>();\n  if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);\n"
    invariant("K_dequant_WGMMA_weight_and_x_slot_pipeline", seg_start, drain,
              (deferred,) if mode == "B" else ())
    _require(changed.count(drain) == cuda.count(drain) == 1, "Mandatory drain/slot release changed")
    sections["mandatory_drain_and_x_release"] = {"baseline_sha256": _sha(drain), "oracle_sha256": _sha(drain)}
    if mode == "A":
        invariant("ordinary_epilogue_body", epi_start, flush_start)
    ranges = "// cost-weighted stage ranges: per-stage cost by the job's n-tiles"
    final = "  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]"
    invariant("ranges_kernel_route_gather_calls_phase_order_final_grid_barrier", ranges, final,
              (done_reset,) if mode == "B" else ())
    invariant("final_output_combine_and_cuda_host_launch", final)
    _require("red.global.add.v4.f32" not in changed[_at(changed, seg_start):_at(changed, phase_start)],
             "Split partial publication survived")
    if mode == "B":
        _require(not re.search(r"\bwgs_(?:epi_tile|flush)\s*(?:<|\()", changed),
                 "Unit completion survived B")
    else:
        _require(changed.count("wgs_epi_tile<PB>(") == 1, "A must have one ordinary epilogue callsite")
    marker = re.search(r"(?m)^#\s*okc:[^\n]*", source)
    _require(marker is not None and marker.end() < lo, "Missing first submission marker before literal")
    result = source[:lo] + changed + source[hi:]
    result = result[:marker.start()] + WARNING + result[marker.end():]
    _, _, parsed_result = _literal(result)
    _require(parsed_result == changed, "Generated literal did not round-trip through AST")
    untouched = source[:marker.start()] + source[marker.end():lo] + source[hi:]
    manifest = {
        "mode": mode, "diagnostic_only": True, "never_submit": True,
        "numerically_valid": False, "baseline_sha256": _sha(source),
        "oracle_sha256": _sha(result), "literal_style": 'r"""',
        "unchanged_outside_literal_and_first_marker_sha256": _sha(untouched),
        "edits": [{"name": name, "cuda_start": a, "cuda_end": b,
                   "removed_sha256": _sha(cuda[a:b]), "replacement": replacement}
                  for a, b, replacement, name in patches],
        "unchanged_sections": sections,
        "owner_semantics": "Current GPU-derived segment bounds only: phase A s1==16; phase B s0==0. Full units own themselves. A publishes only this owner's unsummed partial through the original ordinary epilogue and deferred kind-1 readiness, once per unit; all contributors retain their real K work.",
        "invalid_output_reason": "A omits other contributors' partials; B also omits all unit epilogues and readiness, so phase B reads freshly zeroed xB and final combine reads freshly zeroed expert-output scratch. Neither is a numerical implementation.",
        "B_preserved_pipeline_work": ["GPU routing/ranges/order", "phase-A gather and xready acquire/reset", "weight and x bulk transfers", "K loop/dequant/WGMMA", "weight-slot release", "mandatory WGMMA drain and x-slot release", "final grid barrier and final combine"],
        "layout_policy": "Params, Smem, WgsPend, scratch declarations/accessors and geometry retained byte-for-byte; unused declarations remain to preserve layout, not as fallback execution.",
        "harness_requirements": ["Prepare compliant baseline, never oracle.prepare; bind private oracle ma_ext/ma_scratch afterward", "Fresh zeroed scratch; never reuse prepared expert outputs", "No global A-to-B barrier", "Native verifier must match registers/shared/block/grid and tensor instruction counts; fail on mismatch, never add dummy computation", "Resource probe is expert-stage verification, not decode timing"],
        "native_resource_equivalence": "UNVERIFIED: source preservation alone cannot establish compiled footprint or retained tensor instructions",
    }
    return result, manifest
