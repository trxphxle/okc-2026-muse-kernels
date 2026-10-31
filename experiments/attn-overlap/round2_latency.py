"""Diagnostic-only D1f/V1f critical-path stamps; NEVER a submission or optimization.

Upload with decode_current.py (or --baseline decode_control.py for raw Round-1
calibration), compare_decode_public.py and attention reference.py/checks.py.
Run: python round2_latency.py --output round2_latency.json --seed 731283 --pairs 3

Uses the selected module's unchanged _launch7(..., pdl=False), no layer prepare.
Two sparse source-derived builds separate prologue and tail landmarks. Existing
first-tile-only DBG(12) is allowed; DBG(13), tile counters and PROF stay disabled.
Tail additions execute once per CTA (finalizer landmarks only in its first row
pass). All numerical arithmetic, splits, barriers and memory layout are retained.

Stamps are thread-0 issue landmarks, NOT instruction-retirement measurements.
In particular sum_copies interleaves reads with workspace resets, and compiler
scheduling can cross a timer read. No new fence is added to fake completion.
Parallel CTA durations cannot be summed into savings. No phase-off arm is used:
these measurements establish exposed intervals, not realizable deletion gains.
Derived CUDA artifacts retain the selected source's complete attribution header.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import traceback

FLUSH_BYTES = 256 * 1024 * 1024
STRIDE = 32
ARMS = ("native", "native_repeat", "prologue", "tail")
MASKS = {
    "prologue": (0, 1, 8, 9, 10, 11, 2, 12, 3, 5, 22),
    "tail": (0, 3, 7, 6, 16, 17, 18, 4, 19, 20, 21, 5, 22),
}
STAMP_MAP = {
    0: "Existing attn_kernel entry, after shared-memory base/address setup; before seq_lens load and range construction.",
    1: "Existing segment page-index cp_wait and CTA sync completed; speculative seq length checked; before initial K/V issue.",
    8: "Existing initial STAGES-1 K/V issue/commit and future-index issue completed; K/V need not have arrived.",
    9: "Existing q/constants cp_wait and CTA sync, optional RoPE setup and next CTA sync completed (full has no RoPE).",
    10: "Existing query normalization/conversion completed; before new-token K/V preparation.",
    11: "Existing new-token K/V normalization, cache writes/shared staging and CTA sync completed; before patching in-flight new-token tiles.",
    2: "Existing prologue/new-token patch ends; before consumer accumulator initialization and first tile wait.",
    12: "Existing first tile ONLY: cp.async wait plus CTA sync completed, before first tile math. This is observed consumer readiness, not exact HBM arrival.",
    3: "Existing main attention loop and any stream combination completed; before split merge rescaling/atomics.",
    7: "Existing OKC_MRGDBG: this thread's merge atomics issued; no global completion fence implied.",
    6: "Existing OKC_MRGDBG: optional per-thread fence and CTA sync passed. With MERGE3=1 the per-thread fence is absent; release is the later acq_rel atomic. Value 1 is the legacy noncoop last-finisher flag, NOT a timestamp.",
    16: "Added after cooperative finalizer slice arithmetic / PINDIV, immediately before barrier branch.",
    17: "Added parity branch only: thread 0's acq_rel arrival atomic returned; before last-arrival test. Other barrier schemes do not emit it.",
    18: "Added parity LAST ARRIVER only: other counter reset and epoch update issued, before leaving last-arrival branch; NOT a release-store completion fence.",
    4: "Existing cooperative barrier wait and following CTA sync completed; before finalizer row loop. Noncoop instead denotes last-arrival test and CTA sync.",
    19: "Added first finalizer row pass only: gate load and sum_copies returned. Reads, accumulation and workspace resets are fused; not separately attributable.",
    20: "Added first finalizer row pass only: denominator normalization, precision rounding, square reduction and rsqrt expression evaluated at source level; before sigmoid/gate/output packing.",
    21: "Added first finalizer row pass only: gated BF16 output store issued; no global-store completion fence.",
    5: "Existing finalizer end, including legacy departure/reset branch if enabled. Coop CTAs with no output rows also emit this.",
    22: "Added after segment's existing final CTA sync, before return true; approximate CTA completion landmark, not kernel retirement.",
}
PHASES = {
    "entry_to_page_indices": (0, 1),
    "initial_kv_issue": (1, 8),
    "q_constants_wait_and_rope": (8, 9),
    "query_preparation": (9, 10),
    "new_kv_preparation_and_sync": (10, 11),
    "new_kv_patch": (11, 2),
    "consumer_init_and_first_kv_wait": (2, 12),
    "main_attention_after_first_ready": (12, 3),
    "all_prologue_to_first_ready": (0, 12),
    "merge_rescale_and_issue": (3, 7),
    "merge_cta_sync_optional_fence": (7, 6),
    "finalizer_slice_arithmetic": (6, 16),
    "arrival_atomic": (16, 17),
    "arrival_to_wait_return_and_cta_sync": (17, 4),
    "barrier_total_including_arrival": (16, 4),
    "final_readback_gate_load_and_workspace_reset": (4, 19),
    "denominator_norm_and_rms": (19, 20),
    "sigmoid_gate_pack_output_issue": (20, 21),
    "post_output_finalizer_tail": (21, 5),
    "final_cta_sync_and_exit_landmark": (5, 22),
}
LIMITS = [
    "No measured phase is proven removable: guaranteed realizable saving is 0 us; stage deletion was not performed.",
    "Entry/page-index/KV intervals overlap query preparation and include address/control work, not isolated DRAM latency.",
    "Per-CTA wait times overlap other CTAs' useful work; neither their sum nor median is an additive kernel cost.",
    "Tail after last observed loop end is an optimistic whole-tail deletion envelope, not a legal implementation or an isolated merge cost.",
    "End-spread max-minus-median is only a hypothetical balancing envelope; balancing is a closed negative experiment, not a proposal.",
    "Source-level readback/norm/gate subdivisions may cross compiler scheduling and outstanding memory operations; no added fences serialize them.",
    "Prologue and tail builds are separate executions. Never splice their individual rows into a claimed exact single-launch critical path.",
    "CUPTI durations and globaltimer timestamps have different origins. Compare spans, not absolute timestamps.",
    "If occupancy, launch mode or local-memory use changes, or paired timing shifts >3%, report biased stamps, not an accurate native critical path.",
]


def save(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stats(values):
    values = sorted(float(x) for x in values)
    if not values:
        return None
    def q(p):
        x = (len(values) - 1) * p
        lo, hi = math.floor(x), math.ceil(x)
        return values[lo] + (values[hi] - values[lo]) * (x - lo)
    return dict(n=len(values), min=min(values), median=statistics.median(values),
                p10=q(.1), p90=q(.9), max=max(values),
                max_minus_median=max(values) - statistics.median(values))


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Source anchor must occur exactly once: {old!r}; found {source.count(old)}")
    return source.replace(old, new, 1)


def diagnostic_source(original, arm):
    source = original
    for name, value in (("OKC_DBGON", 1), ("OKC_MRGDBG", 1)):
        source, n = re.subn(rf"(?m)^(#define {name} )0\b", rf"\g<1>{value}", source)
        if n != 1:
            raise RuntimeError(f"Unexpected {name} definition")
    source = replace_once(source, "#define DBGSTRIDE (OKC_SEGDBG ? 64 : 16)", f"#define DBGSTRIDE {STRIDE}")
    mask = sum(1 << slot for slot in MASKS[arm])
    source = replace_once(source, "if (OKC_DBGON && p.dbg && threadIdx.x == 0)",
                          f"if (OKC_DBGON && (({mask}u >> (k)) & 1u) && p.dbg && threadIdx.x == 0)")
    # Only the cooperative per-request branch receives new fine-grained landmarks.
    begin = source.index("  } else if (!FLAT && M_COOP) {")
    end = source.index("    DBG(5);\n  } else {", begin)
    segment = source[begin:end]
    segment = replace_once(segment,
        '    if (OKC_PINDIV) asm volatile("" : "+r"(nact), "+r"(R));',
        '    if (OKC_PINDIV) asm volatile("" : "+r"(nact), "+r"(R));\n    DBG(16);')
    segment = replace_once(segment, "      else old = atomicAdd(c, mine);",
                           "      else old = atomicAdd(c, mine);\n      DBG(17);")
    segment = replace_once(segment, "        p.ws_cnt[24576 + bp] = ep + 1;",
                           "        p.ws_cnt[24576 + bp] = ep + 1;\n        DBG(18);")
    for anchor, slot in (
        ("      sum_copies<L>(p, bp, j, hh, part, act, y, lsum);", 19),
        ("      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);", 20),
        ("      if (act) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;", 21),
    ):
        segment = replace_once(segment, anchor, anchor + f"\n      if (rbase == r_lo) {{ DBG({slot}); }}")
    source = source[:begin] + segment + source[end:]
    anchor = "  else __syncthreads();                   // shared memory is reused by the next segment\n  }\n  return true;"
    source = replace_once(source, anchor, anchor.replace("  return true;", "  DBG(22);\n  return true;"))
    # Host-only resource query of the ACTUAL selected instance, after occupancy guard.
    source = replace_once(source, "template <int L>\nvoid launch(",
        "static std::vector<int64_t> r2_metadata;\n"
        "std::vector<int64_t> latency_metadata() { return r2_metadata; }\n\n"
        "template <int L>\nvoid launch(")
    anchor = "  if (cl || pdl) {"
    metadata = '''  cudaFuncAttributes r2_attr;
  TORCH_CHECK(cudaFuncGetAttributes(&r2_attr, kern) == cudaSuccess, "resource query failed");
  int r2_occ = 0;
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r2_occ, kern,
      p.flat ? Cfg<L, true>::NTX : Cfg<L, false>::NTX, sm) == cudaSuccess, "occupancy query failed");
  r2_metadata = {(int64_t)grid.x, (int64_t)grid.y, (int64_t)grid.z,
      p.flat ? Cfg<L, true>::NTX : Cfg<L, false>::NTX, sm,
      r2_attr.numRegs, (int64_t)r2_attr.localSizeBytes, (int64_t)r2_attr.sharedSizeBytes,
      r2_occ, q.coop, p.flat, p.nc, L, p.n_splits};
'''
    return replace_once(source, anchor, metadata + anchor)


RESOURCE_KEYS = ("grid_x", "grid_y", "grid_z", "threads", "dynamic_shared_bytes",
                 "registers_per_thread", "local_bytes_per_thread", "static_shared_bytes",
                 "active_ctas_per_sm", "actual_coop", "flat", "workspace_copies", "q_len", "splits")


def build(control, arm, artifact, report, save_report):
    from torch.utils.cpp_extension import load_inline
    source = diagnostic_source(control.ATT7_CUDA_SRC, arm)
    baseline_text = Path(report["baseline"]["path"]).read_text()
    # Preserve the complete attributed Python header as comments in each derivative.
    header = baseline_text[:baseline_text.index('ATT7_CUDA_SRC = r"""')]
    attributed = "\n".join("// " + line for line in header.splitlines()) + "\n" + source
    path = artifact(f"{arm}.cu")
    path.write_text(attributed)
    cpp = control.ATT7_CPP_SRC + "\nstd::vector<int64_t> latency_metadata();\n"
    cpp_path = artifact(f"{arm}_bindings.cpp")
    cpp_path.write_text(cpp)
    major, minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}" + ("a" if major == 9 else "")
    flags = ["-DOKC_QDIRECT=0", "-DOKC_PK2=0", "-DOKC_W0ENABLE=1", "-O3",
             f"-DOKC_SM90={int(major >= 9)}", "--expt-relaxed-constexpr"]
    name = "round2_latency_" + arm + "_" + digest(source + repr(flags))[:16]
    report["builds"][arm] = dict(name=name, source_sha256=digest(source),
        source_artifact=path.name, bindings_artifact=cpp_path.name, flags=flags,
        enabled_slots=MASKS[arm], status="building")
    save_report()
    module = load_inline(name=name, cpp_sources=cpp, cuda_sources=source,
                         functions=["run", "latency_metadata"], extra_cuda_cflags=flags, verbose=False)
    report["builds"][arm]["status"] = "built"
    save_report()
    return module


def phase_summary(raw, arm):
    rows = [(i, row) for i, row in enumerate(raw) if row[0] > 1]
    if not rows:
        return dict(valid=False, reason="No entry stamps")
    origin = min(row[0] for _, row in rows)
    stamps = {str(slot): stats((row[slot] - origin) / 1000 for _, row in rows if row[slot] > 1)
              for slot in MASKS[arm]}
    phases = {}
    for name, (start, end) in PHASES.items():
        if start not in MASKS[arm] or end not in MASKS[arm]:
            continue
        valid = [(i, row) for i, row in rows if row[start] > 1 and row[end] > 1]
        phases[name] = dict(start=start, end=end,
            per_cta_us=stats((r[end] - r[start]) / 1000 for _, r in valid),
            reversed_ctas=[i for i, r in valid if r[end] < r[start]])
    ends = [(i, r) for i, r in rows if r[22] > 1]
    loops = [(i, r) for i, r in rows if r[3] > 1]
    output = [(i, r) for i, r in rows if r[21] > 1]
    critical = {}
    if ends and loops:
        last_i, last = max(ends, key=lambda item: item[1][22])
        loop_i, loop = max(loops, key=lambda item: item[1][3])
        critical.update(last_completion_cta=last_i, last_loop_cta=loop_i,
            stamped_envelope_us=(last[22] - origin) / 1000,
            last_loop_to_last_completion_us=(last[22] - loop[3]) / 1000,
            last_loop_cta_phases_us={name: (loop[b] - loop[a]) / 1000
                for name, (a, b) in PHASES.items() if loop[a] > 1 and loop[b] > 1},
            last_completion_cta_phases_us={name: (last[b] - last[a]) / 1000
                for name, (a, b) in PHASES.items() if last[a] > 1 and last[b] > 1})
    if output:
        output_i, out = max(output, key=lambda item: item[1][21])
        critical.update(last_output_issue_cta=output_i,
            last_output_cta_phases_us={name: (out[b] - out[a]) / 1000
                for name, (a, b) in PHASES.items() if out[a] > 1 and out[b] > 1})
    publishers = [(i, r) for i, r in rows if r[18] > 1]
    critical["last_arrival_publish_ctas"] = [i for i, _ in publishers]
    if publishers and output:
        publish = max(r[18] for _, r in publishers)
        critical["last_arrival_publish_to_last_output_issue_us"] = (max(r[21] for _, r in output) - publish) / 1000
    return dict(valid=True, origin_globaltimer_ns=origin, active_ctas=len(rows),
                observed_output_ctas=len(output), stamp_relative_us=stamps,
                phases=phases, critical_chain=critical,
                guaranteed_removable_us=0.0, limits=LIMITS)


def tensor_compare(got, want):
    a, b = got.float(), want.float()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return dict(finite=finite, bit_equal=bool(torch.equal(got.view(torch.uint8), want.view(torch.uint8))),
        max_abs=float((a-b).abs().max()) if finite else None,
        relative_l2=float(torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(b).clamp_min(1e-30)) if finite else None)


def public_check(reference, data, output):
    try:
        raw = reference.collect(data, output)
        passed, message = reference.check_implementation(data, raw)
        return dict(passed=bool(passed), message=str(message), method="attention.reference.check_implementation + collect (output and new FP8 K/V)")
    except Exception:
        return dict(passed=False, error=traceback.format_exc())


def trace(fn, data, bench, flush, path):
    captured = bench.capture(fn, data, None)
    flush.sum()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                          torch.profiler.ProfilerActivity.CUDA]) as prof:
        captured["graph"].replay()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    result = bench.summarize_trace(path)
    result["artifact"] = path.name
    return result


def run(args, report, artifact, save_report):
    global torch
    import torch
    control = load(args.baseline, "round2_latency_control")
    bench = load("compare_decode_public.py", "round2_latency_bench")
    bench.torch = torch
    reference = load("reference.py", "round2_latency_reference")
    report["baseline"]["embedded_cuda_sha256"] = digest(control.ATT7_CUDA_SRC)
    report["device"] = dict(name=torch.cuda.get_device_name(), torch=torch.__version__,
        cuda=torch.version.cuda, capability=torch.cuda.get_device_capability(),
        sms=torch.cuda.get_device_properties(0).multi_processor_count)
    native = control._att7()
    modules = {}
    for arm in ("prologue", "tail"):
        try:
            modules[arm] = build(control, arm, artifact, report, save_report)
        except Exception:
            report["errors"].append(dict(stage="build", arm=arm, error=traceback.format_exc()))
            save_report()
    flush = torch.ones(FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")
    map_path = artifact("stamp_map.json")
    source_lines = control.ATT7_CUDA_SRC.splitlines()
    save(map_path, dict(stride=STRIDE, slots=STAMP_MAP, phases=PHASES, masks=MASKS,
        baseline_source_dbg_sites=[dict(cuda_line=i+1, text=line) for i, line in enumerate(source_lines)
                                   if re.search(r"\bDBG\(", line)], limits=LIMITS))
    report["stamp_map_artifact"] = map_path.name
    for load_name, length in (("D1f", 1), ("V1f", 8)):
        case = dict(load=load_name, pairs=[], status="running")
        report["cases"][load_name] = case
        save_report()
        try:
            for pair in range(args.pairs):
                seed = args.seed + pair * 1009 + (0 if length == 1 else 100003)
                data = reference.generate_input(batch=1, q_len=length, ctx=80000, heads="tp8",
                    layer="full", page=1, spread="fixed", dist="coherent", seed=seed, device="cuda")
                plan = control._plan7(1, 1, length, 0, data["page_table"].shape[1], 1, data["qkv"].device)
                plan.update(control._TUNED7.get((1, length, False, 1), {}))
                if plan["flat"] or plan["cs"] > 1:
                    raise RuntimeError("Diagnostic landmarks require current nonflat/noncluster full-attention plan")
                count = plan["n_splits"]
                dbg = {arm: torch.zeros((count, STRIDE), dtype=torch.int64, device="cuda") for arm in modules}
                functions = {arm: (lambda d, _, m=native: control._launch7(d, _mod=m, pdl=False))
                             for arm in ("native", "native_repeat")}
                for arm, module in modules.items():
                    functions[arm] = lambda d, _, m=module, stamps=dbg[arm]: control._launch7(d, dbg=stamps, _mod=m, pdl=False)
                captures = {arm: bench.capture(fn, data, None, flush) for arm, fn in functions.items()}
                entry = dict(pair=pair, seed=seed, config=data["config"], plan=plan,
                    seq_lens=data["seq_lens"].cpu().tolist(), cache_loc=data["cache_loc"].cpu().tolist(),
                    page_table_sha256=hashlib.sha256(data["page_table"].cpu().numpy().tobytes()).hexdigest(),
                    resources={arm: dict(zip(RESOURCE_KEYS, module.latency_metadata())) for arm, module in modules.items()},
                    timing_rows=[], checks={}, traces={}, stamp_samples={arm: [] for arm in modules})
                case["pairs"].append(entry)
                # A/A shares exact addresses and selected module, but is separately captured.
                # Each fixture rotates all starts in paired forward/reverse rounds.
                names = list(functions)
                for repetition in range(4 * len(names)):
                    shift = (pair + repetition // 2) % len(names)
                    order = names[shift:] + names[:shift]
                    if repetition % 2:
                        order = list(reversed(order))
                    row = dict(repetition=repetition, order=order, us={})
                    for arm in order:
                        row["us"][arm] = bench.measure(captures[arm])
                    entry["timing_rows"].append(row)
                bench.measure(captures["native"])
                base_output = captures["native"]["output"].clone()
                base_parts = reference.unpack(reference.collect(data, base_output), data["config"])
                for arm in names:
                    bench.measure(captures[arm])
                    entry["checks"][arm] = public_check(reference, data, captures[arm]["output"])
                    entry["checks"][arm]["versus_native_output"] = tensor_compare(captures[arm]["output"], base_output)
                    parts = reference.unpack(reference.collect(data, captures[arm]["output"]), data["config"])
                    entry["checks"][arm]["versus_native_new_kv"] = {
                        name: tensor_compare(parts[index], base_parts[index])
                        for name, index in (("key", 1), ("value", 2))}
                save_report()
                # Raw stamp rows from separate cold-L2 replays, not contaminated by .cpu in timed call.
                for arm in modules:
                    for sample in range(args.samples):
                        dbg[arm].zero_()
                        measured = bench.measure(captures[arm])
                        raw = dbg[arm].cpu().tolist()
                        path = artifact(f"{load_name}_p{pair}_{arm}_s{sample}_stamps.json")
                        summary = phase_summary(raw, arm)
                        save(path, dict(load=load_name, seed=seed, arm=arm, sample=sample,
                            internal_external_event_us=measured, stride=STRIDE, raw_globaltimer_rows=raw,
                            parsing_map=STAMP_MAP, resources=entry["resources"][arm], summary=summary))
                        entry["stamp_samples"][arm].append(dict(artifact=path.name,
                            event_us=measured, summary=summary))
                save_report()
                if pair == 0:
                    for arm in names:
                        path = artifact(f"{load_name}_{arm}_native_trace.json")
                        try:
                            entry["traces"][arm] = trace(functions[arm], data, bench, flush, path)
                        except Exception:
                            entry["traces"][arm] = dict(error=traceback.format_exc(), artifact=path.name)
                            report["errors"].append(dict(stage="trace", load=load_name, arm=arm, error=traceback.format_exc()))
                        save_report()
                del captures, functions, dbg, data
            summarize_case(case)
            case["status"] = "complete"
        except Exception:
            case["status"] = "failed"
            case["error"] = traceback.format_exc()
            report["errors"].append(dict(stage="case", load=load_name, error=case["error"]))
            summarize_case(case)
        save_report()


def summarize_case(case):
    rows = [row["us"] for pair in case["pairs"] for row in pair["timing_rows"]]
    summary = dict(timing_us={arm: stats(row[arm] for row in rows if arm in row) for arm in ARMS})
    summary["aa_ratio_repeat_over_native"] = stats(row["native_repeat"] / row["native"] for row in rows)
    aa = summary["aa_ratio_repeat_over_native"]
    summary["aa_within_one_percent"] = bool(aa and abs(aa["median"] - 1) <= .01)
    traced = next((pair["traces"] for pair in case["pairs"] if pair["traces"]), {})
    native_launches = traced.get("native", {}).get("launches", [])
    summary["native_launch_metadata"] = [dict(name=x["name"], args=x["args"]) for x in native_launches]
    summary["instrumentation"] = {}
    for arm in ("prologue", "tail"):
        ratios = stats(row[arm] / statistics.mean((row["native"], row["native_repeat"]))
                       for row in rows if arm in row)
        resources = [pair["resources"][arm] for pair in case["pairs"] if arm in pair["resources"]]
        bad_resource = any(x["actual_coop"] != 1 or x["local_bytes_per_thread"] > 0 for x in resources)
        instrumented_launches = traced.get(arm, {}).get("launches", [])
        trace_comparison = dict(
            native_symbols=[x["name"] for x in native_launches],
            instrumented_symbols=[x["name"] for x in instrumented_launches],
            instrumented_launch_args=[x["args"] for x in instrumented_launches])
        trace_comparison["same_kernel_instances"] = bool(native_launches and instrumented_launches and
            trace_comparison["native_symbols"] == trace_comparison["instrumented_symbols"])
        bad_resource = bad_resource or not trace_comparison["same_kernel_instances"]
        native_durations = [sum(x["duration_us"] for x in traced.get(name, {}).get("launches", []))
                            for name in ("native", "native_repeat")]
        native_mean = statistics.mean(native_durations) if all(native_durations) else 0.0
        trace_ratio = (sum(x["duration_us"] for x in instrumented_launches) / native_mean
                       if native_mean and instrumented_launches else None)
        perturbed = bool((ratios and abs(ratios["median"] - 1) > .03) or
                         (trace_ratio is not None and abs(trace_ratio - 1) > .03))
        samples = [s for pair in case["pairs"] for s in pair["stamp_samples"].get(arm, [])]
        summary["instrumentation"][arm] = dict(paired_ratio_over_native_mean=ratios,
            materially_perturbed=perturbed,
            native_duration_ratio_over_control_mean=trace_ratio,
            local_memory_or_noncoop=bad_resource,
            native_trace_comparison=trace_comparison,
            interpretation="biased source landmarks; no accurate native attribution" if
                bad_resource or not ratios or perturbed or not summary["aa_within_one_percent"]
                else "sparse landmarks within timing gate; source scheduling and cross-CTA caveats still apply",
            stamped_envelope_us=stats(s["summary"].get("critical_chain", {}).get("stamped_envelope_us")
                for s in samples if "stamped_envelope_us" in s["summary"].get("critical_chain", {})),
            last_loop_to_last_completion_us=stats(s["summary"].get("critical_chain", {}).get("last_loop_to_last_completion_us")
                for s in samples if "last_loop_to_last_completion_us" in s["summary"].get("critical_chain", {})))
    summary["public_reference_passed"] = bool(case["pairs"]) and all(
        len(pair["checks"]) == 2 + len(pair["resources"]) and
        all(check["passed"] for check in pair["checks"].values()) for pair in case["pairs"])
    summary["guaranteed_removable_us"] = 0.0
    summary["limits"] = LIMITS
    case["summary"] = summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="round2_latency.json")
    parser.add_argument("--seed", type=int, default=731283)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--baseline", default="decode_current.py")
    args = parser.parse_args()
    if args.pairs < 1 or args.samples < 2:
        parser.error("--pairs must be positive and --samples at least 2")
    output = Path(args.output)
    report = dict(kind="round2_full_attention_latency_diagnostic", status="running",
        submission_allowed=False, official_score=False, seed=args.seed, pairs=args.pairs,
        artifact_files=[], baseline=dict(path=args.baseline), builds={}, cases={}, errors=[],
        timing_contract=dict(cuda_graph=True, internal_events_external_nodes=True,
            cold_l2_read_flush_bytes=FLUSH_BYTES, timed_region="_launch7 only, flush before begin event",
            pdl=False, co_running_gemm=False, rotating_aa=True, same_input_addresses_all_arms=True),
        stamp_map=STAMP_MAP, phase_map=PHASES, limits=LIMITS)
    def artifact(suffix):
        path = output.parent / (output.stem + "_" + suffix)
        if path.name not in report["artifact_files"]:
            report["artifact_files"].append(path.name)
        return path
    def save_report():
        save(output, report)
    save_report()
    try:
        report["baseline"]["sha256"] = hashlib.sha256(Path(args.baseline).read_bytes()).hexdigest()
        for filename in ("reference.py", "checks.py", "compare_decode_public.py"):
            report.setdefault("input_sha256", {})[filename] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        run(args, report, artifact, save_report)
        complete = len(report["cases"]) == 2 and all(c["status"] == "complete" and c["summary"]["public_reference_passed"] for c in report["cases"].values())
        report["status"] = "complete" if complete and not report["errors"] else "partial_failure"
    except Exception:
        report["status"] = "partial_failure"
        report["errors"].append(dict(stage="driver", error=traceback.format_exc()))
    finally:
        report["artifact_files"] = [name for name in report["artifact_files"] if (output.parent / name).is_file()]
        save_report()
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
