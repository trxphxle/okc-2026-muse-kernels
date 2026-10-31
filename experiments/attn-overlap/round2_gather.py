"""Round-2 exact FP8 gather-read ceiling diagnostic; NEVER a submission.

Upload this file, round2_gather.cu, decode_current.py, compare_decode_public.py,
and attention/reference.py + attention/checks.py (uploaded by basename). Run:
  python round2_gather.py --output gather.json --seed 731284 --pairs 11
The optional --baseline decode_control.py selects the unchanged Round-1 snapshot;
the default is the precision-repaired current control. No layer prepare is used.

Past-cache mode reads exactly seq_lens[b] logical rows per request, once each,
using the input page table. The baseline synthesizes its new K/V on chip and
writes cache_loc; it does NOT read those new rows from the cache. The separate
past_plus_new mode additionally reads those materialized rows after the unchanged
baseline has written them. Neither mode performs attention or is submittable.

All three launch geometries and both byte sets are preregistered below. Geometry
selection is reported, never hidden. Source-byte accounting is logical, NOT NCU
DRAM traffic. Timed reads retain two additive checksum moments of every loaded
uint32 word and exact counts; independent bounded NumPy validation checks every
warp sink before AND after timing. SHA256 records the independently read expected
bytes. A checksum is not a proof against adversarial collisions; no XOR is used.

Internal external=True graph events exclude a captured 256-MiB read flush.
Separately captured native traces contain ONLY the measured operation, after an
untimed cold flush. Raw kernel durations, not graph/event overhead, drive the
reported observed read-gap proxy. That proxy is not a proven hardware lower
bound, predicted optimization speedup, or attention/decode board result.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent
CASES = (("V16f", 16, 8), ("V32f", 32, 8), ("D16f", 16, 1), ("D32f", 32, 1))
GEOMETRIES = (
    dict(id=0, name="t256_u4", threads=256, unroll=4, target_blocks_per_sm=8),
    dict(id=1, name="t128_u4", threads=128, unroll=4, target_blocks_per_sm=12),
    dict(id=2, name="t256_u2", threads=256, unroll=2, target_blocks_per_sm=8),
)
MODES = ("past", "past_plus_new")
FLUSH_BYTES = 256 * 1024 * 1024
VALIDATION_ROWS = 4096
KNOWN_BASELINES = {
    "decode_current.py": "053dbacc8f0de4e6111ccb89cb91f629840852cf482a6ca423fe0eaff419c086",
    "decode_control.py": "4531458701f45eb713fcc885447149a563826316edea1f2a419a25b4f7c96964",
}


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save(path, report):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def tensor_meta(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), stride=list(value.stride()),
                device=str(value.device), data_ptr=value.data_ptr(),
                storage_ptr=value.untyped_storage().data_ptr(),
                allocation_bytes=value.untyped_storage().nbytes())


def runtime_metadata():
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    info = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                numpy=np.__version__, cuda_runtime=torch.version.cuda, device=str(props),
                name=props.name, sm_count=props.multi_processor_count,
                total_memory=props.total_memory, compute_capability=list(torch.cuda.get_device_capability()))
    command = ["nvidia-smi", "--query-gpu=name,uuid,driver_version,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu",
               "--format=csv,noheader"]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        info["nvidia_smi"] = dict(command=command, returncode=proc.returncode,
                                  stdout=proc.stdout, stderr=proc.stderr)
    except Exception as exc:
        info["nvidia_smi"] = dict(command=command, error=repr(exc))
    return info


def address_accounting(data):
    cfg = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    require((cfg["page_size"], cfg["num_kv_heads"], cfg["head_dim"], cfg["window"])
            == (1, 2, 64, 0), "Only full TP8 page-1 cases are supported")
    require(data["k_cache"].dtype == data["v_cache"].dtype == torch.float8_e4m3fn,
            "Caches must remain real FP8")
    for key in ("k_cache", "v_cache", "page_table", "seq_lens", "cache_loc"):
        require(data[key].is_contiguous(), f"Noncontiguous fixture {key}")
    require(data["page_table"].dtype == data["seq_lens"].dtype == torch.int32,
            "Expected int32 page table and past lengths")
    require(data["cache_loc"].dtype == torch.int64, "Expected int64 cache_loc")
    seq = data["seq_lens"].cpu().numpy().astype(np.int64)
    loc = data["cache_loc"].cpu().numpy().reshape(B, L)
    pool_rows = data["k_cache"].numel() // 128
    require(data["v_cache"].numel() == data["k_cache"].numel(), "Cache sizes differ")
    require(all(data[k].data_ptr() % 16 == 0 for k in ("k_cache", "v_cache")), "Unaligned cache")
    seen = np.zeros(pool_rows, dtype=np.bool_)
    requests = []
    metadata_digest = hashlib.sha256()
    metadata_digest.update(seq.astype("<i4").tobytes())
    metadata_digest.update(loc.astype("<i8").tobytes())
    for b, past in enumerate(seq.tolist()):
        table = data["page_table"][b].cpu().numpy()
        require(0 <= past and past + L <= table.size, f"Request {b}: length outside table")
        rows = table[:past + L].astype(np.int64)
        require(bool(((rows >= 0) & (rows < pool_rows)).all()), f"Request {b}: invalid physical row")
        # This sorting is validation ONLY. The timed kernel always uses original logical order.
        require(np.unique(rows).size == rows.size and not bool(seen[rows].any()),
                f"Request {b}: aliased pages in supposedly disjoint scored fixture")
        seen[rows] = True
        require(np.array_equal(loc[b], rows[past:]), f"Request {b}: cache_loc disagrees with page table")
        metadata_digest.update(table.astype("<i4", copy=False).tobytes())
        requests.append(dict(request=b, seq_len=past, past_k_bytes=past * 128,
                             past_v_bytes=past * 128, past_kv_bytes=past * 256,
                             past_uint32_words=past * 64, new_cache_rows=loc[b].tolist(),
                             new_cache_write_bytes=L * 256, additional_new_read_bytes=L * 256,
                             past_page_index_bytes=past * 4,
                             new_cache_loc_bytes=L * 8, physical_min=int(rows.min()),
                             physical_max=int(rows.max()),
                             mapped_rows_sha256=hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest(),
                             unused_table_entries=int(table.size - rows.size),
                             padding_not_loaded=True, mapping_matches_cache_loc=True))
    total = int(seq.sum())
    return dict(per_request=requests, sum_seq_lens=total, past_kv_bytes=total * 256,
                past_uint32_words=total * 64, new_rows=B * L, new_write_bytes=B * L * 256,
                additional_new_read_bytes=B * L * 256, past_plus_new_kv_bytes=(total + B * L) * 256,
                unique_physical_rows=int(seen.sum()), no_past_new_aliases=True,
                no_cross_request_aliases=True, cache_alignment_bytes=16,
                cache_allocation_bytes_per_cache=data["k_cache"].numel(),
                metadata=dict(past_page_table_logical_bytes=total * 4,
                              seq_lens_unique_bytes=B * 4, cache_loc_unique_bytes=B * L * 8,
                              page_table_allocation_bytes=data["page_table"].numel() * 4,
                              new_page_table_mapping_validation_bytes=B * L * 4,
                              sha256=metadata_digest.hexdigest(),
                              note="Unique/semantic bytes, not baseline transaction counts or NCU DRAM bytes"),
                checks="All actual rows in bounds, aligned, unique; every cache_loc equals page_table[b,seq+j]; padding excluded"), seq


def geometry_specs(B, sm_count):
    return [dict(g, blocks_per_request=math.ceil(sm_count * g["target_blocks_per_sm"] / B),
                 warps=g["threads"] // 32) for g in GEOMETRIES]


def independent_expected(data, seq, specs):
    """Bounded reference: gather <=4096 physical rows, interpret EVERY raw word on CPU.

    Temporary cache payload <=1 MiB per chunk (both caches combined), with bounded
    NumPy moments/weights and one request's page table. No whole-cache CPU clone,
    dense max-length stream, or pre-gathered/reordered timed input is constructed.
    """
    B, L = data["config"]["batch"], data["config"]["q_len"]
    expected = {(g["id"], mode): np.zeros((B, g["blocks_per_request"], g["warps"], 17), dtype=np.uint32)
                for g in specs for mode in MODES}
    byte_views = [data[k].view(torch.uint8).reshape(-1, 128) for k in ("k_cache", "v_cache")]
    records = []
    for b, past in enumerate(seq.tolist()):
        table = data["page_table"][b].cpu().numpy()
        loc = data["cache_loc"].reshape(B, L)[b].cpu().numpy()
        digests = {region: [hashlib.sha256(), hashlib.sha256()] for region in ("past", "new")}
        words_checked = {region: 0 for region in digests}
        for region, start, end in (("past", 0, past), ("new", past, past + L)):
            for at in range(start, end, VALIDATION_ROWS):
                stop = min(end, at + VALIDATION_ROWS)
                pos = np.arange(at, stop, dtype=np.int64)
                rows = table[at:stop].astype(np.int64) if region == "past" else loc[at - past:stop - past]
                indices = torch.as_tensor(rows, dtype=torch.int64, device=data["k_cache"].device)
                weights = (pos[:, None, None] * 32 + np.arange(32).reshape(1, 8, 4) + 1).astype(np.uint32)
                sums, moments = [], []
                for kind, view in enumerate(byte_views):
                    raw = view.index_select(0, indices).cpu().numpy()
                    digests[region][kind].update(raw.tobytes())
                    words = raw.view("<u4").reshape(-1, 8, 4)
                    sums.append(words.sum(axis=1, dtype=np.uint64).astype(np.uint32))
                    moments.append(np.multiply(words, weights, dtype=np.uint32).sum(axis=1, dtype=np.uint64).astype(np.uint32))
                    words_checked[region] += int(words.size)
                contribution = np.concatenate((*sums, *moments, np.full((pos.size, 1), 8, dtype=np.uint32)), axis=1)
                for g in specs:
                    rpb, blocks, warps = g["threads"] // 8, g["blocks_per_request"], g["warps"]
                    # Independent inverse of the CTA's fixed strided row ownership.
                    block = (pos // (rpb * g["unroll"])) % blocks
                    warp = (pos % rpb) // 4
                    slots = block * warps + warp
                    modes = MODES if region == "past" else ("past_plus_new",)
                    for mode in modes:
                        target = expected[g["id"], mode][b].reshape(-1, 17)
                        np.add.at(target, slots, contribution)
        require(words_checked["past"] == past * 64 and words_checked["new"] == L * 64,
                "Independent full-loop byte count failed")
        records.append(dict(request=b, past_words_checked=words_checked["past"],
                            new_words_checked=words_checked["new"],
                            expected_raw_sha256={region: dict(zip(("k", "v"), (d.hexdigest() for d in ds)))
                                                 for region, ds in digests.items()}))
    return expected, dict(per_request=records, chunk_rows=VALIDATION_ROWS,
                         checksum="16 uint32 additive moments per warp: 8 component sums + 8 index-weighted sums, modulo 2^32; 17th word exact vector-pair count",
                         expected_sink_sha256={f"g{gid}_{mode}": hashlib.sha256(array.tobytes()).hexdigest()
                                               for (gid, mode), array in expected.items()},
                         limitation="Additive moments detect errors non-adversarially, not collision-free cryptographic proof; SHA256 hashes independent expected bytes, not an invented device SHA")


def check_sink(sink, expected, account, mode):
    actual = sink.cpu().numpy().view(np.uint32).reshape(expected.shape)
    bad = np.argwhere(actual != expected)
    require(bad.size == 0, f"Checksum mismatch ({mode}), first [b,block,warp,channel]: {bad[:8].tolist()}")
    counts = actual[..., 16].astype(np.uint64).sum(axis=(1, 2))
    per_request = []
    for b, vectors in enumerate(counts.tolist()):
        want = account["per_request"][b]["past_kv_bytes"]
        if mode == "past_plus_new":
            want += account["per_request"][b]["additional_new_read_bytes"]
        require(vectors * 32 == want, f"Request {b}: full-loop byte mismatch")
        per_request.append(dict(request=b, loaded_vector_pairs=vectors, loaded_bytes=vectors * 32,
                                loaded_uint32_words=vectors * 8, expected_bytes=want))
    return dict(passed=True, all_warp_moments_match=True, per_request=per_request,
                runtime_sink_sha256=hashlib.sha256(actual.tobytes()).hexdigest())


def paired_summary(samples, names):
    medians = {name: statistics.median(s[name] for s in samples) for name in names}
    controls = {}
    for name in names:
        if name.endswith("_repeat"):
            original = name.removesuffix("_repeat")
            ratios = [s[name] / s[original] for s in samples]
            controls[original] = dict(repeat_over_original=medians[name] / medians[original],
                                     paired_ratio_median=statistics.median(ratios), raw_paired_ratios=ratios,
                                     within_one_percent=abs(medians[name] / medians[original] - 1) <= 0.01)
    return dict(median_us=medians, aa=controls)


def oracle(reference, data, output):
    passed, message = reference.check_implementation(data, reference.collect(data, output))
    return dict(passed=bool(passed), message=str(message), api="attention.reference.check_implementation(data, collect(data, output))")


def native_batch(bench, captures, flush, orders, path, report, output):
    # One profiler session avoids the observed intermittent loss of GPU events
    # when starting/stopping Kineto around hundreds of sub-millisecond replays.
    # Cold flushes stay outside each captured graph. Graph IDs, not CPU ranges,
    # identify the measured kernels; retain the complete raw trace including flushes.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        for order in orders:
            for arm in order:
                cold_sink = flush.sum()
                captures[arm]["graph"].replay()
                torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    report["artifact_files"].append(path.name)
    save(output, report)
    raw = json.loads(path.read_text())
    kernels = sorted((event for event in raw["traceEvents"]
                      if event.get("cat") == "kernel" and event.get("ph") == "X"
                      and event.get("args", {}).get("graph id", 0)), key=lambda e: e["ts"])
    scheduled = [(arm, repeat) for repeat, order in enumerate(orders) for arm in order]
    require(len(kernels) == len(scheduled),
            f"Expected {len(scheduled)} graph kernels, observed {len(kernels)}")
    results = {arm: [] for arm in captures}
    graph_ids = {}
    raw_sha = sha_file(path)
    for event, (arm, repeat) in zip(kernels, scheduled):
        graph_id = event["args"]["graph id"]
        require(graph_ids.get(arm, graph_id) == graph_id, "Graph identity changed between repeats")
        require(all(name == arm or value != graph_id for name, value in graph_ids.items()),
                "Distinct captured arms unexpectedly share a graph ID")
        graph_ids[arm] = graph_id
        sliced = path.with_name(f"{path.stem}-{arm}-{repeat}-slice.json")
        save(sliced, dict(traceEvents=[event], source_trace=path.name,
                          source_trace_sha256=raw_sha, deviceProperties=raw.get("deviceProperties")))
        report["artifact_files"].append(sliced.name)
        summary = bench.summarize_trace(sliced)
        summary.update(trace_file=sliced.name, trace_sha256=sha_file(sliced),
                       raw_trace=path.name, flush_bytes=FLUSH_BYTES,
                       flush_outside_captured_graph=True, graph_id=graph_id)
        results[arm].append(summary)
    del cold_sink
    return results


def run_case(args, entry, report, output, baseline, extension, reference, bench, sm_count, flush):
    name, B, L = next(c for c in CASES if c[0] == entry["case"])
    fixture_args = dict(batch=B, ctx=80000, heads="tp8", layer="full", seed=entry["seed"],
                        q_len=L, spread="agentic", dist="coherent", page=1, device="cuda")
    entry["fixture_args"] = fixture_args
    save(output, report)
    data = reference.generate_input(**fixture_args)
    entry["config"] = data["config"]
    entry["fixture_pointers"] = {key: tensor_meta(value) for key, value in data.items() if isinstance(value, torch.Tensor)}
    account, seq = address_accounting(data)
    entry["accounting"] = account
    specs = geometry_specs(B, sm_count)
    entry["geometries"] = specs
    # Unmodified current baseline is responsible for materializing its own new rows.
    base_fn = lambda d, state: baseline._launch7(d, pdl=False)
    first_output = base_fn(data, None)
    torch.cuda.synchronize()
    entry["baseline_oracle_before"] = oracle(reference, data, first_output)
    save(output, report)
    require(entry["baseline_oracle_before"]["passed"], "Unmodified baseline failed the attention reference")
    expected, validation = independent_expected(data, seq, specs)
    entry["independent_validation"] = validation
    functions = {"baseline": base_fn, "baseline_repeat": base_fn}
    sinks, descriptions = {}, {}
    for g in specs:
        for mode in MODES:
            key = f"{g['name']}_{mode}"
            sink = torch.empty(expected[g["id"], mode].shape, device="cuda", dtype=torch.int32)
            sinks[key] = sink
            def gather(d, state, g=g, mode=mode, sink=sink):
                extension.run(d["k_cache"], d["v_cache"], d["page_table"], d["seq_lens"],
                              d["cache_loc"], sink, g["id"], g["blocks_per_request"], L,
                              mode == "past_plus_new")
                return sink
            functions[key] = functions[key + "_repeat"] = gather
            descriptions[key] = dict(mode=mode, geometry_id=g["id"],
                                     total_blocks=B * g["blocks_per_request"],
                                     sink=tensor_meta(sink), native_attributes=dict(extension.info(g["id"], mode == "past_plus_new")),
                                     logical_kv_bytes=account["past_kv_bytes"] + (account["additional_new_read_bytes"] if mode == "past_plus_new" else 0),
                                     checksum_sink_written_bytes=sink.numel() * sink.element_size(),
                                     metadata=dict(page_index_load_bytes=account["sum_seq_lens"] * 4,
                                                   seq_lens_issued_load_bytes=B * g["blocks_per_request"] * 4,
                                                   new_cache_loc_load_bytes=B * L * 8 if mode == "past_plus_new" else 0),
                                     expected_checksum=validation["expected_sink_sha256"][f"g{g['id']}_{mode}"])
    entry["arms"] = descriptions
    captures = {key: bench.capture(fn, data, None, flush=flush) for key, fn in functions.items()}
    entry["checksum_before"] = {}
    for key, desc in descriptions.items():
        captures[key]["graph"].replay()
        captures[key]["end"].synchronize()
        entry["checksum_before"][key] = check_sink(sinks[key], expected[desc["geometry_id"], desc["mode"]], account, desc["mode"])
    entry["event_samples"] = []
    entry["event_orders"] = []
    names = list(functions)
    for pair in range(args.pairs):
        shift = pair % len(names)
        order = names[shift:] + names[:shift]
        if (pair // len(names)) % 2:
            order.reverse()
        sample = {key: bench.measure(captures[key]) for key in order}
        entry["event_samples"].append(sample)
        entry["event_orders"].append(order)
        entry["event_timing"] = paired_summary(entry["event_samples"], names)
        save(output, report)
    entry["native_orders"] = []
    # Reuse each graph and all fixture addresses for native repetitions.
    native_captures = {key: bench.capture(fn, data, None) for key, fn in functions.items()}
    for repeat in range(args.native_repeats):
        shift = (repeat * 3) % len(names)
        order = names[shift:] + names[:shift]
        if repeat % 2:
            order.reverse()
        entry["native_orders"].append(order)
    trace = output.with_name(f"{output.stem}-{name}-native-batch.json")
    entry["native"] = native_batch(bench, native_captures, flush, entry["native_orders"],
                                  trace, report, output)
    save(output, report)
    native_samples = [{key: entry["native"][key][r]["summed_kernel_us"] for key in names}
                      for r in range(args.native_repeats)]
    entry["native_kernel_timing"] = paired_summary(native_samples, names)
    entry["native_kernel_timing"]["metric"] = "Sum of actual native kernel durations; NOT event span or graph replay overhead"
    entry["native_span_median_us"] = {key: statistics.median(s["kernel_timeline_span_us"] for s in summaries)
                                       for key, summaries in entry["native"].items()}
    entry["checksum_after"] = {}
    for key, desc in descriptions.items():
        # Re-exercise both identical captured arms after all mixed base/read timing.
        entry["checksum_after"][key] = {}
        for arm in (key, key + "_repeat"):
            captures[arm]["graph"].replay()
            captures[arm]["end"].synchronize()
            entry["checksum_after"][key][arm] = check_sink(sinks[key], expected[desc["geometry_id"], desc["mode"]], account, desc["mode"])
    captures["baseline"]["graph"].replay()
    captures["baseline"]["end"].synchronize()
    entry["baseline_oracle_after"] = oracle(reference, data, captures["baseline"]["output"])
    require(entry["baseline_oracle_after"]["passed"], "Post-timing baseline oracle failed")
    med = entry["native_kernel_timing"]["median_us"]
    base_us = med["baseline"]
    entry["observed_non_submittable_proxies"] = {}
    for key, desc in descriptions.items():
        read_us = med[key]
        entry["observed_non_submittable_proxies"][key] = dict(
            baseline_native_kernel_us=base_us, gather_native_kernel_us=read_us,
            baseline_minus_gather_us=base_us - read_us,
            fraction_of_full_attention_time=(base_us - read_us) / base_us,
            logical_kv_TB_per_s=desc["logical_kv_bytes"] / read_us / 1e6,
            baseline_past_logical_kv_TB_per_s=account["past_kv_bytes"] / base_us / 1e6,
            dram_throughput_measured=False, includes_new_cache_reads=desc["mode"] == "past_plus_new",
            aa_within_one_percent=all(entry[t]["aa"][arm]["within_one_percent"]
                                      for t in ("event_timing", "native_kernel_timing") for arm in ("baseline", key)))
    past_arms = [k for k, d in descriptions.items() if d["mode"] == "past"]
    best = min(past_arms, key=lambda k: med[k])
    entry["best_observed_past_geometry"] = dict(arm=best, selection="Minimum of all three preregistered native medians; all results retained",
                                                **entry["observed_non_submittable_proxies"][best])
    entry["fixture_pointers_unchanged"] = all(value.data_ptr() == entry["fixture_pointers"][key]["data_ptr"]
                                               for key, value in data.items() if isinstance(value, torch.Tensor))
    require(entry["fixture_pointers_unchanged"], "Fixture pointer reuse violated")
    entry["status"] = "complete"
    save(output, report)
    del captures, native_captures, functions, sinks, first_output, data
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="round2_gather.json")
    parser.add_argument("--seed", type=int, default=731284)
    parser.add_argument("--pairs", type=int, default=11)
    parser.add_argument("--baseline", default="decode_current.py")
    parser.add_argument("--native-repeats", type=int, default=3)
    parser.add_argument("--cases", nargs="+", choices=[c[0] for c in CASES], default=[c[0] for c in CASES])
    args = parser.parse_args()
    require(args.pairs >= 2 and args.native_repeats >= 2, "At least two A/A rounds and native repetitions required")
    require(Path(args.output).name == args.output, "--output must be a basename for artifact collection")
    output = Path(args.output).resolve()
    report = dict(diagnostic_only=True, submittable=False, status="started", arguments=vars(args),
                  artifact_files=[], results=[], start_time_unix=time.time(),
                  preregistration=dict(cases=CASES, geometries=GEOMETRIES, modes=MODES,
                      baseline="Unmodified selected module._launch7(data,pdl=False); no layer prepare, GEMM or co-runner",
                      old_cache_address="page_table[b,pos//P]*P+pos%P for 0<=pos<seq_lens[b]; P=1, bytes=row*128+16*chunk in each of K and V",
                      new_cache_contract="Baseline writes cache_loc[b*L+j], equals mapped row at seq_lens[b]+j. Baseline uses on-chip new K/V; past_plus_new alone reads these additional cache rows.",
                      byte_contract="128 K bytes + 128 V bytes for every actual row; no padding, maximum-length substitution, or contiguous surrogate",
                      cold_flush_bytes=FLUSH_BYTES, timing="CUDA graphs, captured internal external=True events; rotated identical A/A; separate native traces",
                      exclusions="Pure gather omits Q/K normalization, cache stores, QK/PV, softmax, split merge, output RMS/gate and their workspace/metadata traffic",
                      ceiling_limit="Observed checksum-bearing exact-logical-read proxy only. Not a strict physical lower bound, achieved DRAM throughput, production speedup, or board potential. Parent performs weighting/gating.",
                      reference_fixture="Full scored dist=coherent, TP8, context=80000, page=1, batched spread=agentic; no sliding cases"))
    save(output, report)
    try:
        require(sys.byteorder == "little", "Independent raw-word checksum requires little endian host")
        require(torch.cuda.is_available(), "CUDA GPU required; no CPU fallback")
        baseline_path = Path(args.baseline)
        if not baseline_path.is_file():
            baseline_path = HERE / args.baseline
        files = dict(driver=Path(__file__), cuda=HERE / "round2_gather.cu", baseline=baseline_path,
                     timing_helpers=HERE / "compare_decode_public.py", reference=HERE / "reference.py", checks=HERE / "checks.py")
        report["sources"] = {key: dict(path=str(path), sha256=sha_file(path)) for key, path in files.items()}
        if baseline_path.name in KNOWN_BASELINES:
            require(report["sources"]["baseline"]["sha256"] == KNOWN_BASELINES[baseline_path.name], "Known baseline source hash mismatch")
        report["baseline_identity"] = "current_precision_repaired" if baseline_path.name == "decode_current.py" else "explicit_alternative_or_round1_calibration"
        report["runtime"] = runtime_metadata()
        reference = module_from(files["reference"], "reference")
        bench = module_from(files["timing_helpers"], "round2_gather_bench")
        bench.torch = torch
        baseline = module_from(baseline_path, "round2_gather_baseline")
        report["attention_cuda_sha256"] = hashlib.sha256(baseline.ATT7_CUDA_SRC.encode()).hexdigest()
        report["baseline_att7_builder_source"] = __import__("inspect").getsource(baseline._att7)
        save(output, report)
        # Build only on explicit parent execution of this driver, never while authoring.
        baseline._att7()
        flags = ["-O3", "-lineinfo"]
        extension = load(name="round2_gather_" + report["sources"]["cuda"]["sha256"][:12],
                         sources=[str(files["cuda"])], extra_cuda_cflags=flags, verbose=False)
        report["gather_build"] = dict(cuda_flags=flags, arch_list=os.environ.get("TORCH_CUDA_ARCH_LIST"),
                                     extension_path=extension.__file__)
        flush = torch.ones(FLUSH_BYTES // 4, device="cuda", dtype=torch.float32)
        report["flush"] = tensor_meta(flush)
        for case_index, (name, _, _) in enumerate(CASES):
            if name not in args.cases:
                continue
            entry = dict(case=name, seed=args.seed + 1009 * case_index, status="started")
            report["results"].append(entry)
            save(output, report)
            try:
                run_case(args, entry, report, output, baseline, extension, reference, bench,
                         report["runtime"]["sm_count"], flush)
            except Exception:
                entry["status"] = "failed"
                entry["error"] = traceback.format_exc()
                save(output, report)
                gc.collect()
        report["status"] = "complete" if all(r["status"] == "complete" for r in report["results"]) else "partial_failure"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
    finally:
        report["elapsed_seconds"] = time.time() - report["start_time_unix"]
        save(output, report)
    print(json.dumps(dict(output=output.name, status=report["status"], cases=[(r["case"], r["status"]) for r in report["results"]])))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
