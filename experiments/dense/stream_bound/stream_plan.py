"""Untimed, frozen-fixture byte ceiling planner; NEVER submission preparation.

The stream reads original BF16/FP32/NVFP4 weight allocations and the exact union
of physically addressed FP8 K/V rows, including baseline-written new tokens.
Experts are read once per layer, irrespective of route multiplicity. Routes and
page selection are a free oracle. Arithmetic, dependencies, activations, cache
writes, and ordinary routing/page-table reads are deliberately not modelled.

Only the two 4-byte KV scales are copied: a disclosed 16-byte static metadata
packet contains their exact bytes followed by eight zero padding bytes. Global
expert scales instead read their original full 32-entry arrays conservatively.
Native means the original public byte format, NOT the baseline's retained
physical representation. No unverified compression savings are credited.

CPU hashing transfers at most 8 MiB of payload per batch, never expands dense
weights to integers, and gathers scattered KV rows in bounded GPU batches.
The returned keepalive must survive every launch. Rewriting the same KV buffers
keeps addresses valid, but validation XOR is valid only for the frozen values.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch


_HASH_BYTES = 8 * 1024 * 1024
_DENSE = ("qkv_gate", "o", "pooler", "router", "unpooler")
_VECTORS = (
    "attn_residual_gate", "pre_ffn_norm", "router_bias", "pre_expert_norm",
    "post_expert_norm", "post_ffn_norm", "ffn_residual_gate", "next_pre_attn_norm",
)
_EXPERT = ("w13", "w13_sf", "w2", "w2_sf")
_RETAINED_REASON = (
    "Native-only diagnostic: repaired prepare retains multiple tuned dense/SK2 "
    "streams and escape tables; their presence is not proof of the active per-load "
    "consumer. Counts are original public BF16/FP32/NVFP4 bytes, not retained "
    "baseline traffic. Router remains original unrounded FP32."
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _cpu_i64(tensor):
    _require(tensor.dtype in (torch.int32, torch.int64), "metadata must be integer")
    return tensor.detach().to(device="cpu", dtype=torch.int64).numpy()


def _digest(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array, dtype="<i8"))).hexdigest()


def _xor(raw):
    # uint8 CPU tensor/numpy view: no numerical casts of the weight values.
    array = raw.numpy() if isinstance(raw, torch.Tensor) else raw
    return int(np.bitwise_xor.reduce(array.reshape(-1).view("<u4"), initial=np.uint32(0)))


def _runs(rows):
    """Sorted unique row ids -> exact half-open physical row intervals."""
    if not len(rows):
        return np.empty((0, 2), dtype=np.int64)
    cuts = np.flatnonzero(np.diff(rows) != 1) + 1
    starts = np.concatenate((np.array([0]), cuts))
    ends = np.concatenate((cuts, np.array([len(rows)])))
    return np.column_stack((rows[starts], rows[ends - 1] + 1))


def _routes(output, data):
    if isinstance(output, torch.Tensor):
        ids = output
    elif isinstance(output, dict):
        ids = output["ids"]
    else:
        _require(isinstance(output, (tuple, list)) and len(output) == 3,
                 "baseline output must be (out, stream2, ids), {'ids': ids}, or ids")
        ids = output[2]
    cfg = data["config"]
    ids_cpu = _cpu_i64(ids)
    _require(ids_cpu.shape == (int(cfg["tokens"]), int(cfg["top_k"])),
             "actual baseline ids shape does not match tokens/top_k")
    n_experts, n_local = int(cfg["n_experts"]), int(cfg["local_experts"])
    _require(n_experts == 256 and n_local == 32, "planner expects public EP8 expert geometry")
    _require(bool(((ids_cpu >= 0) & (ids_cpu < n_experts)).all()), "invalid global expert ids")
    _require(all(len(np.unique(row)) == len(row) for row in ids_cpu), "duplicate top-k ids")
    local_start_values = _cpu_i64(data["local_start"]).reshape(-1)
    _require(len(local_start_values) == 1, "local_start must be scalar")
    local_start = int(local_start_values[0])
    _require(0 <= local_start < n_experts, "local_start outside global expert range")
    local = (ids_cpu - local_start) % n_experts
    histogram = np.bincount(local[local < n_local], minlength=n_local)
    touched = np.flatnonzero(histogram)
    return touched, {
        "local_start": local_start,
        "global_ids": ids_cpu.tolist(),
        "global_histogram": np.bincount(ids_cpu.reshape(-1), minlength=n_experts).tolist(),
        "local_histogram": histogram.tolist(),
        "local_experts": touched.tolist(),
        "global_local_experts": ((touched + local_start) % n_experts).tolist(),
        "local_route_pairs": int(histogram.sum()),
        "unique_local_experts": len(touched),
        "expert_reads_per_layer": 1,
        "ids_sha256": _digest(ids_cpu),
    }


def _kv_coverage(data):
    cfg = data["config"]
    batch, q_len = int(cfg["batch"]), int(cfg["q_len"])
    page, window = int(cfg["page_size"]), int(cfg["window"])
    _require(batch > 0 and q_len > 0 and page > 0 and window >= 0, "invalid attention geometry")
    _require(int(cfg["tokens"]) == batch * q_len, "tokens != batch*q_len")
    seq = _cpu_i64(data["seq_lens"]).reshape(-1)
    _require(len(seq) == batch and bool((seq >= 0).all()), "invalid sequence lengths")
    pt = data["page_table"]
    _require(pt.ndim == 2 and pt.shape[0] == batch, "invalid page table shape")
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    expected_tail = (page, int(cfg["num_kv_heads"]), int(cfg["head_dim"]))
    _require(tuple(k_cache.shape[1:]) == expected_tail and k_cache.shape == v_cache.shape,
             "KV shape does not match page/head geometry")
    _require(k_cache.dtype == torch.float8_e4m3fn and v_cache.dtype == torch.float8_e4m3fn,
             "KV must retain original FP8 E4M3 bytes")
    row_bytes = int(cfg["num_kv_heads"]) * int(cfg["head_dim"])
    _require(row_bytes == 128, "planner expects public 2x64 KV rows")
    all_rows, pages_by_request, requests = [], [], []
    expected_new = []
    for b, length_value in enumerate(seq):
        length = int(length_value)
        first = max(0, length - window + 1) if window else 0
        stop = length + q_len
        pfirst, pstop = first // page, (stop + page - 1) // page
        _require(pstop <= pt.shape[1], "attended pages exceed page table")
        # Transfer the selected table slice once, not once per physical KV row.
        pages = _cpu_i64(pt[b, pfirst:pstop]).reshape(-1)
        _require(bool(((pages >= 0) & (pages < k_cache.shape[0])).all()), "invalid attended physical page")
        positions = np.arange(first, stop, dtype=np.int64)
        rows = pages[positions // page - pfirst] * page + positions % page
        new_positions = np.arange(length, stop, dtype=np.int64)
        expected_new.append(pages[new_positions // page - pfirst] * page + new_positions % page)
        all_rows.append(rows)
        pages_by_request.append(pages)
        requests.append({
            "request": b, "past_tokens": length, "query_tokens": q_len,
            "logical_token_interval": [first, stop], "logical_tokens": stop - first,
            "logical_page_interval": [pfirst, pstop], "page_count": len(pages),
            "physical_pages_sha256": _digest(pages), "physical_rows_sha256": _digest(rows),
            "new_physical_rows": expected_new[-1].tolist(),
        })
    new_rows = np.concatenate(expected_new)
    _require(np.array_equal(_cpu_i64(data["cache_loc"]).reshape(-1), new_rows),
             "cache_loc disagrees with causal new-token page mapping")
    if "cache_seqlens" in data:
        _require(np.array_equal(_cpu_i64(data["cache_seqlens"]).reshape(-1), seq + q_len),
                 "cache_seqlens disagrees with seq_lens+q_len")
    logical_rows = sum(len(rows) for rows in all_rows)
    rows = np.unique(np.concatenate(all_rows))
    ranges = _runs(rows)
    _require(int((ranges[:, 1] - ranges[:, 0]).sum()) == len(rows), "KV interval coverage mismatch")
    return rows, ranges, pages_by_request, {
        "requests": requests, "row_bytes_per_k_or_v": row_bytes,
        "logical_token_rows": logical_rows, "unique_physical_token_rows": len(rows),
        "shared_physical_token_references_deduplicated": logical_rows - len(rows),
        "logical_bytes_including_shared_references": logical_rows * row_bytes * 2,
        "physical_payload_bytes": len(rows) * row_bytes * 2,
        "unique_physical_pages": int(len(np.unique(rows // page))),
        "physical_pages_sha256": _digest(np.unique(rows // page)),
        "physical_rows_sha256": _digest(rows), "physical_ranges_sha256": _digest(ranges),
        "physical_range_count_per_cache": len(ranges),
        "all_new_tokens_in_union": bool(np.isin(new_rows, rows).all()),
        "causal_union": "[max(0,seq-window+1),seq+q_len) if window else [0,seq+q_len)",
        "coalescing": "sorted physically adjacent covered rows only; no holes or whole-cache shortcut",
    }


def make_plan(weights, data, output, state=None, representation="native", stage_bytes=32768):
    """Return CUDA int64 descriptors, independent CPU byte/XOR oracle and manifest.

    chunks[n,3] = (original source pointer, size, packed destination stage offset).
    stages[n,3] = (first chunk, chunk count, exact payload bytes), with no stage
    padding. All descriptor values are checked before upload. CPU ranges/pages
    are also returned for independent coverage inspection, not timed GPU reads.
    state is accepted for the harness API but no retained buffers are inferred.
    """
    if representation != "native":
        raise ValueError(_RETAINED_REASON)
    _require(isinstance(stage_bytes, int) and 0 < stage_bytes <= 32768 and stage_bytes % 16 == 0,
             "stage_bytes must be a positive multiple of 16 no larger than 32768")
    required = set(_DENSE + _VECTORS + _EXPERT + ("w13_gs", "w2_gs"))
    _require(set(weights) == required,
             f"unaccounted weight schema: missing={sorted(required-set(weights))}, extra={sorted(set(weights)-required)}")
    device = weights["qkv_gate"].device
    _require(device.type == "cuda", "stream plan requires CUDA source allocations")
    touched, routing = _routes(output, data)
    kv_rows, kv_ranges, kv_pages, kv_manifest = _kv_coverage(data)
    tokens = int(data["config"]["tokens"])
    keepalive, sources = [], []
    categories = {}

    def add(name, category, tensor, ranges=None, semantic_bytes=None, row_ids=None):
        _require(isinstance(tensor, torch.Tensor) and tensor.device == device,
                 f"{name}: source must be on {device}")
        _require(tensor.is_contiguous(), f"{name}: refusing a copying contiguous conversion")
        nbytes = tensor.numel() * tensor.element_size()
        raw = tensor.detach().reshape(-1).view(torch.uint8)
        intervals = [(0, nbytes)] if ranges is None else ranges
        physical = 0
        previous_end = 0
        for begin, end in intervals:
            begin, end = int(begin), int(end)
            _require(0 <= begin < end <= nbytes and begin >= previous_end, f"{name}: invalid/overlapping range")
            _require((tensor.data_ptr() + begin) % 16 == 0 and (end - begin) % 16 == 0,
                     f"{name}: source address and size must be 16-byte aligned")
            physical += end - begin
            previous_end = end
        semantic = physical if semantic_bytes is None else int(semantic_bytes)
        _require(0 <= semantic <= physical, f"{name}: semantic bytes exceed physical bytes")
        entry = {"name": name, "category": category, "shape": list(tensor.shape),
                 "dtype": str(tensor.dtype), "source_address": tensor.data_ptr(),
                 "allocation_view_bytes": nbytes, "range_count": len(intervals),
                 "physical_bytes": physical, "semantic_bytes": semantic,
                 "conservative_extra_bytes": physical - semantic}
        sources.append((raw, intervals, row_ids, entry))
        keepalive.append(tensor)
        totals = categories.setdefault(category, {"physical_bytes": 0, "semantic_bytes": 0,
                                                   "conservative_extra_bytes": 0, "source_count": 0})
        for key in ("physical_bytes", "semantic_bytes", "conservative_extra_bytes"):
            totals[key] += entry[key]
        totals["source_count"] += 1

    for name in _DENSE:
        tensor = weights[name]
        _require(tensor.dtype == (torch.float32 if name == "router" else torch.bfloat16),
                 f"{name}: wrong native weight precision")
        if name == "unpooler" and tokens <= 8:
            _require(tensor.shape[0] >= 1024, "unpooler missing small-T output slice")
            tensor = tensor[:1024]
        add(name, "dense." + name, tensor)
    for name in _VECTORS:
        add(name, "norm_gate_bias", weights[name])
    for name in _EXPERT:
        tensor = weights[name]
        _require(tensor.shape[0] == 32 and tensor.element_size() == 1,
                 f"{name}: expected packed NVFP4/FP8 scale rows for 32 experts")
        per_expert = tensor[0].numel() * tensor.element_size()
        intervals = [(int(e) * per_expert, (int(e) + 1) * per_expert) for e in touched]
        add(name, "experts." + name, tensor, intervals)
    for name in ("w13_gs", "w2_gs"):
        tensor = weights[name]
        _require(tensor.dtype == torch.float32 and tensor.numel() == 32,
                 f"{name}: expected 32 original FP32 global scales")
        add(name, "experts.global_scales", tensor, [] if not len(touched) else None,
            semantic_bytes=len(touched) * 4)
    for name in ("k_cache", "v_cache"):
        add(name, "kv." + name, data[name], kv_ranges * 128, row_ids=kv_rows)
    # Scalar copies are bounded static metadata, not a synthetic weight bank.
    scalar_packet = torch.zeros(16, dtype=torch.uint8, device=device)
    for offset, name in ((0, "k_scale"), (4, "v_scale")):
        tensor = data[name]
        _require(tensor.device == device and tensor.dtype == torch.float32 and tensor.numel() == 1,
                 f"{name}: expected one FP32 scale on source device")
        scalar_packet[offset:offset + 4].copy_(tensor.detach().reshape(-1).view(torch.uint8))
        keepalive.append(tensor)
    add("kv_scale_static_packet", "kv.scales", scalar_packet, semantic_bytes=8)

    chunks, stages, physical_ranges = [], [], []
    stage_first = 0
    stage_used = 0
    expected_bytes = 0
    expected_xor = 0
    for raw, intervals, row_ids, entry in sources:
        source_xor = 0
        # CPU oracle reads source ranges independently of descriptor construction.
        if row_ids is not None:
            rows_per_batch = _HASH_BYTES // 128
            byte_rows = raw.view(-1, 128)
            for at in range(0, len(row_ids), rows_per_batch):
                selected = torch.from_numpy(row_ids[at:at + rows_per_batch]).to(device=device)
                cpu = byte_rows.index_select(0, selected).cpu()
                source_xor ^= _xor(cpu)
        else:
            for begin, end in intervals:
                for at in range(int(begin), int(end), _HASH_BYTES):
                    source_xor ^= _xor(raw[at:min(int(end), at + _HASH_BYTES)].cpu())
        entry["expected_xor_u32"] = source_xor
        expected_xor ^= source_xor
        source_chunk_bytes = 0
        for begin, end in intervals:
            address = raw.data_ptr() + int(begin)
            remaining = int(end) - int(begin)
            physical_ranges.append((address, address + remaining, entry["name"]))
            while remaining:
                count = min(remaining, stage_bytes - stage_used)
                chunks.append((address, count, stage_used))
                source_chunk_bytes += count
                address += count
                remaining -= count
                stage_used += count
                if stage_used == stage_bytes:
                    stages.append((stage_first, len(chunks) - stage_first, stage_used))
                    stage_first = len(chunks)
                    stage_used = 0
        _require(source_chunk_bytes == entry["physical_bytes"], "source descriptor coverage mismatch")
        expected_bytes += source_chunk_bytes
    if stage_used:
        stages.append((stage_first, len(chunks) - stage_first, stage_used))
    ordered = sorted(physical_ranges)
    for previous, current in zip(ordered, ordered[1:]):
        _require(previous[1] <= current[0], f"duplicate physical bytes: {previous[2]} / {current[2]}")
    consumed_chunks = 0
    for first, count, total in stages:
        _require(first == consumed_chunks and count > 0, "noncontiguous stage chunk headers")
        offset = 0
        for address, size, destination in chunks[first:first + count]:
            _require(address % 16 == 0 and size > 0 and size % 16 == 0 and destination == offset,
                     "invalid source alignment or stage packing")
            offset += size
        _require(offset == total and total <= stage_bytes, "stage byte count mismatch")
        consumed_chunks += count
    _require(consumed_chunks == len(chunks), "unreferenced chunks")
    _require(sum(stage[2] for stage in stages) == sum(chunk[1] for chunk in chunks) == expected_bytes,
             "stage/chunk/global byte totals disagree")
    chunks_cpu = np.asarray(chunks, dtype=np.int64).reshape(-1, 3)
    stages_cpu = np.asarray(stages, dtype=np.int64).reshape(-1, 3)
    chunks_cuda = torch.from_numpy(chunks_cpu).to(device=device)
    stages_cuda = torch.from_numpy(stages_cpu).to(device=device)
    keepalive.extend((chunks_cuda, stages_cuda))
    semantic_bytes = sum(entry[3]["semantic_bytes"] for entry in sources)
    manifest = {
        "format": "native_public_weights_and_fp8_kv", "representation": representation,
        "oracle_only": True, "preprocessing_timed": False,
        "retained_representation": {"supported": False, "reason": _RETAINED_REASON},
        "tokens": tokens, "stage_capacity_bytes": stage_bytes,
        "stage_count": len(stages), "chunk_count": len(chunks),
        "expected_bytes": expected_bytes, "expected_xor_u32": expected_xor,
        "physical_bytes": expected_bytes, "semantic_bytes": semantic_bytes,
        "conservative_extra_bytes": expected_bytes - semantic_bytes,
        "semantic_byte_definition": "unique original needed payload; shared KV physical rows counted once; all norm/gate/bias vectors included",
        "categories": categories, "sources": [source[3] for source in sources],
        "routing": routing, "kv": kv_manifest,
        "unpooler_rows": 1024 if tokens <= 8 else int(weights["unpooler"].shape[0]),
        "expert_global_scales": "full original 32-entry array per projection if any local expert; inactive entries conservative extras",
        "metadata": {
            "kv_scale_packet": {"physical_bytes": 16, "semantic_bytes": 8, "zero_padding_bytes": 8,
                                "layout": "k_scale raw FP32, v_scale raw FP32, eight zero bytes",
                                "provenance": "bounded lossless static copy of fixture scalar tensors"},
            "page_table_payload_bytes": 0, "route_id_payload_bytes": 0,
            "ordinary_fixture_metadata_payload_bytes": 0,
            "descriptor_storage_bytes": int(chunks_cpu.nbytes + stages_cpu.nbytes),
            "descriptor_traffic": "kernel loads headers in addition to payload; count not included in expected_bytes; DRAM traffic requires NCU",
            "oracle_metadata_excluded": ["page_table", "seq_lens", "cache_seqlens", "positions",
                                         "cache_loc", "cu_seqlens_q", "local_start", "scheduler_metadata", "actual routing ids"],
        },
        "coverage": {"all_weight_keys_accounted": True, "all_routed_local_experts_included": True,
                     "both_expert_codes_block_scales_global_scales": True,
                     "full_causal_kv_union": True, "all_new_tokens_included": kv_manifest["all_new_tokens_in_union"],
                     "duplicate_physical_ranges": 0, "no_out_of_allocation_reads": True,
                     "all_chunks_16_byte_aligned": True, "stage_offsets_contiguous": True,
                     "stage_sum_equals_chunk_sum": True, "stage_padding_bytes": 0},
        "hash": {"algorithm": "XOR of every little-endian uint32 physical payload word",
                 "payload_transfer_batch_bytes": _HASH_BYTES,
                 "method": "independent bounded raw CPU range copies; KV batched uint8 gather over exact physical row union"},
        "idealizations": ["routing and page oracle is free", "unique weights read once per layer",
                          "baseline has already written new KV", "no real math or activation traffic",
                          "no cache-write or inter-stage dependency cost", "no compression credit"],
    }
    return {"chunks": chunks_cuda, "stages": stages_cuda, "keepalive": keepalive,
            "expected_bytes": expected_bytes, "expected_xor_u32": expected_xor,
            "manifest": manifest, "chunks_cpu": chunks_cpu, "stages_cpu": stages_cpu,
            "kv_rows_cpu": kv_rows, "kv_ranges_cpu": kv_ranges, "kv_pages_cpu": kv_pages}
