"""Shared external-only Round4 fixture/measurement contract; no candidate code."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

SOURCE_SHA = '053dbacc8f0de4e6111ccb89cb91f629840852cf482a6ca423fe0eaff419c086'
WEIGHT_SEEDS = (731283, 731284, 731285)
LOADS = (('D32', 32, 1), ('V16', 16, 8), ('V32', 32, 8))
LAYERS = ('sliding', 'full')
FLUSH_BYTES = 256 * 1024 * 1024
WEIGHT_BYTES_PER_JOB = 28311552
REL_L2_TOL = .015


def save(path, report):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def descriptors():
    rows = []
    for seed_index, weight_seed in enumerate(WEIGHT_SEEDS):
        for load_name, batch, q_len in LOADS:
            for layer in LAYERS:
                input_seed = 919233 + batch * 100 + q_len * 1000 + 10000 * (layer == 'full') + seed_index + 1
                case_id = f'r4-{load_name}-{layer}-w{weight_seed}-i{input_seed}'
                rows.append(dict(case_id=case_id, load=load_name, batch=batch, q_len=q_len,
                    m=batch*q_len, layer=layer, ctx=80000, page=1, spread='agentic', dist='soft',
                    weight_seed=weight_seed, input_seed=input_seed,
                    fixture_file=case_id + '.pt'))
    return rows


def plan(counts):
    if len(counts) != 32 or any(n < 0 for n in counts):
        raise ValueError('Expected32 nonnegative local routing counts')
    k = sum(n > 0 for n in counts)
    j64 = sum((n + 63) // 64 for n in counts)
    j128 = sum((n + 127) // 128 for n in counts)
    offsets, carry = [], 0
    for count in counts:
        offsets.append(carry)
        carry += count
    # Exact source ordering: fat experts in ID order, then skinny experts.
    jobs = [dict(expert=e, count=min(64, n-start), first_pair=offsets[e]+start)
            for e, n in enumerate(counts) if n > 8 for start in range(0, n, 64)]
    jobs += [dict(expert=e, count=n, first_pair=offsets[e])
             for e, n in enumerate(counts) if 0 < n <= 8]
    if len(jobs) != j64:
        raise RuntimeError('Job arithmetic disagrees with source route ordering')
    padded = sum((job['count'] + 7)//8*8 for job in jobs)
    return dict(n_e=counts, K=k, R=sum(counts), J64=j64, J128=j128,
        D=j64-k, q=(j64-k)/j64 if j64 else 0.0,
        jobs=jobs, job_list_origin='Host reconstruction of unchanged GPU route(), from captured actual layer routing',
        bytes_per_complete_expert_stream=WEIGHT_BYTES_PER_JOB,
        unique_weight_bytes=k*WEIGHT_BYTES_PER_JOB,
        planned_weight_bytes=j64*WEIGHT_BYTES_PER_JOB,
        duplicated_planned_weight_bytes=(j64-k)*WEIGHT_BYTES_PER_JOB,
        J128_removed_weight_bytes=(j64-j128)*WEIGHT_BYTES_PER_JOB,
        J128_weight_fraction=(j64-j128)/j64 if j64 else 0.0,
        padded_route_rows=padded,
        planned_x_tma_bytes=padded*393216,
        planned_x_note='Requested activation payload over48 units x16 stages x512 bytes per padded row; largely L2-resident, NOT an HBM prediction')


def load_fixture(path, device='cuda'):
    import torch
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['source_sha256'] != SOURCE_SHA:
        raise RuntimeError('Frozen fixture source mismatch')
    tensors = {key: saved[key].to(device).contiguous() for key in ('x','ids','tw','expected','gold')}
    return saved, tensors


def expert_fn(module, state, tensors, extension=None, debug=None):
    import torch
    m = tensors['x'].shape[0]
    native = extension if extension is not None else module._moe(m, state)
    native.okc_set_pdl(0)  # Isolated stage, no predecessor/consumer PDL overlap.
    scratch = list(state['ma_scratch']) + ([] if debug is None else [debug])
    def call():
        out = torch.empty_like(tensors['x'])
        native.okc_run(state['ma_w'], scratch, tensors['x'], tensors['ids'], tensors['tw'],
                       out, 0, 32, state['nsm'])
        return out
    return call


def capture(fn, flush=None):
    import torch
    current, stream = torch.cuda.current_stream(), torch.cuda.Stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    current.wait_stream(stream)
    begin = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    begin.record(); end.record(); torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream.wait_stream(current)
    with torch.cuda.graph(graph, stream=stream):
        sink = flush.sum() if flush is not None else None
        begin.record()
        out = fn()
        end.record()
    current.wait_stream(stream)
    for _ in range(3):
        graph.replay()
    end.synchronize()
    return dict(graph=graph, output=out, begin=begin, end=end, sink=sink)


def measure(captured):
    captured['graph'].replay()
    captured['end'].synchronize()
    us = captured['begin'].elapsed_time(captured['end']) * 1000
    if not math.isfinite(us) or us <= 0:
        raise RuntimeError('Invalid internal-event duration')
    return us
