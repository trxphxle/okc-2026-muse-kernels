"""Unchanged-layer timing/path census for the joint SK2 deletion diagnostic.

Never submit. No deletion, frozen-output binding, or numerical change here.
"""
import argparse
import json
from pathlib import Path
import statistics
import time
import traceback

import torch
import reference
import compare_decode_public as bench

bench.torch = torch
bench.reference = reference
LOADS = (('D1', 1, 1), ('D16', 16, 1), ('D32', 32, 1),
         ('V1', 1, 8), ('V16', 16, 8), ('V32', 32, 8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='dense_joint_control.py')
    parser.add_argument('--output', default='dense-joint-baseline.json')
    parser.add_argument('--seed', type=int, default=731283)
    args = parser.parse_args()
    out = Path(args.output)
    report = dict(kind='unchanged_joint_dense_baseline', submittable=False,
                  source_sha256=bench.sha(args.baseline), seed=args.seed,
                  results=[], status='running')
    started = time.monotonic()
    bench.save(out, report)
    try:
        module = bench.load(args.baseline, 'joint_baseline')
        weights = reference.generate_weights(args.seed)
        prepare_start = time.monotonic()
        state = module.prepare(weights)
        report['prepare_seconds'] = time.monotonic() - prepare_start
        report['choices'] = bench.metadata({k: state.get(k) for k in
                                           ('mm_pick', 'moe_pick', 'compose_graft_enabled')})
        report['bfp_choices'] = bench.metadata(state['bfp'].get('choice')) if state.get('bfp') else None
        if state.get('bfp') is None:
            raise RuntimeError('BFP backend disabled; cannot measure requested SK2 deletions')
        originals = {name: getattr(module, name) for name in
                     ('bfp__run_sk2', 'bfp_fused_pooler_router', '_gle_frt')}
        calls = []
        recording = False

        def observe(name, fn):
            def wrapped(*pos, **kw):
                result = fn(*pos, **kw)
                if recording:
                    x = pos[0]
                    item = dict(function=name, input_shape=list(x.shape), input_dtype=str(x.dtype))
                    if name == 'bfp__run_sk2':
                        _, layer, m, n, k, bst = pos[:6]
                        key = module.bfp__sk2_name(layer, m)
                        stream, table = bst['sk2'][key][1:3]
                        item.update(layer=layer, m=m, n=n, k=k, plan=key,
                                    family='narrow' if layer == 'qkv_gate' and 2 <= m <= 8 else 'main',
                                    weight_stream_bytes=stream.numel() * stream.element_size(),
                                    escape_table_bytes=table.numel() * table.element_size())
                    else:
                        item.update(layer='pooler_router', family='gle' if name == '_gle_frt' else 'main_fused')
                    calls.append(item)
                return result
            return wrapped

        for name, fn in originals.items():
            setattr(module, name, observe(name, fn))
        flush = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
        for load, batch, length in LOADS:
            for layer in ('sliding', 'full'):
                seed = 919233 + batch * 100 + length * 1000 + (layer == 'full') * 10000
                data = reference.generate_input(batch=batch, q_len=length, ctx=80000, layer=layer,
                                                page=1, spread='agentic' if batch > 1 else 'fixed',
                                                dist='soft', seed=seed)
                eager = module.kernel_fn(data, state)
                counts = torch.bincount((eager[2].long() // 32).flatten(), minlength=8)
                data['local_start'].fill_(32 * int(counts.argmax()))
                pristine = bench.clone_data(data)
                calls.clear()
                recording = True
                eager = module.kernel_fn(data, state)
                recording = False
                row = dict(load=load, layer=layer, seed=seed, calls=list(calls),
                           oracle=bench.oracle(data, eager, weights, pristine), samples=[])
                report['results'].append(row)
                if not row['oracle']['passed']:
                    raise RuntimeError(f'Unchanged baseline oracle failed: {load}/{layer}')
                graphs = {name: bench.capture(module.kernel_fn, data, state, flush)
                          for name in ('base', 'repeat')}
                for sample in range(11):
                    order = ('base', 'repeat') if sample % 2 == 0 else ('repeat', 'base')
                    row['samples'].append({name: bench.measure(graphs[name]) for name in order})
                row['median_us'] = {name: statistics.median(s[name] for s in row['samples']) for name in graphs}
                row['aa_ratio'] = statistics.median(s['repeat'] / s['base'] for s in row['samples'])
                row['routes'] = bench.routes(graphs['base']['output'], int(data['local_start']))
                print(json.dumps({k: row[k] for k in ('load', 'layer', 'median_us', 'aa_ratio', 'calls')}), flush=True)
                bench.save(out, report)
                del pristine, data, eager, graphs
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        bench.save(out, report)


if __name__ == '__main__':
    main()
