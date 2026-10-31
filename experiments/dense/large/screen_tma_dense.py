"""Paired large-T-only diagnostic for tma_dense.py versus unchanged alygg77.
Capture/check helpers are imported from the lane's full-board screen. Timings
must never be called a complete-board speedup until unchanged loads are included.
"""
import argparse
import hashlib
import json
import statistics
import time
import traceback
from pathlib import Path
import torch
import screen_dense as screen
import tma_dense as tma
import reference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='dense-tma-screen.json')
    parser.add_argument('--pairs',type=int,default=18)
    parser.add_argument('--seed',type=int,default=5151)
    args=parser.parse_args()
    out=Path(args.output)
    report=dict(status='running',official_score=False,diagnostic_only=True,weight_seed=args.seed,
                rows=[],artifact_files=['dense_tma_generated.cu'],
                timing='captured external events after256MiB cold flush; no subtraction',
                configs=tma.CONFIGS,errors=[])
    begin=time.monotonic()
    try:
        base=screen.load(Path('muse_bf16_proj.py'),'tma_baseline')
        weights=reference.generate_weights(args.seed)
        before=time.monotonic()
        state=base.prepare(weights)
        torch.cuda.synchronize()
        report['baseline_prepare_seconds']=time.monotonic()-before
        screen.save(out,report)
        before=time.monotonic()
        ext=tma.build()
        report['compile_seconds']=time.monotonic()-before
        report['resources']={str(i):dict(zip(('shared_bytes','registers','local_bytes','resident_ctas_per_sm'),ext.dense_info(i))) for i in range(len(tma.CONFIGS))}
        report['source_sha256']=hashlib.sha256(tma.source().encode()).hexdigest()
        screen.save(out,report)
        flush=torch.ones(256*1024*1024//4,device='cuda',dtype=torch.float32)
        for m in (128,256):
            for layer in ('qkv_gate','o','pooler','unpooler'):
                initial=reference.generate_input(m,layer,args.seed+10000+m)
                # The measured raw roof chooses full-K geometry, not K splits.
                # Skip CL2 when it would add wholly padded token tiles.
                configs=[i for i,(bn,wg,st,cl) in enumerate(tma.CONFIGS)
                         if bn*cl<=m and (bn!=32 or layer=='qkv_gate')]
                row=dict(m=m,layer=layer,configs=configs,status='running',records=[])
                report['rows'].append(row)
                screen.save(out,report)
                graphs={}
                functions={'base':lambda d,s:base.kernel_fn(d,s),
                           'base_repeat':lambda d,s:base.kernel_fn(d,s)}
                for cfg in configs:
                    functions[str(cfg)]=lambda d,s,cfg=cfg:tma.run(ext,cfg,weights[layer],d['x'])
                for name,fn in functions.items():
                    data={'x':initial['x'].clone(),'config':dict(initial['config'])}
                    graphs[name]=screen.capture(fn,data,state,flush)
                for rep in range(args.pairs):
                    fresh=reference.generate_input(m,layer,args.seed+20000+m*100+rep)
                    gold=reference.ref_kernel(fresh,weights)
                    for data,*_ in graphs.values():
                        data['x'].copy_(fresh['x'])
                    # Every candidate paired with immediate base/control; alternate order.
                    rec=dict(rep=rep,variants={})
                    row['records'].append(rec)
                    for cfg in (configs if rep%2==0 else list(reversed(configs))):
                        name=str(cfg)
                        order=('base',name,'base_repeat') if rep%2==0 else ('base_repeat',name,'base')
                        times={}
                        checks={}
                        for arm in order:
                            _,graph,_,a,b,_=graphs[arm]
                            graph.replay(); b.synchronize()
                            times[arm]=a.elapsed_time(b)*1000
                        for arm in order:
                            checks[arm]=screen.check(reference,fresh,graphs[arm][2],gold)
                            if not checks[arm]['passed']:
                                raise RuntimeError(f'{layer} m{m} arm{arm}: {checks[arm]}')
                        rec['variants'][name]=dict(us=times,checks=checks,
                            ratio=times['base']/times[name],control=times['base']/times['base_repeat'])
                row['variants']={}
                for cfg in configs:
                    name=str(cfg)
                    entries=[r['variants'][name] for r in row['records']]
                    row['variants'][name]=dict(config=tma.CONFIGS[cfg],
                        median_us=statistics.median(x['us'][name] for x in entries),
                        base_median_us=statistics.median(x['us']['base'] for x in entries),
                        median_ratio=statistics.median(x['ratio'] for x in entries),
                        control_ratio=statistics.median(x['control'] for x in entries))
                row['status']='passed'
                row['best_cfg']=max(row['variants'],key=lambda key:row['variants'][key]['median_ratio'])
                print(json.dumps(dict(m=m,layer=layer,best_cfg=row['best_cfg'],variants=row['variants'])),flush=True)
                screen.save(out,report)
                graphs.clear()
        report['status']='complete'
    except BaseException as exc:
        report['status']='failed';report['errors'].append(traceback.format_exc())
        print(repr(exc),flush=True)
    finally:
        report['elapsed_seconds']=time.monotonic()-begin
        screen.save(out,report)
        print(json.dumps(dict(status=report['status'],output=str(out))),flush=True)
    return int(report['status']=='failed')


if __name__=='__main__':
    raise SystemExit(main())
