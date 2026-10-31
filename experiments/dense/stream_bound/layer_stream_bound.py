"""Ceiling-only continuous layer-byte diagnostic; NEVER a submission.

Free-oracle selection of actual routed expert/page ranges is outside timing.
One persistent stream reads native weight bytes+KV with no layer math, dependency
barriers, output generation or KV writes. Current decoder is unmodified.
Captured external-event spans and independently profiled native spans remain
separate; no replay-cost subtraction. D1/V1 replacement keeps four other loads
unchanged in the six-load geomean. Optional D16 is measured, never extrapolated.

Uses the approved compare_decode_public.py timing/reference/cache conventions.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import torch
import reference

LOADS={'D1':(1,1),'V1':(1,8),'D16':(16,1)}
WEIGHTS={'sliding':46,'full':16}


def save(path,report):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def clone(data):
    return {k:v.clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v) for k,v in data.items()}


def capture(call,flush=None):
    current=torch.cuda.current_stream();stream=torch.cuda.Stream();stream.wait_stream(current)
    a=torch.cuda.Event(enable_timing=True,external=True);b=torch.cuda.Event(enable_timing=True,external=True)
    with torch.cuda.stream(stream):
        for _ in range(3):call()
        a.record();b.record()
    current.wait_stream(stream);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph();sink=None
    with torch.cuda.graph(graph,stream=stream):
        if flush is not None:sink=flush.sum()
        a.record();out=call();b.record()
    current.wait_stream(stream)
    result=dict(graph=graph,a=a,b=b,out=out,sink=sink,stream=stream)
    for _ in range(5):measure(result)
    return result


def measure(cap):
    cap['graph'].replay();cap['b'].synchronize();us=cap['a'].elapsed_time(cap['b'])*1000
    if not math.isfinite(us) or us<=0:raise RuntimeError('Invalid captured time')
    return us


def native(call,flush,path):
    cap=capture(call)
    flush.sum();torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
        cap['graph'].replay();torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    events=[e for e in json.loads(Path(path).read_text())['traceEvents'] if e.get('cat')=='kernel' and e.get('ph')=='X']
    if not events:raise RuntimeError('Native graph trace has no GPU kernels')
    lo=min(e['ts'] for e in events);hi=max(e['ts']+e['dur'] for e in events)
    return dict(span_us=hi-lo,kernel_count=len(events),summed_kernel_us=sum(e['dur'] for e in events),
                launches=[dict(name=e['name'],start_us=e['ts']-lo,duration_us=e['dur'],args=e.get('args',{})) for e in events],
                trace=Path(path).name)


def oracle(pristine,data,output,weights):
    raw=reference.collect(data,output)
    ok,message=reference.check_implementation(pristine,raw,weights)
    if not ok:raise RuntimeError('Current decoder failed public reference: '+str(message))
    return dict(passed=True,message=str(message))


def stream_check(plan,sink,mode):
    rows=sink.cpu().tolist()
    completed=sum(r[0] for r in rows);xor=0
    for r in rows:xor^=int(r[1])&0xffffffff
    stages=sum(r[2] for r in rows);chunks=sum(r[3] for r in rows)
    good=completed==plan['expected_bytes'] and stages==len(plan['stages']) and chunks==len(plan['chunks'])
    if mode:good=good and xor==plan['expected_xor_u32']
    per_cta=True
    for block,row in enumerate(rows):
        owned=plan['stages_cpu'][block::len(rows)]
        want=[int(owned[:,2].sum()),len(owned),int(owned[:,1].sum())]
        per_cta=per_cta and [row[0],row[2],row[3]]==want
    good=good and per_cta
    result=dict(passed=good,completed_bytes=completed,completed_stages=stages,completed_chunks=chunks,
                actual_xor_u32=xor,expected_xor_u32=plan['expected_xor_u32'],mode=mode,
                per_cta_counters_pass=per_cta,per_cta_counters=rows)
    if not good:raise RuntimeError('Byte stream validation failed: '+str(result))
    return result


def stream_smoke(ext,flatten=None):
    """Exercise real TMA boundary/ring cases, not a wiring-only unit test."""
    import numpy as np
    rng=np.random.default_rng(87123)
    checks=[]
    for sizes in ([],[16],[32768 if i%3 else 128 for i in range(22)]+[16]):
        original=rng.integers(0,256,size=sum(sizes),dtype=np.uint8)
        source=torch.from_numpy(original).cuda()
        chunks=[];stages=[];offset=0
        for index,size in enumerate(sizes):
            first=len(chunks);destination=0
            packet=128 if index%2 else 16384
            while destination<size:
                count=min(packet,size-destination)
                chunks.append((source.data_ptr()+offset+destination,count,destination))
                destination+=count
            stages.append((first,len(chunks)-first,size));offset+=size
        chunks_cpu=np.array(chunks,dtype=np.int64).reshape(-1,3)
        stages_cpu=np.array(stages,dtype=np.int64).reshape(-1,3)
        plan=dict(chunks=torch.from_numpy(chunks_cpu).cuda(),stages=torch.from_numpy(stages_cpu).cuda(),
                  stages_cpu=stages_cpu,chunks_cpu=chunks_cpu,expected_bytes=len(original),
                  expected_xor_u32=int(np.bitwise_xor.reduce(original.view('<u4'),initial=np.uint32(0))))
        if flatten is not None:
            plan=flatten(plan)
            sizes=plan['chunks_cpu'][:,1].tolist()
        sink=torch.full((4,4),-1,device='cuda',dtype=torch.int64)
        for mode in (0,1):
            ext.stream_launch(plan['chunks'].data_ptr(),plan['stages'].data_ptr(),sink.data_ptr(),0,len(sizes),4,mode,torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            result=stream_check(plan,sink,mode)
            if mode:
                starts=np.concatenate(([0],np.cumsum(sizes)))
                for block in range(4):
                    expected=0
                    for stage in range(block,len(sizes),4):
                        expected^=int(np.bitwise_xor.reduce(original[starts[stage]:starts[stage+1]].view('<u4'),initial=np.uint32(0)))
                    if int(sink[block,1])!=expected:raise RuntimeError('Per-CTA XOR mismatch')
            checks.append(dict(stage_count=len(sizes),mode=mode,passed=result['passed']))
    return checks


def summarize(report):
    report['loads']={}
    for name in report['requested_loads']:
        cases=[r for r in report['rows'] if r['load']==name]
        if len(cases)!=2 or any(r['status']!='complete' for r in cases):continue
        medians={}
        for layer in WEIGHTS:
            row=next(r for r in cases if r['layer']==layer)
            keys=row['fixtures'][0]['event_median_us'].keys()
            row['event_median_us']={k:statistics.mean(f['event_median_us'][k] for f in row['fixtures']) for k in keys}
            row['aa_ratio']=statistics.median(f['aa_ratio'] for f in row['fixtures'])
            row['aa_within_one_percent']=abs(row['aa_ratio']-1)<=.01
            row['native_mean_us']={k:statistics.mean(f['native'][k]['span_us'] for f in row['fixtures'])
                                    for k in row['fixtures'][0].get('native',{})}
            medians[layer]=row
        keys=cases[0]['event_median_us'].keys()
        weighted={k:sum(WEIGHTS[r['layer']]*r['event_median_us'][k] for r in cases)/62 for k in keys}
        record=dict(weighted_event_us=weighted,aa_ratio=weighted['base']/weighted['base_repeat'],
                    aa_cases_within_one_percent=all(r['aa_within_one_percent'] for r in cases))
        if all('native' in f for r in cases for f in r['fixtures']):
            native_keys=cases[0]['native_mean_us'].keys()
            record['weighted_native_us']={k:sum(WEIGHTS[r['layer']]*r['native_mean_us'][k] for r in cases)/62 for k in native_keys}
        for metric,key in (('event','weighted_event_us'),('native','weighted_native_us')):
            if key in record and 'stream' in record[key]:
                b,s=record[key]['base'],record[key]['stream']
                record[metric+'_gap_us']=b-s;record[metric+'_ideal_ratio']=b/s
        report['loads'][name]=record
    report['ceilings']={}
    for selected in (('D1','V1'),('D1','V1','D16')):
        if not all(s in report['loads'] for s in selected):continue
        for metric in ('event','native'):
            key=metric+'_ideal_ratio'
            if all(key in report['loads'][s] for s in selected):
                ratio=math.prod(report['loads'][s][key] for s in selected)**(1/6)
                report['ceilings']['+'.join(selected)+'_'+metric]=dict(geomean_ratio=ratio,
                    improvement_percent=(ratio-1)*100,passes_five_percent=ratio>=1.05,
                    unchanged_load_count=6-len(selected),scope='zero-math native-byte oracle bound, NOT achievable implementation gain')
    if report['loads']:
        report['aa_accepted']=all(v['aa_cases_within_one_percent'] and abs(v['aa_ratio']-1)<=.01 for v in report['loads'].values())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',default='decode_stream_base.py')
    parser.add_argument('--output',default='layer-stream-bound.json')
    parser.add_argument('--weight-seed',type=int,default=5353)
    parser.add_argument('--fixtures',type=int,default=3)
    parser.add_argument('--samples',type=int,default=24)
    parser.add_argument('--loads',default='D1,V1,D16')
    parser.add_argument('--baseline-only',action='store_true')
    parser.add_argument('--ncu-child',action='store_true')
    parser.add_argument('--stream-kind',choices=('grouped','flat'),default='grouped')
    args=parser.parse_args();output=Path(args.output)
    report=dict(kind='whole_layer_continuous_TMA_ceiling',status='running',official_score=False,production_candidate=False,
                requested_loads=args.loads.split(','),weight_seed=args.weight_seed,reader_kind=args.stream_kind,rows=[],artifact_files=[],errors=[],
                representation='native original precision; unique actual local experts and union of required KV reads',
                omissions=['routing computation/page selection is a free frozen-fixture oracle','no arithmetic or inter-stage dependencies',
                           'no layer outputs or KV writes','activation intermediates, output traffic and synchronization not modeled',
                           'native BF16 weight bytes are not baseline compressed physical traffic'],
                timing='matched external events after256MiB cold flush; native profiler spans independent; no subtraction')
    started=time.monotonic()
    try:
        report['sources']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (args.baseline,__file__,'reference.py','checks.py','nvfp4.py')}
        report['environment']=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
            clocks=subprocess.check_output(['nvidia-smi','--query-gpu=name,clocks.sm,clocks.mem,power.limit','--format=csv'],text=True))
        save(output,report)
        base=load(args.baseline,'_stream_bound_base')
        weights=reference.generate_weights(args.weight_seed)
        prepare_weights={k:v.clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v) for k,v in weights.items()}
        before=time.monotonic();state=base.prepare(prepare_weights);torch.cuda.synchronize()
        report['prepare_seconds']=time.monotonic()-before
        report['router_bias_unchanged']=bool(torch.equal(prepare_weights['router_bias'],weights['router_bias']))
        print(json.dumps(dict(stage='prepared',seconds=report['prepare_seconds'])),flush=True)
        flush=torch.ones(256*1024*1024//4,device='cuda',dtype=torch.float32)
        if not args.baseline_only:
            if args.stream_kind=='flat':
                import stream_flat as stream_kernel
                import stream_flat as stream_plan
            else:
                import stream_kernel
                import stream_plan
            before=time.monotonic();ext=stream_kernel.build();report['stream_compile_seconds']=time.monotonic()-before
            report['stream_resources']={str(mode):ext.stream_info(mode) for mode in (0,1)}
            for path in (Path(stream_kernel.__file__),Path(stream_kernel.__file__).with_suffix('.cu'),Path('stream_kernel.py'),Path('stream_plan.py')):
                report['sources'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            report['stream_smoke']=stream_smoke(ext,getattr(stream_kernel,'flatten',None))
            blocks=torch.cuda.get_device_properties(0).multi_processor_count
            sink=torch.empty((blocks,4),device='cuda',dtype=torch.int64)
            stamps=torch.empty((blocks,4),device='cuda',dtype=torch.int64)
        save(output,report)
        for name in report['requested_loads']:
            batch,q_len=LOADS[name]
            for layer in WEIGHTS:
                shape=dict(batch=batch,q_len=q_len,ctx=80000,layer=layer,page=1,
                           spread='agentic' if batch>1 else 'fixed',dist='soft')
                row=dict(load=name,layer=layer,shape=shape,status='running',fixtures=[])
                report['rows'].append(row);save(output,report)
                for fixture in range(args.fixtures):
                    seed=919233+batch*100+q_len*1000+(layer=='full')*10000+fixture+1
                    pristine=reference.generate_input(seed=seed,**shape)
                    data=clone(pristine)
                    probe=base.kernel_fn(data,state)
                    counts=torch.bincount((probe[2].long()//32).flatten(),minlength=8)
                    local_start=32*int(counts.argmax());pristine['local_start'].fill_(local_start);data['local_start'].fill_(local_start)
                    result=base.kernel_fn(data,state)
                    frozen_ids=result[2].clone()
                    rec=dict(seed=seed,local_start=local_start,ids=frozen_ids.cpu().tolist(),status='running',timing_samples=[])
                    row['fixtures'].append(rec)
                    rec['reference']=oracle(pristine,data,result,weights)
                    calls={'base':lambda:base.kernel_fn(data,state),'base_repeat':lambda:base.kernel_fn(data,state)}
                    if not args.baseline_only:
                        before=time.monotonic()
                        plan=stream_plan.make_plan(weights,data,result,state,representation='native')
                        rec['plan_seconds']=time.monotonic()-before;rec['manifest']=plan['manifest']
                        rec['expected_bytes']=plan['expected_bytes'];rec['expected_xor_u32']=plan['expected_xor_u32']
                        def stream_call(mode=0):
                            ext.stream_launch(plan['chunks'].data_ptr(),plan['stages'].data_ptr(),sink.data_ptr(),stamps.data_ptr(),len(plan['stages']),blocks,mode,torch.cuda.current_stream().cuda_stream)
                            return sink
                        stream_call(1);torch.cuda.synchronize();rec['checksum_check']=stream_check(plan,sink,1)
                        calls['stream']=stream_call
                    if args.ncu_child:
                        flush.sum();torch.cuda.synchronize()
                        torch.cuda.profiler.start();calls['stream']();torch.cuda.synchronize();torch.cuda.profiler.stop()
                        rec['byte_check']=stream_check(plan,sink,0)
                        rec['status']='ncu_profiled';save(output,report)
                        print(json.dumps(dict(load=name,layer=layer,fixture=fixture,expected_bytes=plan['expected_bytes'])),flush=True)
                        del plan,result,probe,data,pristine,frozen_ids;gc.collect()
                        continue
                    caps={k:capture(fn,flush) for k,fn in calls.items()}
                    keys=list(calls)
                    for rep in range(args.samples):
                        order=keys[rep%len(keys):]+keys[:rep%len(keys)]
                        if (rep//len(keys))%2:order=list(reversed(order))
                        times={k:measure(caps[k]) for k in order}
                        rec['timing_samples'].append(times)
                    rec['event_median_us']={k:statistics.median(r[k] for r in rec['timing_samples']) for k in keys}
                    rec['aa_ratio']=statistics.median(r['base']/r['base_repeat'] for r in rec['timing_samples'])
                    if not torch.equal(caps['base']['out'][2],frozen_ids):raise RuntimeError('Routing changed on frozen fixture')
                    rec['reference_after']=oracle(pristine,data,caps['base']['out'],weights)
                    if not args.baseline_only:rec['byte_check']=stream_check(plan,sink,0)
                    rec['native']={}
                    for arm in ('base',) if args.baseline_only else ('base','stream'):
                        path=output.with_name(output.stem+f'-{name}-{layer}-{fixture}-{arm}-native.json')
                        rec['native'][arm]=native(calls[arm],flush,path)
                        report['artifact_files'].append(path.name)
                        if arm=='stream' and rec['native'][arm]['kernel_count']!=1:raise RuntimeError('Stream is not a single native kernel')
                    if not args.baseline_only:
                        stream_call(1);torch.cuda.synchronize();rec['checksum_after']=stream_check(plan,sink,1)
                    rec['status']='complete';save(output,report)
                    print(json.dumps(dict(load=name,layer=layer,fixture=fixture,us=rec['event_median_us'],aa=rec['aa_ratio'],bytes=rec.get('expected_bytes'))),flush=True)
                    caps.clear();calls.clear()
                    if not args.baseline_only:del plan
                    del result,probe,data,pristine,frozen_ids;gc.collect()
                row['status']='complete';
                if not args.ncu_child:summarize(report)
                save(output,report)
        report['status']='complete'
        report['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
    except BaseException:
        report['status']='failed';report['errors'].append(traceback.format_exc());raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started;save(output,report)
        print(json.dumps(dict(status=report['status'],output=str(output),aa=report.get('aa_accepted'),ceilings=report.get('ceilings'))),flush=True)


if __name__=='__main__':main()
