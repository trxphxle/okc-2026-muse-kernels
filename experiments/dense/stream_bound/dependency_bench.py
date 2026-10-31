"""Dependency-aware ceiling diagnostic; NOT a production megakernel/submission.
Real KV writes, BF16-boundary reductions and top8; frozen REAL heavy-math outputs.
Compare unchanged layer, validated flat stream, same-interface feature-off, and
strict/bounded-prefetch dependency modes. Native spans remain separate events.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import traceback
import torch
import reference
import layer_stream_bound as common
import stream_flat
import dependency_abi as abi
import dependency_kernel
import dependency_plan

MODES={'strict':0,'prefetch':1,'off':4}
CHECK_KEYS=('O_LIVE','POOLED_LIVE','SCORES_LIVE','UN_LIVE','H_LIVE','HR_LIVE','CARRY','TAIL','OUT','SOUT','IDS_OUT','XM_LIVE')
EXACT={'O_LIVE','POOLED_LIVE','SCORES_LIVE','UN_LIVE','IDS_OUT'}


def numcheck(actual,want,key):
    if key in EXACT:
        passed=torch.equal(actual,want)
        return dict(passed=bool(passed),exact=True)
    if actual.shape!=want.shape or actual.dtype!=want.dtype:
        return dict(passed=False,error='shape/dtype mismatch')
    a,b=actual.float(),want.float()
    finite=bool(torch.isfinite(a).all())
    if not finite:return dict(passed=False,error='nonfinite glue result')
    delta=(a-b).abs();max_abs=float(delta.max());rel=float(delta.norm()/b.norm().clamp_min(1e-12))
    if actual.dtype==torch.bfloat16:
        # Allow one neighboring BF16 value only, plus a small FP32 absolute floor;
        # require fewer than1% changed values. Captured reference may use another
        # legal reduction tree at a BF16 tie. No tolerance change on failure.
        up=torch.nextafter(want,torch.full_like(want,float('inf'))).float()
        down=torch.nextafter(want,torch.full_like(want,float('-inf'))).float()
        step=torch.maximum((up-b).abs(),(down-b).abs())
        floor=8*torch.finfo(torch.float32).eps*torch.maximum(b.abs(),torch.ones_like(b))
        changed=float((actual!=want).float().mean())
        passed=bool((delta<=torch.maximum(step,floor)).all()) and changed<=.01
        return dict(passed=passed,max_abs=max_abs,relative_l2=rel,changed_fraction=changed,criterion='<=one BF16 neighbor or8 FP32eps floor, <=1% nonexact')
    passed=bool(torch.allclose(actual,want,rtol=2e-5,atol=2e-6))
    return dict(passed=passed,max_abs=max_abs,relative_l2=rel,rtol=2e-5,atol=2e-6)


def payload_check(plan,mode):
    sink=plan['tensors']['SINK'].cpu()
    total=int(sink[:,:,0].sum())
    expected=plan['expected_payload_bytes']
    result=dict(total_bytes=total,expected_bytes=expected,passed=total==expected)
    if mode!=4:
        per=sink[:,:,0].sum(0).tolist()
        result['phase_bytes']=per;result['expected_phase_bytes']=plan['expected_bytes_per_phase']
        result['passed'] &= per==plan['expected_bytes_per_phase']
        if mode in (2,3):
            xors=[]
            for phase in range(9):
                value=0
                for x in sink[:,phase,1].tolist():value^=int(x)&0xffffffff
                xors.append(value)
            result['phase_xors']=xors;result['expected_phase_xors']=plan['expected_xor_per_phase']
            result['passed'] &= xors==plan['expected_xor_per_phase']
    if not result['passed']:raise RuntimeError('Dependency payload mismatch: '+str(result))
    return result


def ordering_check(plan):
    stamps=plan['tensors']['STAMPS'].cpu();ready=plan['tensors']['READY_TIMES'].cpu().tolist()
    checks=[]
    # All phase1(KV+gate) requests are conservatively gated by KV_WRITTEN.
    for phase,flag,field in ((1,'KV_WRITTEN',0),(2,'ATTN_DONE',2),(3,'O_DONE',2),
                            (4,'B1_DONE',2),(5,'RP_DONE',2),(6,'ROUTES_READY',0),
                            (7,'EXPERTS_DONE',2),(8,'UNPOOL_DONE',2)):
        values=stamps[:,phase,field];values=values[values>0]
        r=int(ready[abi.FLAG[flag]])
        passed=r>0 and len(values)>0 and int(values.min())>=r
        checks.append(dict(phase=abi.PHASES[phase],prerequisite=flag,publication_ns=r,
                           first_observed_ns=int(values.min()) if len(values) else None,passed=bool(passed)))
    ordered=all(int(b)>=int(a)>0 for a,b in zip(ready,ready[1:]))
    if not ordered or not all(x['passed'] for x in checks):raise RuntimeError('Dependency ordering violation: '+str(dict(ready=ready,checks=checks)))
    return dict(passed=True,ready_ns=ready,checks=checks,
                stamped_phase_delta_us={abi.FLAGS[i]:(ready[i]-ready[i-1])/1000 for i in range(1,10)},
                warning='Validation-mode clocks/XOR alter timing; never use these spans as primary latency')


def poison(plan,data):
    for key in CHECK_KEYS:
        tensor=plan['tensors'][key]
        if tensor.is_floating_point():tensor.fill_(float('nan'))
        else:tensor.fill_(-1)
    rows=plan['tensors']['CACHE_LOC'].long()
    for name in ('k_cache','v_cache'):
        data[name].view(torch.uint8).reshape(-1,128).index_fill_(0,rows,0)


def check_results(plan,data,mode,instrumented=False):
    result=dict(payload=payload_check(plan,mode))
    if mode==4:return result
    error=int(plan['tensors']['ERROR'].item())
    if error:raise RuntimeError('Device protocol error: '+str(error))
    actual_mask=int(plan['tensors']['ACTIVE_MASK'].item())&0xffffffff
    if actual_mask!=plan['expected_mask']:raise RuntimeError(f'Active expert mask{actual_mask} != {plan["expected_mask"]}')
    result['active_mask']=actual_mask;result['numerical']={}
    for key in CHECK_KEYS:
        check=numcheck(plan['tensors'][key],plan['references'][key],key)
        result['numerical'][key]=check
        if not check['passed']:raise RuntimeError('Glue/oracle mismatch '+key+': '+str(check))
    rows=plan['tensors']['CACHE_LOC'].long()
    for index,name in enumerate(('k_cache','v_cache')):
        got=data[name].view(torch.uint8).reshape(-1,128).index_select(0,rows)
        if not torch.equal(got,plan['tensors']['NEW_KV'][index]):raise RuntimeError('NewKV publication mismatch: '+name)
    flags=plan['tensors']['FLAGS'].cpu().tolist()
    arrivals=plan['tensors']['ARRIVALS'].cpu()[:,:,0]
    if len(set(flags))!=1:raise RuntimeError('Readiness epochs disagree: '+str(flags))
    for flag,count in enumerate(abi.arrival_producers(plan['scalars']['T'],plan['scalars']['BLOCKS'])):
        if not bool((arrivals[flag,:count]==flags[0]).all()):raise RuntimeError('Missing producer arrival for '+abi.FLAGS[flag])
    result['epoch']=flags[0];result['arrival_epochs_pass']=True;result['new_kv_exact']=True
    if instrumented:result['ordering']=ordering_check(plan)
    return result


def aggregate(report):
    loads={}
    for name in report['requested_loads']:
        rows=[r for r in report['rows'] if r['load']==name]
        if len(rows)!=2 or any(r['status']!='complete' for r in rows):continue
        for row in rows:
            keys=row['fixtures'][0]['event_us'].keys()
            row['event_mean_us']={k:statistics.mean(f['event_us'][k] for f in row['fixtures']) for k in keys}
            row['native_mean_us']={k:statistics.mean(f['native'][k]['span_us'] for f in row['fixtures']) for k in row['fixtures'][0]['native']}
            row['aa_ratio']=statistics.median(f['aa_ratio'] for f in row['fixtures'])
        value={}
        for metric,key in (('event','event_mean_us'),('native','native_mean_us')):
            value[metric]={k:sum(common.WEIGHTS[r['layer']]*r[key][k] for r in rows)/62 for k in rows[0][key]}
        value['aa_ratio']=value['event']['base']/value['event']['base_repeat']
        value['aa_within_one_percent']=abs(value['aa_ratio']-1)<=.01 and all(abs(r['aa_ratio']-1)<=.01 for r in rows)
        loads[name]=value
    report['loads']=loads;report['ceilings']={}
    if all(name in loads for name in ('D1','V1')):
        for arm in ('strict','prefetch','off','flat'):
            report['ceilings'][arm]={}
            for metric in ('event','native'):
                ratio=math.prod(loads[n][metric]['base']/loads[n][metric][arm] for n in ('D1','V1'))**(1/6)
                report['ceilings'][arm][metric]=dict(geomean_ratio=ratio,improvement_percent=100*(ratio-1))
        report['aa_accepted']=all(v['aa_within_one_percent'] for v in loads.values())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='dependency-bound.json');p.add_argument('--baseline',default='decode_stream_base.py')
    p.add_argument('--loads',default='D1,V1');p.add_argument('--layers',default='sliding,full')
    p.add_argument('--fixtures',type=int,default=3);p.add_argument('--samples',type=int,default=24)
    p.add_argument('--weight-seed',type=int,default=5353);p.add_argument('--ncu-child',action='store_true')
    p.add_argument('--profile-modes',default='0,1,4')
    args=p.parse_args();output=Path(args.output)
    report=dict(status='running',kind='dependency_aware_stream_ceiling_only',production_build=False,
        requested_loads=args.loads.split(','),requested_layers=args.layers.split(','),rows=[],artifact_files=[],errors=[],
        weight_seed=args.weight_seed,abi_fields=abi.FIELDS,phases=abi.PHASES,flags=abi.FLAGS,
        limitations=['GEMM/attention/expert arithmetic is deleted; real captured producer payloads substitute for heavy math.',
                     'Required boundary reductions, KV stores and top8 are real/timed. Native weights, not retained compression.',
                     'Strict/prefetch are synchronization diagnostics, not production implementations.',
                     'Other4 decode loads unchanged;46:16 sliding/full weighting; no timer subtraction.'])
    started=time.monotonic()
    try:
        paths=[args.baseline,__file__,'dependency_abi.py','dependency_kernel.py','dependency_kernel.cu','dependency_plan.py',
               'stream_flat.py','stream_flat.cu','stream_plan.py','stream_kernel.py','reference.py','checks.py','nvfp4.py']
        report['sources']={Path(x).name:hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in paths}
        report['environment']=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
        common.save(output,report)
        base=common.load(args.baseline,'_dependency_base');weights=reference.generate_weights(args.weight_seed)
        own_weights={k:v.clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v) for k,v in weights.items()}
        before=time.monotonic();state=base.prepare(own_weights);torch.cuda.synchronize();report['prepare_seconds']=time.monotonic()-before
        before=time.monotonic();ext=dependency_kernel.build();report['compile_seconds']=time.monotonic()-before
        report['resources']=ext.dep_info();flat_ext=stream_flat.build()
        flush=torch.ones(256*1024*1024//4,device='cuda',dtype=torch.float32)
        nsm=torch.cuda.get_device_properties(0).multi_processor_count
        flat_sink=torch.empty((nsm,4),device='cuda',dtype=torch.int64);flat_stamps=torch.empty_like(flat_sink)
        common.save(output,report)
        for name in report['requested_loads']:
            batch,q_len=common.LOADS[name]
            if batch*q_len>8:raise ValueError('Dependency diagnostic scoped toT<=8')
            for layer in report['requested_layers']:
                shape=dict(batch=batch,q_len=q_len,ctx=80000,layer=layer,page=1,spread='fixed',dist='soft')
                row=dict(load=name,layer=layer,shape=shape,status='running',fixtures=[]);report['rows'].append(row)
                for fixture in range(args.fixtures):
                    seed=919233+batch*100+q_len*1000+(layer=='full')*10000+fixture+1
                    pristine=reference.generate_input(seed=seed,**shape);data=common.clone(pristine)
                    probe=base.kernel_fn(data,state);counts=torch.bincount((probe[2].long()//32).flatten(),minlength=8)
                    lo=32*int(counts.argmax());pristine['local_start'].fill_(lo);data['local_start'].fill_(lo)
                    actual=base.kernel_fn(data,state)
                    rec=dict(seed=seed,local_start=lo,status='planning',reference=common.oracle(pristine,data,actual,weights))
                    row['fixtures'].append(rec);common.save(output,report)
                    before=time.monotonic();plan=dependency_plan.make_plan(base,weights,data,state,actual)
                    rec['plan_seconds']=time.monotonic()-before;rec['manifest']=plan['manifest']
                    rec['expected_payload_bytes']=plan['expected_payload_bytes'];rec['expected_mask']=plan['expected_mask']
                    fp=plan['flat_plan']
                    def dep(mode):
                        ext.dep_launch(plan['params'].data_ptr(),mode,torch.cuda.current_stream().cuda_stream)
                        return plan['tensors']['OUT'],plan['tensors']['SOUT'],plan['tensors']['IDS_OUT']
                    def flat():
                        flat_ext.stream_launch(fp['chunks'].data_ptr(),fp['stages'].data_ptr(),flat_sink.data_ptr(),flat_stamps.data_ptr(),len(fp['stages']),nsm,0,torch.cuda.current_stream().cuda_stream)
                        return flat_sink
                    rec['validation']={}
                    for mode in (2,3):
                        plan['params'][abi.F['DELAY_STAGE']]=-1
                        poison(plan,data);dep(mode);torch.cuda.synchronize()
                        rec['validation'][str(mode)]=check_results(plan,data,mode,True)
                    # Delayed real producers plus poisoned live buffers exercise
                    # the two data-dependent reads, not merely flag wiring.
                    if fixture==0:
                        rec['delay_validation']=[]
                        for flag in ('KV_WRITTEN','ROUTES_READY'):
                            plan['params'][abi.F['DELAY_STAGE']]=abi.FLAG[flag]
                            poison(plan,data);dep(3);torch.cuda.synchronize()
                            rec['delay_validation'].append(dict(flag=flag,result=check_results(plan,data,3,True)))
                    plan['params'][abi.F['DELAY_STAGE']]=-1
                    common.save(output,report)
                    if args.ncu_child:
                        rec['counter_observations']=[]
                        for mode in map(int,args.profile_modes.split(',')):
                            flush.sum();torch.cuda.synchronize();torch.cuda.profiler.start()
                            dep(mode);torch.cuda.synchronize();torch.cuda.profiler.stop()
                            rec['counter_observations'].append(dict(mode=mode,checks=check_results(plan,data,mode)))
                        rec['status']='counter_profiled';common.save(output,report)
                        del plan,fp,actual,probe,pristine,data;gc_collect()
                        continue
                    calls={'base':lambda:base.kernel_fn(data,state),'base_repeat':lambda:base.kernel_fn(data,state),
                           'flat':flat,**{arm:(lambda mode=mode:dep(mode)) for arm,mode in MODES.items()}}
                    caps={arm:common.capture(fn,flush) for arm,fn in calls.items()}
                    rec['timing_samples']=[];names=list(caps)
                    for rep in range(args.samples):
                        order=names[rep%len(names):]+names[:rep%len(names)]
                        if (rep//len(names))%2:order.reverse()
                        rec['timing_samples'].append({arm:common.measure(caps[arm]) for arm in order})
                    rec['event_us']={arm:statistics.median(x[arm] for x in rec['timing_samples']) for arm in names}
                    rec['aa_ratio']=statistics.median(x['base']/x['base_repeat'] for x in rec['timing_samples'])
                    rec['after']={}
                    for arm,mode in MODES.items():
                        caps[arm]['graph'].replay();caps[arm]['b'].synchronize();rec['after'][arm]=check_results(plan,data,mode)
                    caps['base']['graph'].replay();caps['base']['b'].synchronize()
                    rec['reference_after']=common.oracle(pristine,data,caps['base']['out'],weights)
                    rec['native']={}
                    for arm in ('base','flat','off','strict','prefetch'):
                        path=output.with_name(output.stem+f'-{name}-{layer}-{fixture}-{arm}-native.json')
                        rec['native'][arm]=common.native(calls[arm],flush,path);report['artifact_files'].append(path.name)
                        if arm!='base' and rec['native'][arm]['kernel_count']!=1:raise RuntimeError('Diagnostic must be one persistent kernel')
                    rec['status']='complete';common.save(output,report)
                    print(json.dumps(dict(load=name,layer=layer,fixture=fixture,events=rec['event_us'],aa=rec['aa_ratio'])),flush=True)
                    caps.clear();calls.clear();del plan,fp,actual,probe,pristine,data;gc_collect()
                row['status']='complete'
                if not args.ncu_child:aggregate(report)
                common.save(output,report)
        report['status']='complete'
    except BaseException:
        report['status']='failed';report['errors'].append(traceback.format_exc());raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started;common.save(output,report)
        print(json.dumps(dict(status=report['status'],output=str(output),aa=report.get('aa_accepted'),ceilings=report.get('ceilings'))),flush=True)


def gc_collect():
    import gc
    gc.collect()


if __name__=='__main__':main()
