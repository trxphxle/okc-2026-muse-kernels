"""Independent NCU coverage proof for real dependency/glue stream diagnostics."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='dependency-counter-proof.json')
    p.add_argument('--timing-result',default='dependency-bound.json')
    p.add_argument('--loads',default='D1,V1');p.add_argument('--layers',default='sliding,full')
    p.add_argument('--fixtures',type=int,default=3);p.add_argument('--modes',default='0,1,4')
    a=p.parse_args();out=Path(a.output);stem=out.stem
    child=Path(stem+'-child.json');log=Path(stem+'.log');csvpath=Path(stem+'.csv')
    report=dict(status='running',rows=[],errors=[],artifact_files=[child.name,log.name,csvpath.name],official_score=False)
    started=time.monotonic()
    def save():out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    try:
        timing=json.loads(Path(a.timing_result).read_text())
        if timing['status']!='complete':raise RuntimeError('Timing result incomplete')
        cmd=['/usr/local/cuda/bin/ncu','--profile-from-start','off','--clock-control','none',
             '--cache-control','all','--replay-mode','kernel','--csv','--print-units','base',
             '--kernel-name-base','mangled','--kernel-name','regex:.*dense_dependency.*dependency_stream.*',
             '--metrics','gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_op_read.sum',
             '--log-file',str(csvpath),sys.executable,'dependency_bench.py','--output',str(child),
             '--ncu-child','--profile-modes',a.modes,'--loads',a.loads,'--layers',a.layers,'--fixtures',str(a.fixtures)]
        report.update(command=cmd,matched_timing_result=a.timing_result);save()
        with log.open('w') as stream:
            proc=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=820,check=False)
        report['returncode']=proc.returncode
        if proc.returncode:raise RuntimeError('Counter child failed; inspect '+str(log))
        observed=json.loads(child.read_text())
        if observed['status']!='complete':raise RuntimeError('Counter child incomplete')
        for name in ('dependency_kernel.cu','dependency_kernel.py','dependency_abi.py','dependency_plan.py'):
            if observed['sources'][name]!=timing['sources'][name]:raise RuntimeError('Timing/profile source mismatch: '+name)
        timed={(r['load'],r['layer'],f['seed']):f for r in timing['rows'] for f in r['fixtures']}
        expected=[]
        for row in observed['rows']:
            for fixture in row['fixtures']:
                prior=timed[(row['load'],row['layer'],fixture['seed'])]
                for key in ('expected_payload_bytes','expected_mask'):
                    if fixture[key]!=prior[key]:raise RuntimeError('Profile/timed fixture mismatch: '+key)
                if fixture['manifest']['routing']['ids_sha256']!=prior['manifest']['routing']['ids_sha256']:
                    raise RuntimeError('Profile/timed actual routes differ')
                if fixture['manifest']['flat_manifest']['kv']['physical_rows_sha256']!=prior['manifest']['flat_manifest']['kv']['physical_rows_sha256']:
                    raise RuntimeError('Profile/timed KV coverage differs')
                for observation in fixture['counter_observations']:
                    expected.append((row,fixture,observation))
        header=None;metrics={}
        for fields in csv.reader(io.StringIO(csvpath.read_text())):
            if 'Metric Name' in fields and 'Metric Value' in fields and 'ID' in fields:header=fields;continue
            if not header or len(fields)!=len(header):continue
            row=dict(zip(header,fields))
            if not row['ID'].isdigit():continue
            record=metrics.setdefault(int(row['ID']),dict(kernel=row.get('Kernel Name'),values={}))
            record['values'][row['Metric Name']]=dict(value=float(row['Metric Value'].replace(',','')),unit=row.get('Metric Unit'))
        if len(metrics)!=len(expected):raise RuntimeError('Wrong number of profiled dependency kernels')
        excluded={'K_CACHE','V_CACHE','O_W','POOL_W','ROUTER_W','UN_W','CHUNKS','RANGES','EXP_CHUNKS','FLAT_CHUNKS'}
        sizes={'torch.uint8':1,'torch.bfloat16':2,'torch.float32':4,'torch.int32':4,'torch.uint32':4,'torch.int64':8}
        scales={'ns':.001,'nsecond':.001,'us':1,'usecond':1,'ms':1000,'msecond':1000,'second':1e6,'s':1e6}
        for (_,metric),(case,fixture,observation) in zip(sorted(metrics.items()),expected):
            payload=fixture['expected_payload_bytes'];meta=fixture['manifest']['metadata'];schema=fixture['manifest']['tensor_schema']
            auxiliary=sum(math.prod(v['shape'])*sizes[v['dtype']] for k,v in schema.items() if k not in excluded)
            descriptor=meta['fixed_descriptor_bytes']+meta['expert_bank_descriptor_bytes']+meta['params_bytes']
            values=metric['values'];reads=values['dram__bytes_read.sum']['value'];duration=values['gpu__time_duration.sum']
            if duration['unit'] not in scales:raise RuntimeError('Unknown duration unit')
            us=duration['value']*scales[duration['unit']]
            # Required new KV may be consumed from the cache immediately after
            # its timed store. Other payload remains cold and must be covered.
            lower=payload-meta['new_kv_producer_bytes']-65536
            upper=payload+descriptor+3*auxiliary+1024*1024
            passed=lower<=reads<=upper and observation['checks']['payload']['passed']
            item=dict(load=case['load'],layer=case['layer'],seed=fixture['seed'],mode=observation['mode'],
                      kernel=metric['kernel'],expected_payload_bytes=payload,dram_read_bytes=reads,
                      dram_write_bytes=values['dram__bytes_write.sum']['value'],duration_us=us,
                      payload_ratio=reads/payload,auxiliary_storage_bytes=auxiliary,descriptor_storage_bytes=descriptor,
                      disclosed_min_bytes=lower,disclosed_max_bytes=upper,passed=passed,raw_metrics=values)
            report['rows'].append(item);save()
            if not passed:raise RuntimeError('Dependency traffic outside disclosed coverage envelope: '+str(item))
        report['status']='passed'
        report['scope']='Exact per-phase XOR/byte/ordering/numeric checks plus cold DRAM coverage; auxiliary glue/producer traffic is real and additional. Profiler durations do not replace paired timings.'
    except BaseException:
        report['status']='failed';report['errors'].append(traceback.format_exc());raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started;save()
        print(json.dumps(dict(status=report['status'],rows=len(report['rows']),output=str(out))),flush=True)


if __name__=='__main__':main()
