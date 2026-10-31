"""Independent cold-L2 Nsight byte accounting for the continuous TMA bound.
Runs only the diagnostic, never submits or creates a GPU. Every actual fixture
gets one mode0 launch under the profiler; preparation/checks are outside capture.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='stream-counter-proof.json')
    p.add_argument('--loads',default='D1,V1,D16')
    p.add_argument('--fixtures',type=int,default=3)
    p.add_argument('--timing-result',default='layer-stream-bound.json')
    p.add_argument('--stream-kind',choices=('grouped','flat'),default='grouped')
    a=p.parse_args();output=Path(a.output);stem=output.stem
    csvfile=Path(stem+'.csv');logfile=Path(stem+'.log');child=Path(stem+'-child.json')
    report=dict(status='running',official_score=False,artifact_files=[csvfile.name,logfile.name,child.name],rows=[],errors=[])
    started=time.monotonic()
    def save():output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    try:
        cmd=['/usr/local/cuda/bin/ncu','--profile-from-start','off','--clock-control','none',
             '--cache-control','all','--replay-mode','kernel','--csv','--print-units','base',
             '--kernel-name-base','mangled','--kernel-name','regex:.*dense_stream_.*stream_kernel.*',
             '--metrics','gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_op_read.sum',
             '--log-file',str(csvfile),sys.executable,'layer_stream_bound.py','--output',str(child),
             '--ncu-child','--loads',a.loads,'--fixtures',str(a.fixtures),'--stream-kind',a.stream_kind]
        report['command']=cmd;save()
        with logfile.open('w') as log:
            proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=820,check=False)
        report['returncode']=proc.returncode
        if proc.returncode:raise RuntimeError('Nsight child failed; inspect '+str(logfile))
        observations=json.loads(child.read_text())
        if observations['status']!='complete':raise RuntimeError('Counter child incomplete')
        expected=[(r,f) for r in observations['rows'] for f in r['fixtures']]
        timing=json.loads(Path(a.timing_result).read_text())
        if timing['status']!='complete':raise RuntimeError('Paired timing result incomplete')
        timed={(r['load'],r['layer'],f['seed']):f for r in timing['rows'] for f in r['fixtures']}
        report['matched_timing_result']=a.timing_result
        for case,fixture in expected:
            control=timed[(case['load'],case['layer'],fixture['seed'])]
            if fixture['expected_bytes']!=control['expected_bytes']:
                raise RuntimeError('Profiler payload bytes differ from paired timing fixture')
            for section,field in (('routing','ids_sha256'),('kv','physical_rows_sha256')):
                if fixture['manifest'][section][field]!=control['manifest'][section][field]:
                    raise RuntimeError('Profiler route/page coverage differs from paired timing fixture')
        metrics={};header=None
        for fields in csv.reader(io.StringIO(csvfile.read_text())):
            if 'Metric Name' in fields and 'Metric Value' in fields and 'ID' in fields:
                header=fields;continue
            if header is None or len(fields)!=len(header):continue
            row=dict(zip(header,fields))
            if not row['ID'].isdigit():continue
            record=metrics.setdefault(int(row['ID']),dict(kernel=row.get('Kernel Name'),values={}))
            record['values'][row['Metric Name']]=dict(value=float(row['Metric Value'].replace(',','')),unit=row.get('Metric Unit'))
        if len(metrics)!=len(expected):raise RuntimeError(f'Counter launch count{len(metrics)} != fixture count{len(expected)}')
        for (index,record),(case,fixture) in zip(sorted(metrics.items()),expected):
            values=record['values'];payload=fixture['expected_bytes'];metadata=fixture['manifest']['metadata']['descriptor_storage_bytes']
            reads=values['dram__bytes_read.sum']['value'];duration=values['gpu__time_duration.sum']
            unit=duration['unit'];scale={'nsecond':1e-3,'ns':1e-3,'usecond':1.0,'us':1.0,'msecond':1e3,'ms':1e3,'second':1e6,'s':1e6}
            if unit not in scale:raise RuntimeError('Unknown Nsight duration unit: '+str(unit))
            us=duration['value']*scale[unit]
            allowance=max(1024*1024,.005*payload)
            # Upper allowance covers code fetch, sector overfetch and profiler
            # plumbing; logical model/KV and descriptor traffic stay separate.
            good=reads>=payload*.999 and reads<=payload+metadata+allowance
            result=dict(load=case['load'],layer=case['layer'],seed=fixture['seed'],kernel=record['kernel'],
                        expected_payload_bytes=payload,descriptor_bytes=metadata,dram_read_bytes=reads,
                        dram_write_bytes=values['dram__bytes_write.sum']['value'],payload_ratio=reads/payload,
                        duration_us=us,payload_TB_s=payload/(us*1e6),counter_consistency_passed=good,
                        minimum_read_bytes=payload*.999,maximum_read_bytes=payload+metadata+allowance,
                        checksum_passed=fixture['checksum_check']['passed'],completion_passed=fixture['byte_check']['passed'],
                        raw_metrics=values)
            report['rows'].append(result);save()
            if not good:raise RuntimeError('Read counters outside disclosed coverage allowance: '+str(result))
        report['status']='passed'
        report['scope']='Pure-transfer DRAM proof; profiler durations are independent and do not replace paired event/native timings.'
    except BaseException:
        report['status']='failed';report['errors'].append(traceback.format_exc());raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started;save()
        print(json.dumps(dict(status=report['status'],rows=len(report['rows']),output=str(output))),flush=True)


if __name__=='__main__':main()
