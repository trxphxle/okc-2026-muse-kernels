"""Aggregate measured layer-byte bounds with independent traffic proof.
No GPU work; rejects incomplete controls, coverage or mismatched fixtures.
"""
import argparse
import json
import math
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--timing',type=Path,required=True)
    p.add_argument('--counters',type=Path,required=True)
    p.add_argument('--grouped',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    timing=json.loads(a.timing.read_text());proof=json.loads(a.counters.read_text())
    assert timing['status']=='complete' and proof['status']=='passed'
    assert timing['aa_accepted'],'A/A outside1%: no qualified ceiling decision'
    assert proof['matched_timing_result']==a.timing.name
    profile={(r['load'],r['layer'],r['seed']):r for r in proof['rows']}
    assert len(profile)==sum(len(r['fixtures']) for r in timing['rows'])
    rows=[]
    for row in timing['rows']:
        assert row['aa_within_one_percent']
        for fixture in row['fixtures']:
            assert fixture['status']=='complete'
            assert fixture['reference']['passed'] and fixture['reference_after']['passed']
            assert fixture['checksum_check']['passed'] and fixture['checksum_after']['passed']
            assert fixture['byte_check']['passed'] and fixture['byte_check']['per_cta_counters_pass']
            assert fixture['native']['stream']['kernel_count']==1
            counter=profile[(row['load'],row['layer'],fixture['seed'])]
            assert counter['expected_payload_bytes']==fixture['expected_bytes']
            assert counter['counter_consistency_passed']
            assert all(fixture['manifest']['coverage'][k] for k in ('all_weight_keys_accounted','all_routed_local_experts_included','both_expert_codes_block_scales_global_scales','full_causal_kv_union','all_new_tokens_included','no_out_of_allocation_reads','stage_sum_equals_chunk_sum'))
        base=row['native_mean_us']['base'];stream=row['native_mean_us']['stream']
        rows.append(dict(load=row['load'],layer=row['layer'],fixture_count=len(row['fixtures']),
            native_layer_us=base,native_stream_us=stream,native_gap_us=base-stream,
            event_layer_us=row['event_median_us']['base'],event_stream_us=row['event_median_us']['stream'],
            mean_payload_bytes=statistics.mean(f['expected_bytes'] for f in row['fixtures']),
            payload_bytes_per_fixture=[f['expected_bytes'] for f in row['fixtures']],
            active_experts_per_fixture=[f['manifest']['routing']['unique_local_experts'] for f in row['fixtures']],
            kv_bytes_per_fixture=[f['manifest']['kv']['physical_payload_bytes'] for f in row['fixtures']],
            mean_category_bytes={category:statistics.mean(f['manifest']['categories'][category]['physical_bytes'] for f in row['fixtures']) for category in row['fixtures'][0]['manifest']['categories']},
            native_base_kernel_counts=[f['native']['base']['kernel_count'] for f in row['fixtures']],
            aa_ratio=row['aa_ratio']))
    ceilings={}
    for selected in (('D1','V1'),('D1','V1','D16')):
        if not all(k in timing['loads'] for k in selected):continue
        metrics={}
        for metric,key in (('native','weighted_native_us'),('event','weighted_event_us')):
            ratios={name:1.0 for name in ('D1','D16','D32','V1','V16','V32')}
            for name in selected:
                values=timing['loads'][name][key];ratios[name]=values['base']/values['stream']
            gm=math.prod(ratios.values())**(1/6)
            metrics[metric]=dict(geomean_ratio=gm,improvement_percent=100*(gm-1),load_ratios=ratios)
            if selected==('D1','V1'):
                d,v=(timing['loads'][name][key] for name in selected)
                allowed=d['base']*v['base']/(1.05**6)
                budget=(math.sqrt((d['stream']-v['stream'])**2+4*allowed)-d['stream']-v['stream'])/2
                metrics[metric]['equal_added_work_budget_for_five_percent_us']=budget
        metrics['passes_five_percent_on_both_timers']=min(metrics[m]['geomean_ratio'] for m in ('native','event'))>=1.05
        ceilings['+'.join(selected)]=metrics
    report=dict(status='verified_ceiling_only',production_build=False,official_checks_used=0,
        sources=dict(timing=str(a.timing),counter_proof=str(a.counters),grouped=str(a.grouped) if a.grouped else None),
        baseline_sha256=timing['sources']['decode_stream_base.py'],weight_seed=timing['weight_seed'],
        reader_kind=timing['reader_kind'],rows=rows,loads=timing['loads'],ceilings=ceilings,
        byte_proof=dict(fixture_count=len(profile),all_passed=True,
            dram_over_payload_ratio_range=[min(r['payload_ratio'] for r in profile.values()),max(r['payload_ratio'] for r in profile.values())],
            extra_beyond_payload_and_descriptors_range=[min(r['dram_read_bytes']-r['expected_payload_bytes']-r['descriptor_bytes'] for r in profile.values()),max(r['dram_read_bytes']-r['expected_payload_bytes']-r['descriptor_bytes'] for r in profile.values())]),
        limitations=timing['omissions']+['This is a measured zero-math native-format stand-in, not a universal hardware lower bound or implemented speedup.',
            'Changing a load means replacing its46:16 weighted sliding/full time; all other loads remain exactly1x in the six-load geomean.',
            'Native profile spans and captured-event timings are independently observed; no replay-cost subtraction.',
            'The5% gate is necessary evidence only; this task authorizes no production megakernel.'])
    if a.grouped:
        grouped=json.loads(a.grouped.read_text())
        old={(r['load'],r['layer'],f['seed']):f for r in grouped['rows'] for f in r['fixtures']}
        equal=all(old[(r['load'],r['layer'],f['seed'])]['expected_bytes']==f['expected_bytes'] and old[(r['load'],r['layer'],f['seed'])]['manifest']['routing']['ids_sha256']==f['manifest']['routing']['ids_sha256'] and old[(r['load'],r['layer'],f['seed'])]['manifest']['kv']['physical_rows_sha256']==f['manifest']['kv']['physical_rows_sha256'] for r in timing['rows'] for f in r['fixtures'])
        report['grouped_comparison']=dict(same_payload_and_route_page_coverage=equal,grouped_ceilings=grouped['ceilings'])
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],ceilings=ceilings,byte_proof=report['byte_proof'])))


if __name__=='__main__':main()
