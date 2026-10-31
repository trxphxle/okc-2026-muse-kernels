"""Round4 real-routing V32 shared-ring head-of-line diagnostic (NO CANDIDATE).

Run after the collector has frozen all18 fixtures:
  python round4_hol.py --manifest round4-fixtures.json --output round4-hol.json
  python round4_hol.py --analyze round4-hol.json --output round4-hol-reanalysis.json

Defaults assume decode_round4_control.py, compare_decode_public.py, reference.py,
checks.py and nvfp4.py are beside this script/in the bounded lane sandbox.
The six V32 rows are always required. No synthetic routing/input fallback exists.

Artifacts: JSON report, sparse gzip binary trace per fixture/cohort, schema JSON.
Each replay samples a contiguous cohort (default4 CTAs); cohorts cover all132.
If timing moves >5%, the next full coverage pass uses single-CTA cohorts, retains
both passes' artifacts, and does not promote the noisier pass as a bound. Local
bytes/spills or >5% perturbation always make attribution observer-unreliable.
All intervals are observational evidence, never a two-WG speedup prediction.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback

import round4_common as common
from round4_hol_source import ACTORS, FIELDS, FOOTER, HEADER, build
from round4_hol_analysis import SCHEMA, analyze_cta, read_trace, summarize, write_trace


CAVEATS = [
    'Diagnostic source overrides only; frozen repaired BF16 decoder and all math/barriers/releases/ranges/block geometry remain unchanged. No two-WG implementation or candidate.',
    'Each capacity interval is a timed wait invocation, including ready-on-entry overhead; counts are NOT hardware stall counters.',
    'Last release ownership requires disjoint before/after arrival brackets over all16 consumer warps. Ambiguous ownership contributes only loose overlap.',
    'Direct exposure is a conservative observed triple-intersection and dependency chain, not a counterfactual saving. Full associated consumer-wait envelope is reported separately, not called removable time.',
    'Timestamp stores and compiler/register changes can perturb the sampled CTA even when whole-kernel overhead is small. Passing the observer screen is necessary, never proof of an unperturbed local schedule.',
    'Cohorts are separate launches, not a simultaneous132-CTA trace. Never sum across CTAs or stitch their absolute clocks into a global critical path.',
    'Final global-barrier and normalization work, PDL overlap, added activation traffic, residency and split-unit changes prevent extrapolation to a production two-WG speedup.',
    'Standalone PDL0 expert-stage events exclude the preceding 256MiB flush. They are raw unprofiled event times, not NCU intervals or bandwidth measurements.',
    'Source-identical host-resource-query build supplies baseline resources; the actual original timing arm uses the unchanged native _moe(256,state) extension.',
]


def numerical(output, tensors):
    import torch
    out, gold = output.float(), tensors['gold'].float()
    expected = tensors['expected']
    finite = bool(torch.isfinite(out).all() and torch.isfinite(gold).all())
    if not finite:
        return dict(finite=False, numerical_pass=False)
    error = float(torch.linalg.vector_norm(out - gold) / torch.linalg.vector_norm(gold).clamp_min(1e-30))
    ref_error = float(torch.linalg.vector_norm(out - expected.float()) /
                      torch.linalg.vector_norm(expected.float()).clamp_min(1e-30))
    norms = gold.norm(dim=-1)
    row_error = float(((out-gold).norm(dim=-1) /
                       torch.maximum(norms, (.1*norms.mean()).clamp_min(1e-12))).max())
    equal = bool(torch.equal(output, expected))
    digest = None
    if not equal:
        raw = output.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        digest = hashlib.sha256(raw).hexdigest()
    return dict(finite=True, relative_l2_vs_public_expert=error,
        relative_l2_vs_frozen_native=ref_error, worst_row_relative_l2=row_error,
        tolerance=common.REL_L2_TOL,
        numerical_pass=error <= common.REL_L2_TOL and ref_error <= common.REL_L2_TOL and row_error <= 5*common.REL_L2_TOL,
        byte_exact_vs_frozen_native=equal, output_bytes=output.numel()*output.element_size(),
        output_sha256=digest,
        byte_note='Exact equality is checked on GPU; avoid a full CPU copy/hash when the frozen native output is bit-identical. Nonidentical outputs retain a SHA.')


def check_outputs(captures, tensors):
    import torch
    result = {name: numerical(cap['output'], tensors) for name, cap in captures.items()}
    for name, cap in captures.items():
        result[name]['byte_exact_vs_original'] = bool(torch.equal(cap['output'], captures['original']['output']))
    if not all(row['numerical_pass'] for row in result.values()):
        raise RuntimeError('Before/after public/frozen numerical oracle failed: ' + json.dumps(result))
    return result


def timing(captures, reps, seed):
    """Rotate arm order; no oracle/copy between timed arms in each block."""
    names, rows = list(captures), []
    for rep in range(reps):
        shift = (rep + seed) % len(names)
        order = names[shift:] + names[:shift]
        row = dict(order=order, us={})
        for name in order:
            row['us'][name] = common.measure(captures[name])
        rows.append(row)
    medians = {name: statistics.median(row['us'][name] for row in rows) for name in names}
    ratios = {name: statistics.median(row['us'][name]/row['us']['original']-1 for row in rows)
              for name in names if name != 'original'}
    ratios['traced_vs_feature_off'] = statistics.median(
        row['us']['traced']/row['us']['feature_off']-1 for row in rows)
    return dict(records=rows, median_us=medians, paired_median_fraction=ratios,
                max_absolute_perturbation=max(abs(value) for value in ratios.values()))


def validate_manifest(path, baseline):
    if path.name != 'round4-fixtures.json':
        raise RuntimeError('Only the frozen round4-fixtures.json manifest is accepted')
    manifest = json.loads(path.read_text())
    if common.sha(baseline) != common.SOURCE_SHA or manifest.get('source_sha256') != common.SOURCE_SHA:
        raise RuntimeError('Round4 frozen source SHA mismatch')
    expected = common.descriptors()
    rows = manifest.get('fixtures', [])
    if len(rows) != 18 or len({r['case_id'] for r in rows}) != 18:
        raise RuntimeError('Manifest must contain the18 unique actual-layer fixtures')
    actual = {row['case_id']: row for row in rows}
    for desc in expected:
        row = actual.get(desc['case_id'])
        if row is None or any(row.get(k) != v for k,v in desc.items()):
            raise RuntimeError('Manifest descriptor differs from shared frozen contract')
        fixture = (path.parent / row['fixture_file']).resolve()
        if fixture.parent != path.parent.resolve() or common.sha(fixture) != row['fixture_sha256']:
            raise RuntimeError('Frozen fixture SHA/path mismatch')
        planned = common.plan(row['n_e'])
        if any(row.get(key) != value for key,value in planned.items()):
            raise RuntimeError('Manifest routing/job/byte plan mismatch')
    return [actual[d['case_id']] for d in expected if d['load'] == 'V32']


def qualify(resources, measured):
    base, traced = resources['original']['resources'], resources['instrumented']['resources']
    issues = []
    if base['local_bytes_per_thread'] or traced['local_bytes_per_thread']:
        issues.append('nonzero local bytes (spill/local storage risk); no reliable bound')
    for name, value in [('original', base), ('instrumented', traced)]:
        if value['active_blocks_per_sm'] < 1:
            issues.append(name + ' cannot keep132-block spin-barrier grid resident')
        if (value['block_threads'], value['consumer_warps'], value['weight_slots'], value['activation_slots']) != (576,16,3,3):
            raise RuntimeError('Resource query shows changed block/ring geometry')
    if measured['max_absolute_perturbation'] > .05:
        issues.append('paired median timing perturbation exceeds5%')
    if traced['active_blocks_per_sm'] != base['active_blocks_per_sm']:
        issues.append('instrumentation changes occupancy')
    return dict(observer_reliable=not issues, issues=issues,
        local_bytes_note='cudaFuncAttributes.localSizeBytes flags all local allocation conservatively; no claim that a zero value rules out every observer effect.',
        register_delta=traced['registers_per_thread']-base['registers_per_thread'],
        bound_promoted=False)


def analyze_artifact(path):
    metadata, ctas = read_trace(path)
    rows = [analyze_cta(cta, actors) for cta,actors in sorted(ctas.items())]
    return metadata, rows


def reanalyze(args):
    original = json.loads(args.analyze.read_text())
    if original.get('source_sha256') != common.SOURCE_SHA:
        raise RuntimeError('Cannot analyze traces from another source')
    report = dict(status='complete', mode='offline_reanalysis', source_sha256=common.SOURCE_SHA,
                  source_report=str(args.analyze), caveats=CAVEATS, schema=SCHEMA, fixtures=[])
    for fixture in original['fixtures']:
        out = dict(case_id=fixture['case_id'], passes=[])
        for trace_pass in fixture['passes']:
            rows = []
            for cohort in trace_pass['cohorts']:
                path = args.analyze.parent / cohort['trace_file']
                if common.sha(path) != cohort['trace_sha256']:
                    raise RuntimeError('Trace artifact changed after collection')
                metadata, parsed = analyze_artifact(path)
                if metadata['case_id'] != fixture['case_id'] or metadata['fixture_sha256'] != fixture['fixture_sha256']:
                    raise RuntimeError('Trace frozen fixture identity mismatch')
                rows.extend(parsed)
            out['passes'].append(dict(cohort_size=trace_pass['cohort_size'], per_cta=rows,
                summary=summarize(rows,fixture['J64']), observer_reliable=trace_pass['observer_reliable']))
        report['fixtures'].append(out)
    common.save(args.output, report)


def run(args):
    started = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='starting', source_sha256=common.SOURCE_SHA,
        manifest_sha256=common.sha(args.manifest), submission_allowed=False, candidate_built=False,
        schema=SCHEMA, caveats=CAVEATS, fixtures=[], artifact_files=[],
        measurement=dict(scope='Standalone real whole-layer-derived V32 experts; PDL0; both layers;3 seeds; all132 CTAs per row',
            timing='unprofiled internal CUDA events;256MiB cold read flush outside events; paired rotating arms',
            reps=args.reps, capacity_per_actor=args.capacity, requested_cohort_size=args.cohort_size,
            max_trace_buffer_bytes=64*1024*1024,
            observer_threshold_fraction=.05, counterfactual_bound=False))
    common.save(args.output, report)
    try:
        rows = validate_manifest(args.manifest, args.baseline)
        import torch
        sys.path.insert(0, str(args.reference.resolve().parent))
        reference = common.load(args.reference, 'reference')
        bench = common.load(args.bench, '_round4_hol_bench')
        bench.torch, bench.reference = torch, reference
        module = common.load(args.baseline, '_round4_hol_frozen')
        props = torch.cuda.get_device_properties(0)
        if props.multi_processor_count != 132:
            raise RuntimeError('HOL lane must have exactly132 SMs; grid ownership may not change')
        report['environment'] = dict(python=sys.version, platform=platform.platform(),
            torch=torch.__version__, cuda=torch.version.cuda, gpu=props.name,
            sms=props.multi_processor_count, capability=list(torch.cuda.get_device_capability()))
        report['sources'] = {name: dict(path=str(path), sha256=common.sha(path)) for name,path in {
            'baseline':args.baseline,'reference':args.reference,'bench':args.bench,
            'checks':args.reference.with_name('checks.py'),'nvfp4':args.reference.with_name('nvfp4.py'),
            'runner':Path(__file__),'source_helper':Path(__file__).with_name('round4_hol_source.py'),
            'analysis_helper':Path(__file__).with_name('round4_hol_analysis.py'),
            'common':Path(common.__file__),
        }.items()}
        schema_path = args.output.with_name(args.output.stem + '-schema.json')
        common.save(schema_path, SCHEMA)
        report['artifact_files'].append(schema_path.name)
        flush = torch.ones(common.FLUSH_BYTES//4, dtype=torch.float32, device='cuda')
        stride = FOOTER + args.capacity * FIELDS
        debug_words = HEADER + args.cohort_size * ACTORS * stride
        if debug_words * 8 > 64*1024*1024:
            raise RuntimeError('Requested trace exceeds the64MiB bounded device buffer')
        debug = torch.zeros(debug_words, dtype=torch.int64, device='cuda')
        report['measurement']['allocated_trace_buffer_bytes'] = debug_words*8
        # At most one prepared state/weight seed is resident. The frozen gold is
        # already independently generated; only private clone goes to prepare.
        for seed_index, seed in enumerate(common.WEIGHT_SEEDS):
            report['status'] = 'preparing_seed_' + str(seed)
            common.save(args.output, report)
            weights = reference.generate_weights(seed)
            private = bench.clone_data(weights)
            del weights
            state = module.prepare(private)
            if int(state['nsm']) != 132:
                raise RuntimeError('Prepared grid differs from fixed132-CTA source ownership')
            modules, resources = build(module, state, args.capacity)
            # Fail before launch if resource pressure destroys global-barrier residency.
            for built in resources.values():
                if built['resources']['active_blocks_per_sm'] < 1:
                    raise RuntimeError('Instrumented kernel cannot keep its spin-barrier grid resident')
            native = module._moe(256, state)
            for row in (r for r in rows if r['weight_seed'] == seed):
                saved, tensors = common.load_fixture(args.manifest.parent / row['fixture_file'])
                if tuple(tensors['x'].shape) != (256,4096) or tuple(tensors['ids'].shape) != (256,8):
                    raise RuntimeError('Frozen V32 expert shape differs from current source')
                if not bool(((tensors['ids'] >= -1) & (tensors['ids'] < 32)).all()):
                    raise RuntimeError('Frozen ids must be local0..31 or-1')
                local = tensors['ids'][tensors['ids'] >= 0].long()
                counts = torch.bincount(local, minlength=32).cpu().tolist()
                if counts != row['n_e']:
                    raise RuntimeError('Frozen tensor routes differ from manifest histogram')
                fixture = {key:row[key] for key in ('case_id','fixture_sha256','layer','weight_seed','input_seed','n_e','J64','planned_weight_bytes')}
                fixture.update(passes=[], resources=resources,
                    byte_contract=dict(planned_weight_bytes_unchanged=row['planned_weight_bytes'],
                        routing_histogram_matches=True, weight_source='same generated weight_seed; first32 expert weights; no local_start weight remap',
                        instrumentation_traffic='Diagnostic global trace stores are extra bytes; no DRAM or bandwidth claim'))
                report['fixtures'].append(fixture)
                report['status'] = 'tracing_' + row['case_id']
                common.save(args.output, report)
                debug.zero_()
                debug[:2].copy_(torch.tensor([0,args.cohort_size],dtype=torch.int64,device='cuda'))
                captures = {
                    'original':common.capture(common.expert_fn(module,state,tensors,extension=native),flush),
                    'feature_off':common.capture(common.expert_fn(module,state,tensors,extension=modules['instrumented']),flush),
                    'traced':common.capture(common.expert_fn(module,state,tensors,extension=modules['instrumented'],debug=debug),flush),
                }
                # Host-resource-export original kernel must agree numerically too.
                resource_output = common.expert_fn(module,state,tensors,extension=modules['original'])()
                fixture['resource_mirror_oracle'] = numerical(resource_output,tensors)
                if not fixture['resource_mirror_oracle']['numerical_pass']:
                    raise RuntimeError('Source-identical resource-query mirror failed oracle')
                del resource_output
                fixture['before_oracle'] = check_outputs(captures,tensors)
                sizes = [args.cohort_size]
                for pass_index, cohort_size in enumerate(sizes):
                    trace_pass = dict(cohort_size=cohort_size, cohorts=[], per_cta=[], observer_reliable=True)
                    fixture['passes'].append(trace_pass)
                    for first in range(0,132,cohort_size):
                        count = min(cohort_size,132-first)
                        debug.zero_()
                        debug[:2].copy_(torch.tensor([first,count],dtype=torch.int64,device='cuda'))
                        torch.cuda.synchronize()
                        # Before oracle follows the same selected cohort setup.
                        for cap in captures.values():
                            cap['graph'].replay()
                            cap['end'].synchronize()
                        before = check_outputs(captures,tensors)
                        measured = timing(captures,args.reps,first+seed_index)
                        after = check_outputs(captures,tensors)
                        qualification = qualify(resources,measured)
                        trace_pass['observer_reliable'] &= qualification['observer_reliable']
                        trace_name = f'{args.output.stem}-{row["case_id"]}-c{cohort_size}-{first:03d}.r4hol.gz'
                        path = args.output.parent / trace_name
                        metadata = dict(schema_version=1, source_sha256=common.SOURCE_SHA,
                            case_id=row['case_id'], fixture_sha256=row['fixture_sha256'],
                            manifest_sha256=report['manifest_sha256'], first_cta=first,cta_count=count,
                            capacity=args.capacity, pass_index=pass_index,
                            final_traced_replay_event_us=measured['records'][-1]['us']['traced'],
                            resource_source_sha256=resources['instrumented']['source_sha256'])
                        used = HEADER + count*ACTORS*stride
                        cpu_trace = debug[:used].cpu()
                        write_trace(path,cpu_trace,metadata)
                        del cpu_trace
                        report['artifact_files'].append(trace_name)
                        cohort = dict(first_cta=first,cta_count=count,trace_file=trace_name,
                            trace_sha256=common.sha(path),trace_bytes=path.stat().st_size,
                            timing=measured,qualification=qualification,before_oracle=before,after_oracle=after)
                        trace_pass['cohorts'].append(cohort)
                        # Persist artifact identity even when overflow/order validation rejects it.
                        common.save(args.output,report)
                        _, analyzed = analyze_artifact(path)
                        trace_pass['per_cta'].extend(analyzed)
                        common.save(args.output,report)
                    trace_pass['summary'] = summarize(trace_pass['per_cta'],row['J64'])
                    trace_pass['evidence_qualification'] = ('observer-screen-passed; still observational, not removable time'
                        if trace_pass['observer_reliable'] else 'observer-unreliable; no exposure bound promoted')
                    if cohort_size > 1 and any(c['timing']['max_absolute_perturbation'] > .05 for c in trace_pass['cohorts']):
                        sizes.append(1)
                    common.save(args.output,report)
                fixture['after_oracle'] = check_outputs(captures,tensors)
                fixture['selected_pass'] = len(fixture['passes'])-1
                fixture['observer_reliable'] = fixture['passes'][-1]['observer_reliable']
                fixture['complete'] = True
                del captures,cap,tensors,saved,local
                common.save(args.output,report)
            del state,private,native,modules
            torch.cuda.empty_cache()
        report['status'] = 'complete'
        report['observer_reliable_all_six_fixtures'] = all(f['observer_reliable'] for f in report['fixtures'])
        report['all_six_fixtures_complete'] = len(report['fixtures']) == 6 and all(f['complete'] for f in report['fixtures'])
        if not report['all_six_fixtures_complete']:
            raise RuntimeError('Six V32 fixture coverage was not completed')
    except BaseException as exc:
        report.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        common.save(args.output,report)
    print(json.dumps(dict(report=str(args.output),status=report['status'])),flush=True)
    return 0 if report['status'] == 'complete' else 1


def main():
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path('round4-fixtures.json'))
    parser.add_argument('--baseline',type=Path,default=Path('decode_round4_control.py'))
    parser.add_argument('--bench',type=Path,default=Path('compare_decode_public.py'))
    parser.add_argument('--reference',type=Path,default=Path('reference.py'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cohort-size',type=int,default=4)
    parser.add_argument('--capacity',type=int,default=1024)
    parser.add_argument('--reps',type=int,default=5)
    parser.add_argument('--analyze',type=Path,help='Offline reanalysis of the retained sparse artifacts; no Torch/CUDA initialization')
    args = parser.parse_args()
    if args.output.suffix != '.json':
        parser.error('--output must end in.json')
    if not (1 <= args.cohort_size <= 132 and 16 <= args.capacity <= 4096 and 3 <= args.reps <= 100):
        parser.error('Require cohort-size1..132, capacity16..4096, reps3..100')
    if args.analyze:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        reanalyze(args)
        return 0
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
