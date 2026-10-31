"""Bounded sparse trace codec and conservative shared-ring HOL attribution.

All clocks remain Python integers until integer differences have been formed.
Strict means a trace-supported ordering/dependency chain, not a measured
counterfactual saving. An ambiguous last arrival cannot be called strict.
"""
import gzip
import json
import statistics
import struct
from collections import Counter, defaultdict

from round4_hol_source import ACTORS, FIELDS, FOOTER, HEADER

MAGIC = b'R4HOL01\n'
RECORD = struct.Struct('<HBBiQ')  # CTA, actor, field, sequence (-1=footer), uint64
SCHEMA = dict(version=1, encoding='gzip; magic R4HOL01\\n; uint32 LE JSON length; JSON; 16-byte sparse records <HBBiQ',
    record=['cta:uint16', 'actor:uint8', 'field:uint8', 'sequence:int32', 'value:uint64'],
    actors='0..15 consumer warp leaders;16 W producer;17 X producer; WG=warp//4',
    sequence='CTA-local, continuous A then B; producer mapping records exact reordered gs; reuse n-3',
    timestamps='raw uint64 %globaltimer ns; subtract integers before converting to float',
    consumer_fields={0:'W data wait begin',1:'W data wait end',2:'X data wait begin',3:'X data wait end',
        4:'W arrival before',5:'W arrival after',6:'X arrival before',7:'X arrival after',
        8:'wgs_flush begin',9:'wgs_flush end',10:'flush metadata: PB<<32 | (unit+1)<<4 | pending-kind',
        11:'unit end begin before wg_wait<0>',12:'unit end exit after epilogue or partial publication',
        13:'unit metadata: PB<<32 | (unit+1)'},
    producer_fields={0:'capacity wait begin',1:'capacity wait end',
        2:'xready begin (X only)',3:'xready end (X only)',4:'doneAs begin (X only)',5:'doneAs end (X only)',
        6:'TMA issue begin',7:'TMA issue end',14:'PB<<32 | (gs+1)'},
    footer_fields={0:'kernel entry',1:'routing complete',2:'phaseA complete',3:'phaseB complete',
        4:'global barrier exit',5:'kernel consumer exit',6:'gather handoff complete',7:'gather complete',8:'overflow'},
    sparse='Zero fields omitted. One artifact contains one replay and one contiguous CTA cohort. All132 CTAs covered across cohorts, NOT simultaneous.',
    phase_attribution='Exposure/wait phase is the requesting producer sequence phase; completion frequencies use the completed unit phase, so boundary-crossing evidence is not silently relabeled.',
    completion_index='Flush indexed by current seq (or tail seq); unit-end indexed by next seq; completion metadata identifies actual unit/phase.')


def union_ns(intervals):
    intervals = sorted((int(a), int(b)) for a, b in intervals if b > a)
    total, start, end = 0, 0, 0
    for a, b in intervals:
        if a > end:
            total += end - start
            start, end = a, b
        else:
            end = max(end, b)
    return total + end - start


def overlap(*intervals):
    a, b = max(v[0] for v in intervals), min(v[1] for v in intervals)
    return (a, b) if b > a else None


def stats(values):
    values = sorted(values)
    if not values:
        return dict(count=0, mean=0.0, median=0.0, p90=0, max=0)
    # Values are durations/counts, never absolute clocks.
    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p90=values[max(0, (9 * len(values) + 9)//10 - 1)], max=values[-1])


def write_trace(path, cpu_words, metadata):
    """Stream sparse fields from the bounded CPU tensor; no Python-int full copy."""
    first, count, capacity = metadata['first_cta'], metadata['cta_count'], metadata['capacity']
    stride = FOOTER + capacity * FIELDS
    if cpu_words.numel() != HEADER + count * ACTORS * stride:
        raise RuntimeError('Trace buffer length mismatch')
    view = cpu_words.numpy().reshape(-1)
    encoded = json.dumps(metadata, separators=(',', ':')).encode()
    with gzip.open(path, 'wb', compresslevel=1) as stream:
        stream.write(MAGIC)
        stream.write(struct.pack('<I', len(encoded)))
        stream.write(encoded)
        # Each chunk is one actor; at most (9 + capacity*16)*16 bytes.
        for c in range(count):
            for actor in range(ACTORS):
                base = HEADER + (c * ACTORS + actor) * stride
                chunk = bytearray()
                for index, value in enumerate(view[base:base+stride]):
                    if value:
                        if index < FOOTER:
                            sequence, field = -1, index
                        else:
                            sequence, field = divmod(index - FOOTER, FIELDS)
                        chunk.extend(RECORD.pack(first+c, actor, field, sequence, int(value)))
                stream.write(chunk)


def read_trace(path):
    with gzip.open(path, 'rb') as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise RuntimeError('Unknown HOL trace encoding')
        size_bytes = stream.read(4)
        if len(size_bytes) != 4:
            raise RuntimeError('Truncated trace header')
        size = struct.unpack('<I', size_bytes)[0]
        if size > 65536:
            raise RuntimeError('Oversized trace metadata')
        metadata = json.loads(stream.read(size))
        capacity, first, count = metadata['capacity'], metadata['first_cta'], metadata['cta_count']
        if not (16 <= capacity <= 4096 and 0 <= first < 132 and 1 <= count <= 132-first):
            raise RuntimeError('Invalid bounded trace dimensions')
        # One cohort only; records have a hard cap even for corrupted inputs.
        maximum = count * ACTORS * (FOOTER + capacity * FIELDS)
        ctas = {c: [dict() for _ in range(ACTORS)] for c in range(first, first+count)}
        records = 0
        while True:
            data = stream.read(RECORD.size)
            if not data:
                break
            if len(data) != RECORD.size:
                raise RuntimeError('Truncated trace record')
            records += 1
            if records > maximum:
                raise RuntimeError('Trace exceeds bounded record capacity')
            cta, actor, field, seq, value = RECORD.unpack(data)
            if cta not in ctas or actor >= ACTORS or not (-1 <= seq < capacity):
                raise RuntimeError('Trace record outside scope')
            if field >= (FOOTER if seq == -1 else FIELDS):
                raise RuntimeError('Trace field outside schema')
            target = ctas[cta][actor].setdefault(seq, {})
            if field in target:
                raise RuntimeError('Duplicate trace field')
            target[field] = value
        return metadata, ctas


def interval(record, field, required=False):
    a, b = record.get(field, 0), record.get(field+1, 0)
    if not a and not b and not required:
        return None
    if not a or not b or b < a:
        raise RuntimeError(f'Missing/reversed interval at field {field}')
    return a, b


def analyze_cta(cta, actors):
    """Require all16 releases; use bracketing to resolve last-warp ownership."""
    for actor in actors:
        if actor.get(-1, {}).get(8, 0):
            raise RuntimeError(f'CTA{cta}: bounded trace overflow; attribution rejected')
    producers = actors[16:18]
    sequences = sorted(n for n, record in producers[0].items() if n >= 0 and 14 in record)
    if not sequences or sequences != list(range(len(sequences))):
        raise RuntimeError(f'CTA{cta}: incomplete/noncontiguous producer sequence')
    nseq = len(sequences)
    mapping = {}
    for n in sequences:
        a, b = producers[0][n].get(14), producers[1].get(n, {}).get(14)
        if a != b:
            raise RuntimeError(f'CTA{cta}: W/X sequence mapping disagreement')
        pb, gs = a >> 32, (a & 0xffffffff) - 1
        if pb not in (0, 1) or gs < 0:
            raise RuntimeError('Invalid phase/stage mapping')
        mapping[n] = (pb, gs)
    if sorted(n for n, r in producers[1].items() if n >= 0 and 14 in r) != sequences:
        raise RuntimeError('X producer sequence coverage differs')
    by_phase = {phase: Counter() for phase in ('A', 'B')}
    stage_ranges = {}
    for pb in (0, 1):
        stages = sorted(gs for p, gs in mapping.values() if p == pb)
        if stages and stages != list(range(stages[0], stages[-1]+1)):
            raise RuntimeError('Duplicated or missing mapped stage inside CTA range')
        stage_ranges['AB'[pb]] = dict(stages=len(stages),
            first=stages[0] if stages else None, end=stages[-1]+1 if stages else None)
    waits, releases, completions = {}, {}, []
    pools = defaultdict(list)
    for warp in range(16):
        actor = actors[warp]
        for n in sequences:
            record = actor.get(n, {})
            for ring, wait_field, release_field in [('W', 0, 4), ('X', 2, 6)]:
                wait = interval(record, wait_field, True)
                release = interval(record, release_field, True)
                if release[0] < wait[1]:
                    raise RuntimeError('Consumer release precedes its data-wait completion')
                if n and release[0] < releases[ring, n-1, warp][1]:
                    raise RuntimeError('Per-warp release sequence order violated')
                waits[ring, n, warp], releases[ring, n, warp] = wait, release
                phase = 'AB'[mapping[n][0]]
                pools['consumer_' + ring + '_' + phase].append(wait)
        for n, record in actor.items():
            if n < 0:
                continue
            for kind, field, meta_field in [('flush', 8, 10), ('unit_end', 11, 13)]:
                iv = interval(record, field)
                if iv is None:
                    if meta_field in record:
                        raise RuntimeError('Completion metadata without interval')
                    continue
                if meta_field not in record:
                    raise RuntimeError('Completion interval missing metadata')
                meta = record[meta_field]
                phase = 'AB'[meta >> 32]
                unit = (((meta & 0xffffffff) >> 4) if kind == 'flush' else (meta & 0xffffffff)) - 1
                pending = (meta & 15) if kind == 'flush' else None
                comp = dict(id=len(completions), warp=warp, wg=warp//4, seq=n,
                            phase=phase, unit=unit, kind=kind, pending_kind=pending, interval=iv)
                completions.append(comp)
                pools['completion_' + phase].append(iv)
                by_phase[phase]['completion_warp_intervals'] += 1
    comp_by_warp = defaultdict(list)
    for comp in completions:
        comp_by_warp[comp['warp']].append(comp)
    strict_comps, loose_comps, strict_caps, loose_caps = set(), set(), set(), set()
    strict_units, loose_units = set(), set()
    strict_waits, loose_waits, cross_strict_waits, cross_loose_waits = set(), set(), set(), set()
    owner_counts, candidate_owner_counts, pair_counts = Counter(), Counter(), Counter()
    examples, ordering = [], Counter()
    for ring_index, ring in enumerate(('W', 'X')):
        producer = producers[ring_index]
        for n in sequences:
            phase = 'AB'[mapping[n][0]]
            pr = producer[n]
            issued = interval(pr, 6, True)
            cap = interval(pr, 0, n >= 3)
            if cap is not None:
                if cap[1] > issued[0]:
                    raise RuntimeError('Capacity wait overlaps subsequent producer issue')
                pools['capacity_' + ring + '_' + phase].append(cap)
                by_phase[phase]['producer_capacity_wait_intervals'] += 1
            if ring == 'X':
                for label, f in [('xready', 2), ('doneAs', 4)]:
                    ready = interval(pr, f)
                    if ready is not None:
                        if (cap and ready[0] < cap[1]) or ready[1] > issued[0]:
                            raise RuntimeError('Readiness/capacity/issue ordering violated')
                        pools[label + '_' + phase].append(ready)
                        by_phase[phase][label + '_wait_intervals'] += 1
            if n < 3:
                continue
            reused = n - 3
            rs = [releases[ring, reused, w] for w in range(16)]
            # The final arrival happened somewhere within each [before,after].
            last_before = max(r[0] for r in rs)
            candidates = [w for w, r in enumerate(rs) if r[1] >= last_before]
            for w in candidates:
                candidate_owner_counts[f'{ring}:warp{w}:wg{w//4}'] += 1
            owner = candidates[0] if len(candidates) == 1 else None
            ordering['capacity_waits'] += 1
            ordering['unique_last_release_owner' if owner is not None else 'ambiguous_last_release_owner'] += 1
            if owner is not None:
                owner_counts[f'{ring}:warp{owner}:wg{owner//4}'] += 1
            if max(r[0] for r in rs) > cap[1]:
                raise RuntimeError('Producer resumed before a required prior-slot release began')
            ordering['release_brackets_not_wholly_inside_capacity'] += int(
                owner is not None and not (cap[0] <= rs[owner][0] <= rs[owner][1] <= cap[1]))
            # All possible last owners are retained for loose overlap, but strict
            # demands disjoint arrival brackets and an explicit issue->wait chain.
            for blocker in candidates:
                rel = rs[blocker]
                previous_end = releases[ring, reused-1, blocker][1] if reused else 0
                for comp in comp_by_warp[blocker]:
                    ci = comp['interval']
                    if comp['seq'] > reused+1 or ci[1] <= previous_end or ci[0] >= rel[1]:
                        continue
                    # Unit-end at seq=reused+1 can delay that stage's X retirement.
                    # Its later epilogue suffix is NOT charged before this release.
                    owner_prefix = (max(ci[0], previous_end), min(ci[1], rel[0]))
                    loose_overlap = overlap(ci, cap)
                    if loose_overlap is None:
                        continue
                    strict_chain = (owner == blocker and cap[0] <= rel[0] <= rel[1] <= cap[1]
                                    and owner_prefix[0] < owner_prefix[1])
                    for waiter in range(16):
                        if waiter//4 == blocker//4:
                            continue
                        wi = waits[ring, n, waiter]
                        loose = overlap(wi, cap, ci)
                        if loose is None:
                            continue
                        cap_id = ring, n
                        unit_id = (comp['phase'], comp['unit'], comp['kind'], comp['wg'])
                        loose_comps.add(comp['id']); loose_caps.add(cap_id); loose_units.add(unit_id)
                        loose_waits.add((ring,n,waiter))
                        pools['loose_' + phase].append(loose)
                        pools['associated_envelope_' + phase].append(wi)
                        by_phase[phase]['loose_overlap_tuples_including_strict'] += 1
                        cross = waiter//8 != blocker//8
                        if cross:
                            cross_loose_waits.add((ring,n,waiter))
                            pools['cross_half_loose_' + phase].append(loose)
                            pools['cross_half_envelope_' + phase].append(wi)
                        # Waiter released its old slot before this blocker, then
                        # waits for the very sequence whose producer awaits reuse.
                        strict = overlap(wi, cap, owner_prefix) if (
                            strict_chain and rs[waiter][1] <= rel[0]
                            and wi[0] < issued[0] and cap[1] <= issued[1] <= wi[1]) else None
                        if strict is not None:
                            strict_comps.add(comp['id']); strict_caps.add(cap_id); strict_units.add(unit_id)
                            strict_waits.add((ring,n,waiter))
                            pools['direct_' + phase].append(strict)
                            by_phase[phase]['strict_chain_tuples'] += 1
                            pair = f'{blocker//4}->{waiter//4}'
                            pair_counts[pair] += 1
                            pools['pair_' + pair].append(strict)
                            if cross:
                                cross_strict_waits.add((ring,n,waiter))
                                pools['cross_half_direct_' + phase].append(strict)
                                by_phase[phase]['strict_cross_half_tuples'] += 1
                            if len(examples) < 32:
                                examples.append(dict(ring=ring, requested_sequence=n, reused_sequence=reused,
                                    requested_phase=phase, blocker_warp=blocker, blocker_wg=blocker//4,
                                    waiter_warp=waiter, waiter_wg=waiter//4, cross_half=cross,
                                    completion={k:v for k,v in comp.items() if k != 'interval'},
                                    completion_interval_ns=list(ci), capacity_interval_ns=list(cap),
                                    release_bracket_ns=list(rel), issue_interval_ns=list(issued),
                                    consumer_interval_ns=list(wi), direct_interval_ns=list(strict),
                                    direct_duration_ns=strict[1]-strict[0]))
    def aggregate(prefix):
        ivs = pools[prefix + '_A'] + pools[prefix + '_B']
        ns = union_ns(ivs)
        return dict(ns=ns, us=ns / 1000, phaseA_ns=union_ns(pools[prefix + '_A']),
                    phaseB_ns=union_ns(pools[prefix + '_B']))
    complete_units = {(c['phase'], c['unit'], c['kind'], c['wg']) for c in completions}
    frequency = dict(completion_warp_intervals=len(completions),
        completion_wg_operations=len(complete_units), producer_capacity_waits=ordering['capacity_waits'],
        strict_completion_warp_intervals=len(strict_comps), loose_completion_warp_intervals=len(loose_comps),
        strict_completion_wg_operations=len(strict_units), loose_completion_wg_operations=len(loose_units),
        strict_producer_capacity_waits=len(strict_caps), loose_producer_capacity_waits=len(loose_caps),
        consumer_data_wait_intervals=32*nseq,
        strict_other_wg_wait_intervals=len(strict_waits), loose_other_wg_wait_intervals=len(loose_waits),
        strict_cross_half_wait_intervals=len(cross_strict_waits), loose_cross_half_wait_intervals=len(cross_loose_waits))
    frequency['strict_per_completion'] = len(strict_comps)/len(completions) if completions else 0.0
    frequency['strict_per_producer_capacity_wait'] = len(strict_caps)/ordering['capacity_waits'] if ordering['capacity_waits'] else 0.0
    for pb, phase in enumerate('AB'):
        phase_completions = [c for c in completions if c['phase'] == phase]
        for label, selected in [('strict', strict_comps), ('loose', loose_comps)]:
            by_phase[phase][label + '_completion_warp_intervals'] = sum(c['id'] in selected for c in phase_completions)
        for label, selected in [('strict', strict_caps), ('loose', loose_caps)]:
            by_phase[phase][label + '_producer_capacity_waits'] = sum(mapping[n][0] == pb for ring,n in selected)
        by_phase[phase]['consumer_data_wait_intervals'] = 32*stage_ranges[phase]['stages']
        for label, selected in [('strict_other_wg',strict_waits),('loose_other_wg',loose_waits),
                                ('strict_cross_half',cross_strict_waits),('loose_cross_half',cross_loose_waits)]:
            by_phase[phase][label + '_wait_intervals'] = sum(mapping[n][0] == pb for ring,n,warp in selected)
        by_phase[phase]['strict_per_completion'] = (by_phase[phase]['strict_completion_warp_intervals'] /
            len(phase_completions) if phase_completions else 0.0)
        denominator = by_phase[phase]['producer_capacity_wait_intervals']
        by_phase[phase]['strict_per_producer_capacity_wait'] = (by_phase[phase]['strict_producer_capacity_waits'] /
            denominator if denominator else 0.0)
    footers = [actor.get(-1, {}) for actor in actors[:16]]
    if any(any(k not in f for k in (0,2,3,4,5)) for f in footers):
        raise RuntimeError('Missing consumer phase/final-stage landmarks')
    # Span over16 observed leaders, not an asserted whole-CTA or grid completion.
    landmarks = dict(consumer_leader_observed_span_ns=max(f[5] for f in footers)-min(f[0] for f in footers),
        last_phaseB_to_last_barrier_exit_ns=max(f[4] for f in footers)-max(f[3] for f in footers),
        phaseB_leader_exit_spread_ns=max(f[3] for f in footers)-min(f[3] for f in footers),
        post_barrier_observed_final_stage_ns=max(f[5] for f in footers)-min(f[4] for f in footers))
    exposures = {name: aggregate(name) for name in (
        'direct', 'loose', 'associated_envelope', 'cross_half_direct', 'cross_half_loose', 'cross_half_envelope')}
    wait_unions = {name: aggregate(name) for name in ('capacity_W','capacity_X','consumer_W','consumer_X','xready','doneAs','completion')}
    return dict(cta=cta, sequence_count=nseq,
        phases={phase: dict(by_phase[phase]) for phase in ('A','B')},
        mapping_summary=stage_ranges,
        exposed=exposures, wait_unions=wait_unions, frequency=frequency,
        last_release_owners=dict(owner_counts), candidate_last_release_owners=dict(candidate_owner_counts),
        release_order_checks=dict(ordering), cross_half_partition=[[0,1],[2,3]],
        strict_wg_pairs={key: dict(tuples=value, union_ns=union_ns(pools['pair_'+key])) for key,value in pair_counts.items()},
        landmarks=landmarks, strict_examples=examples,
        evidence_note='Direct is an integer interval-union of completion-prefix ∩ producer capacity ∩ other-WG data wait with a unique last-release bracket and same-sequence issue chain. Loose includes ambiguous overlap. Neither is removable time.')


def summarize(rows, jobs):
    if sorted(r['cta'] for r in rows) != list(range(132)):
        raise RuntimeError('Full fixture attribution requires each of all132 CTAs exactly once')
    coverage = {}
    for phase, stages_per_job in [('A',512),('B',256)]:
        cursor = 0
        spans = sorted((r['mapping_summary'][phase]['first'], r['mapping_summary'][phase]['end'])
                       for r in rows if r['mapping_summary'][phase]['stages'])
        for first,end in spans:
            if first != cursor:
                raise RuntimeError('CTA ranges overlap or omit actual planned stages')
            cursor = end
        if cursor != jobs*stages_per_job:
            raise RuntimeError('Observed stage count differs from frozen J64 job plan')
        coverage[phase] = dict(stages=cursor, expected_stages=jobs*stages_per_job, exact=True)
    names = rows[0]['exposed']
    duration_stats = {name: dict(ns=stats([r['exposed'][name]['ns'] for r in rows]),
                               us=stats([r['exposed'][name]['us'] for r in rows]),
                               phaseA_ns=stats([r['exposed'][name]['phaseA_ns'] for r in rows]),
                               phaseB_ns=stats([r['exposed'][name]['phaseB_ns'] for r in rows])) for name in names}
    frequencies = Counter()
    for row in rows:
        frequencies.update({k:v for k,v in row['frequency'].items() if not k.startswith('strict_per_')})
    frequencies['strict_per_completion'] = (frequencies['strict_completion_warp_intervals'] / frequencies['completion_warp_intervals']
                                           if frequencies['completion_warp_intervals'] else 0.0)
    frequencies['strict_per_producer_capacity_wait'] = (frequencies['strict_producer_capacity_waits'] / frequencies['producer_capacity_waits']
                                                      if frequencies['producer_capacity_waits'] else 0.0)
    phases = {}
    for phase in 'AB':
        total = Counter()
        for row in rows:
            total.update({k:v for k,v in row['phases'][phase].items() if not k.startswith('strict_per_')})
        total['strict_per_completion'] = (total['strict_completion_warp_intervals'] / total['completion_warp_intervals']
                                         if total['completion_warp_intervals'] else 0.0)
        total['strict_per_producer_capacity_wait'] = (total['strict_producer_capacity_waits'] / total['producer_capacity_wait_intervals']
                                                    if total['producer_capacity_wait_intervals'] else 0.0)
        phases[phase] = dict(total)
    return dict(per_cta_exposed=duration_stats, frequency=dict(frequencies), ctas=132,
        phases=phases, exact_planned_stage_coverage=coverage,
        final_stage={k:stats([r['landmarks'][k] for r in rows]) for k in rows[0]['landmarks']},
        interpretation='Cohorts are separate executions. Never add CTA/WG/warp overlap as elapsed time or infer a whole-grid critical path from stitched clocks. Global barrier and final-stage dependence may hide or relocate all local exposure; no production speedup or two-WG counterfactual bound is asserted.')
