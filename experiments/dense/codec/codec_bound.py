"""Ceiling-only BF16 entropy / register decode diagnostic, never a submission.

Uploads (flattened into /workspace): codec_bound.py, codec_probe.cu,
reference.py, checks.py. Run /opt/sglang/bin/python codec_bound.py
  --result codec-bound.json --seeds 5151,6161,7171
No baseline import, prepare(), GEMV, or official check is run.

Formats are concrete streams, not entropy estimates:
* Window W=2/4/8: raw low8 + sign + log2(W) offset, 32-byte header,
  sorted uint32 global escape indices and uint16 original BF16 bits. Decode
  clamps exponents, then overwrites escaped bit patterns (not FP32 correction).
* Tile512 alternative:9-bit local index +7-bit original e71 =16bits/escape,
  plus uint32 prefix offsets for every512values and a32-byte header. Exact
  bytes and CPU bit reconstruction are measured; full-format GPU read timing
  includes these arrays. No escape execution is timed in the core decoder.
* Full-exponent window8 allows odd alignment: mant7+sign1+exp3. Its tile
  records use local9+exponent8=17bits, with explicit chunk-tail byte padding.
* Dictionary10/11: top (2**B-1) uint16 patterns, B-bit symbols, last symbol
  reserved for escapes, uint16 escape values in encounter order, 32-byte header.
Every chunk is actually packed/unpacked; chunk boundaries are byte aligned.
All 65,536 BF16 codes are additionally round-tripped, including signed zeros,
subnormals, infinities and NaNs. Matrix-level raw16 fallback is available when
encoding expands; baseline correction safety is reported, never assumed.

Peak temporaries scale with --chunk-values (default262144), not matrix size.
At most one packed matrix buffer is on GPU alongside reference weights and a
256MiB read-flush buffer. Histograms have65536 bins; no full integer matrix.
The only decoded outputs in GPU microbenches are per-thread scalar checksums.
GPU modes11/12 intentionally OMIT escapes and measure clipped core payloads;
full-format byte-read floors INCLUDE the header and exact escape arrays.
Cycle metrics include checksum/loop cost and are not isolated instruction
latency. Register-only inputs remain opaque to the optimizer each iteration.

Attribution: codec_probe.cu contains approved Apache-2.0 vjeux/alygg77 decode8.
See its source header; all other diagnostic code is new lane code.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import statistics
import struct
import sys
import time
import traceback

import numpy as np
import torch

HEADER_BYTES = 32
LAYERS = ('qkv_gate', 'o', 'pooler', 'unpooler')
CPP = '''#include <torch/extension.h>
#include <vector>
void codec_run(int64_t,int64_t,int64_t,int64_t,bool,int64_t,int64_t,int64_t,int64_t,int64_t);
std::vector<int64_t> codec_resources(int64_t,bool);
'''


def save(path, report):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def progress(event, **fields):
    print(json.dumps(dict(event=event, **fields)), flush=True)


def entropy(hist):
    p = hist[hist > 0].astype(np.float64)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def histogram_report(hist):
    codes = np.arange(65536, dtype=np.uint32)
    def marginal(labels, size):
        out = np.zeros(size, dtype=np.int64)
        np.add.at(out, labels, hist)
        return out
    low = marginal(codes & 255, 256)
    sign = marginal(codes >> 15, 2)
    exponent = marginal((codes >> 7) & 255, 256)
    e71 = marginal((codes >> 8) & 127, 128)
    high = marginal(codes >> 8, 256)
    h = entropy(hist)
    return dict(values=int(hist.sum()), unique_codes=int(np.count_nonzero(hist)),
                shannon_bits_per_value=h,
                shannon_bytes_lower_bound=float(hist.sum()*h/8),
                low_byte_entropy=entropy(low), sign_entropy=entropy(sign),
                exponent8_entropy=entropy(exponent), e71_entropy=entropy(e71),
                high_byte_entropy=entropy(high),
                high_given_low_entropy=h-entropy(low),
                raw_low8_plus_ideal_conditional_high_bits=8+h-entropy(low),
                raw_low8_plus_independent_high_bits=8+entropy(high),
                low_byte_histogram=low.tolist(), sign_histogram=sign.tolist(),
                exponent8_histogram=exponent.tolist(), e71_histogram=e71.tolist())


def chunks(weight, count):
    flat = weight.view(torch.int16).reshape(-1)
    for start in range(0, flat.numel(), count):
        yield start, flat[start:start+count].cpu().numpy().view(np.uint16)


def fixed_pack(codes, width):
    dtype = np.uint32 if width > 16 else np.uint16
    bits = ((codes[:, None].astype(dtype) >> np.arange(width, dtype=dtype)) & 1).astype(np.uint8)
    return np.packbits(bits.reshape(-1), bitorder='little')


def fixed_unpack(data, width, n):
    bits = np.unpackbits(data, bitorder='little')[:n*width].reshape(n, width)
    dtype = np.uint32 if width > 16 else np.uint16
    return (bits.astype(dtype) << np.arange(width, dtype=dtype)).sum(1, dtype=dtype)


def window_encode(bits, width, base):
    # width is the total nominal bits/value, including the unchanged low8.
    expbits = width-9
    window = 1 << expbits
    e = (bits >> 8) & 127
    off = np.clip(e.astype(np.int16)-base, 0, window-1).astype(np.uint16)
    code = ((bits >> 15) << expbits) | off
    lo = (bits & 255).astype(np.uint8)
    if width == 12:
        g = code.astype(np.uint8).reshape(-1, 8)
        hi = (g[:, :4] | (g[:, 4:] << 4)).reshape(-1)
        unpacked = np.empty_like(g)
        hg = hi.reshape(-1, 4)
        unpacked[:, :4] = hg & 15
        unpacked[:, 4:] = hg >> 4
        got_code = unpacked.reshape(-1).astype(np.uint16)
    else:
        hi = fixed_pack(code, width-8)
        got_code = fixed_unpack(hi, width-8, len(bits))
    got = lo.astype(np.uint16) | ((base + (got_code & (window-1))) << 8) | ((got_code >> expbits) << 15)
    escaped = np.flatnonzero((e < base) | (e >= base+window)).astype(np.uint32)
    original = bits[escaped].copy()
    # Explicit serialize/parse side arrays to check exact representation.
    indices = np.frombuffer(escaped.tobytes(), dtype='<u4')
    originals = np.frombuffer(original.astype('<u2').tobytes(), dtype='<u2')
    clipped = got.copy()
    got[indices] = originals
    return lo, hi, escaped, original, got, clipped


def exponent_window_encode(bits, width, base):
    """11bit variant with odd/even full exponent window alignment allowed."""
    exponent = (bits >> 7) & 255
    offset = np.clip(exponent.astype(np.int16)-base, 0, 7).astype(np.uint16)
    code = (bits & 127) | (offset << 7) | ((bits >> 15) << 10)
    payload = fixed_pack(code, 11)
    decoded = fixed_unpack(payload, 11, len(bits))
    got = (decoded & 127) | ((base+((decoded >> 7) & 7)) << 7) | ((decoded >> 10) << 15)
    ix = np.flatnonzero((exponent < base) | (exponent >= base+8)).astype(np.uint32)
    original = np.frombuffer(bits[ix].astype('<u2').tobytes(), dtype='<u2')
    clipped = got.copy()
    got[ix] = original
    return payload, np.empty(0, dtype=np.uint8), ix, original, got, clipped


def tile_roundtrip(bits, clipped, ix, exponent8=False):
    """Concrete512value blocks: prefix32 offsets and packed local9+high7/exp8."""
    counts = np.bincount(ix//512, minlength=(len(bits)+511)//512)
    offsets = np.concatenate((np.zeros(1, dtype=np.uint32), counts.cumsum(dtype=np.uint32)))
    high = ((bits[ix] >> (7 if exponent8 else 8)) & (255 if exponent8 else 127)).astype(np.uint32)
    record = (ix % 512) | (high << 9)
    width = 17 if exponent8 else 16
    wire = fixed_pack(record, width) if exponent8 else record.astype('<u2').view(np.uint8)
    parsed = fixed_unpack(wire, width, len(ix)) if exponent8 else np.frombuffer(wire, dtype='<u2').astype(np.uint32)
    positions = np.repeat(np.arange(len(counts), dtype=np.uint32), np.diff(offsets).astype(np.int64))*512+(parsed & 511)
    got = clipped.copy()
    if exponent8:
        got[positions] = (got[positions] & 0x807f) | ((parsed >> 9) << 7)
    else:
        got[positions] = (got[positions] & 0x80ff) | ((parsed >> 9) << 8)
    return int(np.count_nonzero(got != bits))


def dictionary_encode(bits, width, table, lut):
    code = lut[bits]
    payload = fixed_pack(code, width)
    decoded_code = fixed_unpack(payload, width, len(bits))
    sentinel = (1 << width)-1
    escaped = code == sentinel
    raw = np.frombuffer(bits[escaped].astype('<u2').tobytes(), dtype='<u2')
    safe_table = np.append(table, np.uint16(0))
    got = safe_table[decoded_code]
    got[decoded_code == sentinel] = raw
    return payload.nbytes, raw.nbytes, got, int(escaped.sum())


def correction_exact(bits, clipped, escaped):
    if not len(escaped):
        return 0
    x = (bits[escaped].astype(np.uint32) << 16).view(np.float32)
    y = (clipped[escaped].astype(np.uint32) << 16).view(np.float32)
    with np.errstate(invalid='ignore', over='ignore'):
        d = x-y
        z = y+d
    return int(np.count_nonzero(~np.isfinite(d) | (z.view(np.uint32) != x.view(np.uint32))))


def analyze(weight, chunk_values, hist):
    n, k = weight.shape
    count = weight.numel()
    stats = histogram_report(hist)
    stats.update(shape=[n, k], raw_bytes=2*count, formats={})
    codes = np.arange(65536, dtype=np.uint16)
    e_hist = np.asarray(stats['e71_histogram'], dtype=np.int64)
    specs = []
    for width in (10, 11, 12):
        window = 1 << (width-9)
        base = int(np.convolve(e_hist, np.ones(window, dtype=np.int64), 'valid').argmax())
        specs.append((f'window{width}', width, base, None, None))
    full_hist = np.asarray(stats['exponent8_histogram'], dtype=np.int64)
    full_base = int(np.convolve(full_hist, np.ones(8, dtype=np.int64), 'valid').argmax())
    specs.append(('exponent_window11', 11, full_base, None, None))
    order = np.argsort(-hist, kind='stable')
    for width in (10, 11):
        table = order[:(1 << width)-1].astype(np.uint16)
        lut = np.full(65536, (1 << width)-1, dtype=np.uint16)
        lut[table] = np.arange(len(table), dtype=np.uint16)
        specs.append((f'dictionary{width}', width, None, table, lut))
    for name, width, base, table, lut in specs:
        began = time.perf_counter()
        payload_bytes = escapes = errors = side_bytes = correction_errors = 0
        tile_errors = tile_records_bytes = 0
        encoder = exponent_window_encode if name == 'exponent_window11' else window_encode
        row_escapes = np.zeros(n, dtype=np.int64)
        digest = hashlib.sha256()
        if base is not None:
            exhaustive = encoder(codes, width, base)
            generic_errors = int(np.count_nonzero(exhaustive[4] != codes))
            generic_correction_errors = correction_exact(codes, exhaustive[5], exhaustive[2])
            generic_tile_errors = tile_roundtrip(codes, exhaustive[5], exhaustive[2], name == 'exponent_window11')
        else:
            generic_errors = int(np.count_nonzero(dictionary_encode(codes, width, table, lut)[2] != codes))
            generic_correction_errors = None
        for start, bits in chunks(weight, chunk_values):
            digest.update(memoryview(bits))
            if base is not None:
                lo, hi, ix, original, got, clipped = encoder(bits, width, base)
                tile_errors += tile_roundtrip(bits, clipped, ix, name == 'exponent_window11')
                tile_record_bits = 17 if name == 'exponent_window11' else 16
                tile_records_bytes += (tile_record_bits*len(ix)+7)//8
                payload_bytes += lo.nbytes + hi.nbytes
                escapes += len(ix)
                side_bytes += ix.nbytes + original.nbytes
                np.add.at(row_escapes, (start+ix)//k, 1)
                correction_errors += correction_exact(bits, clipped, ix)
            else:
                pbytes, sbytes, got, ne = dictionary_encode(bits, width, table, lut)
                payload_bytes += pbytes
                side_bytes += sbytes
                escapes += ne
            errors += int(np.count_nonzero(got != bits))
        table_bytes = 0 if table is None else table.nbytes
        total = HEADER_BYTES + payload_bytes + side_bytes + table_bytes
        record = dict(nominal_bits=width, base=base, escape_count=escapes,
                      escape_fraction=escapes/count, payload_bytes=payload_bytes,
                      side_bytes=side_bytes, dictionary_bytes=table_bytes,
                      header_bytes=HEADER_BYTES, total_bytes=total,
                      actual_bits_per_value=total*8/count,
                      at_most_11_bits_including_overhead=total*8 <= 11*count,
                      raw_fallback_bytes=min(total, 2*count+HEADER_BYTES),
                      raw_fallback_selected=total > 2*count+HEADER_BYTES,
                      bit_errors=errors, generic_65536_bit_errors=generic_errors,
                      original_bits_sha256=digest.hexdigest(),
                      encode_roundtrip_seconds=time.perf_counter()-began)
        if base is not None:
            tile_record_bits = 17 if name == 'exponent_window11' else 16
            tile_offsets_bytes = 4*((count+511)//512+1)
            tile_total = HEADER_BYTES + payload_bytes + tile_offsets_bytes + tile_records_bytes
            record['tile512'] = dict(
                record_bits=tile_record_bits, offset_bytes=tile_offsets_bytes,
                records_bytes=tile_records_bytes, total_bytes=tile_total,
                tight_contiguous_records_bytes=(tile_record_bits*escapes+7)//8,
                record_chunk_values=chunk_values,
                chunk_padding_bytes=tile_records_bytes-(tile_record_bits*escapes+7)//8,
                actual_bits_per_value=tile_total*8/count,
                at_most_11_bits_including_overhead=tile_total*8 <= 11*count,
                bit_errors=tile_errors, generic_65536_bit_errors=generic_tile_errors,
                note='local9 index + original e71_7 (or exponent8); sign and retained low bits reused. Prefix32 per512block. Escape execution OMITTED from GPU core.')
            packed_index_width = max(1, (count-1).bit_length())
            record.update(correction_nonexact_count=correction_errors,
                          generic_correction_nonexact_count=generic_correction_errors,
                          baseline_escape_threshold_pass=escapes <= count//200,
                          tight_bitpacked_global_index_bits=packed_index_width,
                          tight_index_side_bytes=((escapes*packed_index_width+7)//8)+2*escapes,
                          escape_bitmap_rank_side_bytes=(count+7)//8+2*escapes+4*((count+255)//256+1),
                          upstream_int2_unpadded_bytes=8*escapes)
            record['upstream_sk2_tables_unbalanced'] = {}
            for rows in (8, 16, 32, 64):
                counts = row_escapes.reshape(-1, rows).sum(1)
                pmax = int(counts.max())+1
                pmax += pmax % 2
                tb = len(counts)*pmax*8
                record['upstream_sk2_tables_unbalanced'][str(rows)] = dict(
                    rows_per_cta=rows, ctas=len(counts), max_escapes=int(counts.max()),
                    pmax=pmax, table_bytes=tb, payload_plus_table_bytes=payload_bytes+tb,
                    note='Exact upstream sk2_table allocation for these rows, before optional balancing/permutation metadata; not all production routes.')
        stats['formats'][name] = record
        if errors or generic_errors or tile_errors or (base is not None and generic_tile_errors):
            raise RuntimeError(f'{name}: matrix/generic/tile reconstruction errors')
    return stats


def prepare_gpu_payload(weight, chunk_values, width, info, tiled=False):
    count = weight.numel()
    payload_bytes = count*width//8
    total = info['tile512']['total_bytes'] if tiled else info['total_bytes']
    # Wire layout for timing: low plane | high plane | index32 array |
    # original16 array | fixed32 header. Position of header does not affect size.
    data = torch.empty(total, dtype=torch.uint8, device='cuda')
    escaped_total = info['escape_count']
    escaped_done = 0
    expected = 0
    block_counts = np.zeros((count+511)//512, dtype=np.uint32) if tiled else None
    for start, bits in chunks(weight, chunk_values):
        lo, hi, ix, original, _, clipped = window_encode(bits, width, info['base'])
        def put(offset, arr):
            raw = np.ascontiguousarray(arr).view(np.uint8)
            data[offset:offset+raw.size].copy_(torch.from_numpy(raw).to('cuda'))
        put(start, lo)
        put(count+start*(width-8)//8, hi)
        if tiled:
            record = ((ix % 512) | (((original >> 8) & 127).astype(np.uint32) << 9)).astype('<u2')
            put(payload_bytes+2*escaped_done, record)
            local_counts = np.bincount(ix//512, minlength=(len(bits)+511)//512)
            block_counts[start//512:start//512+len(local_counts)] = local_counts
        else:
            put(payload_bytes+4*escaped_done, (ix+start).astype('<u4'))
            put(payload_bytes+4*escaped_total+2*escaped_done, original.astype('<u2'))
        escaped_done += len(ix)
        expected += int(clipped.astype(np.uint64).sum())
    if tiled:
        offsets = np.concatenate((np.zeros(1, dtype=np.uint32), block_counts.cumsum(dtype=np.uint32)))
        put(payload_bytes+2*escaped_total, offsets.astype('<u4'))
    header = struct.pack('<8sQQII', b'CDTIL001' if tiled else b'CDWIN001', count, escaped_total, width, info['base'])
    data[-HEADER_BYTES:].copy_(torch.from_numpy(np.frombuffer(header, dtype=np.uint8).copy()).to('cuda'))
    return data, payload_bytes, expected


def byte_sum(data, count, chunk_values):
    total = 0
    for start in range(0, count, chunk_values):
        raw = data[start:min(count, start+chunk_values)].cpu().numpy()
        if len(raw) % 4:
            raw = np.pad(raw, (0, 4-len(raw)%4))
        total += int(raw.view('<u4').astype(np.uint64).sum())
    return total


def register_expected(data, n, width, base, threads, repeats):
    low = data[:threads*8].cpu().numpy()
    high = data[n:n+threads*(width-8)].cpu().numpy()
    if width == 12:
        hg = high.reshape(-1, 4)
        code = np.concatenate((hg & 15, hg >> 4), axis=1).reshape(-1).astype(np.uint16)
    else:
        code = fixed_unpack(high, 3, threads*8)
    ebits=width-9
    decoded = low.astype(np.uint16) | ((base+(code & ((1 << ebits)-1))) << 8) | ((code >> ebits) << 15)
    return int(decoded.astype(np.uint64).sum())*repeats


class Bench:
    def __init__(self, ext, args):
        self.ext, self.args = ext, args
        self.blocks = torch.cuda.get_device_properties(0).multi_processor_count
        self.sums = torch.empty(self.blocks*256, dtype=torch.int64, device='cuda')
        self.cycles = torch.empty(self.blocks, dtype=torch.int64, device='cuda')
        self.flush = torch.ones(256*1024*1024, dtype=torch.uint8, device='cuda')
        self.flush_sums = torch.empty_like(self.sums)
        self.flush_cycles = torch.empty_like(self.cycles)

    def run(self, data, n, base, mode, expected, cold, registers=False):
        repeats = self.args.register_repeats if registers else 1
        def call():
            self.ext.codec_run(data.data_ptr(), n, base, mode, registers, repeats,
                               self.sums.data_ptr(), self.cycles.data_ptr(), self.blocks,
                               torch.cuda.current_stream().cuda_stream)
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        begin = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            if cold:
                self.ext.codec_run(self.flush.data_ptr(), self.flush.numel(), 0, 0, False, 1,
                                   self.flush_sums.data_ptr(), self.flush_cycles.data_ptr(), self.blocks,
                                   torch.cuda.current_stream().cuda_stream)
            begin.record()
            call()
            end.record()
        samples, cycle_samples = [], []
        for _ in range(self.args.samples):
            graph.replay()
            end.synchronize()
            samples.append(begin.elapsed_time(end)*1000)
            cycles = self.cycles.cpu().tolist()
            cycle_samples.append(dict(min=min(cycles), median=statistics.median(cycles), max=max(cycles)))
        actual = int(self.sums.sum().item())
        if actual != expected:
            raise RuntimeError(f'GPU checksum mode{mode} reg{registers}: {actual} != {expected}')
        values = self.blocks*256*8*repeats if registers else n if mode else None
        result = dict(mode=mode, registers=registers, cold=cold,
                      timing='external CUDA events inside graph, after read-only256MiB flush if cold; no subtraction',
                      samples_us=samples, median_us=statistics.median(samples),
                      checksum=actual, expected_checksum=expected, checksum_pass=True,
                      cycles_per_cta_last_replay=cycles, cycle_samples=cycle_samples,
                      resources=dict(zip(('registers_per_thread','local_bytes','static_shared_bytes','max_threads'),
                                         self.ext.codec_resources(mode, registers))),
                      grid_blocks=self.blocks, threads_per_block=256, register_repeats=repeats)
        if values:
            result.update(values=values,
                          summed_cta_cycles_per_value=sum(cycles)/values,
                          median_cta_cycles_per_thread_value=statistics.median(cycles)/(8*repeats) if registers else None,
                          cycle_caveat='CTA elapsed cycles/values, includes checksum and loop; NOT dependency latency or GEMV slowdown')
        read_bytes = (12 if mode == 12 else 11)*n//8 if mode else n
        if not registers:
            result.update(read_bytes=read_bytes, effective_TB_per_second=read_bytes/(result['median_us']*1e6))
        return result


@contextlib.contextmanager
def compiler_log(path):
    sys.stdout.flush()
    sys.stderr.flush()
    with open(path, 'w') as log:
        old = [os.dup(1), os.dup(2)]
        try:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            for fd, saved in zip((1,2), old):
                os.dup2(saved, fd)
                os.close(saved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', default='codec-bound.json')
    parser.add_argument('--seeds', default='5151,6161,7171')
    parser.add_argument('--chunk-values', type=int, default=262144)
    parser.add_argument('--samples', type=int, default=15)
    parser.add_argument('--register-repeats', type=int, default=4096)
    parser.add_argument('--entropy-only', action='store_true', help='Skip compilation/timing, still generate actual CUDA reference weights')
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(',')]
    if len(seeds) != 3 or len(set(seeds)) != 3 or seeds[0] != 5151:
        parser.error('Need seed5151 followed by two distinct independent seeds')
    if args.chunk_values < 512 or args.chunk_values % 512 or args.samples < 3 or args.register_repeats < 1:
        parser.error('chunk-values must be positive multiple512; samples>=3; register-repeats>=1')
    report = dict(kind='diagnostic_ceiling_not_submission', status='running',
                  seeds=seeds, artifact_files=[], matrices=[], errors=[],
                  argv=sys.argv, torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  device=torch.cuda.get_device_name(0), chunk_values=args.chunk_values,
                  prior_fact_scope='H200_MEASURED_FACTS section7:10.46bits,~1.3% lower-format escapes,+2.4% free-decode; exact seeds/aggregation unspecified. Recompute per-matrix and parent six-load bound.',
                  limitations=['No production candidate or GEMV timings.',
                               'GPU11/12 omit all escape lookup/correction and decode clipped BF16 codes.',
                               'Dictionary candidates have exact CPU packing/roundtrip but no GPU decoder.',
                               'Pure-read includes observable checksum; it is a measured stream proxy, not physical bandwidth guarantee.',
                               'FP32 router is unmodified; excluded from BF16 entropy and compression gain.'])
    save(args.result, report)
    try:
        import reference
        bench = None
        if not args.entropy_only:
            from torch.utils.cpp_extension import load_inline
            source_path = Path(__file__).with_name('codec_probe.cu')
            source = source_path.read_text()
            report['cuda_source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
            log_path = Path(args.result).with_name(Path(args.result).stem+'-compile.log')
            report['artifact_files'].append(log_path.name)
            save(args.result, report)
            progress('compile_start')
            began = time.perf_counter()
            with compiler_log(log_path):
                ext = load_inline(name='dense_codec_'+report['cuda_source_sha256'][:12],
                                  cpp_sources=CPP, cuda_sources=source,
                                  functions=['codec_run', 'codec_resources'],
                                  extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90,code=sm_90', '-Xptxas=-v'],
                                  verbose=True)
            report['compile_seconds'] = time.perf_counter()-began
            bench = Bench(ext, args)
            progress('compile_done', seconds=report['compile_seconds'])
            save(args.result, report)
        for seed in seeds:
            progress('generate_weights', seed=seed)
            weights = reference.generate_weights(seed, device='cuda')
            torch.cuda.synchronize()
            histograms = {}
            aggregate = np.zeros(65536, dtype=np.int64)
            for layer in LAYERS + ('unpooler_prefix1024',):
                weight = weights['unpooler'][:1024] if layer == 'unpooler_prefix1024' else weights[layer]
                if weight.dtype != torch.bfloat16 or not weight.is_contiguous():
                    raise RuntimeError(f'{layer}: expected contiguous actual BF16 reference weights')
                hist = np.zeros(65536, dtype=np.int64)
                for _, bits in chunks(weight, args.chunk_values):
                    hist += np.bincount(bits, minlength=65536)
                histograms[layer] = hist
                if layer != 'unpooler_prefix1024':
                    aggregate += hist
                progress('histogram_done', seed=seed, layer=layer, entropy=entropy(hist))
                row = dict(seed=seed, layer=layer, **analyze(weight, args.chunk_values, hist), microbench={})
                report['matrices'].append(row)
                save(args.result, report)
                if bench:
                    for width in (12, 11):
                        info = row['formats'][f'window{width}']
                        data, payload_bytes, expected = prepare_gpu_payload(weight, args.chunk_values, width, info)
                        n = weight.numel()
                        floor_payload = byte_sum(data, payload_bytes, args.chunk_values)
                        floor_total = byte_sum(data, data.numel(), args.chunk_values)
                        trials = row['microbench'][f'window{width}'] = {}
                        for cold in (False, True):
                            suffix = 'cold' if cold else 'warm'
                            trials['decode_core_'+suffix] = bench.run(data, n, info['base'], width, expected, cold)
                            trials['read_payload_'+suffix] = bench.run(data, payload_bytes, 0, 0, floor_payload, cold)
                            trials['read_full_format_'+suffix] = bench.run(data, data.numel(), 0, 0, floor_total, cold)
                            save(args.result, report)
                        reg_expected = register_expected(data, n, width, info['base'], bench.blocks*256, args.register_repeats)
                        trials['register_decode'] = bench.run(data, n, info['base'], width, reg_expected, False, True)
                        if width == 12:
                            t = bench.blocks*256
                            low_sum = byte_sum(data, t*8, args.chunk_values)
                            high_sum = byte_sum(data[n:], t*4, args.chunk_values)
                            trials['register_checksum_floor'] = bench.run(data, n, 0, 0, (low_sum+high_sum)*args.register_repeats, False, True)
                        save(args.result, report)
                        del data
                        tiled, _, _ = prepare_gpu_payload(weight, args.chunk_values, width, info, tiled=True)
                        tiled_sum = byte_sum(tiled, tiled.numel(), args.chunk_values)
                        for cold in (False, True):
                            suffix = 'cold' if cold else 'warm'
                            trials['read_tile512_'+suffix] = bench.run(tiled, tiled.numel(), 0, 0, tiled_sum, cold)
                        del tiled
                        save(args.result, report)
                progress('matrix_done', seed=seed, layer=layer,
                         bits={k:v['actual_bits_per_value'] for k,v in row['formats'].items()})
                save(args.result, report)
            histograms['aggregate_nonoverlapping_bf16'] = aggregate
            hist_path = Path(args.result).with_name(f'{Path(args.result).stem}-hist-{seed}.npz')
            np.savez_compressed(hist_path, **histograms)
            report['artifact_files'].append(hist_path.name)
            router = weights['router']
            router_hash = hashlib.sha256()
            raw = router.view(torch.uint8).reshape(-1)
            for start in range(0, raw.numel(), args.chunk_values):
                router_hash.update(memoryview(raw[start:start+args.chunk_values].cpu().numpy()))
            report.setdefault('seed_summaries', []).append(dict(
                seed=seed, aggregate_bf16=histogram_report(aggregate),
                router=dict(dtype=str(router.dtype), shape=list(router.shape), raw_bytes=router.numel()*router.element_size(),
                            sha256=router_hash.hexdigest(), unchanged=True)))
            del weight, weights, router, raw
            save(args.result, report)
        report['status'] = 'complete'
    except Exception:
        report['status'] = 'failed'
        report['errors'].append(traceback.format_exc())
        raise
    finally:
        report['peak_host_ru_maxrss_platform_units'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        save(args.result, report)
        progress('finished', status=report['status'], result=args.result)


if __name__ == '__main__':
    main()
