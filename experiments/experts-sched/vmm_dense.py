"""Allocation-only dense diagnostic; never a replacement projection kernel.

Packing uses the unchanged baseline helpers. Reconstruction and FP64 oracle
work in 64-row chunks; the baseline compressor itself uses its existing full
matrix workspace once per layer. Pure-read counts are logical, not DRAM bytes.
"""
import statistics
import torch


def _nbytes(tensor):
    return tensor.numel() * tensor.element_size()


def _require(flags):
    if not all(flags.values()):
        raise RuntimeError(f'Dense exact-representation validation failed: {flags}')
    return flags


def _reconstruct(module, raw, comp, exact):
    """Decode all 12-bit values, then apply every FP32 escape delta exactly."""
    n, k = raw.shape
    base = comp['base4'] & 255
    escapes = 0
    for start in range(0, n, 64):
        end = min(start + 64, n)
        packed = comp['packed'][start // 8:end // 8]
        low = packed[..., :512].reshape(-1, k // 64, 32, 16).to(torch.int32)
        codes = packed[..., 512:].reshape(-1, k // 64, 32, 8).to(torch.int32)
        code = torch.cat((codes[..., :4] & 15, codes[..., :4] >> 4,
                          codes[..., 4:] & 15, codes[..., 4:] >> 4), -1)
        bits = ((((code >> 3) << 7) | (base + (code & 7))) << 8) | low
        units = bits.to(torch.int16).view(torch.bfloat16)
        decoded = units.reshape(-1, k // 64, 8, 4, 4, 2, 2).permute(
            0, 2, 1, 4, 5, 3, 6).reshape(end - start, k).float()
        keep = (comp['rows'] >= start) & (comp['rows'] < end)
        rows = comp['rows'][keep] - start
        cols = comp['cols'][keep]
        decoded[rows, cols] += comp['delta'][keep]
        escapes += rows.numel()
        _require({'reconstructed_bf16_exact': exact(decoded.to(raw.dtype), raw[start:end]),
                  'reconstructed_fp32_exact': exact(decoded, raw[start:end].float())})
    return dict(reconstructed_bf16_exact=True, reconstructed_fp32_exact=True,
                escape_count=escapes, reconstruction_chunk_rows=64)


def _oracle(x, raw):
    result = torch.empty((x.shape[0], raw.shape[0]), device=x.device, dtype=torch.float64)
    xd = x.double()
    for start in range(0, raw.shape[0], 64):
        result[:, start:start + 64] = xd @ raw[start:start + 64].double().T
    return result


def _numerics(output, reference, exact):
    finite = bool(torch.isfinite(output).all())
    if not finite:
        return dict(finite=False, diagnostic_allclose=False)
    error = output.double() - reference
    return dict(finite=True, max_abs_error=float(error.abs().max()),
                rms_error=float(error.square().mean().sqrt()),
                relative_l2_error=float(error.norm() / reference.norm().clamp_min(1e-300)),
                rounded_fp64_bit_exact=exact(output, reference.to(output.dtype)),
                diagnostic_allclose=bool(torch.allclose(output.double(), reference, rtol=.02, atol=.02)))


def _gates(result):
    # Gate on >=3% stage speedup; retain the raw paired ratios for interpretation.
    result['stage_gain_at_least_three_percent'] = {
        name: bool(result['aa_within_one_percent'] and ratio >= 1.03)
        for name, ratio in result['median_paired_ratio'].items()}
    result['gate_is_diagnostic_only'] = True
    return result


def _layer(module, raw, layer, state, ext, flush, time_arms, exact, report, checkpoint):
    if raw.dtype != torch.bfloat16 or not raw.is_contiguous():
        raise ValueError('Dense probe requires the original contiguous BF16 weights')
    n, k = raw.shape
    entry = report['dense'][layer] = dict(shape=[n, k], raw_logical_bytes=_nbytes(raw),
        api_caveat='Codec is unchanged custom SK2; raw arms all use torch.mm(x, weight.T). '
                   'Codec versus raw is not an allocation-only comparison.',
        numerical_tolerance=dict(rtol=.02, atol=.02, relative_l2=.003, qualification=False), gemv={})
    raw_snapshot = raw.clone()
    compressed = None
    caps = report['capabilities']
    if not caps.get('vmm_supported', False) or not caps.get('generic_compression_supported', False):
        entry['compression'] = dict(status='skipped_unsupported', capabilities=caps)
    else:
        try:
            storage, metadata = ext.allocate(_nbytes(raw), True, True)
        except RuntimeError as exc:
            entry['compression'] = dict(status='allocation_denied', error=str(exc))
        else:
            entry['compression'] = dict(metadata=metadata, logical_bytes=_nbytes(raw))
            if metadata.get('compression_granted_is_generic') is not True:
                entry['compression']['status'] = 'not_granted_no_compressed_arm'
                del storage
            else:
                if metadata.get('compression_requested_is_generic') is not True:
                    raise RuntimeError('Compression grant metadata lacks requested compression evidence')
                ext.copy_into(storage, raw)
                compressed = storage.view(raw.dtype).view(raw.shape)
                _require({'compressed_copy_bit_exact': exact(compressed, raw)})
                entry['compression'].update(status='granted', initial_bit_exact=True)
                del storage  # The typed view retains the allocation through all captures.

    bfp = state.get('bfp')
    names = {m: module.bfp__sk2_name(layer, m) for m in (1, 8)}
    available = bfp is not None and all(name in bfp.get('sk2', {}) for name in names.values())
    expected = {}
    if available:
        comp = module.bfp_sk2_compress(raw)
        if comp is None:
            raise RuntimeError('Prepared codec exists but original weights no longer compress losslessly')
        entry['representation'] = _reconstruct(module, raw, comp, exact)
        entry['codec_paths'] = {}
        for name in dict.fromkeys(names.values()):
            ci, stream, table, pmax, base4, n_rows = bfp['sk2'][name]
            cfg = module.bfp_SK2_CONFIGS[ci]
            _, cfg_layer, cfg_rows, r, nt, nwr, wsk, upw, _, _ = cfg
            _require({'config_matches_layer': cfg_layer == layer,
                      'config_matches_rows': cfg_rows == n_rows == n,
                      'config_matches_name': cfg[0] == name,
                      'base4_exact': base4 == comp['base4']})
            want_stream = module.bfp_sk2_stream(comp['packed'], n_rows, r, nt, nwr, wsk, upw)
            want_table, want_pmax = module.bfp_sk2_table(comp, n_rows, r)
            _require({'prepared_stream_exact': exact(stream, want_stream),
                      'prepared_table_exact': exact(table, want_table), 'pmax_exact': pmax == want_pmax})
            expected[name] = (want_stream, want_table)
            entry['codec_paths'][name] = dict(stream_logical_bytes=_nbytes(stream),
                escape_table_logical_bytes=_nbytes(table), total_logical_bytes=_nbytes(stream) + _nbytes(table),
                escape_count=comp['rows'].numel(), escape_payload_bytes=comp['rows'].numel() * 8,
                pmax=pmax, base4=base4, prepared_stream_exact=True, prepared_table_exact=True)
        del comp
    else:
        entry['representation'] = dict(status='skipped_no_prepared_SK2_path')

    def backing_checks():
        flags = {'raw_weights_unchanged': exact(raw, raw_snapshot)}
        if compressed is not None:
            flags['compressed_weights_exact'] = exact(compressed, raw_snapshot)
        for name, (want_stream, want_table) in expected.items():
            flags[name + '_stream_exact'] = exact(bfp['sk2'][name][1], want_stream)
            flags[name + '_table_exact'] = exact(bfp['sk2'][name][2], want_table)
        return _require(flags)

    # Each stream/table read has its own completion buffer; no checksum/reduction
    # is included inside the timed interval. Count sums happen only in validation.
    read_inputs = {'raw_normal': (raw,), 'raw_repeat': (raw,)}
    if compressed is not None:
        read_inputs['raw_compressible'] = (compressed,)
    if available:
        for name in expected:
            read_inputs['codec_' + name] = tuple(bfp['sk2'][name][1:3])
    counts = {name: tuple(torch.empty(state['nsm'], device=raw.device, dtype=torch.int64)
                          for _ in tensors) for name, tensors in read_inputs.items()}

    def read_arm(name):
        def run():
            for tensor, out in zip(read_inputs[name], counts[name]):
                ext.read_bytes(tensor, out)
            return counts[name]
        return run

    def validate_reads(outputs):
        flags = backing_checks()
        completed = {}
        for name, buffers in outputs.items():
            got = [int(buf.sum()) for buf in buffers]
            wanted = [_nbytes(t) for t in read_inputs[name]]
            _require({name + '_completed_logical_bytes': got == wanted})
            completed[name] = got
        return dict(exact=flags, completed_logical_bytes=completed)

    read_arms = {name: read_arm(name) for name in read_inputs}
    entry['read'] = _gates(time_arms(read_arms, flush, 'raw_normal', 'raw_repeat', validate_reads,
                                   reps=12 * len(read_arms)))
    entry['read']['note'] = 'Codec reads actual stream plus padded escape table in two launches; logical bytes are not physical DRAM traffic.'
    entry['read']['logical_bytes'] = {name: [_nbytes(t) for t in tensors] for name, tensors in read_inputs.items()}
    checkpoint()

    for m in (1, 8):
        generator = torch.Generator(device=raw.device).manual_seed(731283 + m)
        x = torch.randn((m, k), device=raw.device, generator=generator).to(torch.bfloat16)
        reference = _oracle(x, raw)
        arms = {'raw_normal': lambda: torch.mm(x, raw.T),
                'raw_repeat': lambda: torch.mm(x, raw.T)}
        if compressed is not None:
            arms['raw_compressible'] = lambda: torch.mm(x, compressed.T)
        if available:
            arms['codec'] = lambda: module.bfp__run_sk2(x, layer, m, n, k, bfp)
        eager = {name: arm().clone() for name, arm in arms.items()}

        def validate_gemv(outputs):
            flags = backing_checks()
            for name, out in outputs.items():
                flags[name + '_replay_exact'] = exact(out, eager[name])
                if name.startswith('raw_'):
                    flags[name + '_allocation_only_exact'] = exact(out, outputs['raw_normal'])
            _require(flags)
            numerical = {name: _numerics(out, reference, exact) for name, out in outputs.items()}
            _require({name + '_reference_valid': item['finite'] and item['diagnostic_allclose']
                      and item.get('relative_l2_error', float('inf')) <= .003
                      for name, item in numerical.items()})
            return dict(exact=flags, numerical=numerical)

        result = _gates(time_arms(arms, flush, 'raw_normal', 'raw_repeat', validate_gemv))
        if available:
            result['codec_path'] = names[m]
            result['raw_over_codec_median_paired_ratio'] = result['median_paired_ratio']['codec']
        else:
            result['codec_status'] = 'skipped_no_prepared_SK2_path'
        result['compression_win_eligible'] = compressed is not None
        if available and compressed is not None:
            ratio = statistics.median(r['us']['codec'] / r['us']['raw_compressible']
                                      for r in result['records'])
            result['codec_over_compressible_median_paired_ratio'] = ratio
            result['beats_current_codec_by_three_percent'] = bool(result['aa_within_one_percent'] and ratio >= 1.03)
        else:
            result['beats_current_codec_by_three_percent'] = False
        entry['gemv'][str(m)] = result
        checkpoint()
    entry['final_exact'] = backing_checks()
    checkpoint()


def run_dense(module, weights, state, ext, flush, time_arms, exact, report, checkpoint):
    """Run exactly two layers and M1/M8; release all allocation views on return."""
    previous_pdl = module.bfp_SK2_PDL
    module.bfp_SK2_PDL = 0  # Isolated stages have no preceding dependent launch.
    try:
        for layer in ('pooler', 'qkv_gate'):
            _layer(module, weights[layer], layer, state, ext, flush, time_arms, exact, report, checkpoint)
    finally:
        module.bfp_SK2_PDL = previous_pdl
