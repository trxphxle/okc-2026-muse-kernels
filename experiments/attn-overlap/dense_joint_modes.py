# SPDX-License-Identifier: Apache-2.0
"""Timing-only SK2 joint deletions; never a submission implementation.

The unchanged control generates every translation unit before any patch is applied.
Mode 1 deletes epilogues/fixups, mode 2 deletes consumer arithmetic, and mode 3
combines those deletions in one captured kernel. Native producers, PDL positions,
launch geometry, and required ring handshakes are retained. The caller must freeze
correct SK2 return tensors outside timing; diagnostic outputs are discard-only.

Derived CUDA artifacts carry the control's complete attribution/license header.
No compilation or CUDA work occurs merely by importing this builder.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


_CONTROL_SHA256 = "053dbacc8f0de4e6111ccb89cb91f629840852cf482a6ca423fe0eaff419c086"
_KEYS = ("main", "narrow", "gle")
_ATTRIBUTES = dict(zip(_KEYS, ("bfp__ext", "_DE_EXT", "_GLE_EXT")))
_MAIN_EXPORTS = ["skinny_tma_a", "gemm90", "wide", "sk2", "gq", "sk2f", "sk2_set_pdl"]
_SETTER = "sk2_set_joint_diag"
_SKIP = "if (dgv & 2) { __syncwarp(); if (lane == 0) mbar_arrive(empty0 + 8 * slot); continue; }"


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _replace(source, old, new, *, count=1):
    actual = source.count(old)
    if actual != count:
        raise RuntimeError(f"joint source anchor: expected {count}, found {actual}: {old[:160]!r}")
    return source.replace(old, new)


def _bounds(source, start, end):
    if source.count(start) != 1 or source.count(end) != 1:
        raise RuntimeError(f"joint region anchors are not unique: {start[:100]!r}, {end[:100]!r}")
    a = source.index(start)
    b = source.index(end)
    if b <= a:
        raise RuntimeError("joint region anchors are out of order")
    return a, b


def _wrap(source, start, end, keepalive):
    a, b = _bounds(source, start, end)
    return (source[:a] + "    if (dgv & 1) {\n" + keepalive +
            "    } else {\n" + source[a:b] + "    }\n" + source[b:])


def _keepalive(array, sizes):
    """Empty volatile asm keeps *all* omitted-epilogue inputs live, without stores."""
    text = ""
    indices = []
    for i, size in enumerate(sizes):
        index = f"j{i}"
        indices.append(index)
        text += f"#pragma unroll\n      for (int {index} = 0; {index} < {size}; ++{index})\n"
    expression = array + "".join(f"[{i}]" for i in indices)
    return text + f'        asm volatile("" :: "f"({expression}));\n'


def _producer(source):
    a, b = _bounds(
        source,
        "  if (tid == 0) {\n    // stage 0 first of all",
        "  if (warp >= NW + 1) {\n    // NESC escape warps",
    )
    return source[a:b]


def _patch(source, family):
    original_producer = _producer(source)
    source = _replace(
        source, "#define SK2_DIAG 0",
        "#define SK2_DIAG 1\n"
        "// Timing-only: copied by value into each kernel launch (and CUDA graph node).\n"
        "static int g_sk2_joint_diag = 0;\n"
        "void sk2_set_joint_diag(int64_t mode) {\n"
        '  TORCH_CHECK(mode >= 0 && mode <= 3, "joint diagnostic mode must be 0..3");\n'
        "  g_sk2_joint_diag = (int)mode;\n}",
    )
    source = _replace(
        source,
        "  const int dgv = SK2_DIAG ? dg : 0;   // diagnostics (dg) compiled out of production builds",
        "  const int dgv = SK2_DIAG ? (dg & 3) : 0;   // no x-load omission, fake reads, ALU substitution, or tracing",
    )
    source = _replace(
        source, "  auto go = [&](auto pkc) {\n    constexpr bool PK = decltype(pkc)::value;",
        "  const int joint_dg = g_sk2_joint_diag;\n"
        "  auto go = [&](auto pkc) {\n    constexpr bool PK = decltype(pkc)::value;",
    )
    source = _replace(source, "M, N, K, (int)dg, dbg,", "M, N, K, joint_dg, dbg,", count=2)

    # Escape helpers retain their completion barrier and all producer dependencies.
    # Narrow's direct reads have NO full/empty participation: adding it deadlocks.
    helper_start = "    if (COMP) {\n      float* cr = corr + ew * CSZ;"
    helper_end = "    }\n    mbar_arrive(cbar);\n    __syncwarp();"
    a, b = _bounds(source, helper_start, helper_end)
    original_helper = source[a + len("    if (COMP) {\n"):b]
    if family == "narrow":
        if "mbar_arrive(empty0" in original_helper or "SK2_WAIT(full0" in original_helper:
            raise RuntimeError("narrow escape helper unexpectedly participates in ring barriers")
        deleted_helper = r'''      mbar_wait_sleep(pbar, 0);
      const int cnt = ptab_s[0].x;
      constexpr int KST = UPS * 64;
      // Preserve the original sparse activation addresses/bytes, not correction math.
      for (int pe = 1 + ew; pe <= cnt; pe += NESC) {
        const int2 e = ptab_s[pe];
        const int kk = e.x & 65535;
        if (lane < M && kk >= gs0 * KST && kk < (gs0 + nst) * KST) {
          unsigned short bits;
          asm volatile("ld.global.b16 %0, [%1];" : "=h"(bits)
                       : "l"(x + (size_t)lane * K + kk) : "memory");
        }
      }
'''
    else:
        deleted_helper = r'''      mbar_wait_sleep(pbar, 0);   // escape-table bulk copy must finish too
      for (int st = 0; st < nstt; ++st) {
        const int slot = st % NSTAGE;
        SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
'''
    source = (source[:a] + "    if (COMP) {\n    if (dgv & 1) {\n" + deleted_helper +
              "    } else {\n" + original_helper + "    }\n" + source[b:])

    # The BF16 fused router has its own stage loop, outside the original bit-2 hook.
    router_wait = ("    for (int st = nst; st < nstt; ++st) {\n"
                   "      const int slot = st % NSTAGE;\n"
                   "      mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);\n")
    source = _replace(source, router_wait, router_wait + "      " + _SKIP + "\n")
    if family == "gle":
        # Interleaved FP32 work occurs BEFORE the original bit-2 hook. Move that
        # hook to immediately after the common wait, covering either kind of stage.
        source = _replace(source, "\n    " + _SKIP + "\n", "\n")
        ilv_wait = ("  for (int st = 0; st < (ilv ? nstt : nst); ++st) {\n"
                    "    const int slot = st % NSTAGE;\n"
                    "    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);\n")
        source = _replace(source, ilv_wait, ilv_wait + "    " + _SKIP + "\n")
        fp32_wait = ("      for (int st = nst; st < nstt; ++st) {\n"
                     "        const int slot = st % NSTAGE;\n"
                     "        mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);\n")
        source = _replace(source, fp32_wait, fp32_wait + "        " + _SKIP + "\n")

        # Keep FP32 accumulators live in mode 1 while removing every reduction,
        # partial store, quarter counter, and router output store.
        source = _wrap(
            source,
            "#pragma unroll\n      for (int t = 0; t < 8; ++t)\n#pragma unroll\n"
            "        for (int d = 16; d; d >>= 1) afi[t] += __shfl_xor_sync(0xffffffffu, afi[t], d);",
            "    } else if (ep.fpr) {\n", _keepalive("afi", (8,)),
        )
        source = _wrap(
            source,
            "      // reduction over the 128 column groups through the (now idle) ring:",
            "    } else {\n    // router piece:", _keepalive("af", (32,)),
        )

    reduction_end = ("    }\n    __shared__ int s_rlast;" if family == "gle"
                     else "    __shared__ int s_rlast;")
    source = _wrap(
        source, "    // k-warp reduction of the 8 x TOKS piece through the (now idle) ring:",
        reduction_end, _keepalive("accr", (2, "MT", 4)),
    )
    # Counter/fence protocol belongs to the deleted reduction, not the load ring.
    # All CTAs skip it together, so counters remain unchanged and reusable.
    common_start = "    __shared__ int s_rlast;"
    common_end = "    named_bar(1, NC);   // the ring (redr) is reused by the pooler epilogue below\n"
    a, b = _bounds(source, common_start, common_end)
    b += len(common_end)
    source = (source[:a] + "    if (!(dgv & 1)) {\n" + source[a:b] +
              "    }\n" + source[b:])
    source = _replace(
        source,
        "  if (dgv & 1) {\n"
        "    if (acc[0][0][0][0] == 1.2345e-30f) out[tid] = cvt_out<OutT>(acc[NACC - 1][0][0][1]);\n"
        "    return;\n  }",
        "  if (dgv & 1) {\n" + _keepalive("acc", ("NACC", "MT", "NT", 4)) +
        "    if constexpr (COMP) mbar_wait(cbar, 0);\n"
        '    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");\n'
        "    return;\n  }",
    )
    if _producer(source) != original_producer:
        raise RuntimeError(f"{family}: joint patch changed the producer/prologue")
    return source


def _launches(source, family, control):
    """Per-instantiation payload counts; pmax is intentionally fixture-dependent."""
    launches = []
    pattern = r"launch_sk2<([^<>]+)>\(Ws, x, out, ptab, pmax, base4, M, ([^;]+)"
    for match in re.finditer(pattern, source):
        args = [x.strip() for x in match.group(1).split(",")]
        if len(args) not in (12, 13):
            raise RuntimeError(f"unexpected SK2 template arity: {args}")
        mt, tokr, nt, nwr, wsk, upw, stages = map(int, args[:7])
        comp, hl = args[7] == "true", args[8] == "true"
        nacc, cl = map(int, args[9:11])
        fused = len(args) == 13 and args[12] == "true"
        prefix = source[source.rfind("\n", 0, match.start()) + 1:match.start()]
        config_match = re.search(r"cfg == (\d+)", prefix)
        if fused:
            config = next(c for c in control.bfp_SK2_CONFIGS if c[0] == "pooler")
        elif family == "narrow":
            config = ("qkv8n_r144", "qkv_gate", 2304, 24, 3, 1, 4, 2, 2, 1)
        elif config_match:
            config = control.bfp_SK2_CONFIGS[int(config_match.group(1))]
        else:
            raise RuntimeError(f"cannot attribute SK2 dispatch: {prefix}")
        name, layer, n, r = config[:4]
        k = control.bfp_SK2_K[layer]
        nw, ups = nwr * wsk, wsk * upw
        tu = 768 if comp else 2048 if hl else 1024
        wby = nw * upw * nt * tu
        xby = ups * tokr * 128
        sst = (xby + wby + 1023) & ~1023
        nesc = (4 if family == "narrow" else min(stages, 4)) if comp else 1
        total_stages = k // (ups * 64)
        if k % (ups * 64) or n % r or r != nwr * nt * 8:
            raise RuntimeError("source-derived stage geometry is not integral")
        entry = {
            "template": match.group(1), "config": name, "layer": layer,
            "fused": fused, "tokr": tokr, "mt": mt, "n": n, "k": k,
            "rows_per_cta_group": r, "cluster_size": cl, "ctas": n // r * cl,
            "consumer_warps": nw, "escape_warps": nesc,
            "threads": (nw + 1 + nesc) * 32, "ring_slots": stages,
            "ring_slot_bytes": sst, "weight_bytes_per_primary_stage": wby,
            "dynamic_shared_bytes_without_ptab": (
                1024 + stages * sst + 16 * stages + 32 +
                nesc * ((16 * mt * (r + 1) + 3) & ~3) * 4 +
                (cl - 1) * wsk * 16 * mt * (r + ((40 - r % 32) % 32)) * 4 + 128
            ),
            "dynamic_shared_ptab_bytes": "pmax * 8",
            "primary_stages_by_cluster_rank": [
                (rank + 1) * total_stages // cl - rank * total_stages // cl for rank in range(cl)
            ],
            "primary_weight_payload_bytes": n // r * total_stages * wby,
            "primary_activation_payload_bytes_per_token": n // r * total_stages * ups * 128,
            "primary_activation_stage_bytes": "UPS * (PK ? M : TOKR) * 128",
            "escape_table_payload_bytes": f"{n // r * cl} * pmax * 8" if comp else "0",
            "full_barrier_arrivals": 1,
            "empty_barrier_arrivals": nw + (nesc if comp and family != "narrow" else 0),
            "escape_completion_arrivals": 32 * nesc,
            "cluster_peer_completion_bytes": (cl - 1) * nw * 32 * mt * nt * 16,
            "nacc": nacc,
        }
        if fused:
            entry["router"] = {
                "stages_per_cta": 4,
                "weight_bytes_per_stage": nw * 2048,
                "weight_payload_bytes": n // r * 4 * nw * 2048,
                "bf16_activation_payload_bytes_per_token": n // r * 4 * ups * 128,
                "fp32_activation_payload_bytes_per_token": n // r * 4 * 2048 if family == "gle" else None,
                "fp32_activation_bytes_per_stage": "M * 2048" if family == "gle" else None,
                "fp32_interleave": "unchanged GleEpi.ilv/PER/OFF" if family == "gle" else None,
            }
        launches.append(entry)
    if not launches:
        raise RuntimeError(f"{family}: no launch instantiations found")
    return launches


def _binding(declarations, functions):
    # This is load_inline's default error-handled binding convention, emitted
    # explicitly so the artifact is the complete C++ translation unit compiled.
    lines = ["#include <torch/extension.h>\n", declarations,
             "\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"]
    lines.extend(f'  m.def("{name}", torch::wrap_pybind_function({name}), "{name}");\n'
                 for name in functions)
    lines.append("}\n")
    return "".join(lines)


def originals(control):
    """Return original modules prepared by the control, even after installation."""
    saved = getattr(control, "_dense_joint_original_modules", None)
    if saved is None:
        saved = {key: getattr(control, attr) for key, attr in _ATTRIBUTES.items()}
        if any(module is None for module in saved.values()):
            raise RuntimeError("prepare the exact original control before building/installing joint modes")
        control._dense_joint_original_modules = dict(saved)
    return dict(saved)


def install(control, modules):
    """Swap only native module caches; never prepare, mutate routes, or wrap hooks."""
    if set(modules) != set(_KEYS) or any(modules[key] is None for key in _KEYS):
        raise ValueError("joint modules must contain main, narrow, and gle")
    originals(control)
    for key, attr in _ATTRIBUTES.items():
        setattr(control, attr, modules[key])


def set_mode(modules, mode):
    """Set before capture: replay retains the mode copied into each launch node."""
    if type(mode) is not int or not 0 <= mode <= 3:
        raise ValueError("joint diagnostic mode must be an integer in 0..3")
    if set(modules) != set(_KEYS):
        raise ValueError("joint modules must contain main, narrow, and gle")
    for key in _KEYS:
        getattr(modules[key], _SETTER)(mode)


def build_all(control, artifact_prefix: Path, on_artifact):
    """Build the three native diagnostic modules, returning (modules, metadata).

    The callback receives each artifact basename immediately after it is written,
    before any compilation. Build failures therefore leave retrievable sources.
    This function is for the parent GPU harness only; it does not prepare inputs.
    """
    control_path = Path(control.__file__)
    control_bytes = control_path.read_bytes()
    control_hash = hashlib.sha256(control_bytes).hexdigest()
    if control_hash != _CONTROL_SHA256:
        raise RuntimeError(f"joint control SHA mismatch: {control_hash}")
    control_text = control_bytes.decode()
    header = control_text.split('"""Muse Spark decode layer:', 1)
    if len(header) != 2 or "# END NATHEN FERNANDES MIT LICENSE" not in header[0]:
        raise RuntimeError("control attribution boundary mismatch")
    attribution = "\n".join("//" + line[1:] if line.startswith("#") else "// " + line
                            for line in header[0].splitlines()) + "\n"
    attribution += "// Timing-only joint-deletion diagnostic; not for submission.\n"

    # DO NOT change bfp_SK2_SRC: _gle_source has exact SK2_DIAG=0 anchors.
    common = _replace(control.bfp_CUDA_SRC, "@@MBAR@@", control.bfp_MBAR_HELPERS)
    common = _replace(common, "@@BODY@@", control.bfp_WG_HELPERS)
    base_sources = {
        "main": (common + control.bfp_wgmma_defs([8, 16, 32, 64, 128]) +
                 control.bfp__tma_srcs(True) + control.bfp_MAP_SRC +
                 control.bfp__gemm_src(True) + control.bfp__wide_src(True) +
                 control.bfp__sk2_src(True) + control.bfp__gq_src(True)),
        "narrow": control._de_source(),
        "gle": control._gle_source(),
    }
    declarations = {
        "main": control.bfp_CPP_SRC,
        "narrow": (control.bfp_CPP_SRC[control.bfp_CPP_SRC.index("void sk2("):
                                        control.bfp_CPP_SRC.index("void sk2f(")] +
                   "\nvoid sk2_set_pdl(int64_t n);\n"),
        "gle": (
            "void gle_pr(torch::Tensor Ws, torch::Tensor Wsr, torch::Tensor x, torch::Tensor out, torch::Tensor outr,"
            " torch::Tensor ptab, int64_t pmax, int64_t base4, torch::Tensor part, torch::Tensor rcnt, int64_t M,"
            " torch::Tensor x32, torch::Tensor w32, int64_t wpf);\nvoid sk2_set_pdl(int64_t n);\n"
        ),
    }
    exports = {"main": _MAIN_EXPORTS, "narrow": ["sk2", "sk2_set_pdl"],
               "gle": ["gle_pr", "sk2_set_pdl"]}
    metadata = {
        "control_sha256": control_hash,
        "mode_source": "host setter -> launch_sk2 joint_dg -> by-value captured kernel dg & 3",
        "original_dg_argument": "retained API parameter; overridden by the host-set joint mode",
        "modes": {"0": "feature off", "1": "epilogue and fixup deletion",
                  "2": "consumer and decoder deletion", "3": "joint deletion in the same kernels"},
        "raw_original_bit_coverage": {
            "1": "primary reduction/output only; escape helper computation and fused-router reductions were NOT covered",
            "2": "primary consumer loads/decode/MMA only; fused BF16/FP32 router math was NOT covered",
        },
        "complete_category_coverage": {
            "1": ["all primary accumulator reductions, corrections, conversion, and output stores",
                  "main/GLE escape-table traversal and correction FMAs (producer table copy retained)",
                  "narrow direct-escape correction FMAs and correction-buffer initialization",
                  "fused BF16, FP32 tail, and interleaved FP32 reductions/partial stores/output stores",
                  "fused quarter counters and release/acquire fences used solely by the deleted reduction"],
            "2": ["primary consumer shared loads, activation ldmatrix, codec decode, and MMA",
                  "fused BF16 hi/lo consumer shared loads, ldmatrix, and both MMAs",
                  "fused FP32 tail and interleaved consumer shared loads and all FMAs"],
            "3": "union of complete categories 1 and 2, measured directly, never summed",
        },
        "retained_protocol": [
            "identical producer/prologue source and same real pointers, bulk/TMA bytes, ordering, and stage count",
            "every consumer observes every full stage and arrives at empty exactly once per warp",
            "main/GLE escape warps observe and release every stage; narrow escape helpers never touch ring barriers",
            "escape-table pbar completion and escape cbar arrivals/waits",
            "cluster arrive/acquire participation; deleted cross-CTA epilogue qbar exchange is not awaited",
            "original PDL waits and launch_dependents positions, grid/block/shared memory, stream, and launch attributes",
        ],
        "limitations": [
            "timing-only: modes 1..3 do not return valid dense outputs; harness must return frozen correct tensors",
            "narrow mode1 retains sparse table traversal and volatile loads of the original BF16 activation addresses/bytes; only fixup math and correction stores are deleted",
            "mode2 deliberately retains escape-fixup work; mode1 deliberately retains all consumer accumulations via empty volatile-asm keepalives",
            "only SK2-family launches are altered; cuBLAS/GQ/other dense routes (normally V16/V32) remain original and must be reported as uncovered",
            "payload counts are source-derived transfer requests, not measured HBM traffic; pmax and actual route coverage come from each fixture",
            "runtime mode branches and register/codegen effects require a separately measured diagnostic mode0 versus production A/A control",
            "host setters are process-local and must not race captures or launches on other host threads",
        ],
        "artifact_files": [], "modules": {},
    }
    artifact_prefix = Path(artifact_prefix)
    artifact_prefix.parent.mkdir(parents=True, exist_ok=True)
    pending = {}
    for family in _KEYS:
        original = base_sources[family]
        if "@@" in original:
            raise RuntimeError(f"{family}: unresolved generator placeholder")
        patched = _patch(original, family)
        digest = _sha(patched)[:16]
        kernel_name = f"sk2_kernel_joint_{family}_{digest}"
        launch_name = f"launch_sk2_joint_{family}_{digest}"
        launches = _launches(original, family, control)
        # Match the original GLE generator's private host-symbol separation too.
        patched = _replace(patched, "sk2_kernel", kernel_name, count=2)
        patched = _replace(patched, "launch_sk2", launch_name, count=len(launches) + 1)
        if family == "gle":
            patched = _replace(patched, "void sk2(", f"void sk2_joint_gle_{digest}(")
            patched = _replace(patched, "void sk2f(", f"void sk2f_joint_gle_{digest}(")
            patched = _replace(patched, "  sk2f(Ws,", f"  sk2f_joint_gle_{digest}(Ws,")
        source = attribution + patched
        functions = exports[family] + [_SETTER]
        cpp = attribution + _binding(declarations[family] + "\nvoid sk2_set_joint_diag(int64_t mode);\n", functions)
        flags = ["-O3", "-gencode=arch=compute_90a,code=sm_90a" if family == "main"
                 else "-gencode=arch=compute_90,code=sm_90"]
        if family == "main" and control.bfp__SK2_DELAY:
            flags.append(f"-DSK2_DELAY_ESC={control.bfp__SK2_DELAY}")
        name = f"dense_joint_{family}_{_sha(source + cpp + repr(flags))[:16]}"
        for suffix, content in (("cu", source), ("cpp", cpp)):
            path = artifact_prefix.with_name(f"{artifact_prefix.name}-{family}.{suffix}")
            path.write_text(content)
            metadata["artifact_files"].append(path.name)
            on_artifact(path.name)
        metadata["modules"][family] = {
            "extension_name": name, "kernel_name": kernel_name,
            "original_generated_source_sha256": _sha(original),
            "diagnostic_source_sha256": _sha(source), "binding_sha256": _sha(cpp),
            "unchanged_producer_sha256": _sha(_producer(original)),
            "exports": functions, "extra_cuda_cflags": flags,
            "no_implicit_headers": True,
            "launches": launches,
        }
        pending[family] = (name, cpp, source, flags)
    modules = {}
    for family, (name, cpp, source, flags) in pending.items():
        modules[family] = control.load_inline(
            name=name, cpp_sources=cpp, cuda_sources=source, functions=None,
            no_implicit_headers=True, extra_cuda_cflags=flags, verbose=False,
        )
    set_mode(modules, 0)
    return modules, metadata
