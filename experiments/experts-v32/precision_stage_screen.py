"""Diagnostic-only single-repair stage attribution on fixed repaired inputs.

Uses actual repaired decoder stage arguments, not synthetic GEMMs or changed
routing. Numerical ablations are deliberately noncompliant and never candidates.
Full-layer native timing is collected separately; stage deltas are not additive
critical-path savings. Run one adapter/group batch per bounded Modal call.
"""
import argparse
import ast
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import compare_decode_public as common

common.torch = torch
LOADS = common.LOADS
JIT_NAMES = ("_route", "_route_a", "_route_b", "_route_split", "_graft_route", "_sl_graft_route",
             "_sl_tile_route", "_boundary1", "_boundary2", "_sl_cluster_finish",
             "_graft_row_boundary", "_graft_row_boundary_pre", "_pexp_norm_sum",
             "_qk_prep", "_attn_combine", "_norm_gate", "_norm_gate_part")
NATIVE_GETTERS = ("_sl_cluster_module", "_graft_glue_module", "_graft_b2_module")


def save(path, value):
    common.save(path, value)


def namespace_extensions(source, tag):
    raw = source.encode()
    lines = raw.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    changes = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
        if name != "load_inline":
            continue
        for kw in node.keywords:
            if kw.arg == "name":
                expr = kw.value
                begin = offsets[expr.lineno - 1] + expr.col_offset
                end = offsets[expr.end_lineno - 1] + expr.end_col_offset
                changes.append((begin, end, repr("p3_" + tag + "_").encode() + b" + (" + raw[begin:end] + b")"))
    if not changes:
        raise RuntimeError("No native extension identities found")
    for begin, end, value in sorted(changes, reverse=True):
        raw = raw[:begin] + value + raw[end:]
    result = raw.decode()
    ast.parse(result)
    return result


def clone_values(values):
    memo = {}
    def clone(value):
        if isinstance(value, torch.Tensor):
            key = (value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype)
            if key not in memo:
                memo[key] = value.clone()
            return memo[key]
        if isinstance(value, tuple):
            return tuple(clone(v) for v in value)
        if isinstance(value, list):
            return [clone(v) for v in value]
        if isinstance(value, dict):
            return {k: clone(v) for k, v in value.items()}
        return value
    return clone(values)


def tensors(values):
    if isinstance(values, torch.Tensor):
        return [values]
    if isinstance(values, (tuple, list)):
        return [t for value in values for t in tensors(value)]
    return []


def compare(actual, expected):
    results = []
    for a, b in zip(actual, expected):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError("Stage tensor layout mismatch")
        if a.is_floating_point():
            af, bf = a.float(), b.float()
            finite = bool(torch.isfinite(af).all() & torch.isfinite(bf).all())
            error = float((af - bf).norm() / bf.norm().clamp_min(1e-12)) if finite else None
            close = finite and error <= 0.0001
        else:
            finite, error, close = True, None, bool(torch.equal(a, b))
        results.append(dict(equal=bool(torch.equal(a, b)), finite=finite, relative_l2=error, close=close))
    return dict(close=len(actual) == len(expected) and all(r["close"] for r in results), tensors=results)


class Observer:
    def __init__(self, module):
        self.module, self.originals, self.calls = module, {}, []

    def record(self, kind, name, invoke, args, kwargs, grid=None):
        before = clone_values(args)
        result = invoke(*args, **kwargs)
        after = clone_values(args)
        self.calls.append(dict(kind=kind, name=name, grid=grid, args=before,
                               kwargs=dict(kwargs), expected=tensors(after)))
        return result

    def install(self):
        observer = self
        for name in JIT_NAMES:
            original = getattr(self.module, name, None)
            if original is None or not hasattr(original, "__getitem__"):
                continue
            self.originals[name] = original
            class JitProxy:
                def __init__(self, symbol, target):
                    self.symbol, self.target = symbol, target
                def __getitem__(self, grid):
                    def call(*args, **kwargs):
                        return observer.record("triton", self.symbol, self.target[grid], args, kwargs, grid)
                    return call
                def __getattr__(self, key):
                    return getattr(self.target, key)
            setattr(self.module, name, JitProxy(name, original))
        for name in NATIVE_GETTERS:
            original = getattr(self.module, name, None)
            if original is None:
                continue
            self.originals[name] = original
            class NativeProxy:
                def __init__(self, getter, target):
                    self.getter, self.target = getter, target
                def __getattr__(self, method):
                    target = getattr(self.target, method)
                    if method not in ("b1", "b2s", "b2c"):
                        return target
                    def call(*args, **kwargs):
                        return observer.record("native_glue", self.getter + "." + method, target, args, kwargs)
                    return call
            def getter(original=original, name=name):
                return NativeProxy(name, original())
            setattr(self.module, name, getter)
        original = self.module._moe
        self.originals["_moe"] = original
        class ExpertProxy:
            def __init__(self, target):
                self.target = target
            def okc_run(self, weights, scratch, x, ids, tw, output, lo, count, grid):
                before = clone_values((x, ids, tw))
                result = self.target.okc_run(weights, scratch, x, ids, tw, output, lo, count, grid)
                observer.calls.append(dict(kind="experts", name="_moe.okc_run", args=before,
                    kwargs=dict(lo=lo, count=count, grid=grid), expected=[output.clone()]))
                return result
            def __getattr__(self, key):
                return getattr(self.target, key)
        def moe(*args, **kwargs):
            return ExpertProxy(original(*args, **kwargs))
        self.module._moe = moe
        attention = self.module._decode_attention7
        self.originals["_decode_attention7"] = attention
        def attn(data, state, qkv):
            before = qkv.clone()
            result = attention(data, state, qkv)
            observer.calls.append(dict(kind="attention", name="_decode_attention7", args=before,
                                       kwargs={}, expected=[result.clone()]))
            return result
        self.module._decode_attention7 = attn

    def restore(self):
        for name, value in self.originals.items():
            setattr(self.module, name, value)


def relevant(group, call):
    if group["id"].startswith("attn_native_"):
        return call["kind"] == "attention"
    if group["id"].startswith("experts_"):
        return call["kind"] == "experts" or call["name"] == "_pexp_norm_sum"
    if call["name"] in ("_graft_row_boundary", "_graft_row_boundary_pre"):
        post = call["kwargs"].get("POST", call["args"][10])
        if group["id"] == "glue_b1_projection_bf16" and post:
            return False
        if group["id"] == "glue_b2_projection_bf16" and not post:
            return False
    return any(call["name"] == symbol or call["name"].split(".")[0] == symbol
               for symbol in group["source_symbols"])


def stage_function(module, state, data, call):
    kind = call["kind"]
    if kind == "attention":
        private = common.clone_data(data)
        qkv = call["args"].clone()
        # Runtime plans remain the repaired plans; each module owns mutable WS7/CL7.
        def run(_, __):
            return [module._decode_attention7(private, state, qkv)]
        return run
    args = clone_values(call["args"])
    if kind == "experts":
        x, ids, tw = args
        output = torch.empty_like(x)
        ext = module._moe(x.shape[0], state)
        ext.okc_set_pdl(0)
        meta = call["kwargs"]
        def run(_, __):
            ext.okc_run(state["ma_w"], state["ma_scratch"], x, ids, tw, output,
                        meta["lo"], meta["count"], meta["grid"])
            return [output]
        return run
    kwargs = dict(call["kwargs"])
    if "launch_pdl" in kwargs:
        kwargs["launch_pdl"] = False
    if "USE_PDL" in kwargs:
        kwargs["USE_PDL"] = False
    if kind == "triton":
        target = getattr(module, call["name"])[call["grid"]]
    else:
        getter, method = call["name"].split(".")
        target = getattr(getattr(module, getter)(), method)
    def run(_, __):
        target(*args, **kwargs)
        return tensors(args)
    return run


def benchmark(base, variant, state, variant_state, data, call, flush, samples):
    fns = {"base": stage_function(base, state, data, call),
           "variant": stage_function(variant, variant_state, data, call)}
    caps = {name: common.capture(fn, None, None, flush) for name, fn in fns.items()}
    caps["repeat"] = common.capture(fns["base"], None, None, flush)
    observed = compare(caps["base"]["output"], call["expected"])
    if not observed["close"]:
        raise RuntimeError("Isolated repaired stage does not reproduce observed stage: " + call["name"] + " " + str(observed))
    rows = []
    names = ("base", "variant", "repeat")
    for index in range(samples):
        shift = index % 3
        order = list(names[shift:] + names[:shift])
        if (index // 3) % 2:
            order.reverse()
        us = {name: common.measure(caps[name]) for name in order}
        rows.append(dict(us=us, saved_us=us["base"] - us["variant"], control_saved_us=us["base"] - us["repeat"]))
    final = compare(caps["base"]["output"], caps["repeat"]["output"])
    if not final["close"]:
        raise RuntimeError("Repaired A/A stage output mismatch")
    return dict(status="measured", kind=call["kind"], symbol=call["name"],
                median_us={name: statistics.median(r["us"][name] for r in rows) for name in names},
                median_saved_us=statistics.median(r["saved_us"] for r in rows),
                median_control_saved_us=statistics.median(r["control_saved_us"] for r in rows),
                median_ratio=statistics.median(r["us"]["base"] / r["us"]["variant"] for r in rows),
                control_ratio=statistics.median(r["us"]["base"] / r["us"]["repeat"] for r in rows),
                observed_stage_reproduction=observed, baseline_repeat_comparison=final,
                ablation_output_comparison=compare(caps["variant"]["output"], caps["base"]["output"]), samples=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="decode_alygg77_bf16.py")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--groups", required=True, help="Comma-separated semantic group IDs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=18)
    args = parser.parse_args()
    report = dict(status="starting", diagnostic_only=True, submission_allowed=False, groups=[], cases=[], artifact_files=[],
                  baseline_sha256=common.sha(args.baseline), adapter_sha256=common.sha(args.adapter),
                  limitations=["Isolated fixed-input stage deltas are NOT additive exposed whole-layer savings.",
                               "Numerical variants deliberately violate one audited requirement and cannot be submitted.",
                               "Inactive/structural/host-only repairs require separate attribution rather than invented zero timings.",
                               "Standalone Triton/native expert launches disable PDL where exposed; both arms use identical isolation."])
    save(args.output, report)
    try:
        adapter = common.load(args.adapter, "adapter")
        selected = {name: next(g for g in adapter.GROUPS if g["id"] == name) for name in args.groups.split(",")}
        raw = Path(args.baseline).read_text()
        variants = {}
        for name, group in selected.items():
            entry = dict(group)
            report["groups"].append(entry)
            try:
                transformed = adapter.transform(raw, name)
            except ValueError as exc:
                entry.update(transform_status="not_single_toggle", reason=str(exc))
                continue
            if transformed == raw:
                raise RuntimeError("Ablation transform silently did nothing: " + name)
            tag = hashlib.sha256(transformed.encode()).hexdigest()[:10]
            path = "diagnostic_" + name + ".py"
            Path(path).write_text(namespace_extensions(transformed, tag))
            report["artifact_files"].append(path)
            variants[name] = common.load(path, "ablation_" + name)
            entry.update(transform_status="generated", path=path, generated_sha256=common.sha(path))
        base = common.load(args.baseline, "repaired")
        reference = common.load("reference.py", "reference")
        common.reference = reference
        weights = reference.generate_weights(731283)
        before = time.monotonic()
        state = base.prepare(weights)
        torch.cuda.synchronize()
        report["prepare_seconds"] = time.monotonic() - before
        report["fixed_mode"] = common.mode_state(base, state)
        report["environment"] = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
        variant_states = {}
        for name, module in variants.items():
            variant_states[name] = dict(state, ma_ext={},
                                       ma_scratch=[tensor.clone() for tensor in state["ma_scratch"]])
            if hasattr(base, "_TUNED7"):
                module._TUNED7 = dict(base._TUNED7)
        flush = torch.ones(common.FLUSH_BYTES // 4, device="cuda")
        for load, batch, length in LOADS:
            for layer in ("sliding", "full"):
                seed = 919233 + batch * 100 + length * 1000 + (layer == "full") * 10000
                data = reference.generate_input(batch=batch, q_len=length, ctx=80000, layer=layer, page=1,
                                                spread="agentic" if batch > 1 else "fixed", dist="soft", seed=seed)
                baseline_data = common.clone_data(data)
                expected = base.kernel_fn(baseline_data, state)
                observed_data = common.clone_data(data)
                observer = Observer(base)
                observer.install()
                try:
                    observed = base.kernel_fn(observed_data, state)
                    torch.cuda.synchronize()
                finally:
                    observer.restore()
                context = dict(load=load, layer=layer, input_seed=seed,
                               config=common.metadata(data["config"]),
                               observation_comparison=common.comparison(observed, expected),
                               observed_stages=[dict(kind=c["kind"], symbol=c["name"]) for c in observer.calls], groups={})
                if not context["observation_comparison"]["close"]:
                    raise RuntimeError("Stage observation changed repaired layer behavior")
                report["cases"].append(context)
                save(args.output, report)
                for name, group in selected.items():
                    calls = [call for call in observer.calls if relevant(group, call)]
                    if name not in variants:
                        context["groups"][name] = dict(status="structural_or_host_only", stages=[])
                    elif not calls:
                        context["groups"][name] = dict(status="not_executed_in_observed_dispatch", stages=[])
                    else:
                        result = dict(status="measuring", stages=[])
                        context["groups"][name] = result
                        for call in calls:
                            result["stages"].append(benchmark(base, variants[name], state, variant_states[name],
                                                              observed_data, call, flush, args.samples))
                            save(args.output, report)
                        result["status"] = "measured"
                        result["sum_isolated_median_saved_us"] = sum(r["median_saved_us"] for r in result["stages"])
                        result["sum_isolated_control_saved_us"] = sum(r["median_control_saved_us"] for r in result["stages"])
                    save(args.output, report)
                print(json.dumps(dict(load=load, layer=layer, groups={name:{k:v for k,v in value.items() if k!="stages"}
                                                                     for name,value in context["groups"].items()})),flush=True)
                del data, baseline_data, observed_data, expected, observed, observer
                gc.collect()
        report["status"] = "complete"
    except BaseException as exc:
        import traceback
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        save(args.output, report)
        print(json.dumps(dict(status=report["status"], output=args.output)),flush=True)


if __name__ == "__main__":
    main()
