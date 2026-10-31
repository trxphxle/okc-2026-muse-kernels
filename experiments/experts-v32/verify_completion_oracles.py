"""NEVER SUBMIT: isolated native completion-oracle resource/traffic proof.

Six real repaired-layer fixtures, ordered D32/V16/V32 x sliding/full, each
followed by base/A/B. This is NOT a decode benchmark or an oracle correctness
test. Only the compliant baseline is checked numerically. Timing and NCU run in
separate bounded processes; every isolated launch starts with zeroed scratch.
Missing counters, changed native footprints, and changed executed tensor counts
are failures, not reasons to relax the contract. Partial artifacts survive.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

BASE_SHA = "053dbacc8f0de4e6111ccb89cb91f629840852cf482a6ca423fe0eaff419c086"
LOADS = (("D32", 32, 1), ("V16", 16, 8), ("V32", 32, 8))
ARMS = ("base", "A", "B")
REQUIRED = (
    "gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
    "lts__t_sectors_op_read.sum", "smsp__inst_executed_pipe_tensor.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
)
RESOURCES = (
    "launch__registers_per_thread", "launch__shared_mem_per_block_allocated",
    "launch__shared_mem_per_block_static", "launch__shared_mem_per_block_dynamic",
    "launch__block_size", "launch__grid_size",
)
TENSOR = "smsp__inst_executed_pipe_tensor.sum"


def save(path, report):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def source_parts(path):
    source = Path(path).read_text()
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "MOE_CUDA_SRC" for t in n.targets)]
    if len(nodes) != 1 or not isinstance(nodes[0].value, ast.Constant):
        raise RuntimeError("Expected one literal MOE_CUDA_SRC: " + str(path))
    node = nodes[0].value
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line.encode()))
    raw = source.encode()
    start = offsets[node.lineno - 1] + node.col_offset
    end = offsets[node.end_lineno - 1] + node.end_col_offset
    outside = (raw[:start] + b"<MOE_CUDA_SRC>" + raw[end:]).decode()
    # The builder replaces only the first submission marker with its warning.
    outside = outside.split("\n", 1)[1]
    return source, node.value, outside


def non_wgs_projection(source):
    """Conservative source invariant, NOT a compiler or numerical proof.

    Ignore only bodies lexically guarded by a positive OKC_WGS conjunction.
    Retain every other line and all outer branch directives. In particular,
    alternative (#else) implementations remain visible to this comparison.
    """
    stack, result = [], []
    for line in source.splitlines(keepends=True):
        match = re.match(r"\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)", line)
        excluded = any(stack)
        if match:
            op, expression = match.groups()
            if op in ("if", "ifdef", "ifndef"):
                if not excluded:
                    result.append(line)
                stack.append(op == "if" and bool(re.match(r"\s*OKC_WGS\s*(?:&&|$)", expression)))
            elif op in ("elif", "else"):
                if not stack:
                    raise RuntimeError("Unbalanced native preprocessor branches")
                stack[-1] = op == "elif" and bool(re.match(r"\s*OKC_WGS\s*(?:&&|$)", expression))
                if not any(stack[:-1]):
                    result.append(line)
            else:
                if not stack:
                    raise RuntimeError("Unbalanced native preprocessor branches")
                stack.pop()
                if not any(stack):
                    result.append(line)
        elif not excluded:
            result.append(line)
    if stack:
        raise RuntimeError("Unclosed native preprocessor branch")
    return "".join(result)


def source_coverage(paths):
    parts = {arm: source_parts(path) for arm, path in paths.items()}
    if sha(paths["base"]) != BASE_SHA:
        raise RuntimeError("Baseline is not the pinned repaired source")
    projection = non_wgs_projection(parts["base"][1])
    for arm in ("A", "B"):
        if "NEVER SUBMIT" not in parts[arm][0].splitlines()[0]:
            raise RuntimeError("Missing diagnostic-only first-line warning: " + arm)
        if parts[arm][2] != parts["base"][2]:
            raise RuntimeError("Changes outside the native MOE literal: " + arm)
        if parts[arm][1] == parts["base"][1]:
            raise RuntimeError("Oracle did not change native source: " + arm)
        if non_wgs_projection(parts[arm][1]) != projection:
            raise RuntimeError("Oracle changes non-WGS source coverage: " + arm)
    return dict(status="passed", source_sha256={k: sha(v) for k, v in paths.items()},
                outside_literal_sha256=digest(parts["base"][2]),
                non_wgs_source_sha256=digest(projection),
                small_loads=["D1", "D16", "V1"],
                scope="Unchanged host/helper/front/back source and non-WGS branches; no oracle numerical claim")


def run_logged(command, log, report, output, timeout, optional=False):
    report.setdefault("commands", []).append(dict(argv=list(map(str, command)), log=Path(log).name,
                                                  timeout_seconds=timeout))
    report["artifact_files"].append(Path(log).name)
    save(output, report)
    with open(log, "w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise RuntimeError("Timed out; process group killed: " + str(log))
    if code and not optional:
        raise RuntimeError(f"{log}: exit {code}")
    return code


class ExpertObserver:
    """Observe the actual repaired layer's native boundary, never its router."""
    def __init__(self, module):
        self.module, self.original, self.calls = module, module._moe, []

    def install(self):
        observer = self
        class Proxy:
            def __init__(self, target):
                self.target = target

            def okc_run(self, weights, scratch, x, ids, tw, output, lo, count, grid):
                inputs = tuple(t.clone() for t in (x, ids, tw))
                result = self.target.okc_run(weights, scratch, x, ids, tw, output, lo, count, grid)
                observer.calls.append(dict(inputs=inputs, expected=output.clone(),
                                           lo=int(lo), count=int(count), grid=int(grid)))
                return result

            def __getattr__(self, name):
                return getattr(self.target, name)

        def moe(*args, **kwargs):
            return Proxy(self.original(*args, **kwargs))
        self.module._moe = moe

    def restore(self):
        self.module._moe = self.original


def tensor_identity(tensor, torch):
    raw = tensor.detach().contiguous().reshape(-1).cpu().view(torch.uint8).numpy().tobytes()
    return dict(shape=list(tensor.shape), stride=list(tensor.stride()), dtype=str(tensor.dtype),
                sha256=hashlib.sha256(raw).hexdigest())


def compare_tensor(actual, expected, torch):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return dict(close=False, reason="shape/dtype mismatch")
    a, b = actual.float(), expected.float()
    finite = bool(torch.isfinite(a).all() & torch.isfinite(b).all())
    error = float((a - b).norm() / b.norm().clamp_min(1e-12)) if finite else None
    return dict(close=finite and error <= 0.0001, finite=finite, relative_l2=error,
                equal=bool(torch.equal(actual, expected)))


def child(args, report):
    import torch
    import compare_decode_public as common
    common.torch = torch
    paths = dict(base=args.baseline, A=args.oracle_a, B=args.oracle_b)
    modules = {arm: common.load(path, "completion_" + arm) for arm, path in paths.items()}
    reference = common.load(str(Path(common.__file__).with_name("reference.py")), "completion_reference")
    common.reference = reference
    base = modules["base"]
    weights = reference.generate_weights(731283)
    started = time.monotonic()
    state = base.prepare(weights)  # Never call either oracle's prepare.
    torch.cuda.synchronize()
    report.update(prepare_seconds=time.monotonic() - started,
                  environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name()),
                  fixed_mode=common.mode_state(base, state), weight_seed=731283)
    report["small_load_dispatch"] = []
    for load, tokens, key in (("D1", 1, "tiny"), ("V1", 8, "tiny2"), ("D16", 16, "mid")):
        target = base._moe(tokens, state)
        selected = [k for k, v in state["ma_ext"].items() if v is target]
        if selected != [key]:
            raise RuntimeError("Unexpected small-load native dispatch: " + repr(selected))
        native = target._module
        definitions = [list(k) for k, v in base._MA_EXTS.items() if v is native]
        if len(definitions) != 1 or "-DOKC_WGS=1" in definitions[0]:
            raise RuntimeError("Small-load build is not the unchanged non-WGS path")
        report["small_load_dispatch"].append(dict(load=load, tokens=tokens, key=key,
                                                 defines=definitions[0], extension=str(native.__file__)))
    flush = torch.ones(common.FLUSH_BYTES // 4, device="cuda")
    report["extensions"] = []
    seen_extensions = set()
    for load, batch, length in LOADS:
        for layer in ("sliding", "full"):
            seed = 919233 + batch * 100 + length * 1000 + (layer == "full") * 10000
            frozen_path = f"completion-frozen-{load}-{layer}.pt"
            report["artifact_files"].append(frozen_path)
            data = expected = observed = observer = None
            if args.mode == "timing":
                data = reference.generate_input(batch=batch, q_len=length, ctx=80000, layer=layer, page=1,
                                                spread="agentic", dist="soft", seed=seed)
                input_identity = {key: tensor_identity(value, torch) for key, value in data.items()
                                  if isinstance(value, torch.Tensor)}
                expected = base.kernel_fn(common.clone_data(data), state)
                observer = ExpertObserver(base)
                observer.install()
                try:
                    observed = base.kernel_fn(common.clone_data(data), state)
                    torch.cuda.synchronize()
                finally:
                    observer.restore()
                observation = common.comparison(observed, expected)
                if not observation["close"] or len(observer.calls) != 1:
                    raise RuntimeError("Observation changed output or did not yield exactly one native expert call")
                call = observer.calls[0]
                x, ids, tw = call["inputs"]
                fixture = dict(load=load, layer=layer, input_seed=seed, input_identity=input_identity,
                               config=common.metadata(data["config"]),
                               native_inputs={k: tensor_identity(t, torch) for k, t in zip(("xm", "lids", "wts"), (x, ids, tw))},
                               lo=call["lo"], count=call["count"], grid=call["grid"],
                               observed_output=tensor_identity(call["expected"], torch),
                               observation_comparison=observation)
                torch.save(dict(baseline_sha256=BASE_SHA, weight_seed=731283, fixture=fixture,
                                x=x.cpu(), ids=ids.cpu(), tw=tw.cpu(), expected=call["expected"].cpu()), frozen_path)
            else:
                frozen = torch.load(frozen_path, map_location="cpu", weights_only=True)
                fixture = frozen["fixture"]
                if (frozen["baseline_sha256"] != BASE_SHA or frozen["weight_seed"] != 731283
                        or (fixture["load"], fixture["layer"], fixture["input_seed"]) != (load, layer, seed)):
                    raise RuntimeError("Frozen real-input fixture identity mismatch")
                x, ids, tw = (frozen[key].to("cuda") for key in ("x", "ids", "tw"))
                call = dict(lo=fixture["lo"], count=fixture["count"], grid=fixture["grid"],
                            expected=frozen["expected"].to("cuda"))
                del frozen
            for arm in ARMS:
                module = modules[arm]
                private = dict(state, ma_ext={}, ma_scratch=[torch.zeros_like(t) for t in state["ma_scratch"]])
                if any(a.data_ptr() == b.data_ptr() for a, b in zip(private["ma_scratch"], state["ma_scratch"])):
                    raise RuntimeError("Scratch aliases prepared state")
                ext = module._moe(x.shape[0], private)
                ext.okc_set_pdl(0)
                key = next(k for k, value in private["ma_ext"].items() if value is ext)
                wanted = "b4" if load == "D32" else "main"
                if key != wanted:
                    raise RuntimeError(f"Expected {wanted} native variant; got {key}")
                extension = ext._module
                definitions = [list(k) for k, value in module._MA_EXTS.items() if value is extension]
                if len(definitions) != 1 or "-DOKC_WGS=1" not in definitions[0]:
                    raise RuntimeError("Native fixture did not select real WGS build")
                extension_path = str(Path(extension.__file__).resolve())
                if (arm, key) not in seen_extensions:
                    report["extensions"].append(dict(arm=arm, variant=key, path=extension_path,
                                                     sha256=sha(extension_path), defines=definitions[0]))
                    seen_extensions.add((arm, key))
                output = torch.empty_like(x)
                row = dict(fixture, arm=arm, variant=key, defines=definitions[0],
                           native_pdl=False, scratch=common.metadata(private["ma_scratch"]),
                           status="launching", oracle_correctness="not claimed" if arm != "base" else None)
                report["records"].append(row)
                save(args.output, report)

                def reset():
                    for tensor in private["ma_scratch"]:
                        tensor.zero_()
                    output.zero_()

                def invoke():
                    ext.okc_run(private["ma_w"], private["ma_scratch"], x, ids, tw, output,
                                call["lo"], call["count"], call["grid"])

                # A warmup validates the baseline on the observed native boundary.
                # Scratch is zeroed again before EVERY measured invocation.
                reset()
                invoke()
                torch.cuda.synchronize()
                if arm == "base":
                    row["baseline_observed_comparison"] = compare_tensor(output, call["expected"], torch)
                    if not row["baseline_observed_comparison"]["close"]:
                        raise RuntimeError("Isolated baseline does not reproduce the observed expert output")
                if args.mode == "timing":
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    samples = []
                    for _ in range(args.samples):
                        reset()
                        sink = flush.sum()
                        torch.cuda.synchronize()
                        begin.record()
                        invoke()
                        end.record()
                        end.synchronize()
                        samples.append(begin.elapsed_time(end) * 1000)
                        del sink
                    row.update(samples_us=samples, median_us=statistics.median(samples))
                else:
                    reset()
                    sink = flush.sum()
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStart()
                    try:
                        invoke()
                        torch.cuda.synchronize()
                    finally:
                        torch.cuda.cudart().cudaProfilerStop()
                    del sink
                if arm == "base":
                    row["baseline_after_comparison"] = compare_tensor(output, call["expected"], torch)
                    if not row["baseline_after_comparison"]["close"]:
                        raise RuntimeError("Baseline native output changed after measurement")
                after_identity = {k: tensor_identity(t, torch) for k, t in zip(("xm", "lids", "wts"), (x, ids, tw))}
                if after_identity != fixture["native_inputs"]:
                    raise RuntimeError("Native invocation mutated frozen real inputs")
                row["status"] = "complete"
                save(args.output, report)
                del private, output, ext
            del data, expected, observed, observer, call, x, ids, tw
    report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()


def parent(args, report):
    from profile_large import parse_csv
    prefix = Path(args.output).stem
    common = [sys.executable, str(Path(__file__).resolve()), "--baseline", str(Path(args.baseline).resolve()),
              "--oracle-a", str(Path(args.oracle_a).resolve()), "--oracle-b", str(Path(args.oracle_b).resolve()),
              "--samples", str(args.samples)]
    ncu = args.ncu or shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
    timing, profiled = prefix + "-timing.json", prefix + "-profiled.json"
    csv_file, ncu_report = prefix + "-ncu.csv", prefix + "-native"
    capture_log, metric_file = prefix + "-capture.log", prefix + "-metrics.txt"
    report["artifact_files"].extend([timing, profiled, capture_log, ncu_report + ".ncu-rep"])
    run_logged([ncu, "--query-metrics", "--query-metrics-mode", "all"], metric_file,
               report, args.output, 60)
    installed = Path(metric_file).read_text()
    missing = [metric for metric in REQUIRED if not re.search(r"(?<![\w.])" + re.escape(metric) + r"(?![\w.])", installed)]
    if missing:
        raise RuntimeError("Missing required native metrics: " + repr(missing))
    report["selected_metrics"] = list(REQUIRED)
    report["required_launchstats"] = list(RESOURCES)
    run_logged(common + ["--mode", "timing", "--output", timing], prefix + "-timing.log",
               report, args.output, args.child_timeout)
    run_logged([ncu, "--profile-from-start", "off", "--clock-control", "none", "--cache-control", "all",
                "--replay-mode", "kernel", "--kernel-name", "regex:.*moe_fused.*", "--metrics", ",".join(REQUIRED),
                "--section", "LaunchStats", "--csv", "--print-units", "base", "--log-file", capture_log,
                "-f", "-o", ncu_report, *common, "--mode", "profile", "--output", profiled],
               prefix + "-profile.log", report, args.output, args.child_timeout)
    # Raw import exposes technical launch__ keys; Details uses display labels.
    run_logged([ncu, "--import", ncu_report + ".ncu-rep", "--page", "raw", "--csv", "--print-units", "base"],
               csv_file, report, args.output, 120)
    native = parse_csv(csv_file)
    timing_data = json.loads(Path(timing).read_text())
    profile_data = json.loads(Path(profiled).read_text())
    report["raw_native_records"] = native
    report["children"] = dict(timing=timing_data, profile=profile_data)
    report["artifact_files"].extend(timing_data.get("artifact_files", []))
    report["artifact_files"].extend(profile_data.get("artifact_files", []))
    save(args.output, report)
    for data in (timing_data, profile_data):
        if data["status"] != "complete" or data["source_coverage"] != report["source_coverage"]:
            raise RuntimeError("Incomplete child or source identity mismatch")
    if len(native) != 18 or len(timing_data["records"]) != 18 or len(profile_data["records"]) != 18:
        raise RuntimeError("Expected exactly 18 ordered native records and fixtures per process")
    expected_order = [(load, layer, arm) for load, _, _ in LOADS for layer in ("sliding", "full") for arm in ARMS]
    failures = []
    for index, (n, t, p, order) in enumerate(zip(native, timing_data["records"], profile_data["records"], expected_order)):
        for row in (t, p):
            if tuple(row[k] for k in ("load", "layer", "arm")) != order or row["status"] != "complete":
                raise RuntimeError("Native record ordering/child completion mismatch")
        for field in ("input_seed", "input_identity", "native_inputs", "config", "lo", "count", "grid",
                      "defines", "scratch", "variant", "observed_output"):
            if t[field] != p[field]:
                raise RuntimeError(f"Timing/NCU fixture mismatch at {order}: {field}")
        metrics = n["metrics"]
        missing = set(REQUIRED + RESOURCES) - metrics.keys()
        if missing:
            raise RuntimeError(f"Missing measured native metrics at {order}: {sorted(missing)}")
        if any(not math.isfinite(metrics[m]) or metrics[m] < 0 for m in REQUIRED + RESOURCES):
            raise RuntimeError("Invalid native metric value")
        if metrics[TENSOR] <= 0:
            raise RuntimeError("No executed tensor instructions measured")
        row = dict(t, native=n, profile_us=metrics["gpu__time_duration.sum"] / 1000,
                   dram_read_bytes=metrics["dram__bytes_read.sum"], dram_write_bytes=metrics["dram__bytes_write.sum"],
                   l2_read_sectors=metrics["lts__t_sectors_op_read.sum"],
                   l2_read_bytes=metrics["lts__t_sectors_op_read.sum"] * 32,
                   executed_tensor_instructions=metrics[TENSOR],
                   resources={key: metrics[key] for key in RESOURCES})
        report["records"].append(row)
        baseline = report["records"][index - index % 3]
        checks = dict(kernel_name=n["kernel"] == baseline["native"]["kernel"],
                      resources=row["resources"] == baseline["resources"],
                      executed_tensor_instructions=metrics[TENSOR] == baseline["executed_tensor_instructions"],
                      native_inputs=row["native_inputs"] == baseline["native_inputs"],
                      defines=row["defines"] == baseline["defines"], scratch=row["scratch"] == baseline["scratch"])
        row["baseline_exact_checks"] = checks
        if not all(checks.values()):
            failures.append(dict(load=order[0], layer=order[1], arm=order[2], checks=checks))
        save(args.output, report)
    report["exact_match_failures"] = failures
    # Export real extension paths, not guessed cache/build names, even on mismatch.
    cuobjdump = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    report["compiled_exports"] = []
    for entry in profile_data["extensions"]:
        exported = dict(entry, exports=[])
        report["compiled_exports"].append(exported)
        if not Path(entry["path"]).is_file() or sha(entry["path"]) != entry["sha256"]:
            exported["status"] = "extension_unavailable_or_changed"
            continue
        for option, suffix in (("--dump-resource-usage", "resources.txt"), ("--dump-sass", "sass.txt")):
            filename = f"{prefix}-{entry['arm']}-{entry['variant']}-{suffix}"
            try:
                code = run_logged([cuobjdump, option, entry["path"]], filename, report, args.output, 90, optional=True)
                exported["exports"].append(dict(file=filename, exit_code=code))
            except (OSError, RuntimeError) as exc:
                exported["exports"].append(dict(file=filename, error=repr(exc)))
    if failures:
        raise RuntimeError("Native footprint/work-count contract rejected: " + json.dumps(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="decode_alygg77_bf16.py")
    parser.add_argument("--oracle-a", default="oracle_split_A.py")
    parser.add_argument("--oracle-b", default="oracle_completion_B.py")
    parser.add_argument("--output", default="completion-resource.json")
    parser.add_argument("--mode", choices=("parent", "timing", "profile"), default="parent")
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--child-timeout", type=int, default=800)
    parser.add_argument("--ncu")
    args = parser.parse_args()
    if not 1 <= args.samples <= 31 or not 1 <= args.child_timeout <= 3600:
        parser.error("samples must be 1..31 and child-timeout 1..3600 seconds")
    # Artifacts live beside the report and are enumerated by basename for upload.
    args.baseline, args.oracle_a, args.oracle_b = [str(Path(p).resolve()) for p in
                                                 (args.baseline, args.oracle_a, args.oracle_b)]
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(output.parent)
    args.output = output.name
    report = dict(status="running", mode=args.mode, diagnostic_only=True, submission_allowed=False,
                  scope="Isolated native expert launches on observed repaired full-layer inputs; NOT decode timing",
                  resource_acceptance="Exact registers/shared/block/grid and executed tensor instructions vs baseline",
                  numeric_acceptance="Baseline reproduces observed expert output; A/B correctness deliberately unclaimed",
                  scratch_policy="Fresh same-shape zeroed allocation per arm/fixture; zero before every invocation",
                  records=[], artifact_files=[], expected_native_records=18)
    save(args.output, report)
    try:
        report["source_coverage"] = source_coverage(dict(base=args.baseline, A=args.oracle_a, B=args.oracle_b))
        save(args.output, report)
        (parent if args.mode == "parent" else child)(args, report)
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
    finally:
        # Include files produced before a child failure/timeout, including partial
        # child JSON and profiler logs, without claiming missing files exist.
        report["artifact_files"] = sorted({Path(p).name for p in report["artifact_files"] if Path(p).is_file()})
        save(args.output, report)
        print(json.dumps(dict(status=report["status"], output=args.output)), flush=True)
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
