"""Diagnostic only: delete seven/eight FP4 unpack instructions, retain all bytes.

This is deliberately NOT a valid MoE implementation and must never be submitted.
The baseline and idealized graph share the original packed weights and scratch,
execute serially, and retain routing, TMA, scales, WGMMA, epilogues and finalization.
All three arms are captured; only unchanged arms are checked against the oracle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch
import screen_moe as screen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="muse_moe_w4a16.py")
    parser.add_argument("--output", default="ceiling-decode.json")
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=731283)
    args = parser.parse_args()
    report = dict(status="running", diagnostic_only=True, official_score=False,
                  source_sha256=hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
                  hypothesis="remove FP4 unpack issue work only; no claim of correct output or achievable gain",
                  changes="WGS dq4 becomes one AND shared by all four operands; same weight/scale loads and math schedule",
                  weight_seed=args.seed, pairs_per_case=args.pairs, artifact_files=[], cases=[])
    screen.save(Path(args.output), report)
    base = screen.load(Path(args.source), "ceiling_base")
    floor = screen.load(Path(args.source), "ceiling_floor")
    old = "          dq4(q, r0, r1, r2, r3);"
    new = "          r0 = r1 = r2 = r3 = q & 0x81C081C0u;  // INVALID diagnostic instruction ceiling"
    if floor.CUDA_SRC.count(old) != 1:
        raise RuntimeError("Expected exactly one WGS unpack call")
    floor.CUDA_SRC = floor.CUDA_SRC.replace(old, new)
    # The CUDA source content is part of load_inline's version key, but giving
    # this diagnostic its own define also makes its on-disk extension distinct.
    floor.MAIN_DEFINES = tuple(floor.MAIN_DEFINES) + ("-DEXPERTS_V32_CEILING=1",)
    reference = screen.load(Path("reference.py"), "ceiling_reference")
    weights = reference.generate_weights(args.seed)
    start = time.monotonic()
    state = base.prepare(weights)
    report["baseline_prepare_seconds"] = time.monotonic() - start
    start = time.monotonic()
    ext = floor._ext(tuple(floor.MAIN_DEFINES) + state["sfx"])
    report["ceiling_compile_seconds"] = time.monotonic() - start
    report["extension"] = ext.__file__
    report["gpu"] = torch.cuda.get_device_name()
    flush = torch.ones(screen.FLUSH_BYTES // 4, device="cuda")
    for index, (load, m) in enumerate(screen.LOADS):
        row = dict(load=load, m=m, records=[])
        report["cases"].append(row)
        initial = screen.make_input(reference, m, args.seed + m * 10000)
        captures, events = {}, {}
        for name in screen.NAMES:
            data = screen.stable_input(initial)
            mod = floor if name == "candidate" and m > base.MID_MAX_M else base
            graph, out = screen.capture(mod.kernel_fn, data, state)
            captures[name] = (data, graph, out)
            events[name] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            a, b = events[name]; a.record(); b.record(); b.synchronize()
        for pair in range(args.pairs):
            seed = args.seed + m * 10000 + pair + 1
            fresh = screen.make_input(reference, m, seed)
            gold = reference.ref_kernel(fresh, weights)
            record = dict(input_seed=seed, routing=screen.routing(fresh), us={}, correctness={})
            for data, _, _ in captures.values():
                screen.copy_input(data, fresh)
            order = screen.ORDERS[(pair + 3 * (index % 2)) % len(screen.ORDERS)]
            record["order"] = list(order)
            for name in order:
                sink = flush.sum()
                a, b = events[name]; a.record(); captures[name][1].replay(); b.record(); b.synchronize()
                record["us"][name] = a.elapsed_time(b) * 1000
                del sink
            for name in ("base", "base_repeat"):
                passed, message = reference.compare(captures[name][2], gold, screen.REL_L2_TOL)
                record["correctness"][name] = dict(passed=bool(passed), message=str(message))
                if not passed:
                    raise RuntimeError(name + ": " + str(message))
            record["paired_ratio"] = {name: record["us"]["base"] / record["us"][name]
                                      for name in screen.VARIANTS}
            row["records"].append(record)
        row.update(screen.summarize(row["records"]))
        print(json.dumps({k: row[k] for k in ("load", "median_us", "median_ratio")}), flush=True)
        screen.save(Path(args.output), report)
        captures.clear(); events.clear()
        del graph, out, data, initial, fresh, gold
    report["geomean_ratio"] = screen.aggregate(report["cases"])
    report["control_passed"] = abs(report["geomean_ratio"]["base_repeat"] - 1) <= 0.01
    report["ceiling_at_least_three_percent"] = report["geomean_ratio"]["candidate"] >= 1.03
    report["status"] = "complete"
    report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    for option, suffix in (("--dump-resource-usage", "-resources.txt"), ("--dump-sass", "-sass.txt")):
        path = Path(args.output).stem + suffix
        with open(path, "w") as output:
            subprocess.run(["/usr/local/cuda/bin/cuobjdump", option, ext.__file__],
                           stdout=output, stderr=subprocess.STDOUT, check=True, timeout=60)
        report["artifact_files"].append(path)
    screen.save(Path(args.output), report)
    print(json.dumps({k: report[k] for k in ("status", "geomean_ratio", "control_passed", "ceiling_at_least_three_percent")}), flush=True)


if __name__ == "__main__":
    main()
