"""Round-two H200 transfer-only pairing/multicast mechanism screen.

Not a candidate. All arms read the same task plan and source bank. Independent
pairs already have optimal same-address alignment; native-MoE extrapolations
are therefore kept separate. No larger per-CTA token tile or decoded weights.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

GRID, PAIRS, SB, STAGES = 132, 66, 36864, 16
WORDS = SB // 4
BANK_UNITS = 32 * (32 + 16)
A, B, MASK = 1664525, 1013904223, 2**32 - 1
ARMS = ("independent", "cluster_independent", "multicast", "independent_repeat")
MODES = dict(independent=0, cluster_independent=1, multicast=2, independent_repeat=0)
METRICS = ("gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
           "lts__t_sectors_op_read.sum", "lts__t_sector_hit_rate.pct")


def save(path, report):
    Path(path).write_text(json.dumps(report, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_plan(counts):
    plan = []
    for phase in (0, 1):
        units = 32 if phase == 0 else 16
        offset = 0 if phase == 0 else 32 * 32
        singletons = []
        for expert, count in enumerate(counts):
            jobs = (count + 63) // 64
            for _ in range(jobs // 2):
                plan.extend((offset + expert * units + unit,) * 2 for unit in range(units))
            if jobs % 2:
                singletons.extend(offset + expert * units + unit for unit in range(units))
        for first in range(0, len(singletons), 2):
            plan.append((singletons[first], singletons[first + 1] if first + 1 < len(singletons) else -1))
    return plan


def expected(plan, mode, full):
    output = [[0, 0, 0] for _ in range(GRID)]
    for group, pair in enumerate(plan):
        for rank, unit in enumerate(pair):
            if unit < 0:
                continue
            row = output[(group % PAIRS) * 2 + rank]
            start = unit * STAGES * WORDS
            if full:
                checksum = 0  # CHECK builds count per-word mismatches, not a colliding modular sum.
            else:
                checksum = sum((A * (start + stage * WORDS) + B) +
                               (A * (start + stage * WORDS + WORDS - 1) + B)
                               for stage in range(STAGES)) & MASK
            row[0] = (row[0] + checksum) & MASK
            row[1] += STAGES
            row[2] += 0 if mode == 2 and rank == 1 and pair[0] == pair[1] else STAGES
    return output


def fixtures(bound):
    return [*bound["microbench_fixtures"], dict(load="all_paired_control", input_seed=None,
                                                counts=[128] * 32, base_us=None)]


def child(args):
    import torch
    from torch.utils.cpp_extension import load_inline
    from screen_moe import capture
    source = Path("multicast_read.cu").read_text()
    report = dict(status="preparing", mode=args.mode, transfer_only=True,
                  cuda_sha256=sha("multicast_read.cu"), bound_sha256=sha("reuse-bound.json"),
                  fixtures=[], artifact_files=[])
    save(args.output, report)
    start = time.monotonic()
    ext = load_inline(name="experts_v32_multicast_" + report["cuda_sha256"][:12],
                      cpp_sources="""#include <torch/extension.h>
std::vector<int64_t> configure();
void initialize(torch::Tensor source);
void run(torch::Tensor source, torch::Tensor plan, torch::Tensor output, int64_t mode, bool check);
""", cuda_sources=source, functions=["configure", "initialize", "run"],
                      extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    report["compile_seconds"] = time.monotonic() - start
    report["occupancy"] = dict(zip(("sms", "per_cta_shared_bytes", "max_active_two_cta_clusters"), ext.configure()))
    report["extension"] = ext.__file__
    report["gpu"] = torch.cuda.get_device_name()
    bank = torch.empty(BANK_UNITS * STAGES * WORDS, dtype=torch.int32, device="cuda")
    ext.initialize(bank)
    flush = torch.ones(256 * 1024 * 1024 // 4, device="cuda")
    bound = json.loads(Path("reuse-bound.json").read_text())
    # A non-endpoint corruption must be observed by both recipients, not hidden
    # by the timed endpoint checksum or by a permutation-invariant full sum.
    fault_plan = torch.tensor(make_plan([128] * 32), dtype=torch.int32, device="cuda")
    fault_out = torch.empty((GRID, 3), dtype=torch.int64, device="cuda")
    bank[123].bitwise_xor_(1)
    report["validator_fault_detection"] = {}
    for mode in (0, 2):
        ext.run(bank, fault_plan, fault_out, mode, True)
        mismatches = int(fault_out[:, 0].sum().item())
        if mismatches != 2:
            raise RuntimeError("Per-word validator did not detect both corrupted-word recipients")
        report["validator_fault_detection"][str(mode)] = mismatches
    bank[123].bitwise_xor_(1)
    del fault_plan, fault_out
    for index, fixture in enumerate(fixtures(bound)):
        plan = make_plan(fixture["counts"])
        table = torch.tensor(plan, dtype=torch.int32, device="cuda")
        logical = sum(unit >= 0 for pair in plan for unit in pair) * STAGES * SB
        saved = sum(pair[0] == pair[1] and pair[0] >= 0 for pair in plan) * STAGES * SB
        row = dict(fixture, group_count=len(plan), logical_consumer_bytes=logical,
                   multicast_issued_bytes=logical - saved, multicast_saved_request_bytes=saved,
                   sharing_group_fraction=sum(a == b for a, b in plan) / len(plan), checks={}, samples=[])
        report["fixtures"].append(row)
        save(args.output, report)
        captures, events = {}, {}
        for name in ARMS:
            mode = MODES[name]
            out = torch.empty((GRID, 3), dtype=torch.int64, device="cuda")
            ext.run(bank, table, out, mode, True)
            torch.cuda.synchronize()
            if out.cpu().tolist() != expected(plan, mode, True):
                row["failure_actual"] = out.cpu().tolist()
                row["failure_expected"] = expected(plan, mode, True)
                save(args.output, report)
                raise RuntimeError("Exact per-word validation failure: " + name)
            row["checks"][name] = {"all_words_exact": True}
            def run(_, __, actual_mode=mode, actual_out=out):
                ext.run(bank, table, actual_out, actual_mode, False)
                return actual_out
            graph, result = capture(run, {}, {})
            captures[name] = (graph, result)
            events[name] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            a, b = events[name]; a.record(); b.record(); b.synchronize()
        if args.mode == "timing":
            for sample in range(args.samples):
                offset = sample % 4
                order = list(ARMS[offset:] + ARMS[:offset])
                if (sample // 4) % 2:
                    order.reverse()
                record = dict(order=order, us={})
                for name in order:
                    sink = flush.sum()
                    a, b = events[name]
                    a.record(); captures[name][0].replay(); b.record(); b.synchronize()
                    record["us"][name] = a.elapsed_time(b) * 1000
                    del sink
                record["ratio"] = {name: record["us"]["independent"] / record["us"][name]
                                   for name in ARMS[1:]}
                record["multicast_vs_cluster"] = record["us"]["cluster_independent"] / record["us"]["multicast"]
                row["samples"].append(record)
            row["median_us"] = {name: statistics.median(r["us"][name] for r in row["samples"]) for name in ARMS}
            row["median_ratio"] = {name: statistics.median(r["ratio"][name] for r in row["samples"])
                                   for name in ARMS[1:]}
            row["multicast_vs_cluster"] = statistics.median(r["multicast_vs_cluster"] for r in row["samples"])
        else:
            row["profile_order"] = list(ARMS[:3])
            for name in ARMS[:3]:
                sink = flush.sum()
                torch.cuda.synchronize()
                torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
                captures[name][0].replay()
                torch.cuda.synchronize()
                torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStop())
                del sink
        for name, (_, result) in captures.items():
            if result.cpu().tolist() != expected(plan, MODES[name], False):
                raise RuntimeError("Timed stage checksum/completion failure: " + name)
            row["checks"][name]["timed_stage_checksum_and_counts"] = True
        row["status"] = "passed"
        print(json.dumps({k: row[k] for k in ("load", "input_seed", "logical_consumer_bytes", "multicast_saved_request_bytes")}
                         | {k: row[k] for k in ("median_us", "median_ratio", "multicast_vs_cluster") if k in row}), flush=True)
        save(args.output, report)
        captures.clear(); events.clear()
        del table, out, graph, result
    if args.mode == "timing":
        report["control_geomean"] = math.prod(r["median_ratio"]["independent_repeat"] for r in report["fixtures"]) ** (1 / len(report["fixtures"]))
        report["control_within_one_percent"] = abs(report["control_geomean"] - 1) <= 0.01
    report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["status"] = "complete"
    save(args.output, report)


def parent(args):
    from profile_large import parse_csv
    prefix = Path(args.output).stem
    report = dict(status="running", transfer_only=True, official_score=False, artifact_files=[],
                  limitations=["No MoE compute or dependency graph: transfer timings are not an integrated speedup.",
                               "Independent arm uses the same perfectly aligned pair plan; L2 may already coalesce repeated reads.",
                               "Cluster-independent arm isolates multicast while retaining the exact per-stage cluster synchronization.",
                               "All arms reserve native-sized shared memory but use a trivial256-thread consumer, not WGS576-thread compute."])
    save(args.output, report)
    def run(command, log, timeout=700):
        report["artifact_files"].append(log)
        save(args.output, report)
        with open(log, "w") as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
        if result.returncode:
            raise RuntimeError(log + ": exit " + str(result.returncode))
    timing, profiled, counters = prefix + "-timing.json", prefix + "-profiled.json", prefix + "-ncu.csv"
    report["artifact_files"].extend([timing, profiled, counters])
    common = [sys.executable, __file__, "--samples", str(args.samples)]
    try:
        run(common + ["--mode", "timing", "--output", timing], prefix + "-timing.log")
        measured = json.loads(Path(timing).read_text())
        run(["/usr/local/cuda/bin/ncu", "--profile-from-start", "off", "--clock-control", "none",
             "--cache-control", "all", "--replay-mode", "kernel", "--csv", "--print-units", "base",
             "--metrics", ",".join(METRICS), "--log-file", counters,
             *common, "--mode", "profile", "--output", profiled], prefix + "-profile.log")
        profile = json.loads(Path(profiled).read_text())
        native = parse_csv(counters)
        if len(native) != len(measured["fixtures"]) * 3 or len(profile["fixtures"]) != len(measured["fixtures"]):
            raise RuntimeError("Native launch count mismatch")
        report["fixtures"] = []
        for index, (t, p) in enumerate(zip(measured["fixtures"], profile["fixtures"])):
            for key in ("counts", "load", "input_seed", "logical_consumer_bytes", "multicast_issued_bytes"):
                if t[key] != p[key]:
                    raise RuntimeError("Timing/profile identity mismatch: " + key)
            row = dict(t, native={})
            for name, record in zip(ARMS[:3], native[index * 3:index * 3 + 3]):
                if not set(METRICS) <= record["metrics"].keys() or "stream<" not in record["kernel"]:
                    raise RuntimeError("Unexpected native record")
                values = record["metrics"]
                row["native"][name] = dict(record, profile_us=values["gpu__time_duration.sum"] / 1000,
                                           dram_MB=values["dram__bytes_read.sum"] / 1e6,
                                           l2_read_MB=values["lts__t_sectors_op_read.sum"] * 32 / 1e6)
            report["fixtures"].append(row)
        for key in ("control_geomean", "control_within_one_percent", "occupancy", "compile_seconds", "gpu",
                    "cuda_sha256", "bound_sha256", "peak_cuda_allocated_bytes", "validator_fault_detection"):
            report[key] = measured[key]
        run(["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", measured["extension"]], prefix + "-resources.txt", 60)
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        raise
    finally:
        report["artifact_files"] = list(dict.fromkeys(p for p in report["artifact_files"] if Path(p).exists()))
        save(args.output, report)
        print(json.dumps(dict(status=report["status"], output=args.output, control=report.get("control_geomean"))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="multicast-result.json")
    parser.add_argument("--mode", choices=("parent", "timing", "profile"), default="parent")
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    (parent if args.mode == "parent" else child)(args)
