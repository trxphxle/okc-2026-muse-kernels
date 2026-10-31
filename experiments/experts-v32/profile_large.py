"""Lane-only unchanged MoE traffic/issue attribution; never a submission.

Run with run_layer.py, screen_moe.py, upstream source and public oracle helpers.
Three fresh inputs/load, one public weight seed. Timings and NCU launches are
separate processes with identical source/fixture hashes. No replay subtraction.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import time

LOADS = (("D32", 32), ("V16", 128), ("V32", 256))
WEIGHT_BYTES = 28311552
REQUIRED = (
    "gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
    "lts__t_sectors_op_read.sum", "lts__t_sector_hit_rate.pct",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__inst_executed.sum", "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_membar_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
)
OPTIONAL = (
    "launch__registers_per_thread", "launch__shared_mem_per_block_allocated",
    "smsp__inst_executed_pipe_alu.sum", "smsp__inst_executed_pipe_fma.sum",
    "smsp__inst_executed_pipe_fmaheavy.sum", "smsp__inst_executed_pipe_fp16.sum",
    "smsp__inst_executed_pipe_lsu.sum", "smsp__inst_executed_pipe_uniform.sum",
    "smsp__inst_executed_pipe_tensor.sum", "smsp__inst_executed_pipe_xu.sum",
    "smsp__sass_thread_inst_executed_op_integer_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fp16_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_conversion_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_control_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_memory_pred_on.sum",
)


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def child(args):
    import torch
    import screen_moe as screen
    base = screen.load(Path(args.source), "profile_base")
    reference = screen.load(Path("reference.py"), "profile_reference")
    report = dict(status="preparing", mode=args.mode, source_sha256=sha(args.source),
                  weight_seed=731283, fixtures=[], artifact_files=[])
    save(args.output, report)
    weights = reference.generate_weights(731283)
    start = time.monotonic()
    state = base.prepare(weights)
    report["prepare_seconds"] = time.monotonic() - start
    report["sfx"] = list(state["sfx"])
    report["gpu"] = torch.cuda.get_device_name()
    report["torch"] = torch.__version__
    report["cuda"] = torch.version.cuda
    report["extensions"] = [str(Path(ext.__file__)) for ext in base._EXTS.values()]
    flush = torch.ones(screen.FLUSH_BYTES // 4, device="cuda")
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record(); end.record(); end.synchronize()
    for load, m in LOADS:
        for pair in range(3):
            seed = 731283 + m * 10000 + pair + 1
            data = screen.make_input(reference, m, seed)
            counts = screen.routing(data)["local_expert_histogram"]
            chunks = [min(64, n - j) for n in counts for j in range(0, n, 64)]
            row = dict(load=load, m=m, input_seed=seed, counts=counts,
                       input_sha256=hashlib.sha256(b"".join(data[k].cpu().view(torch.uint8).numpy().tobytes()
                           for k in ("hidden_states", "topk_ids", "topk_weights"))).hexdigest(),
                       experts=sum(n > 0 for n in counts), jobs=len(chunks), routes=sum(counts),
                       job_counts=chunks, ntt_histogram=dict(Counter((n + 7) // 8 for n in chunks)),
                       unique_weight_bytes=sum(n > 0 for n in counts) * WEIGHT_BYTES,
                       planned_weight_bytes=len(chunks) * WEIGHT_BYTES,
                       weight_reread_bytes=(len(chunks) - sum(n > 0 for n in counts)) * WEIGHT_BYTES,
                       logical_x_tma_bytes=sum((n + 7) // 8 * 8 for n in chunks) * 512 * 16 * 48)
            gold = reference.ref_kernel(data, weights)
            graph, output = screen.capture(base.kernel_fn, data, state)
            passed, message = reference.compare(output, gold, screen.REL_L2_TOL)
            row["correctness_before"] = dict(passed=bool(passed), message=str(message))
            if not passed:
                raise RuntimeError(str(message))
            if args.mode == "timing":
                samples = []
                for _ in range(21):
                    sink = flush.sum()
                    begin.record(); graph.replay(); end.record(); end.synchronize()
                    samples.append(begin.elapsed_time(end) * 1000)
                    del sink
                row["samples_us"] = samples
                row["median_us"] = sorted(samples)[len(samples) // 2]
            else:
                sink = flush.sum()
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                graph.replay()
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                del sink
            passed, message = reference.compare(output, gold, screen.REL_L2_TOL)
            row["correctness_after"] = dict(passed=bool(passed), message=str(message))
            if not passed:
                raise RuntimeError(str(message))
            report["fixtures"].append(row)
            save(args.output, report)
            print(json.dumps({k: row[k] for k in ("load", "input_seed", "experts", "jobs", "routes")}), flush=True)
            del graph, output, gold, data
    report["status"] = "complete"
    report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    save(args.output, report)


def parse_csv(path):
    rows = list(csv.reader(io.StringIO(Path(path).read_text())))
    start = next(i for i, r in enumerate(rows) if "Kernel Name" in r and "ID" in r)
    header = rows[start]
    if "Metric Name" not in header:
        # ncu --import --page raw uses one metric column per native record.
        units = dict(zip(header, rows[start + 1]))
        records = []
        for values in rows[start + 2:]:
            if len(values) != len(header):
                continue
            r = dict(zip(header, values))
            metrics = {}
            for name, text in r.items():
                if "__" not in name:
                    continue
                try:
                    metrics[name] = float(text.replace(",", ""))
                except ValueError:
                    continue
            records.append(dict(kernel=r["Kernel Name"], metrics=metrics, units=units))
        return records
    records = {}
    for values in rows[start + 1:]:
        if len(values) > len(header):
            continue
        values = values + [""] * (len(header) - len(values))
        r = dict(zip(header, values))
        try:
            value = float(r["Metric Value"].replace(",", ""))
        except ValueError:
            continue
        record = records.setdefault(r["ID"], dict(kernel=r["Kernel Name"], metrics={}, units={}))
        record["metrics"][r["Metric Name"]] = value
        record["units"][r["Metric Name"]] = r["Metric Unit"]
    return list(records.values())


def parent(args):
    prefix = Path(args.output).stem
    report = dict(status="running", official_score=False, source_sha256=sha(args.source),
                  policy="unchanged native graph, cold L2, native clocks; dynamic counters are not additive latency savings",
                  artifact_files=[])
    save(args.output, report)
    def run(command, name, timeout=800, allow_failure=False):
        report["artifact_files"].append(name)
        save(args.output, report)
        with open(name, "w") as out:
            result = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, timeout=timeout)
        if result.returncode and not allow_failure:
            raise RuntimeError(f"{name}: exit {result.returncode}")
        return result.returncode
    try:
        timing, profiled = prefix + "-timing.json", prefix + "-profiled.json"
        common = [sys.executable, __file__, "--source", args.source]
        if args.reuse_timing:
            timing = args.reuse_timing
            retained = json.loads(Path(timing).read_text())
            if retained["status"] != "complete" or retained["source_sha256"] != sha(args.source):
                raise RuntimeError("Retained timing/source mismatch")
        else:
            run(common + ["--mode", "timing", "--output", timing], prefix + "-timing.log")
        report["artifact_files"].append(timing)
        ncu = "/usr/local/cuda/bin/ncu"
        metric_file = prefix + "-metrics.txt"
        if not args.reuse_profile:
            run([ncu, "--query-metrics", "--query-metrics-mode", "all"], metric_file, 30)
        installed = Path(metric_file).read_text()
        exists = lambda m: re.search(r"(?<![\w.])" + re.escape(m) + r"(?![\w.])", installed) is not None
        missing = [m for m in REQUIRED if not exists(m)]
        if missing:
            raise RuntimeError("Missing required metrics: " + repr(missing))
        selected = list(REQUIRED) + [m for m in OPTIONAL if exists(m)]
        report["selected_metrics"] = selected
        csv_file, ncu_report = prefix + "-ncu.csv", prefix + "-native"
        report["artifact_files"].extend([profiled, csv_file, ncu_report + ".ncu-rep"])
        if not args.reuse_profile:
            run([ncu, "--profile-from-start", "off", "--clock-control", "none",
                 "--cache-control", "all", "--replay-mode", "kernel", "--csv",
                 "--kernel-name", "regex:.*moe_fused.*", "--metrics", ",".join(selected),
                 "--section", "SourceCounters", "--section", "InstructionStats",
                 "--print-units", "base", "--log-file", prefix + "-capture.log", "-f", "-o", ncu_report,
                 *common, "--mode", "profile", "--output", profiled], prefix + "-profile.log")
        run([ncu, "--import", ncu_report + ".ncu-rep", "--page", "raw", "--csv", "--print-units", "base"],
            csv_file, 90)
        native = parse_csv(csv_file)
        timing_rows = json.loads(Path(timing).read_text())
        profile_rows = json.loads(Path(profiled).read_text())
        if len(native) != 9 or len(profile_rows["fixtures"]) != 9 or len(timing_rows["fixtures"]) != 9:
            raise RuntimeError("Expected exactly nine native records and fixtures")
        report["fixtures"] = []
        for observed, t, p in zip(native, timing_rows["fixtures"], profile_rows["fixtures"]):
            for key in ("input_sha256", "counts", "input_seed"):
                if t[key] != p[key]:
                    raise RuntimeError("Fixture mismatch: " + key)
            missing = set(REQUIRED) - observed["metrics"].keys()
            if missing:
                raise RuntimeError("Missing measured metrics: " + repr(missing))
            values = observed["metrics"]
            row = dict(t, native=observed)
            row["profile_us"] = values["gpu__time_duration.sum"] / 1000
            row["dram_read_bytes"] = values["dram__bytes_read.sum"]
            row["dram_over_unique_bytes"] = row["dram_read_bytes"] - row["unique_weight_bytes"]
            row["dram_over_planned_weights_bytes"] = row["dram_read_bytes"] - row["planned_weight_bytes"]
            row["dram_read_TB_s"] = row["dram_read_bytes"] / row["profile_us"] / 1e6
            report["fixtures"].append(row)
        report["prepare_seconds"] = timing_rows["prepare_seconds"]
        report["gpu"] = timing_rows["gpu"]
        report["peak_cuda_allocated_bytes"] = timing_rows["peak_cuda_allocated_bytes"]
        main = next(p for p in timing_rows["extensions"] if "okc_wgs1" in p)
        run(["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", main], prefix + "-resources.txt", 30)
        run(["/usr/local/cuda/bin/cuobjdump", "--dump-sass", main], prefix + "-sass.txt", 60)
        rc = run([ncu, "--import", ncu_report + ".ncu-rep", "--page", "source", "--print-source", "sass"],
                 prefix + "-source.txt", 90, True)
        report["source_export_exit_code"] = rc
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        raise
    finally:
        report["artifact_files"] = [p for p in report["artifact_files"] if Path(p).exists()]
        save(args.output, report)
        print(json.dumps(dict(status=report["status"], output=args.output)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="muse_moe_w4a16.py")
    parser.add_argument("--output", default="profile-large.json")
    parser.add_argument("--mode", choices=("parent", "timing", "profile"), default="parent")
    parser.add_argument("--reuse-timing", help="Completed timing JSON from this exact source; resume profiling only")
    parser.add_argument("--reuse-profile", action="store_true", help="Parse the already captured native NCU report")
    args = parser.parse_args()
    (parent if args.mode == "parent" else child)(args)
