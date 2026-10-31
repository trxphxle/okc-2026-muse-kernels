"""Recover only the missing V16-full native attribution, not successful timings.

Three native graph replays per arm; partition strictly by cudaGraphLaunch's CUPTI
correlation ID. No CPU timestamp containment and no fastest-trace selection.
"""
import argparse
import ast
import json
from pathlib import Path
import statistics

import torch
import compare_decode_public as common

common.torch = torch


def pin_policy(state, saved):
    for key in ("mm_pick", "moe_pick", "compose_graft_enabled", "compose_boundary_warps"):
        if key not in saved:
            continue
        target = {}
        for name, value in saved[key].items():
            target[ast.literal_eval(name)] = value
        state[key] = target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="precision-raw-vs-repaired.json")
    parser.add_argument("--output", default="precision-native-recovery.json")
    args = parser.parse_args()
    prior = json.loads(Path(args.input).read_text())
    fixture = next(row for row in prior["results"] if row["load"] == "V16" and row["layer"] == "full")
    seed = fixture["records"][-1]["seed"]
    report = dict(status="running", load="V16", layer="full", input_seed=seed,
                  source_report_sha256=common.sha(args.input), arms={}, artifact_files=[],
                  attribution="Native kernels grouped by cudaGraphLaunch CUPTI correlation IDs; no CPU-window inference",
                  limitations="Recovery native timings are a separate capture; successful original paired event timing remains authoritative.")
    common.save(args.output, report)
    ref = common.load("reference.py", "recovery_reference")
    weights = ref.generate_weights(prior["seed"])
    flush = torch.ones(common.FLUSH_BYTES // 4, device="cuda")
    try:
        for arm in ("baseline", "public"):
            path = prior["sources"][arm]["path"]
            module = common.load(path, "recover_" + arm)
            state = module.prepare(common.clone_data(weights))
            pin_policy(state, fixture["mode_after_samples"][arm]["state"])
            data = ref.generate_input(seed=seed, **fixture["shape"])
            cap = common.capture(module.kernel_fn, data, state)
            expected = tuple(t.clone() for t in cap["output"])
            trace_path = Path(args.output).stem + "-" + arm + "-trace.json"
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profiler:
                for _ in range(3):
                    flush.sum()
                    torch.cuda.synchronize()
                    cap["graph"].replay()
                    torch.cuda.synchronize()
            profiler.export_chrome_trace(trace_path)
            report["artifact_files"].append(trace_path)
            trace = json.loads(Path(trace_path).read_text())
            launches = [event for event in trace["traceEvents"] if event.get("cat") == "cuda_runtime"
                        and "cudaGraphLaunch" in event.get("name", "")]
            if len(launches) != 3:
                raise RuntimeError("Expected three graph launch correlation IDs")
            record = dict(source_sha256=common.sha(path), samples=[], partial_samples=[],
                          output_comparison=common.comparison(cap["output"], expected),
                          mode=common.mode_state(module, state))
            for index, launch in enumerate(launches):
                corr = launch["args"]["correlation"]
                kernels = [event for event in trace["traceEvents"] if event.get("cat") == "kernel"
                           and event.get("args", {}).get("correlation") == corr]
                names = [event["name"] for event in kernels]
                complete = (any("attn_kernel" in name for name in names)
                            and any("moe_fused" in name for name in names)
                            and any("_graft_route" in name for name in names)
                            and len(kernels) >= 9)
                if not complete:
                    record["partial_samples"].append(dict(index=index, correlation=corr, names=names))
                    continue
                one_path = Path(args.output).stem + "-" + arm + "-replay" + str(index) + ".json"
                Path(one_path).write_text(json.dumps(dict(traceEvents=kernels)))
                report["artifact_files"].append(one_path)
                record["samples"].append(dict(index=index, correlation=corr, trace=one_path,
                                               **common.summarize_trace(one_path)))
            if not record["samples"] or not record["output_comparison"]["close"]:
                raise RuntimeError("No complete native recovery or graph output mismatch")
            record["status"] = "complete"
            report["arms"][arm] = record
            common.save(args.output, report)
            del cap, state, module, data, expected
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        common.save(args.output, report)
        print(json.dumps(dict(status=report["status"], output=args.output,
                              complete_replays={a:len(r["samples"]) for a,r in report["arms"].items()})),flush=True)


if __name__ == "__main__":
    main()
