"""Timing-only WGS completion oracle screen: real full decode, one allocation pad/process.

Prepare ONLY the compliant baseline for every arm, then bind oracle WGS modules
and fresh zeroed scratch. Every timed graph computes its own frontend/routing;
no route tables or expected outputs are supplied to either oracle. All shapes,
full/sliding weighting, cold-L2 work and A/A controls remain explicit.
"""
import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch
import compare_decode_public as common

common.torch = torch
ARMS = ("base", "A", "B", "repeat")
VARIANTS = ("A", "B", "repeat")
LOADS = common.LOADS
WEIGHTS = {"sliding":46,"full":16}


def save(path, report):
    common.save(path, report)


def tiny_fingerprint(data):
    digest = hashlib.sha256()
    for name in ("attn_input", "stream", "local_start", "seq_lens", "cache_loc", "page_table"):
        value = data[name]
        digest.update(name.encode())
        digest.update(value.contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    digest.update(json.dumps(common.metadata(data["config"]),sort_keys=True).encode())
    return digest.hexdigest()


def fixed_policy(base_state, state):
    keys = ("mm_pick", "moe_pick", "compose_graft_enabled", "compose_boundary_warps",
            "marlin_cfg", "pbmax", "att_ok", "ma_ok")
    for name in keys:
        if name in base_state:
            state[name] = copy.deepcopy(base_state[name])
    if base_state.get("bfp") is not None:
        if state.get("bfp") is None:
            raise RuntimeError("BFP eligibility changed for identical weights")
        for name in ("choice", "disabled"):
            if name in base_state["bfp"]:
                state["bfp"][name] = copy.deepcopy(base_state["bfp"][name])
    if state["ma_sfx"] != base_state["ma_sfx"] or state["nsm"] != base_state["nsm"]:
        raise RuntimeError("Expert scale/grid policy changed")
    if not torch.equal(state["bias"],base_state["bias"]):
        raise RuntimeError("Prepared router bias differs")


def summarize_case(row):
    row["median_us"] = {name:statistics.median(r["us"][name] for r in row["records"]) for name in ARMS}
    row["median_ratio"] = {name:statistics.median(r["ratio"][name] for r in row["records"]) for name in VARIANTS}
    row["control_within_one_percent"] = abs(row["median_ratio"]["repeat"]-1)<=0.01
    row["all_route_ids_equal"] = {name:all(r["route_ids_equal"][name] for r in row["records"]) for name in VARIANTS}
    row["status"] = "complete"


def aggregate(report):
    result = {}
    for load,_,_ in LOADS:
        cases = [r for r in report["cases"] if r["load"]==load]
        if len(cases)!=2 or any(r.get("status")!="complete" for r in cases):
            continue
        base = sum(WEIGHTS[r["layer"]]*r["median_us"]["base"] for r in cases)/62
        times = {name:sum(WEIGHTS[r["layer"]]*(r["median_us"]["base"]/r["median_ratio"][name])
                          for r in cases)/62 for name in VARIANTS}
        result[load] = dict(base_us=base, equivalent_us=times,
                            ratio={name:base/value for name,value in times.items()},
                            all_case_controls_passed=all(r["control_within_one_percent"] for r in cases))
    report["loads"] = result
    if len(result)==6:
        report["geomean_ratio"] = {name:math.prod(r["ratio"][name] for r in result.values())**(1/6) for name in VARIANTS}
        report["all_case_controls_passed"] = all(r["control_within_one_percent"] for r in report["cases"])
        report["aggregate_control_passed"] = abs(report["geomean_ratio"]["repeat"]-1)<=0.01


def profile_v32(row, captures, private, prefix, report):
    trace_path = prefix + "-V32-" + row["layer"] + "-native.json"
    identities=[]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profiler:
        for name in ("base","A","B"):
            for replay in range(3):
                captures[name]["graph"].replay()
                torch.cuda.synchronize()
                identities.append(dict(arm=name,replay=replay,
                                       routes=common.routes(captures[name]["output"],int(private[name]["local_start"]))))
    profiler.export_chrome_trace(trace_path)
    report["artifact_files"].append(Path(trace_path).name)
    trace=json.loads(Path(trace_path).read_text())
    launches=[e for e in trace["traceEvents"] if e.get("cat")=="cuda_runtime" and "cudaGraphLaunch" in e.get("name","")]
    if len(launches)!=len(identities):
        raise RuntimeError("Native graph launch/correlation count mismatch")
    measured=[]
    for launch,identity in zip(launches,identities):
        correlation=launch["args"]["correlation"]
        experts=[e for e in trace["traceEvents"] if e.get("cat")=="kernel"
                 and e.get("args",{}).get("correlation")==correlation and "moe_fused" in e.get("name","")]
        if len(experts)!=1:
            raise RuntimeError("Expected one complete native WGS expert event per graph")
        event=experts[0]
        measured.append(dict(identity,correlation=correlation,name=event["name"],us=event["dur"],resources=event["args"]))
    row["native_experts"] = dict(samples=measured,
        median_us={name:statistics.median(r["us"] for r in measured if r["arm"]==name) for name in ("base","A","B")},
        attribution="CUDA graph launch correlation IDs; actual MoE kernel duration, not CPU timestamp containment",
        trace=Path(trace_path).name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",default="decode_alygg77_bf16.py")
    parser.add_argument("--oracle-a",default="oracle_split_A.py")
    parser.add_argument("--oracle-b",default="oracle_completion_B.py")
    parser.add_argument("--output",required=True)
    parser.add_argument("--pad-seed",type=int,required=True)
    parser.add_argument("--weight-seed",type=int,default=731283)
    parser.add_argument("--samples",type=int,default=4)
    args=parser.parse_args()
    if args.samples<4 or args.samples%4:
        parser.error("--samples must be a positive multiple of4 for balanced arm positions")
    prefix=str(Path(args.output).with_suffix(""))
    report=dict(status="starting",diagnostic_only=True,submission_allowed=False,
                pad_seed=args.pad_seed,weight_seed=args.weight_seed,samples_per_case=args.samples,
                cases=[],artifact_files=[],prepare_seconds={},pads_bytes={},
                protocol="Eight-replay palindrome per fresh input; rotate/reverse roles;256MiB read-only flush inside every graph before external timing events",
                policy="Every state prepared by compliant base; freeze policies; oracle WGS handles rebound only after prepare; private zeroed expert scratch; real frontend/routing timed in every graph",
                limitations=["Oracle outputs intentionally fail mathematics; no correctness or compliant speedup claim for A/B.",
                             "Only baseline is public-oracle checked; actual upstream route equality is reported, never supplied to the oracles.",
                             "Native V32 expert spans preserve full-layer PDL overlap and are not added to other stage spans."])
    save(args.output,report)
    try:
        reference=common.load("reference.py","oracle_reference")
        common.reference=reference
        paths={"base":args.baseline,"A":args.oracle_a,"B":args.oracle_b}
        modules={name:common.load(path,"completion_"+name) for name,path in paths.items()}
        modules["repeat"]=modules["base"]
        report["sources"]={name:dict(path=path,sha256=common.sha(path)) for name,path in paths.items()}
        report["environment"]=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
        weights=reference.generate_weights(args.weight_seed)
        states,pads={},[]
        rng=random.Random(args.pad_seed)
        for name in ("base","A","B"):
            nbytes=rng.randrange(0,32*1024*1024//512+1)*512
            pads.append(torch.empty(nbytes,dtype=torch.uint8,device="cuda"))
            report["pads_bytes"][name]=nbytes
            arm_weights=common.clone_data(weights)
            start=time.monotonic()
            states[name]=modules["base"].prepare(arm_weights)
            torch.cuda.synchronize()
            report["prepare_seconds"][name]=time.monotonic()-start
            if name!="base":
                fixed_policy(states["base"],states[name])
                # Only WGS variants differ. Tiny/tiny2/mid remain original modules.
                for key in ("b4","main"):
                    states[name]["ma_ext"].pop(key,None)
                if hasattr(modules["base"],"_TUNED7"):
                    modules[name]._TUNED7=copy.deepcopy(modules["base"]._TUNED7)
                for tokens in (32,128):
                    modules[name]._moe(tokens,states[name])
            for scratch in states[name]["ma_scratch"]:
                scratch.zero_()
            del arm_weights
            save(args.output,report)
        states["repeat"]=states["base"]
        report["fixed_modes"]={name:common.mode_state(modules[name],states[name]) for name in ("base","A","B")}
        report["scratch_layouts"]={name:[common.metadata(t) for t in states[name]["ma_scratch"]] for name in ("base","A","B")}
        flush=torch.ones(common.FLUSH_BYTES//4,device="cuda")
        for index,(load,batch,length) in enumerate(LOADS):
            for layer in ("sliding","full"):
                shape=dict(batch=batch,q_len=length,ctx=80000,layer=layer,page=1,
                           spread="agentic" if batch>1 else "fixed",dist="soft")
                seed=919233+batch*100+length*1000+(layer=="full")*10000
                initial=reference.generate_input(seed=seed,**shape)
                private={name:common.clone_data(initial) for name in ("base","A","B")}
                private["repeat"]=private["base"]
                captures={name:common.capture(modules[name].kernel_fn,private[name],states[name],flush) for name in ARMS}
                row=dict(load=load,layer=layer,shape=shape,capture_seed=seed,records=[],status="timing")
                report["cases"].append(row)
                save(args.output,report)
                for sample in range(args.samples):
                    fresh=reference.generate_input(seed=seed+sample+1,**shape)
                    for name in ("base","A","B"):
                        common.refill(private[name],fresh)
                    shift=(sample+index)%4
                    order=list(ARMS[shift:]+ARMS[:shift])
                    if (sample//4)%2:
                        order.reverse()
                    order=order+list(reversed(order))
                    observations={name:[] for name in ARMS}
                    for name in order:
                        observations[name].append(common.measure(captures[name]))
                    us={name:statistics.mean(values) for name,values in observations.items()}
                    base_output=captures["base"]["output"]
                    ids={name:bool(torch.equal(base_output[2],captures[name]["output"][2])) for name in VARIANTS}
                    check=common.oracle(private["base"],base_output,weights,fresh)
                    if not check["passed"]:
                        raise RuntimeError("Compliant baseline failed public oracle: "+str(check))
                    record=dict(input_seed=seed+sample+1,input_fingerprint=tiny_fingerprint(fresh),order=order,
                                repetitions_us=observations,us=us,ratio={name:us["base"]/us[name] for name in VARIANTS},
                                route_ids_equal=ids,baseline_oracle=check,
                                routes={name:common.routes(captures[name]["output"],int(private[name]["local_start"]))
                                        for name in ("base","A","B")})
                    if batch*length<=16:
                        record["unchanged_output_comparison"]={name:common.comparison(captures[name]["output"],base_output)
                                                                for name in ("A","B")}
                        if not all(value["close"] for value in record["unchanged_output_comparison"].values()):
                            raise RuntimeError("An oracle changed a non-WGS load")
                    row["records"].append(record)
                    save(args.output,report)
                summarize_case(row)
                if load=="V32":
                    profile_v32(row,captures,private,prefix,report)
                aggregate(report)
                save(args.output,report)
                print(json.dumps(dict(pad=args.pad_seed,load=load,layer=layer,ratio=row["median_ratio"],
                                      controls=row["control_within_one_percent"],routes=row["all_route_ids_equal"])),flush=True)
                captures.clear(); private.clear()
                del initial,fresh,base_output
                gc.collect()
        if len(report["cases"])!=12 or any(r["status"]!="complete" for r in report["cases"]):
            raise RuntimeError("Incomplete required load/layer coverage")
        report["status"]="complete"
        report["peak_cuda_allocated_bytes"]=torch.cuda.max_memory_allocated()
        aggregate(report)
    except BaseException as exc:
        import traceback
        report.update(status="failed",error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        save(args.output,report)
        print(json.dumps(dict(status=report["status"],output=args.output,geomean=report.get("geomean_ratio"))),flush=True)


if __name__=="__main__":
    main()
