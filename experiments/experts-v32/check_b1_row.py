"""Exact B1 row/cluster equivalence plus real local-memory counters; not a submission."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

METRICS = ("gpu__time_duration.sum", "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
           "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum")


def save(path, report):
    Path(path).write_text(json.dumps(report, indent=2) + "\n")


def child(args):
    import torch
    import compare_decode_public as common
    common.torch = torch
    reference = common.load("reference.py", "row_reference")
    base = common.load(args.baseline, "row_base")
    candidate = common.load(args.candidate, "row_candidate")
    report = dict(status="preparing", mode=args.mode, cases=[], profile_order=[],
                  sources={"baseline":common.sha(args.baseline),"candidate":common.sha(args.candidate)})
    save(args.output,report)
    weights = reference.generate_weights(731283)
    start = time.monotonic()
    state = candidate.prepare(weights)
    torch.cuda.synchronize()
    report["candidate_prepare_seconds"] = time.monotonic()-start
    gain = state["graft"]
    mods = {"base":base._sl_cluster_module(),"candidate":candidate._sl_cluster_module()}
    gen = torch.Generator(device="cuda").manual_seed(782810)
    flush = torch.ones(common.FLUSH_BYTES//4,device="cuda")
    for tokens in (1,2,4,8,16):
        for fixture in ("random","bf16_sensitive_zero_stream","zero_branch"):
            branch = torch.randn((tokens,8192),device="cuda",generator=gen).to(torch.bfloat16)
            stream = torch.randn((tokens,8192),device="cuda",generator=gen)*0.25
            if fixture == "bf16_sensitive_zero_stream":
                i = torch.arange(tokens*8192,device="cuda").reshape(tokens,8192)
                branch = ((1.0+(i%127).float()/128.0)*torch.exp2(((i//127)%7-3).float())
                          *torch.where(i%3==0,-1.0,1.0)).to(torch.bfloat16)
                stream.zero_()
            elif fixture == "zero_branch":
                branch.zero_()
            for router in (False,True):
                row = dict(tokens=tokens,fixture=fixture,router=router)
                report["cases"].append(row)
                functions, caps = {}, {}
                for name, mod in mods.items():
                    h = torch.empty_like(branch)
                    carry = torch.empty_like(stream)
                    tail = torch.empty((tokens,),device="cuda")
                    hr = torch.empty_like(stream)
                    def run(_,__,mod=mod,h=h,carry=carry,tail=tail,hr=hr):
                        mod.b1(branch,stream,gain["pre1"],gain["alpha1"],gain["beta1"],
                               h,carry,gain["alpha2"],tail,hr,router)
                        return (h,carry,tail,hr) if router else (h,carry,tail)
                    functions[name] = run
                    caps[name] = common.capture(run,None,None,flush)
                row["bitwise_equal"] = []
                for a,b in zip(caps["base"]["output"],caps["candidate"]["output"]):
                    integer = torch.int16 if a.dtype==torch.bfloat16 else torch.int32
                    equal = bool(torch.equal(a.view(integer),b.view(integer)))
                    row["bitwise_equal"].append(equal)
                    if not equal:
                        row["mismatch_max_abs"] = float((a.float()-b.float()).abs().max())
                        save(args.output,report)
                        raise RuntimeError("B1 arithmetic changed: "+str(row))
                if tokens in (1,8) and fixture=="random" and router:
                    if args.mode=="timing":
                        caps["repeat"] = common.capture(functions["base"],None,None,flush)
                        samples = []
                        names=("base","candidate","repeat")
                        for sample in range(24):
                            offset=sample%3
                            order=list(names[offset:]+names[:offset])
                            if (sample//3)%2: order.reverse()
                            us={name:common.measure(caps[name]) for name in order}
                            samples.append(us)
                        row["samples_us"] = samples
                        row["median_us"] = {name:statistics.median(s[name] for s in samples) for name in names}
                        row["candidate_ratio"] = statistics.median(s["base"]/s["candidate"] for s in samples)
                        row["control_ratio"] = statistics.median(s["base"]/s["repeat"] for s in samples)
                    else:
                        for name in ("base","candidate"):
                            # The graph includes its flush before its native event; NCU
                            # targets only the named B1 kernels, not the read-only flush.
                            torch.cuda.synchronize()
                            torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
                            caps[name]["graph"].replay()
                            torch.cuda.synchronize()
                            torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStop())
                            report["profile_order"].append(dict(tokens=tokens,arm=name))
                row["status"]="passed"
                save(args.output,report)
                caps.clear(); functions.clear()
    candidate._check_guards()
    report["status"]="complete"
    report["guards"]="passed"
    save(args.output,report)


def parent(args):
    from profile_large import parse_csv
    prefix=Path(args.output).stem
    source_prefix=args.reuse_prefix or prefix
    timing,profiled,csv=source_prefix+"-timing.json",source_prefix+"-profiled.json",source_prefix+"-ncu.csv"
    report=dict(status="running",artifact_files=[timing,profiled,csv],
                limitations="Standalone B1 timing does not include O-to-B1 PDL overlap; whole-layer screen is authoritative.")
    common=[sys.executable,__file__,"--baseline",args.baseline,"--candidate",args.candidate]
    def run(command,log):
        report["artifact_files"].append(log)
        save(args.output,report)
        with open(log,"w") as output:
            result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,timeout=700)
        if result.returncode: raise RuntimeError(log+": exit "+str(result.returncode))
    try:
        if not args.reuse_prefix:
            run(common+["--mode","timing","--output",timing],prefix+"-timing.log")
            run(["/usr/local/cuda/bin/ncu","--profile-from-start","off","--clock-control","none",
                 "--cache-control","all","--replay-mode","kernel","--csv","--print-units","base",
                 "--kernel-name","regex:.*glue_(row|cluster)_boundary.*","--section","LaunchStats",
                 "--metrics",",".join(METRICS),"--log-file",csv,
                 *common,"--mode","counters","--output",profiled],prefix+"-profile.log")
        measured=json.loads(Path(timing).read_text())
        profile=json.loads(Path(profiled).read_text())
        native=parse_csv(csv)
        if len(native)!=4 or len(profile["profile_order"])!=4:
            raise RuntimeError("Expected four B1 native records")
        report["native"]=[dict(identity,record=record) for identity,record in zip(profile["profile_order"],native)]
        for item in report["native"]:
            values=item["record"]["metrics"]
            if not set(METRICS)<=values.keys(): raise RuntimeError("Missing local-memory counters")
            if item["arm"]=="candidate" and any(values[k] for k in METRICS[1:]):
                raise RuntimeError("Single-CTA B1 spills; reject candidate")
        report["bitwise_cases_per_process"]=len(measured["cases"])
        report["sources"]=measured["sources"]
        report["candidate_prepare_seconds"]=measured["candidate_prepare_seconds"]
        report["standalone_timings"]=[r for r in measured["cases"] if "median_us" in r]
        report["status"]="complete"
    except BaseException as exc:
        report.update(status="failed",error=repr(exc))
        raise
    finally:
        report["artifact_files"]=[p for p in report["artifact_files"] if Path(p).exists()]
        save(args.output,report)
        print(json.dumps(dict(status=report["status"],output=args.output)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",default="decode_alygg77_bf16.py")
    parser.add_argument("--candidate",default="decode_b1_row.py")
    parser.add_argument("--output",default="b1-row-check.json")
    parser.add_argument("--mode",choices=("parent","timing","counters"),default="parent")
    parser.add_argument("--reuse-prefix",help="Parse existing timing/profile/CSV artifacts only; do not repeat GPU work")
    args=parser.parse_args()
    (parent if args.mode=="parent" else child)(args)
