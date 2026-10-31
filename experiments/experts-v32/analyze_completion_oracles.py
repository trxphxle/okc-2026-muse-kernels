"""Analyze all padded oracle screens, retaining every required case and A/A arm.

Bootstrap resamples allocation pads and shared input indices (same indices across
pads) rather than pretending repeated routing seeds are independent. This reports
an oracle ceiling, never a correct kernel speedup or an authorization to submit.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

LOADS=("D1","D16","D32","V1","V16","V32")
LAYERS=("sliding","full")
WEIGHTS={"sliding":46,"full":16}
ARMS=("A","B","repeat")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantile(values,p):
    values=sorted(values)
    index=(len(values)-1)*p
    low=int(index)
    high=min(low+1,len(values)-1)
    return values[low]+(values[high]-values[low])*(index-low)


def ratios(screen, indices=None):
    cases={(c["load"],c["layer"]):c for c in screen["cases"]}
    load_ratios={}
    for load in LOADS:
        baseline=0.0
        variants={name:0.0 for name in ARMS}
        for layer in LAYERS:
            case=cases[load,layer]
            records=case["records"] if indices is None else [case["records"][i] for i in indices[load,layer]]
            base=statistics.median(r["us"]["base"] for r in records)
            baseline+=WEIGHTS[layer]*base
            for arm in ARMS:
                ratio=statistics.median(r["ratio"][arm] for r in records)
                variants[arm]+=WEIGHTS[layer]*base/ratio
        load_ratios[load]={arm:baseline/variants[arm] for arm in ARMS}
    return {arm:math.prod(value[arm] for value in load_ratios.values())**(1/6) for arm in ARMS},load_ratios


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screens",type=Path,nargs="+")
    parser.add_argument("--resource",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--bootstrap",type=int,default=5000)
    args=parser.parse_args()
    if len(args.screens)<4:
        parser.error("At least four distinct allocation pads are required")
    resource=json.loads(args.resource.read_text())
    if resource["status"]!="complete" or resource.get("exact_match_failures"):
        raise RuntimeError("Native resource/work contract did not pass")
    expected_sources=resource["source_coverage"]["source_sha256"]
    screens=[json.loads(path.read_text()) for path in args.screens]
    seeds=[s["pad_seed"] for s in screens]
    if len(set(seeds))!=len(seeds):
        raise RuntimeError("Duplicate allocation pad")
    expected={(load,layer) for load in LOADS for layer in LAYERS}
    count=screens[0]["samples_per_case"]
    reference_inputs={}
    pad_rows=[]
    for path,screen in zip(args.screens,screens):
        if screen["status"]!="complete" or {(c["load"],c["layer"]) for c in screen["cases"]}!=expected:
            raise RuntimeError("Incomplete six-load/two-attention-mode screen")
        if {name:value["sha256"] for name,value in screen["sources"].items()}!=expected_sources:
            raise RuntimeError("Native-proof/screen source mismatch")
        if screen["samples_per_case"]!=count:
            raise RuntimeError("Mismatched paired sample counts")
        for case in screen["cases"]:
            key=case["load"],case["layer"]
            if len(case["records"])!=count or not all(r["baseline_oracle"]["passed"] for r in case["records"]):
                raise RuntimeError("Incomplete baseline correctness coverage")
            identities=[(r["input_seed"],r["input_fingerprint"]) for r in case["records"]]
            if key in reference_inputs and identities!=reference_inputs[key]:
                raise RuntimeError("Input fixtures changed across allocation pads")
            reference_inputs[key]=identities
        point,loads=ratios(screen)
        pad_rows.append(dict(path=str(path),sha256=sha(path),pad_seed=screen["pad_seed"],
                             pads_bytes=screen["pads_bytes"],ratio=point,loads=loads,
                             all_case_controls_passed=screen["all_case_controls_passed"],
                             aggregate_control_passed=screen["aggregate_control_passed"]))
    point={arm:math.prod(row["ratio"][arm] for row in pad_rows)**(1/len(pad_rows)) for arm in ARMS}
    boot={arm:[] for arm in ARMS}
    rng=random.Random(631710)
    for _ in range(args.bootstrap):
        selected=[rng.randrange(len(screens)) for _ in screens]
        indices={key:[rng.randrange(count) for _ in range(count)] for key in expected}
        values=[ratios(screens[i],indices)[0] for i in selected]
        for arm in ARMS:
            boot[arm].append(math.prod(v[arm] for v in values)**(1/len(values)))
    ci={arm:[quantile(boot[arm],0.025),quantile(boot[arm],0.975)] for arm in ARMS}
    controls=(all(row["aggregate_control_passed"] and row["all_case_controls_passed"] for row in pad_rows)
              and abs(point["repeat"]-1)<=0.01)
    v32=[]
    for screen in screens:
        cases={r["layer"]:r for r in screen["cases"] if r["load"]=="V32"}
        times={arm:sum(WEIGHTS[layer]*cases[layer]["native_experts"]["median_us"][arm] for layer in LAYERS)/62
               for arm in ("base","A","B")}
        v32.append(dict(pad_seed=screen["pad_seed"],weighted_native_us=times,
                        saved_us={arm:times["base"]-times[arm] for arm in ("A","B")},
                        stage_cut_percent={arm:100*(1-times[arm]/times["base"]) for arm in ("A","B")}))
    gates=dict(native_contract_passed=True,all_controls_passed=controls,
               A_point_above_three_percent=point["A"]>=1.03,
               A_lower_95pct_above_three_percent=ci["A"][0]>=1.03,
               A_every_pad_above_three_percent=all(row["ratio"]["A"]>=1.03 for row in pad_rows),
               B_point_below_three_percent=point["B"]<1.03)
    gates["ownership_rewrite_gate_passed"]=(controls and gates["A_lower_95pct_above_three_percent"]
                                             and gates["A_every_pad_above_three_percent"] and not gates["B_point_below_three_percent"])
    if not controls:
        decision="inconclusive control calibration; no rewrite authorized"
    elif gates["B_point_below_three_percent"]:
        decision="kill measured completion family: broader B below3%"
    elif gates["ownership_rewrite_gate_passed"]:
        decision="A clears3% with margin; report to PM before any ownership rewrite"
    else:
        decision="A does not clear3% with margin; no ownership rewrite"
    report=dict(status="complete",diagnostic_only=True,submission_allowed=False,
                resource_evidence=dict(path=str(args.resource),sha256=sha(args.resource)),
                pads=pad_rows,ratio=point,ci95=ci,bootstrap_repetitions=args.bootstrap,
                bootstrap_policy="Pad blocks and shared per-case input indices resampled; all cases retained",
                per_load={load:{arm:math.prod(row["loads"][load][arm] for row in pad_rows)**(1/len(pad_rows))
                                for arm in ARMS} for load in LOADS},
                v32_expert_stage=v32,
                v32_mean_saved_us={arm:statistics.mean(r["saved_us"][arm] for r in v32) for arm in ("A","B")},
                v32_requested_target_us=[28,31],gates=gates,decision=decision,
                route_mismatch_records={arm:sum(not r["route_ids_equal"][arm] for s in screens for c in s["cases"] for r in c["records"])
                                        for arm in ARMS},
                limitations=["These are intentionally incorrect deletion oracles, never eligible implementations.",
                             "B retains mandatory real MMA/ring lifetime and final-output combine; this scopes the measured family.",
                             "Native expert durations are observed inside full-layer replay; do not add overlapping stage spans.",
                             "Route mismatches are retained and reported, never filtered out of the primary timing estimate."])
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(ratio=point,ci95=ci,gates=gates,decision=decision,v32_mean_saved_us=report["v32_mean_saved_us"])),flush=True)


if __name__=="__main__":
    main()
