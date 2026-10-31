"""Round-two diagnostic: saved real route counts -> optimistic weight-reuse ceiling.

No new timing, no source mutation, no numerical simulation. Marginal savings are
planned repeated NVFP4 bytes / measured DRAM bandwidth. This deliberately gives
full credit to repeats that may already hit L2 and charges zero clustering cost.
Historical full-layer evidence is kept separate from the current MoE source.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/openkernel-challenge/modal-lab"
LOADS = (("D1", 1), ("D16", 16), ("D32", 32), ("V1", 8), ("V16", 128), ("V32", 256))
EXPERT_BYTES = 28311552


def load(path):
    return json.loads(path.read_text())


def describe(counts):
    jobs = [(n + 63) // 64 for n in counts if n]
    distinct = len(jobs)
    return dict(counts=counts, experts=distinct, jobs64=sum(jobs),
                heavy_experts=sum(n > 1 for n in jobs),
                maximum_jobs_per_expert=max(jobs, default=0),
                repeated_jobs=sum(jobs) - distinct,
                pair_saved_jobs=sum(n // 2 for n in jobs),
                ideal_saved_bytes=(sum(jobs) - distinct) * EXPERT_BYTES,
                pair_saved_bytes=sum(n // 2 for n in jobs) * EXPERT_BYTES,
                planned_weight_bytes=sum(jobs) * EXPERT_BYTES)


def summarize(rows):
    by_load = {}
    for name, m in LOADS:
        selected = [r for r in rows if r["load"] == name]
        if not selected:
            by_load[name] = dict(fixtures=0, speedup_ideal=1.0, speedup_pair=1.0,
                                 reason="unchanged; no 64-token job repeats possible")
            continue
        def avg(key):
            return sum(r[key] * r.get("weight", 1) for r in selected) / sum(r.get("weight", 1) for r in selected)
        baseline = avg("base_us")
        item = dict(m=m, fixtures=len(selected),
                    mean_experts=avg("experts"), mean_jobs64=avg("jobs64"),
                    mean_heavy_experts=avg("heavy_experts"),
                    mean_repeated_jobs=avg("repeated_jobs"),
                    max_repeated_jobs=max(r["repeated_jobs"] for r in selected),
                    mean_planned_weight_bytes=avg("planned_weight_bytes"),
                    mean_ideal_saved_bytes=avg("ideal_saved_bytes"),
                    mean_pair_saved_bytes=avg("pair_saved_bytes"),
                    mean_base_us=baseline)
        for mode in ("ideal", "pair"):
            saved = avg(mode + "_saved_us")
            if saved >= baseline:
                raise ValueError("Deletion bound removes the entire kernel")
            item["mean_" + mode + "_saved_us"] = saved
            item["speedup_" + mode] = baseline / (baseline - saved)
            item["median_fixture_speedup_" + mode] = statistics.median(
                r["base_us"] / (r["base_us"] - r[mode + "_saved_us"]) for r in selected)
        by_load[name] = item
    return dict(per_load=by_load,
                board_speedup_ideal=math.prod(r["speedup_ideal"] for r in by_load.values()) ** (1 / 6),
                board_speedup_pair=math.prod(r["speedup_pair"] for r in by_load.values()) ** (1 / 6),
                total_fixtures=len(rows), fixtures=rows)


def add_savings(row, bandwidth):
    row["bandwidth_TB_s"] = bandwidth
    for mode in ("ideal", "pair"):
        row[mode + "_saved_us"] = row[mode + "_saved_bytes"] / (bandwidth * 1e6)
    return row


def public_rows(cases, bw):
    rows = []
    for case in cases:
        for record in case["records"]:
            row = dict(load=case["load"], m=case["m"], input_seed=record["input_seed"],
                       base_us=(record["us"]["base"] + record["us"]["base_repeat"]) / 2,
                       **describe(record["routing"]["local_expert_histogram"]))
            rows.append(add_savings(row, bw.get(case["load"], 1.0)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("reuse-bound.json"))
    args = parser.parse_args()
    sources = dict(native=OUT / "layer-okc-h200-experts-v32-04/profile-large.json",
                   aa=OUT / "layer-okc-h200-experts-v32-01/aa-upstream.json",
                   repeat=OUT / "layer-okc-h200-experts-v32-04/ceiling-decode-repeat.json",
                   historical=OUT / "layer-okc-h200-18/expert-bandwidth.json")
    data = {k: load(p) for k, p in sources.items()}
    assert all(d["status"] in ("complete", "passed") for d in data.values())
    native = data["native"]["fixtures"]
    bw = {name: sum(r["dram_read_bytes"] for r in native if r["load"] == name) /
                 sum(r["profile_us"] for r in native if r["load"] == name) / 1e6
          for name in ("D32", "V16", "V32")}
    matched = [add_savings(dict(load=r["load"], m=r["m"], input_seed=r["input_seed"],
                                base_us=r["profile_us"], **describe(r["counts"])), r["dram_read_TB_s"])
               for r in native]
    aa = public_rows(data["aa"]["weight_screens"][0]["scored"], bw)
    repeat = public_rows(data["repeat"]["cases"], bw)
    historical = []
    for r in data["historical"]["cases"]:
        name = ("D" if r["q_len"] == 1 else "V") + str(r["batch"])
        historical.append(add_savings(dict(load=name, m=r["batch"] * r["q_len"],
                                           weight_seed=r["weight_seed"], layer=r["layer"],
                                           weight=46 if r["layer"] == "sliding" else 16,
                                           base_us=r["profile_us"], **describe(r["counts"])),
                                      r["dram_read_bytes"] / r["profile_us"] / 1e6))
    report = dict(status="complete", official_score=False, expert_bytes=EXPERT_BYTES,
                  sources={k: dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                           for k, p in sources.items()}, measured_bandwidth_TB_s=bw,
                  formula="saved_us = planned repeated bytes / measured DRAM bytes-per-us; mean load time ratio then equal-six-load geomean",
                  limitations=["Zero overhead and full DRAM credit for planned repeats are optimistic assumptions, not achievable performance.",
                               "Whole-kernel bandwidth includes routing/sync costs; this is a model, not a mathematical hardware bound.",
                               "Historical full-layer routing and timing use a different native source and are never pooled with current source.",
                               "Public routing ranges overlap between captures; cohorts are reported separately, not as independent pooled samples.",
                               "D16's eight-token split is outside this idea and remains unchanged."],
                  cohorts=dict(matched_native=summarize(matched), current_aa=summarize(aa),
                               current_second_weight=summarize(repeat), historical_layer=summarize(historical)),
                  microbench_fixtures=[dict(load=r["load"], input_seed=r["input_seed"], counts=r["counts"],
                                            base_us=r["profile_us"]) for r in native])
    report["any_current_ceiling_at_least_three_percent"] = any(
        report["cohorts"][k]["board_speedup_ideal"] >= 1.03
        for k in ("matched_native", "current_aa", "current_second_weight"))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for name, cohort in report["cohorts"].items():
        print(json.dumps(dict(cohort=name, ideal=cohort["board_speedup_ideal"], pair=cohort["board_speedup_pair"],
                              loads={k: {p: v.get(p, 0) for p in ("mean_repeated_jobs", "mean_ideal_saved_bytes",
                                                                "mean_ideal_saved_us", "speedup_ideal")}
                                     for k, v in cohort["per_load"].items()})))


if __name__ == "__main__":
    main()
