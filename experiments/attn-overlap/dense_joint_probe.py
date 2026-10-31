#!/usr/bin/env python3
"""Timing-only, NON-SUBMITTABLE whole-layer SK2 joint-deletion experiment.

Upload this file, dense_joint_modes.py, dense_joint_control.py,
compare_decode_public.py, and DECODE-LAYER reference.py/checks.py/nvfp4.py.
  python dense_joint_probe.py --output dense-joint.json --inputs 2 --pairs 31
  python dense_joint_probe.py --output dense-build.json --build-only

--inputs counts fresh, independently frozen fixtures per case; --pairs counts
rotating, forward/reverse paired timing repetitions on EACH fixed fixture.
No cached tensor is a legal implementation. All real launches remain at their
original Python callsites, with original inputs, weights, PDL and geometry.
Only selected SK2 return values are substituted; attention, routing, experts,
and glue always execute. Native profiling is deliberately omitted rather than
reuse the single-kernel native_batch protocol on a multi-kernel layer.
"""
from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import importlib.util
import json
import math
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path

ARMS = ("production", "production_repeat", "frozen_production", "diag0",
        "diag0_repeat", "epilogue", "consumer", "both")
MODES = {"diag0": 0, "diag0_repeat": 0, "epilogue": 1, "consumer": 2, "both": 3}
HOOKS = ("bfp__run_sk2", "bfp_fused_pooler_router", "_gle_frt")
SELECTED = {"sk2", "sk2f", "gle_pr"}
LOADS = (("D1", 1, 1), ("D16", 16, 1), ("D32", 32, 1),
         ("V1", 1, 8), ("V16", 16, 8), ("V32", 32, 8))
LAYER_WEIGHT = {"sliding": 46, "full": 16}
BASE_SHA = "053dbacc8f0de4e6111ccb89cb91f629840852cf482a6ca423fe0eaff419c086"


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save(path, report):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def tensor_meta(x):
    return dict(shape=list(x.shape), stride=list(x.stride()), dtype=str(x.dtype),
                device=str(x.device), bytes=x.numel() * x.element_size(),
                data_ptr=x.data_ptr(), storage_ptr=x.untyped_storage().data_ptr())


def describe(x):
    if isinstance(x, torch.Tensor):
        return tensor_meta(x)
    if isinstance(x, (tuple, list)):
        return [describe(v) for v in x]
    if isinstance(x, dict):
        return {str(k): describe(v) for k, v in x.items()}
    if isinstance(x, (str, int, float, bool, type(None))):
        return x
    return type(x).__name__


def clone_tree(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    if isinstance(x, tuple):
        return tuple(clone_tree(v) for v in x)
    if isinstance(x, list):
        return [clone_tree(v) for v in x]
    raise TypeError("Unexpected dense output " + type(x).__name__)


def tensor_delta(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return dict(equal=False, finite=False, max_abs=None, relative_l2=None)
    equal = bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))
    if not a.is_floating_point():
        return dict(equal=equal, finite=True, max_abs=0.0 if equal else None, relative_l2=0.0 if equal else None)
    aa, bb = a.float(), b.float()
    finite = bool(torch.isfinite(aa).all() and torch.isfinite(bb).all())
    if not finite:
        return dict(equal=equal, finite=False, max_abs=None, relative_l2=None)
    diff = aa - bb
    return dict(equal=equal, finite=True, max_abs=float(diff.abs().max()) if diff.numel() else 0.0,
                relative_l2=float(diff.norm() / bb.norm().clamp_min(1e-12)))


def within_observed(delta, noise):
    # Descriptive only: one A/A sample is not an upper bound on future floating
    # reductions. Timing validity uses the independent oracle and exact workload.
    return delta["finite"] and (delta["equal"] or (
        delta["max_abs"] is not None and noise["max_abs"] is not None and
        delta["max_abs"] <= noise["max_abs"] and
        delta["relative_l2"] <= noise["relative_l2"]))


class ExtensionProxy:
    def __init__(self, module, family, audit):
        self.module, self.family, self.audit = module, family, audit

    def __getattr__(self, name):
        fn = getattr(self.module, name)
        if not callable(fn) or "set_" in name:
            return fn

        @functools.wraps(fn)
        def launch(*args, **kwargs):
            self.audit.native(self.family, name, args, kwargs)
            return fn(*args, **kwargs)
        return launch


class BoundaryAudit:
    """Post-order substitution, with extension-level proof of selected launches.

    Recording clones happen only during an untimed original-production pass.
    Captures merely retain references and return those frozen clones AFTER the
    original function launched into its ordinary allocated output. Nested
    fallback wrappers with no direct SK2 launch are never independently frozen.
    """
    def __init__(self, control, modes, originals):
        self.control, self.modes, self.originals = control, modes, originals
        self.functions = {n: getattr(control, n) for n in HOOKS}
        self.moe = control._moe
        self.marlin = control._marlin_experts
        self.frozen = []
        self.expected = None
        self.action = "observe"
        self.arm = "production"
        self.mode = None
        self.reset()

    def reset(self):
        self.stack, self.calls, self.boundaries = [], [], []
        self.activations, self.experts, self.discards = [], [], []
        self.cursor = 0

    def install(self, modules, arm, action):
        self.arm, self.action = arm, action
        self.mode = MODES.get(arm)
        if self.mode is not None:
            # Set ALL three launch-time masks together before graph capture.
            self.modes.set_mode(modules, self.mode)
        proxies = {name: ExtensionProxy(ext, name, self) for name, ext in modules.items()}
        self.modes.install(self.control, proxies)
        for name, fn in self.functions.items():
            setattr(self.control, name, self.wrapper(name, fn))

        def moe(*args, **kwargs):
            return ExtensionProxy(self.moe(*args, **kwargs), "expert", self)
        self.control._moe = moe

        def marlin(x, weights, ids, state):
            self.expert("marlin", x, ids, weights, None)
            return self.marlin(x, weights, ids, state)
        self.control._marlin_experts = marlin

    def restore(self):
        self.modes.install(self.control, self.originals)
        for name, fn in self.functions.items():
            setattr(self.control, name, fn)
        self.control._moe, self.control._marlin_experts = self.moe, self.marlin

    def expert(self, backend, x, ids, weights, geometry):
        self.experts.append(dict(backend=backend, x=x, ids=ids, weights=weights,
                                 geometry=geometry))

    def native(self, family, method, args, kwargs):
        if family == "expert":
            if method == "okc_run":
                self.expert("agent", args[2], args[3], args[4], list(args[6:]))
            return
        selected = method in SELECTED
        owner = self.stack[-1] if self.stack else None
        if selected and owner is None:
            raise RuntimeError("Selected dense launch escaped the three return-boundary hooks")
        if selected:
            owner["direct"] += 1
        roles = {}
        if method == "sk2":
            roles = {0: "weight_stream", 1: "activation", 2: "discard_output", 3: "escape_table"}
        elif method in ("sk2f", "gle_pr"):
            roles = {0: "weight_stream", 1: "router_weight_stream", 2: "activation",
                     3: "discard_output", 4: "discard_router_output", 5: "escape_table"}
            if method == "gle_pr":
                roles.update({11: "router_activation_fp32", 12: "router_weight_fp32"})
        arguments = []
        for i, arg in enumerate(args):
            role = roles.get(i, "argument")
            arguments.append(dict(index=i, role=role, value=describe(arg)))
            if selected and "activation" in role:
                self.activations.append((f"{len(self.calls)}:{role}", arg))
        self.calls.append(dict(index=len(self.calls), family=family, method=method,
            selected=selected, hook=owner["hook"] if owner else None,
            layer=owner["layer"] if owner else "unselected_dense",
            mode=self.mode, arguments=arguments, kwargs=describe(kwargs)))

    def wrapper(self, name, fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            layer = (args[1] if len(args) > 1 else kwargs["layer"]) if name == HOOKS[0] else "pooler+router"
            frame = dict(hook=name, layer=layer, direct=0)
            self.stack.append(frame)
            try:
                output = fn(*args, **kwargs)
            finally:
                self.stack.pop()
            if not frame["direct"]:
                return output
            signature = dict(hook=name, layer=layer, direct_launches=frame["direct"],
                             output_layout=self.layout(output))
            index = self.cursor
            self.cursor += 1
            self.boundaries.append(dict(index=index, **signature))
            if self.action == "record":
                self.frozen.append((signature, clone_tree(output)))
            elif self.action == "freeze":
                if index >= len(self.frozen) or self.frozen[index][0] != signature:
                    raise RuntimeError("Dense post-order boundary changed on fixed fixture")
                self.discards.append(output)
                return self.frozen[index][1]
            return output
        return wrapped

    @staticmethod
    def layout(x):
        if isinstance(x, torch.Tensor):
            return dict(shape=list(x.shape), dtype=str(x.dtype), stride=list(x.stride()))
        return [BoundaryAudit.layout(v) for v in x]

    @staticmethod
    def signature(calls):
        result = []
        for call in calls:
            args = []
            for arg in call["arguments"]:
                value = arg["value"]
                if isinstance(value, dict) and "data_ptr" in value:
                    value = dict(value)
                    if "weight" not in arg["role"] and arg["role"] != "escape_table":
                        value.pop("data_ptr")
                        value.pop("storage_ptr")
                args.append(dict(index=arg["index"], role=arg["role"], value=value))
            result.append(dict(family=call["family"], method=call["method"], layer=call["layer"],
                               arguments=args, kwargs=call["kwargs"]))
        return result

    def run(self, data, state):
        self.reset()
        output = self.control.kernel_fn(data, state)
        if self.action == "freeze" and self.cursor != len(self.frozen):
            raise RuntimeError("Not every frozen dense boundary was visited")
        signature = self.signature(self.calls)
        if self.expected is not None and signature != self.expected:
            raise RuntimeError("Native dense launch order, geometry, or actual weight pointers changed")
        return output

    def observations(self):
        return dict(calls=list(self.calls), boundaries=list(self.boundaries),
                    activations=list(self.activations), experts=list(self.experts),
                    discard_outputs=list(self.discards))


def snapshot_work(observations):
    activations = {name: x.detach().clone() for name, x in observations["activations"]}
    experts = []
    public = []
    for item in observations["experts"]:
        ids = item["ids"].long()
        valid = (ids >= 0) & (ids < 32)
        counts = torch.bincount(ids[valid], minlength=32).cpu().tolist()
        summary = dict(backend=item["backend"], geometry=item["geometry"],
            activation=tensor_meta(item["x"]), ids=tensor_meta(item["ids"]),
            weights=tensor_meta(item["weights"]), total_route_slots=ids.numel(),
            local_routes=int(valid.sum()), nonlocal_routes=int((ids < 0).sum()),
            invalid_routes=int((ids >= 32).sum()), local_expert_counts=counts,
            active_local_experts=sum(n > 0 for n in counts))
        public.append(summary)
        experts.append({k: item[k].detach().clone() for k in ("x", "ids", "weights")})
    return dict(activations=activations, experts=experts, counts=public)


def work_deltas(work, baseline):
    structure = (list(work["activations"]) == list(baseline["activations"]) and
                 len(work["experts"]) == len(baseline["experts"]))
    deltas = {}
    if structure:
        for key, value in work["activations"].items():
            deltas["dense_input/" + key] = tensor_delta(value, baseline["activations"][key])
        for index, (actual, expected) in enumerate(zip(work["experts"], baseline["experts"])):
            for key in actual:
                deltas[f"expert/{index}/{key}"] = tensor_delta(actual[key], expected[key])
    counts = lambda w: [{k: v for k, v in row.items() if k not in ("activation", "ids", "weights")}
                        for row in w["counts"]]
    ids_equal = structure and all(torch.equal(a["ids"], b["ids"])
                                 for a, b in zip(work["experts"], baseline["experts"]))
    zeros_equal = structure and all(torch.equal(a["weights"] == 0, b["weights"] == 0)
                                   for a, b in zip(work["experts"], baseline["experts"]))
    return dict(structure_equal=structure, counts_equal=counts(work) == counts(baseline),
                expert_ids_equal=ids_equal, expert_weight_zero_masks_equal=zeros_equal,
                finite=all(d["finite"] for d in deltas.values()), tensors=deltas)


def aggregate(report, args):
    report["loads"] = {}
    for load, _, _ in LOADS:
        if load not in args.loads:
            continue
        rows = [r for r in report["results"] if r["load"] == load]
        complete = len(rows) == 2 and all(r["status"] == "complete" for r in rows)
        entry = dict(complete=complete)
        report["loads"][load] = entry
        if not complete:
            continue
        times = {arm: sum(LAYER_WEIGHT[r["layer"]] * r["median_us"][arm] for r in rows) / 62
                 for arm in ARMS}
        entry.update(weighted_mean_us=times,
            weighted_62_layer_us={arm: value * 62 for arm, value in times.items()},
            valid=all(r["valid"] for r in rows),
            selected_launches_by_layer={r["layer"]: r["selected_launches"] for r in rows},
            selected_SK2_coverage_present=all(n > 0 for r in rows for n in r["selected_launches"]),
            coverage_scope="Only observed SK2-family launches; non-SK2 dense paths remain unchanged and uncovered",
            production_aa_ratio=times["production"] / times["production_repeat"],
            diag0_aa_ratio=times["diag0"] / times["diag0_repeat"],
            codegen_overhead_us=times["diag0"] - times["frozen_production"],
            frozen_control_overhead_us=times["frozen_production"] - times["production"],
            deletion_observed_us={a: times["diag0"] - times[a] for a in ("epilogue", "consumer", "both")},
            deletion_observed_62_layer_us={a: 62 * (times["diag0"] - times[a]) for a in ("epilogue", "consumer", "both")})
        entry["controls_within_one_percent"] = all(abs(entry[k] - 1) <= .01
            for k in ("production_aa_ratio", "diag0_aa_ratio"))
        entry["valid"] &= entry["controls_within_one_percent"]
        entry["joint_budget_gate"] = dict(target_us={"D16": 12.6, "D32": 16.1}.get(load),
            unit="46:16 weighted mean whole-layer microseconds", accepted=False)
        if load in ("D16", "D32"):
            entry["joint_budget_gate"]["accepted"] = (entry["valid"] and entry["selected_SK2_coverage_present"] and
                entry["deletion_observed_us"]["both"] >= entry["joint_budget_gate"]["target_us"])
    complete = len(report["loads"]) == 6 and all(x["complete"] for x in report["loads"].values())
    target_coverage = complete and all(report["loads"][n]["selected_SK2_coverage_present"] for n in ("D16", "D32"))
    valid = complete and target_coverage and all(x["valid"] for x in report["loads"].values())
    ratios = {}
    if complete:
        for arm in ("epilogue", "consumer", "both"):
            ratios[arm] = math.exp(statistics.mean(math.log(x["weighted_mean_us"]["diag0"] /
                x["weighted_mean_us"][arm]) for x in report["loads"].values()))
    report["summary"] = dict(all_twelve_cases_complete=complete, diagnostic_valid=valid,
        target_SK2_coverage_present=target_coverage,
        observed_six_load_geomean_ratios=ratios, accepted_ceiling_ratios=ratios if valid else None,
        three_percent_joint_gate=valid and ratios.get("both", 0) >= 1.03,
        targeted_budget_gate=valid and all(report["loads"][n]["joint_budget_gate"]["accepted"]
                                         for n in ("D16", "D32")),
        legal_speedup_claim=False, submission_allowed=False,
        limitation="Frozen return buffers change dependencies and cache placement; observed joint deletion is a timing-only diagnostic, not a legal optimization. Individual deletion savings are NEVER added.")


def run_fixture(control, modes, originals, modules, helper, reference, state, weights,
                flush, row, fixture, args, report, persist):
    seed = args.input_seed + row["batch"] * 100 + row["q_len"] * 1000 + (row["layer"] == "full") * 10000 + fixture
    shape = dict(batch=row["batch"], q_len=row["q_len"], ctx=80000, layer=row["layer"],
                 page=1, spread="agentic" if row["batch"] > 1 else "fixed", dist="soft")
    fresh = reference.generate_input(seed=seed, **shape)
    data = helper.clone_data(fresh)
    audit = BoundaryAudit(control, modes, originals)
    captures, observations, checks, snapshots = {}, {}, {}, {}
    record = dict(fixture=fixture, seed=seed, shape=shape, status="running", checks=checks,
                  coverage={}, timing_samples=[], order=[], route_work={}, errors={})
    row["records"].append(record)
    persist()
    try:
        # Match the scored workload convention: choose the busiest shard from an
        # unchanged production probe, once, before freezing this fixed fixture.
        modes.install(control, originals)
        probe = control.kernel_fn(data, state)
        shard_counts = torch.bincount((probe[2].long() // 32).flatten(), minlength=8)
        local_start = 32 * int(shard_counts.argmax())
        fresh["local_start"].fill_(local_start)
        record["local_start"] = local_start
        record["production_probe_shard_counts"] = shard_counts.cpu().tolist()
        del probe
        helper.refill(data, fresh)
        audit.install(originals, "production", "record")
        audit.run(data, state)
        torch.cuda.synchronize()
        audit.expected = audit.signature(audit.calls)
        record["frozen_boundaries"] = [dict(index=i, signature=sig, tensors=describe(value))
                                       for i, (sig, value) in enumerate(audit.frozen)]
        record["frozen_buffer_bytes"] = sum(x.numel() * x.element_size()
            for _, output in audit.frozen for x in (output if isinstance(output, (tuple, list)) else (output,)))
        helper.refill(data, fresh)
        torch.cuda.synchronize()
        # Frozen returns cannot establish that the feature-off kernels compute
        # correctly. Validate an unfrozen mode0 graph independently, including
        # replay after changing the host setter to3 (captured arguments must win).
        audit.install(modules, "diag0", "observe")
        zero_graph = helper.capture(audit.run, data, state)
        modes.set_mode(modules, 3)
        zero_graph["graph"].replay()
        torch.cuda.synchronize()
        record["unfrozen_mode0_after_host_mode3"] = helper.oracle(
            data, zero_graph["output"], weights, fresh)
        if not record["unfrozen_mode0_after_host_mode3"]["passed"]:
            raise RuntimeError("Unfrozen diagnostic mode0/captured-mode oracle failure")
        modes.set_mode(modules, 0)
        del zero_graph
        persist()
        # All graphs share fixed input/weight storage and one prepared state.
        # Cache writes are idempotent at the fixture's fixed new-token slots;
        # complete cache comparisons below reject any other mutation.
        for arm in ARMS:
            audit.install(modules if arm in MODES else originals, arm,
                          "observe" if arm in ("production", "production_repeat") else "freeze")
            helper.refill(data, fresh)
            captures[arm] = helper.capture(audit.run, data, state, flush)
            observations[arm] = audit.observations()
            record["coverage"][arm] = dict(launches=observations[arm]["calls"],
                post_order_boundaries=observations[arm]["boundaries"],
                selected_launches=sum(c["selected"] for c in observations[arm]["calls"]),
                enabled_modes={name: MODES[arm] for name in modules} if arm in MODES else None,
                all_selected_launches_masked=arm in MODES,
                coverage_unit="Observed extension dispatch; one SK2/sk2f/gle_pr call is one selected native dense launch",
                native_profile="not_run: single-kernel native_batch is not valid for this multi-kernel graph")
            output = tuple(x.detach().clone() for x in captures[arm]["output"])
            snapshots[arm] = dict(output=output, work=snapshot_work(observations[arm]))
            record["route_work"][arm] = dict(routes=helper.routes(output, local_start),
                                              expert_launches=snapshots[arm]["work"]["counts"])
            checks[arm] = dict(oracle=helper.oracle(data, output, weights, fresh))
            if arm == "production":
                cache_baseline = {k: data[k].clone() for k in ("k_cache", "v_cache")}
            checks[arm]["cache_equal"] = all(torch.equal(data[k].view(torch.uint8), cache_baseline[k].view(torch.uint8))
                                               for k in cache_baseline)
            persist()
            if arm in ("production", "production_repeat") and not checks[arm]["oracle"]["passed"]:
                raise RuntimeError(arm + " independent oracle failure: " + checks[arm]["oracle"]["message"])
        del cache_baseline
        baseline = snapshots["production"]
        noise = [tensor_delta(a, b) for a, b in zip(snapshots["production_repeat"]["output"], baseline["output"])]
        work_noise = work_deltas(snapshots["production_repeat"]["work"], baseline["work"])
        record["observed_production_nondeterminism"] = dict(output=noise, work=work_noise)
        for arm in ARMS:
            output = snapshots[arm]["output"]
            deltas = [tensor_delta(a, b) for a, b in zip(output, baseline["output"])]
            work = work_deltas(snapshots[arm]["work"], baseline["work"])
            work_ok = (work["structure_equal"] and work["counts_equal"] and
                       work["expert_ids_equal"] and work["expert_weight_zero_masks_equal"] and work["finite"])
            check = checks[arm]
            check.update(output_deltas=deltas, work_deltas=work,
                comparison=helper.comparison(output, baseline["output"]),
                output_within_observed_aa=all(within_observed(d, n) for d, n in zip(deltas, noise)),
                routing_equal=bool(torch.equal(output[2], baseline["output"][2])),
                local_expert_work_equal=work_ok)
            check["valid"] = (check["oracle"]["passed"] and check["cache_equal"] and
                check["comparison"]["close"] and
                check["routing_equal"] and work_ok)
        # Each round brackets the rotated arms by their reverse order. No mode
        # setter, input refresh, freezing, oracle, or work inspection is timed.
        samples = {arm: [] for arm in ARMS}
        for pair in range(args.pairs):
            shift = (pair + fixture) % len(ARMS)
            order = ARMS[shift:] + ARMS[:shift]
            if pair % 2:
                order = tuple(reversed(order))
            round_samples = {arm: [] for arm in ARMS}
            for arm in order + tuple(reversed(order)):
                us = helper.measure(captures[arm])
                round_samples[arm].append(us)
                record["order"].append(arm)
            paired = {arm: statistics.mean(values) for arm, values in round_samples.items()}
            for arm, value in paired.items():
                samples[arm].append(value)
            record["timing_samples"].append(dict(pair=pair, paired_us=paired, raw_us=round_samples,
                joint_delta_us=paired["diag0"] - paired["both"]))
            persist()
        # Validate the actual final timed replay, not only eager/capture warmups.
        # Replay one arm at a time: state/scratch is shared, so snapshot immediately.
        cache_expected = None
        for arm in ARMS:
            helper.measure(captures[arm])
            output = captures[arm]["output"]
            final_work = snapshot_work(observations[arm])
            final_delta = work_deltas(final_work, baseline["work"])
            final_output = [tensor_delta(a, b) for a, b in zip(output, baseline["output"])]
            if arm == "production":
                cache_expected = {k: data[k].clone() for k in ("k_cache", "v_cache")}
            final = dict(oracle=helper.oracle(data, output, weights, fresh),
                output_deltas=final_output, work_deltas=final_delta,
                comparison=helper.comparison(output, baseline["output"]),
                cache_equal=all(torch.equal(data[k].view(torch.uint8), cache_expected[k].view(torch.uint8))
                                for k in cache_expected),
                route_work=helper.routes(output, local_start), expert_counts=final_work["counts"])
            final["valid"] = (final["oracle"]["passed"] and final["cache_equal"] and
                final["comparison"]["close"] and
                final_delta["structure_equal"] and final_delta["counts_equal"] and
                final_delta["expert_ids_equal"] and final_delta["expert_weight_zero_masks_equal"] and
                final_delta["finite"])
            checks[arm]["final_replay"] = final
            checks[arm]["valid"] &= final["valid"]
            persist()
            if arm in ("production", "production_repeat") and not final["oracle"]["passed"]:
                raise RuntimeError(arm + " final replay oracle failure")
        record.update(status="complete", valid=all(checks[a]["valid"] for a in ARMS),
                      median_us={arm: statistics.median(values) for arm, values in samples.items()},
                      paired_joint_delta_us=statistics.median(r["joint_delta_us"] for r in record["timing_samples"]))
        selected = record["coverage"]["production"]["selected_launches"]
        record["coverage_status"] = "selected_SK2_only" if selected else "uncovered_non_SK2_negative_control"
        record["no_selected_launches_is_not_a_dense_ceiling"] = selected == 0
        persist()
    except BaseException as exc:
        record.update(status="failed", valid=False, error=repr(exc), traceback=traceback.format_exc())
        persist()
        raise
    finally:
        audit.restore()
        captures.clear()
        observations.clear()
        snapshots.clear()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path(__file__).with_name("dense_joint_control.py"))
    parser.add_argument("--seed", type=int, default=731283, help="weight seed; original prepare runs exactly once")
    parser.add_argument("--input-seed", type=int, default=919233)
    parser.add_argument("--inputs", type=int, default=2, help="fresh fixed fixtures per load/layer")
    parser.add_argument("--pairs", type=int, default=31, help="forward/reverse timing rounds per fixture")
    parser.add_argument("--loads", nargs="+", default=[n for n, _, _ in LOADS], help="load names, space or comma separated")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    args.loads = [n.upper() for group in args.loads for n in group.split(",")]
    if args.inputs < 1 or args.pairs < 1 or len(set(args.loads)) != len(args.loads) or not set(args.loads) <= {n for n, _, _ in LOADS}:
        parser.error("Positive --inputs/--pairs and unique scored --loads are required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(kind="dense_joint_deletion_frozen_diagnostic_NON_SUBMITTABLE", status="running",
        submission_allowed=False, official_score=False, artifact_files=[], seed=args.seed,
        inputs_per_case=args.inputs, timing_pairs=args.pairs, selected_loads=args.loads,
        arms=list(ARMS), modes=MODES, results=[], build_only=args.build_only,
        protocol=dict(freeze="Only SK2-family post-order Python return boundaries, outside all timed regions; real kernels execute into ordinary retained discard outputs",
            prepare="Exactly one unmodified original prepare, before diagnostic builds; same state and weight pointers for all arms",
            events="compare_decode_public.capture/measure: internal graph external CUDA timing events, after 256MiB cold-L2 flush",
            timing="Rotating forward/reverse paired arms; fixed fixture recaptured after every fresh input",
            workload_gate="Exact native call order/geometry/weight pointers, routing IDs, per-expert counts and route-weight zero masks; finite actual inputs. Floating deltas/A/A envelope are retained descriptively, not equated with execution work.",
            numerical_gate="Independent public oracle plus existing whole-layer baseline comparison (relative L2 <=0.003, row L2 <=0.015), full cache equality, and unfrozen captured-mode0 correctness.",
            aggregation="Median paired round time per fixture; median fixtures per case; 46 sliding:16 full weighted mean per load; equal six-load geomean",
            native_profile="Skipped: no claim of observed native starts/ends or complete kernel-level timeline",
            ceiling="Joint arm directly measured; never sum epilogue and consumer savings"))
    started = time.monotonic()
    persist = lambda: save(args.output, report)
    persist()
    try:
        global torch
        import torch
        import dense_joint_modes as modes
        report["environment"] = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()))
        control = load_file(args.baseline, "_dense_joint_original")
        report["sources"] = {k: dict(path=str(p), sha256=sha(p)) for k, p in
            dict(baseline=args.baseline, driver=Path(__file__), modes=Path(modes.__file__)).items()}
        report["promoted_base_sha_match"] = report["sources"]["baseline"]["sha256"] == BASE_SHA
        def on_artifact(name):
            if Path(name).name != name:
                raise ValueError("build_all callback must supply artifact basename")
            if name not in report["artifact_files"]:
                report["artifact_files"].append(name)
            persist()
        if args.build_only:
            before = time.monotonic()
            modules, metadata = modes.build_all(control, args.output.with_suffix(""), on_artifact)
            report.update(status="build_complete", builds=metadata, build_seconds=time.monotonic() - before,
                          original_prepare_calls=0, weights_generated=False,
                          summary=dict(diagnostic_valid=False, reason="Build-only; no numerical or timing observations"))
            persist()
            return 0
        import reference
        required = ("generate_weights", "generate_input", "collect", "check_implementation", "new_slots", "HIDDEN")
        if not all(hasattr(reference, name) for name in required):
            raise RuntimeError("Upload DECODE-LAYER reference.py/checks.py/nvfp4.py, not attention reference")
        helper_path = Path(__file__).with_name("compare_decode_public.py")
        helper = load_file(helper_path, "_dense_joint_compare")
        helper.torch, helper.reference = torch, reference
        for name, path in dict(helper=helper_path, reference=Path(reference.__file__),
            checks=Path(reference.__file__).with_name("checks.py"), nvfp4=Path(reference.__file__).with_name("nvfp4.py")).items():
            report["sources"][name] = dict(path=str(path), sha256=sha(path))
        before = time.monotonic()
        weights = reference.generate_weights(args.seed)
        torch.cuda.synchronize()
        report["weight_generation_seconds"] = time.monotonic() - before
        prepare_weights = helper.clone_data(weights)
        before = time.monotonic()
        state = control.prepare(prepare_weights)
        torch.cuda.synchronize()
        report["prepare"] = dict(seconds=time.monotonic() - before, calls=1,
                                  unmodified_original=True, state=helper.metadata(state))
        originals = modes.originals(control)
        if set(originals) != {"main", "narrow", "gle"} or any(ext is None for ext in originals.values()):
            raise RuntimeError("Original prepare did not retain all three compiled families")
        report["original_modules"] = {name: getattr(ext, "__name__", type(ext).__name__) for name, ext in originals.items()}
        persist()
        before = time.monotonic()
        modules, metadata = modes.build_all(control, args.output.with_suffix(""), on_artifact)
        report.update(builds=metadata, build_seconds=time.monotonic() - before)
        modes.install(control, originals)
        flush = torch.ones(256 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
        report["results"] = [dict(load=n, batch=b, q_len=q, layer=layer, status="not_run", records=[])
                             for n, b, q in LOADS if n in args.loads for layer in LAYER_WEIGHT]
        persist()
        for row in report["results"]:
            row["status"] = "running"
            for fixture in range(args.inputs):
                run_fixture(control, modes, originals, modules, helper, reference, state, weights,
                            flush, row, fixture, args, report, persist)
            row.update(status="complete", valid=all(r["valid"] for r in row["records"]),
                       median_us={arm: statistics.median(r["median_us"][arm] for r in row["records"]) for arm in ARMS},
                       selected_launches=[r["coverage"]["production"]["selected_launches"] for r in row["records"]])
            row["aa_within_one_percent"] = all(abs(row["median_us"][a] / row["median_us"][b] - 1) <= .01
                for a, b in (("production", "production_repeat"), ("diag0", "diag0_repeat")))
            row["valid"] &= row["aa_within_one_percent"]
            aggregate(report, args)
            persist()
        aggregate(report, args)
        report["status"] = "complete" if report["summary"]["diagnostic_valid"] else "complete_invalid"
        report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        report["summary"] = dict(diagnostic_valid=False, accepted_ceiling_ratios=None,
                                 submission_allowed=False, reason="Incomplete or failed run")
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        persist()
    return 0 if report["status"] in ("complete", "build_complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
