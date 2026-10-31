"""Build one compliant copy: prepare exact BF16 post-expert gain once.

This is a source-generation helper, never imported by the candidate. It changes
weight-only preparation plus every native/helper/Marlin gain consumer together;
all GEMM1/GEMM2 BF16 boundaries and FP32 products/statistics remain unchanged.
"""
import ast
import hashlib

GAINS = (("1.f + __bfloat162float(p.normw[n0])", 2),
         ("1.f + __bfloat162float(p.normw[n0 + 8])", 2),
         ("1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg))", 4),
         ("1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8))", 4),
         ("1.f + __uint_as_float(lnw0 << 16)", 1),
         ("1.f + __uint_as_float(lnw1 << 16)", 1),
         ("1.f + __bfloat162float(__ldg(p.normw + n0))", 1),
         ("1.f + __bfloat162float(__ldg(p.normw + n0 + 8))", 1))


def subscript(node, owner, key):
    return (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and node.value.id == owner and isinstance(node.slice, ast.Constant) and node.slice.value == key)


def literal(source, node, changed):
    original = ast.get_source_segment(source, node)
    for prefix in ('r"""', "r'''", '"""', "'''"):
        if original.startswith(prefix) and prefix[-3:] not in changed:
            return prefix + changed + prefix[-3:]
    return repr(changed)


def build(source):
    tree = ast.parse(source)
    edits, changed_sources = [], []
    source_nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if ((isinstance(target, ast.Name) and target.id == "MOE_CUDA_SRC")
                        or subscript(target, "_MA_ALT_SRCS", "moe_helper0")):
                    source_nodes.append(node.value)
    if len(source_nodes) != 2:
        raise AssertionError("Expected base and alternate native expert sources")
    for node in source_nodes:
        old = ast.literal_eval(node)
        prefix = "__bfloat162float(__float2bfloat16("
        if old.count(prefix) != 38:
            raise AssertionError("Native precision inventory changed")
        changed = old
        for value, count in GAINS:
            anchor = prefix + value + "))"
            if changed.count(anchor) != count:
                raise AssertionError("Native gain anchor changed: " + value)
            changed = changed.replace(anchor, value[len("1.f + "):])
        if changed.count(prefix) != 22:
            raise AssertionError("GEMM rounding sites were not preserved")
        edits.append((node, literal(source, node, changed)))
        changed_sources.append(dict(before_sha256=hashlib.sha256(old.encode()).hexdigest(),
                                    after_sha256=hashlib.sha256(changed.encode()).hexdigest(),
                                    removed_gain_rounds=16, retained_activation_rounds=22))
    prepares = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prepare"]
    prepared_gain = [node.value for fn in prepares for node in fn.body if isinstance(node, ast.Assign)
                     and any(subscript(target, "st", "g_postexp") for target in node.targets)]
    if len(prepared_gain) != 1 or ast.get_source_segment(source, prepared_gain[0]) != 'weights["post_expert_norm"].contiguous()':
        raise AssertionError("Post-expert gain preparation changed")
    edits.append((prepared_gain[0], '(weights["post_expert_norm"].float() + 1.0).to(torch.bfloat16).contiguous()'))
    pack = next(fn for fn in tree.body if isinstance(fn, ast.FunctionDef) and fn.name == "_ma_prepare")
    arrays = [node.value for node in pack.body if isinstance(node, ast.Assign)
              and any(subscript(target, "st", "ma_w") for target in node.targets)]
    if len(arrays) != 1 or not isinstance(arrays[0], ast.List):
        raise AssertionError("Native weight vector changed")
    gain_slot = arrays[0].elts[7]
    if ast.get_source_segment(source, gain_slot) != 'weights["post_expert_norm"].contiguous()':
        raise AssertionError("Native gain slot changed")
    edits.append((gain_slot, 'st["g_postexp"]'))
    finalizer = next(fn for fn in tree.body if isinstance(fn, ast.FunctionDef) and fn.name == "_pexp_norm_sum")
    gain_rhs = [node.value for node in finalizer.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "w" for target in node.targets)]
    old_gain = "(1.0 + tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)"
    if len(gain_rhs) != 1 or ast.get_source_segment(source, gain_rhs[0]) != old_gain:
        raise AssertionError("Marlin gain consumer changed")
    edits.append((gain_rhs[0], "tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)"))
    encoded = source.encode()
    offsets = [0]
    for line in encoded.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    spans = sorted(((offsets[n.lineno - 1] + n.col_offset, offsets[n.end_lineno - 1] + n.end_col_offset,
                     replacement.encode()) for n, replacement in edits), reverse=True)
    previous = len(encoded)
    for start, end, replacement in spans:
        if not 0 <= start < end <= previous:
            raise AssertionError("Overlapping AST edits")
        encoded = encoded[:start] + replacement + encoded[end:]
        previous = start
    result = encoded.decode()
    comments = {
        "# bf16; kernel forms 1 + w": "# prepared BF16 effective gain; shared with Marlin",
        "# BF16 as given; _pexp_norm_sum forms 1 + w": "# exact BF16 effective gain; raw checkpoint remains immutable",
        "# Keep raw checkpoint norms immutable; consumers form BF16 effective gains.":
            "# Keep raw checkpoint norms immutable; cache the BF16 post-expert effective gain.",
    }
    for old, new in comments.items():
        if result.count(old) != 1:
            raise AssertionError("Gain contract comment changed: " + old)
        result = result.replace(old, new)
    first, remainder = result.split("\n", 1)
    result = first + "\n# Precision optimization: prepare exact BF16 post-expert gain once; retain all audited activation boundaries.\n" + remainder
    ast.parse(result)
    return result, dict(kind="prepare_effective_post_expert_gain", native_sources=changed_sources,
                        native_gain_consumers=32, marlin_gain_consumers=1,
                        shared_gain_slots=["state.g_postexp", "state.ma_w[7]"],
                        input_sha256=hashlib.sha256(source.encode()).hexdigest(),
                        output_sha256=hashlib.sha256(result.encode()).hexdigest())
