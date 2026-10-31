"""One compliant structural candidate: localize B1's unchanged32-warp arithmetic.

The eight-CTA implementation uses1024 logical threads and8 values/thread.
The row implementation uses those same1024 logical threads in one CTA, preserving
all lane assignments and reduction trees. Only synchronization/storage ownership
changes; the original scalar FP32 helpers and every BF16 boundary are retained.
"""
import ast
import hashlib


def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise AssertionError("B1 row source drift: " + old[:100])
    return text.replace(old, new)


def replace_function(text, start, end, replacement):
    if text.count(start) != 1 or text.count(end) != 1:
        raise AssertionError("Reduction function boundary drift")
    first = text.index(start)
    last = text.index(end, first)
    return text[:first] + replacement + text[last:]


def build(source):
    tree = ast.parse(source)
    literals = [node.value for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_SL_GLUE_CUDA" for t in node.targets)]
    if len(literals) != 1:
        raise AssertionError("Expected one early-tail B1 source")
    node = literals[0]
    old = ast.literal_eval(node)
    changed = replace(old, "#include <cooperative_groups.h>\n", "")
    changed = replace(changed, "namespace cg = cooperative_groups;\n", "")
    changed = replace_function(changed,
        "template<int NW,int N>\n__device__ __forceinline__ float cluster_squares(",
        "template<int V,int NW,bool POST,bool SLICED,bool ROUTER,bool PRELOADED_BASE>",
        """template<int NW,int N>
__device__ __forceinline__ float block_squares(const float (&x)[N], float* sm) {
  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  float sum=butterfly(local_squares(x),16);
  if(lane==0) sm[warp]=sum;
  __syncthreads();
  return butterfly(sm[lane&(NW-1)],NW/2);
}

template<int NW,int N>
__device__ __forceinline__ float block_squares_pair(const float (&x)[N], const float (&y)[N],
    float* sm, float* smy, float& tail) {
  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  float sx=butterfly(local_squares(x),16);
  float sy=butterfly(local_squares(y),16);
  if(lane==0) { sm[warp]=sx; smy[warp]=sy; }
  __syncthreads();
  tail=butterfly(smy[lane&(NW-1)],NW/2);
  return butterfly(sm[lane&(NW-1)],NW/2);
}

""")
    changed = replace(changed, "glue_cluster_boundary", "glue_row_boundary", 5)
    changed = replace(changed, "__launch_bounds__(128)", "__launch_bounds__(1024)")
    changed = replace(changed, "  constexpr int per_cta=logical_threads/8;\n",
                      '  static_assert(NW==32, "row boundary preserves exactly32 logical warps");\n')
    changed = replace(changed, """  cg::cluster_group cluster=cg::this_cluster();
  int rank=cluster.block_rank(), tid=threadIdx.x;
  int logical=rank*per_cta+(tid%per_cta);
  int row=blockIdx.x/8;
  bool active=tid<per_cta;
""", "  int tid=threadIdx.x, logical=tid, row=blockIdx.x;\n")
    changed = replace(changed, """  // DSMEM destinations must exist before the first remote store. All work
  // above this barrier and the dependency wait is independent of the branch.
  cluster.sync();
""", "  // Register preloads are independent of the branch; no remote shared-memory initialization is needed.\n")
    changed = replace(changed, "cluster_squares<NW>(b,partials[0],cluster,rank)",
                      "block_squares<NW>(b,partials[0])")
    changed = replace(changed, "cluster_squares_pair<NW>(ns,masked,partials[1],partials[2],cluster,rank,tail)",
                      "block_squares_pair<NW>(ns,masked,partials[1],partials[2],tail)")
    changed = replace(changed, "if(rank==0 && tid==0)", "if(tid==0)")
    changed = replace(changed, """    if(active) {
      gsb<V>(a.out+row*8192+col,h);
      gsf<V>(a.next+row*8192+col,carry+r*V);
      if constexpr(ROUTER) gsf<V>(a.router+row*8192+col,h);
    }
""", """    gsb<V>(a.out+row*8192+col,h);
    gsf<V>(a.next+row*8192+col,carry+r*V);
    if constexpr(ROUTER) gsf<V>(a.router+row*8192+col,h);
""")
    changed = replace(changed, """  cfg.gridDim=dim3(a.rows*8);cfg.blockDim=dim3(128);cfg.stream=stream;
  cudaLaunchAttribute attrs[2]{};
  attrs[0].id=cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim={8,1,1};
  attrs[1].id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed=1;
  cfg.attrs=attrs;cfg.numAttrs=2;
""", """  cfg.gridDim=dim3(a.rows);cfg.blockDim=dim3(1024);cfg.stream=stream;
  cudaLaunchAttribute attr{};
  attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed=1;
  cfg.attrs=&attr;cfg.numAttrs=1;
""")
    changed = replace(changed, "// eight 128-thread CTAs. This source is embedded in the single-file candidate.",
                      "// the same1024 logical threads in one CTA per row. Every per-lane FP32/BF16 operation is retained.")
    if any(fragment in changed for fragment in ("cg::", "cluster.", "per_cta", "glue_cluster_boundary")):
        raise AssertionError("Obsolete distributed-boundary machinery remains")
    original_literal = ast.get_source_segment(source, node)
    if not original_literal.startswith("r'''") or "'''" in changed:
        raise AssertionError("Unexpected B1 source quoting")
    raw = source.encode()
    offsets = [0]
    for line in raw.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    begin = offsets[node.lineno - 1] + node.col_offset
    end = offsets[node.end_lineno - 1] + node.end_col_offset
    result = (raw[:begin] + ("r'''" + changed + "'''").encode() + raw[end:]).decode()
    first, rest = result.split("\n", 1)
    result = first + "\n# Precision recovery: same32-warp B1 arithmetic in one CTA per row, with all audited BF16/router boundaries retained.\n" + rest
    ast.parse(result)
    return result, dict(kind="exact_b1_row", logical_threads=1024, values_per_thread=8,
                        original_ctas_per_row=8, candidate_ctas_per_row=1,
                        preserved="ga/gm/gf/gb/gr, local_squares, descending XOR butterflies, column mapping, branch BF16, residual FMA, FP32 router, carry and tail",
                        input_sha256=hashlib.sha256(source.encode()).hexdigest(),
                        output_sha256=hashlib.sha256(result.encode()).hexdigest(),
                        before_cuda_sha256=hashlib.sha256(old.encode()).hexdigest(),
                        after_cuda_sha256=hashlib.sha256(changed.encode()).hexdigest())
