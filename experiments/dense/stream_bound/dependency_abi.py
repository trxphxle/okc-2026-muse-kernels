"""Shared ABI for the dependency-aware zero-heavy-math stream bound.

Mandatory real work: new KV stores; BF16-rounded B1's two sequential full-row
reductions; pooler normalization/top8 selection; sliced B2's two reductions with
B1's precomputed tail. GEMM/attention/expert arithmetic is replaced by frozen
REAL producer tensors captured from the unmodified baseline, never fabricated
scores/output caches claimed as a submission. GPU producer copies and readiness
publication remain timed. Actual top8 produces the active-expert mask; expert
weights are selected on GPU from a bank for all32 experts, not a host routed list.

Phase0 reads Q,K,V rows0:1280 of qkv_gate. Remaining gate rows1280:2304 join
phase1(KV), because attention Q/K/V can become ready before its output gate.
The phase1 completion joins both before O consumes. Every original weight/KV
payload byte from the validated native manifest still appears exactly once.

Fixed chunks are int64[n,3]=(ptr,bytes,0), phases int64[9,2]=(start,count).
Expert bank int64[32,expert_chunks_per,3] contains original NVFP4 codes/scales;
full global-scale vectors and post-expert norm live in fixed phase6.
Phase6 dynamic bytes are read ONLY after timed top8 publishes ACTIVE_MASK.

Params is CUDA int64[len(FIELDS)]. Tensor entries below contain device pointers;
T/BLOCKS/EXP_CHUNKS_PER/DELAY_STAGE/FLAT_COUNT are scalar values. All source/destination
lifetimes are retained by the prepared plan. No timed host-value decisions.

One resident256-thread CTA per SM. Warp0 runs the same4x32KiB TMA ring; other
warps participate only in real glue/copies. Device release/acquire completion
flags avoid grid-wide exit/re-entry barriers. Invocation epoch is flags[QKV_DONE]
+1 loaded before phase0 by every CTA; QKV's all-CTA arrival prevents an epoch
race. Producers release distinct cache-line-padded arrival epochs; one warp
acquires them and publishes readiness. No contended RMW or external reset.

Modes0/1 are timed strict/prefetch;2/3 are their exact-XOR+timestamp validation
versions. Mode4 is a feature-off continuous-read control using FLAT_CHUNKS
and FLAT_COUNT, with the same256-thread/kernel interface resource envelope.
Prefetch keeps at most2 next-operation constant-weight chunks in slots0/1:
O across KV work, router/pooler across B1, unpooler across top8/experts.
Current work uses slots2/3 while these are retained. No KV prefetch before new
KV publication, no expert bank read before top8. Prefetched bytes cannot be
released/discarded until their consumer prerequisite is satisfied.

SINK[blocks,9,2]: physical payload bytes and uint32 XOR(mode2/3), per phase.
STAMPS[blocks,9,3]: first TMA issue,last TMA completion,consumer-enter(ns).
READY_TIMES[10]: release-publication time(ns) per flag; validation modes only.
FLAGS uint32[10]; ARRIVALS uint32[10,blocks,32] (only word0 per row used).
ACTIVE_MASK uint32[1]; ERROR int32[1].
LOCAL_START int32[1], IDS_OUT int32[T,8], CACHE_LOC int64[T].
PRE1/POST2/PRE2/GPREXP are BF16 effective factors(1+w); ALPHA/BETA are FP32.
The original ffn_residual_gate payload belongs to phase3 because B1 uses ALPHA2;
phase8 holds post_ffn_norm and next_pre_attn_norm.
Dense-live tensors are populated from their matching oracle ONLY after each
matching full-K weight-row chunk is consumed; pointer bases identify rows.
B1 rows are owned by CTAs b<T; B2 output tiles by b<T*8 (1024cols each), using
its legal precomputed8192-wide tail from B1, as the repaired baseline does.
DELAY_STAGE=-1 disables optional validation-only producer delay; nonnegative
values inject a bounded delay before a specified ready flag for ordering tests.
"""
FIELDS=(
'CHUNKS','RANGES','EXP_CHUNKS','EXP_CHUNKS_PER','T','BLOCKS','LOCAL_START',
'K_CACHE','V_CACHE','CACHE_LOC','NEW_KV',
'O_ORACLE','O_LIVE','POOLED_ORACLE','POOLED_LIVE','SCORES_ORACLE','SCORES_LIVE',
'UN_ORACLE','UN_LIVE','STREAM0','ALPHA1','BETA1','PRE1','ALPHA2','BETA2','POST2','PRE2',
'H_LIVE','HR_LIVE','CARRY','TAIL','OUT','SOUT','IDS_OUT','ACTIVE_MASK','FLAGS','ARRIVALS',
'SINK','STAMPS','READY_TIMES','O_W','POOL_W','ROUTER_W','UN_W','ERROR','GPREXP','XM_LIVE','DELAY_STAGE',
'FLAT_CHUNKS','FLAT_COUNT')
F={name:i for i,name in enumerate(FIELDS)}
PHASES=('qkv_qk_v','attention_kv_and_gate','o','boundary1','router_pooler','top8_and_preexpert','experts','unpooler','boundary2')
FLAGS=('QKV_DONE','KV_WRITTEN','ATTN_DONE','O_DONE','B1_DONE','RP_DONE','ROUTES_READY','EXPERTS_DONE','UNPOOL_DONE','B2_DONE')
FLAG={name:i for i,name in enumerate(FLAGS)}


def cpp_enum():
    return 'enum ParamIndex {'+','.join(name+'='+str(i) for name,i in F.items())+'};\n'+'enum ReadyIndex {'+','.join(name+'='+str(i) for name,i in FLAG.items())+'};\n'


def arrival_producers(tokens,blocks):
    return [blocks,tokens,blocks,blocks,tokens,blocks,tokens,blocks,blocks,tokens*8]
