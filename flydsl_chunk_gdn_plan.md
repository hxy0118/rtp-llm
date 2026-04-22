# FlyDSL chunk-GDN Megakernel — Session-by-Session Implementation Plan

> **Goal**: Port flashinfer cutedsl `chunk_gated_delta_rule_sm100` to AMD CDNA3/4
> via FlyDSL — one fused megakernel replacing FLA Triton's 7-kernel pipeline,
> drop-in production-ready for RTP serving Qwen3.5-397B-A17B prefill.
>
> **Total estimate**: 13 sessions (~$3-5K at current session cost), spread over 2-3 weeks.
>
> Each session: one focused deliverable, validated vs FLA on MI308X before moving to next.
>
> **Baseline principle**: align with cutedsl architecture. Deviate only with explicit reason
> (AMD primitive missing, or measured zero perf benefit). See `flydsl_chunk_gdn_checkpoint.md` §3
> for the cutedsl reference architecture and §8.5 C5 for the constraint.

---

## Session Plan Overview

| # | Session | Primary Deliverable | Cumulative Algo Coverage | Validation |
|---|---|---|---|---|
| 1 | **Housekeeping** (done) | Condensed checkpoint + this plan | - | - |
| 2 | Skeleton + test harness | Per-step precision-compare framework | Scaffold only | vs FLA per-stage |
| 3 | state-as-B mechanics | `q @ state` and `w @ state` primitives | State read path | Isolated |
| 4 | v_new + state update | Core recurrence with host-pre w/u | Steps G, H (no decay), I | vs FLA |
| 5 | g_decay + final_state | Step H (decay), final_state correct | +E decay | vs FLA |
| 6 | O_inter + O_intra | Full O output matching chunk_fwd_o | +J, K, L | vs FLA |
| 7 | Inverse on GPU | Hierarchical MFMA stages 1-4 | +D (no host inv) | vs FLA |
| 8 | w, u on GPU | Fused recompute_w_u inside kernel | +E, F | vs FLA |
| 9 | kk on GPU | Fused K·β·K^T with L_mask | +C | vs FLA |
| 10 | cumsum on GPU | Inline `cumsumlog` (mirrors cutedsl warp 10) | +A, B | vs FLA end-to-end |
| 11 | DK=DV=128 + multi-warp | Production shape via 4-warp tiled_mma | Production scale | Precision + perf |
| 12 | Sparse checkpoints | Block-boundary state snapshots | +M reuse-cache | RTP integration |
| 13 | MI355X deployment | rocprofv3 A/B vs FLA + precision regression | - | Production validation |

**Out of plan — kept as standalone Triton kernel (matches cutedsl boundary)**:
- `fused_l2norm_qk_kernel` — 49us in commit `3c529f051` (17× over baseline, near hardware peak via rsqrt+mul). cutedsl `chunk_gated_delta_rule_sm100` also takes already-normalized q,k as input (zero `l2norm`/`rsqrt` matches in `blackwell/`). Re-implementing in FlyDSL has zero benefit AND breaks parity with cutedsl boundary.

---

## Session 2: Skeleton + Per-step Precision Test Harness

### Deliverables
1. **Kernel skeleton** (`flydsl_megakernel_v2.py`) with all 13 stages as labeled placeholders
2. **Stage-by-stage test framework** (`test_megakernel_stages.py`):
   - Extract intermediate tensors from FLA (kk, A_inv, w, u, v_new, state history, O_inter, O_intra)
   - Compare FlyDSL intermediate against FLA golden at each stage
3. **Verify RTP production path is FLA-only**
   - Already cleaned in housekeeping (2026-04-21): no FlyDSL code in `rtp_llm/models_py/`
   - Per constraint C4, no experimental dispatch added until Session 12

### Acceptance
- Skeleton compiles (even if stages produce garbage)
- Test harness can report per-stage rel error vs FLA for any algorithm step
- No regression in FLA default path (production tree under `rtp_llm/` unchanged)

### Risks / notes
- Extracting FLA intermediates requires instrumenting FLA kernels (monkey-patch or rebuild Triton versions that save intermediates)
- May need new Triton kernels just to dump intermediate tensors for comparison

---

## Session 3: State-as-B Operand Mechanics (Foundation)

### Deliverables
1. **Validate `state_as_B.py`**: standalone test that
   - Initializes state S in register (frag_C layout)
   - Converts S → bf16 LDS staging
   - Uses S as MFMA B operand in another GEMM (e.g., `q @ S`)
   - Produces correct output matching `torch.matmul(q, S)`
2. **Document lane-layout conversion** between C-frag and B-frag for MFMA 16x16x16 bf16
3. **LDS budget analysis**: Confirm state + staging fits in 64 KB (CDNA3) / 160 KB (CDNA4)

### Acceptance
- `q @ state` returns correct fp32 output, rel err < 1e-3 bf16-quantized
- `w @ state` (another state-as-B GEMM) similarly validates
- Register + LDS pressure stays within 1 CTA / CU

### Risks / notes
- MFMA C-frag → LDS → B-frag requires careful lane remap
- Previously hit `fx.make_tensor` missing API; may need STensor shim manual approach

---

## Session 4: V_new + State Update (Core Recurrence)

### Deliverables
1. **Kernel that takes pre-computed `w, u`** (from FLA `recompute_w_u_fwd` on host or via FLA Triton call):
   - Per chunk: `v_new = u - w @ state` (state as B operand)
   - Per chunk: `state += k^T @ v_new` (k^T raw, NOT k_beta_decay — that's in w already)
2. **Python wrapper** that:
   - Calls FLA `chunk_scaled_dot_kkt_fwd → solve_tril → recompute_w_u_fwd` for w, u
   - Calls FlyDSL kernel for state recurrence
3. **Precision test**: compare final_state vs FLA's final_state (no g decay yet; set g=0)

### Acceptance
- final_state matches FLA rel err < 1e-2 (bf16 noise)
- v_new per chunk can be optionally written to GMEM for debug

### Risks / notes
- 2 state-as-B GEMMs per chunk: `w @ state` and `q @ state` (for O in later sessions)
- Register pressure starts climbing; may need to reduce NT scf.for carry or stage more in LDS

---

## Session 5: g Decay + Accurate Final State

### Deliverables
1. **Per-chunk g_last broadcast** — read `g_cumsum[chunk*BT + BT - 1]` (single fp32 per head)
2. **State decay**: `state *= exp(g_last)` applied in register between chunks
3. **V_new decay**: `v_new *= exp(g_last - g_cumsum_per_token)` (per-token decay before state update)
4. **g_cumsum input** — use FLA's `chunk_local_cumsum` output as kernel input

### Acceptance
- With non-zero g, final_state matches FLA rel err < 1e-2
- Per-chunk g decay precision verified in isolation first

### Risks / notes
- `exp` in FlyDSL — use `rocdl.exp2` or `arith.exp` wrapper
- Fp32 decay computation but state stays in fp32 accumulator (no precision loss)

---

## Session 6: O Output (O_inter + O_intra, replacing chunk_fwd_o)

### Deliverables
1. **O_inter = q @ state** — another state-as-B GEMM (uses LDS staging from Session 3)
2. **O_intra** — intra-chunk attention:
   - `W = q @ k^T * strict_lower_mask * exp(g cross-terms)` (MFMA + ALU)
   - `O_intra = W @ v_new` (MFMA)
3. **O[t] = (O_inter + O_intra) * scale** → per-token GMEM write
4. **l2_norm done externally** by FLA Triton `fused_l2norm_qk` (caller side; matches cutedsl boundary; **permanent** — see "Out of plan" at top)

### Acceptance
- Full O output matches FLA chunk_fwd_o rel err < 5e-3
- No NaN/Inf under realistic input distributions

### Risks / notes
- Intra-chunk: q @ k^T is new GEMM (smaller, [BT, DK] @ [BT, DK] = [BT, BT])
- g cross-term mask is complex: `exp(g_cumsum[i] - g_cumsum[j])` per (i, j) pair
- May need LDS for mask staging

---

## Session 7: Hierarchical Inverse on GPU (Replace Host inv)

### Deliverables
1. **Stage 1** — Gauss-Jordan 8x8 diagonal blocks (`ds_bpermute` or `ds_swizzle` for lane shuffle)
2. **Stage 2** — 8x8 → 16x16 via MFMA 16x16x16
3. **Stage 3** — 16x16 → 32x32 via MFMA 16x16x16
4. **Stage 4** — 32x32 → 64x64 via MFMA 32x32x16
5. **Integration**: kernel no longer requires A_inv as input

### Acceptance
- A_inv computed on GPU matches host `torch.linalg.inv(I + M)` rel err < 1e-3
- Kernel runs stage 1-4 in-place in LDS, total < 1us added per chunk

### Risks / notes
- `ds_bpermute` API had issues previously; may need workaround with `ds_swizzle` or LDS broadcast
- bf16 intermediate precision across 4 stages needs validation (may need f32 intermediates)
- Prior attempt `test_flydsl_v6_inverse_stage1.py` can be resumed

---

## Session 8: Fused recompute_w_u (GPU)

### Deliverables
1. **w = A_inv @ (k_beta_with_decay)** — internal to kernel using A_inv from Session 7
2. **u = A_inv @ (v_beta)** — internal to kernel
3. **Pre-decay k_beta**: `k_with_decay = k * beta * exp(g_cumsum)` (ALU)
4. **Pre-decay v_beta**: `v_beta = v * beta` (ALU)

### Acceptance
- w, u match FLA's recompute_w_u_fwd output rel err < 1e-3
- Kernel now takes (q, k, v, beta, g_cumsum, initial_state); no A_inv input needed

### Risks / notes
- Two more MFMAs per chunk (A_inv @ k, A_inv @ v) add compute
- Register + LDS pressure; may need to spill or reduce tile sizes

---

## Session 9: Fused kk Computation (GPU)

### Deliverables
1. **kk = K · β · K^T** — MFMA with β pre-scaling (ALU before MFMA)
2. **L_mask decay** — `L_mask[i, j] = exp(g_cumsum[i] - g_cumsum[j])` (2D mask from g)
3. **M = -kk with strict_lower + L_mask + diagonal clearing**
4. **Integration**: kernel no longer requires A (kkt result) input

### Acceptance
- kk matches FLA chunk_scaled_dot_kkt_fwd rel err < 1e-3
- M matches FLA's A input to solve_tril

### Risks / notes
- kk is 64x64 MFMA output, 16 atoms per warp (similar to existing GEMMs)
- L_mask requires 2D computation from g (can be done inline per-thread)

---

## Session 10: cumsum on GPU (mirror cutedsl warp 10)

### Deliverables
1. **Inline cumsumlog scan**: per-chunk warp-level prefix sum of `log(gate)` across BT tokens
   - cutedsl uses dedicated warp (warp 10) — we use cooperative inline scan within our CTA
2. **Inline cumprod**: `cumprod[t] = exp2(cumsumlog[t])` (single ALU per token)
3. **Stage cumsumlog/cumprod in LDS** for reuse by downstream MFMA epilogues:
   - T-pairwise (intra-chunk decay mask)
   - kk_epi (W_kk *= T * β)
   - qk_epi (W_qk *= T * scale)
   - decay_v (V_new *= cumprod[BT-1] / cumprod[t])
   - state decay (S *= cumprod[BT-1])
4. **Drop external `chunk_local_cumsum` Triton call** — kernel now takes raw `g` as input

### Acceptance
- Full FlyDSL kernel output matches FLA chunk_gated_delta_rule (with external l2_norm) rel err < 5e-3
- `cumsumlog` LDS values match `chunk_local_cumsum` Triton kernel rel err < 1e-5

### Risks / notes
- BT=64 fp32 cumsum is tiny (256 bytes); warp shuffle scan is 6-step (log2(64))
- Storing as bf16 in LDS would lose precision for exp() — keep as fp32
- Mirrors cutedsl SMEM allocation: `cumsumlog/cumprod/cumprod_scale` 512B each

---

## Session 11: DK=DV=128 + Multi-warp Tiled MMA

### Deliverables
1. **Multi-warp tiled_mma** for production DK=DV=128 output (4 warps × (2,2,1) for 128x128)
2. **Register pressure management** — 64 atoms/warp × 4 f32 frag = 256 VGPR/wave
3. **Precision verification at production shape**
4. **Perf measurement vs FLA at DK=DV=128**

### Acceptance
- DK=DV=128 kernel compiles and runs without VGPR spill
- Precision matches FLA rel err < 5e-3
- Perf ≥ 1.3× FLA on MI308X

### Risks / notes
- 64 atoms/warp × 4 f32/atom = 256 VGPR just for frag_C — very tight
- May need to split state across multiple warps (e.g., 4 warps each hold quarter of state)

---

## Session 12: Sparse State Checkpoints + RTP Integration

### Deliverables
1. **Kernel emits state to `checkpoints` tensor** at `chunk % (SEQ_SIZE_PER_BLOCK / CHUNK_SIZE) == 0` boundaries
2. **New `store_sparse_checkpoints_to_block_map` Triton kernel** (replaces store_ssm_state_to_block_map)
3. **qwen3_next.py** updated to consume sparse checkpoints
4. **Full prefill + reuse-cache test on MI308X**

### Acceptance
- Cold prefill produces correct O + final_state
- Warm prefill (cache hit at SEQ_SIZE_PER_BLOCK boundary) produces identical output to FLA
- No regression in non-FlyDSL path

### Risks / notes
- Sparse checkpoint write is per-chunk conditional; minimal overhead
- RTP block_map addressing logic stays same

---

## Session 13: MI355X Deployment + Validation

### Deliverables
1. **Deploy to MI355X** (CDNA4 gfx950)
2. **rocprofv3 baseline vs FLA** at production shape
3. **Full Qwen3.5-397B prefill regression** (precision + latency)
4. **Final performance report**: expected 1.5-2× speedup on MI355X

### Acceptance
- MI355X numerics match MI308X (bit-exact or rel err similar)
- Measured speedup ≥ 1.5× on representative prefill workload (apples-to-apples per C2: same algorithm, same shape DK=DV=128 H=32, same Python overhead, end-to-end)
- Production-ready: dispatch FlyDSL megakernel from `chunk.py` for supported shapes; FLA fallback for unsupported (Session 12 dispatch design)

### Risks / notes
- CDNA4 has larger LDS (160 KB vs 64 KB) — may allow further optimization
- `mfma_scaled` (FP4/FP6) for future FP8 optimization path (not this plan)

---

## Key cross-session infra

### Testing framework (built in Session 2, used throughout)
- Per-stage golden comparison: intermediate tensor dump from FLA → rel err vs FlyDSL at same stage
- Automated regression: each session re-runs all prior session's tests to catch regressions
- CI-like behavior even without CI

### Precision targets (cumulative, progressively strict)
- Session 4: final_state rel err < 1e-2 (no g)
- Session 5: final_state rel err < 1e-2 (with g)
- Session 6: O rel err < 5e-3
- Session 10: full megakernel (with external l2_norm only) rel err < 5e-3 vs FLA end-to-end

### Performance targets (cumulative)
- Session 4: perf not critical, focus precision
- Session 6: ≥ parity with FLA (equal work)
- Session 11: ≥ 1.3× FLA on MI308X (DK=DV=128)
- Session 13: ≥ 1.5× FLA on MI355X

---

## Out of scope (future / non-goals)

- Backward pass (prefill only, no training support)
- FP8/FP4 MFMA (separate optimization once BF16 path lands)
- Other linear attention variants (Mamba2, RetNet) — chunk_gdn specific
- decode path (single-token, uses fused_recurrent not chunk_gdn)

---

## Risk register

| Risk | Mitigation |
|---|---|
| FlyDSL API friction slows session N | Session budget assumes 20% API friction time |
| `ds_bpermute` API blocking inverse stages 1-4 | Session 7 — try `ds_swizzle` or LDS broadcast fallback |
| bf16 precision drift across stages | f32 intermediates at critical fan-in/out points |
| VGPR spill at DK=DV=128 | Split state across warps; session 11 has contingency |
| Cross-batch varlen not supported | Session 12 RTP integration: dispatch falls back to FLA for unsupported shapes |
| MI355X-only features (async_copy / global_load_lds) needed | CDNA3 baseline runs on MI355X; Session 13 may add CDNA4-only optimization |

---

**Plan version**: 1.2 (2026-04-22)
**Next action**: Session 2 — build skeleton + test harness
**Revision lesson** (codified as constraint C5 in checkpoint §8.5): always grep cutedsl source before deviating from its fusion boundaries.
