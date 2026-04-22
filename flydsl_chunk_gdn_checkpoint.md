# Chunk-GDN FlyDSL Megakernel — Optimization Checkpoint

> **Goal**: Port flashinfer Blackwell cutedsl `chunk_gated_delta_rule_sm100` megakernel
> to AMD CDNA3/4 via FlyDSL, replacing FLA Triton's 6-7 kernel pipeline with
> a single fused kernel.
>
> **Target GPU**: MI355X (CDNA4, gfx950)  |  **Dev GPU**: MI308X (CDNA3, gfx942)
> **Target model**: Qwen3.5-397B-A17B, prefill TP=2 (H_v=32, H_k=8, DK=DV=128, BT=64)

## 1. Architecture (decided, locked in)

- **DSL**: FlyDSL (`flydsl-0.1.2.dev462` wheel; apfloat patchelf fix applied, see memory)
- **Approach**: Fully fused megakernel matching cutedsl structure
- **State**: Resident in registers (VGPR frag_C) across all NT chunks, like cutedsl TMEM
- **Output**: `o` per-token + `final_state` per-batch (no h history, no v_new, no w/u)
- **Reuse-cache**: Sparse `state_checkpoints` at SEQ_SIZE_PER_BLOCK boundaries (matches cutedsl `output_checkpoints`)

## 2. What FLA does today (baseline to replace)

```
chunk_gated_delta_rule_fwd: 29.33 ms/layer on MI308X, 6.66 ms on MI355X
  ├─ chunk_local_cumsum_scalar_kernel        0.18 ms
  ├─ l2norm_fwd_kernel1 (x2)                 0.18 ms
  ├─ chunk_scaled_dot_kkt_fwd_kernel         1.80 ms  (K·β·K^T with L_mask)
  ├─ solve_tril_16x16_kernel                 1.55 ms  (A_inv stages 1-2)
  ├─ merge_16x16_to_64x64_inverse_kernel     1.22 ms  (A_inv stages 3-4)
  ├─ recompute_w_u_fwd_kernel                5.72 ms  (w = A_inv·kβexp(g), u = A_inv·vβ)
  ├─ chunk_gated_delta_rule_fwd_kernel_h    11.43 ms  (state recurrence + h history)
  └─ chunk_fwd_kernel_o                      7.06 ms  (O = q·state + intra attn)
```

## 3. cutedsl reference architecture (THE baseline to mirror)

> Source: `/root/flashinfer/flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py`
> Public entry: `chunk_gated_delta_rule_sm100` (gdn_prefill.py:91)
> Inputs: `q, k, v, gate, beta` — **q,k are pre-normalized by caller** (l2_norm NOT inside)
>
> **Principle (locked in 2026-04-22)**: completely align with cutedsl architecture.
> Only deviate with explicit reason documented (e.g., AMD hardware lacks the primitive,
> or measured perf data shows fusion has zero benefit). No "I think we can skip this"
> calls without checking cutedsl source first.

### Inside-kernel preprocessing (CG0, executed by warp 10 + warps 0-3)
```
cumsumlog[t]    = sum_{l=0}^{t} log(gate_l)            # warp 10 inline scan
cumprod[t]      = exp(cumsumlog[t])                     # warp 10 reg compute
T_pairwise[i,j] = exp(cumsumlog[i] - cumsumlog[j])     # CG0 reg compute, i>=j
                = cumprod[i] / cumprod[j]               # equivalent form
```
**WHY fused (not external Triton)**: cumsumlog feeds 4+ downstream consumers
(T-pairwise, kk_epi, qk_epi, decay_v) within the same kernel — putting it in
SMEM avoids 4× GMEM round-trips. cumprod[BT-1] is also state decay scalar across chunks.

### 7 GEMMs per chunk
| # | GEMM | Computation | Notes |
|---|---|---|---|
| 1 | kk | `W_kk[BT,BT] = K @ K^T` | Lower-triangular intra scores |
| 2 | qk | `W_qk[BT,BT] = Q @ K^T` | Output attention scores |
| 3 | k*state (KS) | `KS[BT,DV] = K @ S_prev` | State as B operand |
| 4 | q*state (QS) | `QS[BT,DV] = Q @ S_prev` | State as B operand |
| 5 | new_v (NV) | `NV[BT,DV] = A_inv @ V` | A_inv = (I + M_kk)^{-1}, M_kk = T·β·W_kk |
| 6 | qkv | `O_intra[BT,DV] = W_qkv @ NV` | W_qkv = T·β·W_qk (CG0 epi) |
| 7 | kv update | `dS[DK,DV] = K^T @ delta` | delta = V - KS (after decay) |

### Epilogue
```
O[BT,DV]  = O_intra + T_col * QS                        # combine intra + inter
S_next    = cumprod[BT-1] * S_prev + dS                 # state decay + update
```

### cutedsl warp specialization (12 warps = 384 threads)
| Warp | Role |
|---|---|
| 0-3 | CG0: T-pairwise, kk_epi, qk_epi, hierarchical inverse |
| 4-7 | CG1: kv_decay_v, v-k*state, state*q_epi, new_v_epi, kv_update_epi, qkv_epilogue |
| 8 | MMA warp: issues all 7 GEMMs |
| 9 | TMA load: q (single-buf), k (double-buf prefetch), v |
| 10 | TMA gate: loads gate+beta, **computes cumsumlog inline** |
| 11 | Epilogue: stores O to GMEM |

### cutedsl SMEM/TMEM budget (Blackwell sm100, 225.5 KB SMEM + 256 KB TMEM)
SMEM:
- q, k(2-stage), v: 32KB each
- A_inverse / new_v: 32KB (overwritten between stages)
- QK output / O store: 32KB (overwritten)
- state / decay_v: 32KB (TMEM↔SMEM staging)
- cumsumlog, cumprod, cumprod_scale: 512B each (BT=64 fp32 × stages)

TMEM:
- state S: 65KB (DK×DV fp32 = 128×128×4)
- q*state acc: 65KB
- shared accumulator (qk/kk/new_v/k*state/kv/qkv): 65KB × 2 stages

### What cutedsl does NOT include (left to caller)
- **l2_norm(q, k)**: pre-processing in caller, NOT in megakernel
  - cutedsl source has zero `l2norm`/`rsqrt` matches in `blackwell/`
  - Public API `chunk_gated_delta_rule_sm100` takes already-normalized q, k

## 4. Our FlyDSL target megakernel (mirror of cutedsl, AMD-adapted)

Single kernel, persistent over NT chunks, grid (H_v, B, 1).

**Pre-processing kept as standalone Triton kernels** (matches cutedsl's external boundary):
- `fused_l2norm_qk` → 49us in Triton (commit `3c529f051`, 17× over baseline,
  beats SGL by 2×, near hardware peak via rsqrt+mul) — **cutedsl also does NOT fuse this**

**Inside the megakernel** (mirrors cutedsl's 13-stage flow):
```
For each chunk t (single AMD CTA, register-resident state):
  A. cumsumlog[t] = inline cumsum of log(gate)   # mirror cutedsl warp 10 (warp-level scan)
  B. cumprod[t]   = exp(cumsumlog[t])            # ALU
  C. kk           = K @ K^T (with β, L_mask)     # MFMA  [GEMM 1]
  D. A_inv        = (I - strict_lower(M_kk))^{-1} # hierarchical MFMA stages 1-4
  E. u            = A_inv @ (v * β)              # MFMA  [part of recompute_w_u]
  F. w            = A_inv @ (k * β * cumprod)   # MFMA  [part of recompute_w_u]
  G. v_new        = u - w @ state                # MFMA; state as B operand  [GEMM 5 equivalent]
  H. state       *= cumprod[BT-1];  v_new *= cumprod[BT-1]/cumprod  # decay
  I. state       += k^T @ v_new                  # MFMA  [GEMM 7 = dS]
  J. O_inter      = q @ state                    # MFMA; state as B operand  [GEMM 4 = QS]
  K. O_intra      = (q @ k^T * strict_lower * T_mask) @ v_new  # 2 MFMAs + ALU  [GEMM 2+6]
  L. O[t]         = (O_inter + O_intra) * scale  → GMEM write
  M. If chunk at SEQ_SIZE_PER_BLOCK boundary: state → sparse checkpoints  [reuse-cache]
```

Per-chunk budget: ~8-10 MFMAs + ALU + 2 MFMA with state-as-B-operand + inverse.

### AMD-specific deviations from cutedsl (with rationale)
| Deviation | Why |
|---|---|
| State in VGPR frag_C, not TMEM | AMD has no TMEM equivalent (only LDS + VGPR). Larger LDS on CDNA4 (160KB vs 64KB CDNA3) helps with staging. |
| Single warp per CTA (or 4-warp tiled_mma) instead of 12-warp specialization | AMD MFMA is synchronous; warp specialization with TMA producer/consumer doesn't map directly. CG0/CG1/MMA/TMA all collapse to warps within 1 wave. |
| LDS bf16 staging for state-as-B operand | AMD MFMA requires B operand in VGPR with specific layout; C-frag has different lane mapping → must stage through LDS bf16. |
| `mfma_f32_16x16x16bf16_1k` (CDNA3) atoms instead of `tcgen5_async_mma` | AMD's MFMA is the only matrix path. CDNA4 adds 16x16x32 / 32x32x16 atoms for higher throughput. |
| No async TMA pipeline | CDNA3 has no TMA; CDNA4 has `global_load_lds` (closer but not descriptor-driven). |

Expected: **1.5-2× vs FLA Triton** on MI355X (full fused + single launch + no h/w/u/A GMEM traffic).

## 5. Current implementation state

### ✅ Validated building blocks (from prior sessions)
- 7 individual GEMMs (kk, qk, KS, QS, NV, qkv, dS) each tested in FlyDSL
- Runtime NT=1024 via `for chunk_idx, state in range(NT, init=[...])` scf.for with carry
- Manual byte offset for K/V per chunk (bypasses zipped_divide runtime-slice limits)
- Manual MFMA via `rocdl.mfma_f32_16x16x16bf16_1k` (no fx.gemm abstraction)
- Register-resident state accumulator (frag_C across 1024 chunks)
- Multi-CTA grid (H, B, 1) for parallel per-head processing
- State as MFMA B operand (via bf16 LDS staging — `test_flydsl_chunk_gdn_v2.py` pattern)
- 16 atoms per warp for 64x64 output at DK=DV=64 (`test_flydsl_v11_full_multicta.py`)

### ⚠️ Not yet implemented (in plan, by session)
- Hierarchical inverse on MFMA stages 1-4 (`test_flydsl_v6_inverse_stage1.py` blocked on `ds_bpermute` API) — Session 7
- `w @ state` and `q @ state` inside same loop (state as B operand × 2 per chunk) — Sessions 4 + 6
- g decay applied between chunks — Session 5
- Inline cumsumlog scan (mirror cutedsl warp 10) — Session 10
- DK=DV=128 production shape (currently 64; need 4-warp tiled_mma) — Session 11
- Sparse checkpoint output at block boundaries — Session 12

### ❌ Out of plan (intentional, with reason)
- L2-norm fused into megakernel — cutedsl `chunk_gated_delta_rule_sm100` also takes pre-normalized q,k. Triton implementation already 49us (commit `3c529f051`, near hardware peak).

### 🧹 RTP integration status (as of 2026-04-21)

**All experimental FlyDSL drop-in code has been REMOVED from RTP.**
The production path is pure FLA Triton — identical to upstream pre-FlyDSL state.

Files reverted / deleted:
- ✗ `rtp_llm/models_py/triton_kernels/fla/flydsl_megakernel.py` — deleted
- ✗ `rtp_llm/models_py/triton_kernels/fla/flydsl_state_update.py` — deleted
- ✓ `chunk.py` — `_try_flydsl_path` dispatch removed, back to FLA-only
- ✓ `qwen3_next.py` — `h=None` branch removed, back to FLA-only
- ✓ `block.py` — `store_final_state_only_*` removed

**Rationale for cleanup**: the prior MVP drop-in produced numerically-incorrect
outputs (missing inverse / g-decay / intra-chunk attention). Even gated behind
`USE_FLYDSL_CHUNK_GDN=0` default, having the wrong-math code in the tree risked
silent activation. Re-integration will happen only at Session 12 (see flydsl_chunk_gdn_plan.md)
after a precision-correct megakernel is validated.

## 6. Key environment + infra (confirmed working)

| Item | Status |
|---|---|
| FlyDSL wheel install | ✅ `/opt/conda310/lib/python3.10/site-packages/flydsl/` |
| apfloat .so patch | ✅ `patchelf --remove-needed libmlir_apfloat_wrappers.so.23.0git libmlir_c_runner_utils.so` |
| MI308X development machine | ✅ gfx942, 80 CU, 64 KB LDS/CU |
| gpu-wiki reference | ✅ `/root/gpu-wiki/` |
| FlyDSL reference kernels | ✅ `/root/atrex/src/flydsl/` (pre-projection GEMM, 1181 lines) |
| flashinfer cutedsl source | ✅ `/root/flashinfer/flashinfer/gdn_kernels/blackwell/` (3733 lines main kernel) |

## 7. Key memory / learnings (see `/root/.claude/projects/-root/memory/`)

- `project_chunk_gdn_target_gpu.md` — MI355X is the production target, not MI308X
- `feedback_use_flydsl_not_gluon.md` — Use FlyDSL, not Gluon (even though Gluon docs more complete)
- `feedback_chunk_gdn_porting_focus.md` — port flashinfer cutedsl, NOT in-place optimize FLA
- `feedback_flydsl_apfloat_patch.md` — patchelf fix for wheel .so deps
- `feedback_flydsl_persistent_runtime_loop.md` — manual byte offset required for runtime NT loops

## 8. Related docs

- `flydsl_chunk_gdn_plan.md` — **detailed session-by-session plan** for real megakernel
- `TODAY_RETROSPECTIVE.md` — single-page status + next-steps (read this first)
- `flydsl_chunk_gdn_checkpoint_history.md` — 1820-line original checkpoint (historical detail)
- `chunk_gdn_perf/README.md` — index of validated FlyDSL building-block test files
- `flashinfer_gdn_porting.md` — original algorithm analysis

## 8.5 Process constraints (LOCKED-IN, apply to every future session)

These constraints were committed on 2026-04-21 after the housekeeping retrospective
revealed a pattern of premature performance claims and scope underestimation.
They override any impulse to ship faster or claim bigger numbers.

### C1 — Honest per-session status update (no exceptions)
Every session MUST end with a status block containing:
- ✅ What was actually validated (ran + matched reference)
- ⚠️ What was implemented but NOT fully validated
- ❌ What was NOT implemented (even if originally scoped)
- 🐞 Any regressions introduced in other paths
- 💰 Actual session cost vs budgeted

No "basically done" / "mostly working" language — every claim cites a test + rel err + condition.

### C2 — No "X× speedup" claim unless apples-to-apples end-to-end
A speedup number may ONLY be reported if ALL of these hold:
1. **Same algorithm**: the FlyDSL path produces output within rel err < 5e-3 of FLA
   on the same input (precision-correct, no "simplified" variants)
2. **Same shape**: DK=DV=128, H=32, BT=64, NT ≥ 256 (production scale)
3. **End-to-end**: measures the full FLA pipeline (6 kernels) vs full FlyDSL megakernel
   (not "FLA chunk_h only vs FlyDSL state-only")
4. **Same overhead counted**: Python-side transposes, reshape, and kernel launch
   count for both sides
5. **MI308X (dev) or MI355X (prod)**, identified explicitly

Prior misleading numbers retired: "4×", "6.1×", "113×", "1.92×" (all failed ≥1 condition).

### C3 — Stop-and-replan on scope underestimate
If at any point during a session it becomes clear the deliverable will take
>1.5× the originally estimated effort, STOP and:
1. Write current state (what works, what's blocked) to checkpoint
2. Split the remaining work into two sessions in flydsl_chunk_gdn_plan.md
3. Ask the user to confirm the new split before proceeding

Past failures: tried to fuse 6-7 kernels in one go, hit hierarchical inverse API
friction, ended up with simplified-algo MVP that still shipped to production
code path. Won't repeat.

### C4 — No production-tree pollution with experimental code
Experimental FlyDSL code lives in `/root/rtp-llm/chunk_gdn_perf/`, NOT in
`/root/rtp-llm/rtp_llm/`. Integration into the RTP production tree
(`rtp_llm/models_py/`) happens ONCE, at Session 12, after precision-correct end-to-end
megakernel validation. No env-var-gated drop-in "escape hatches" meanwhile.

### C5 — cutedsl is the baseline. Deviate only with explicit reason.
Before deciding to skip / merge / drop / re-scope any algorithm step:
1. **Check cutedsl source first** (`/root/flashinfer/flashinfer/gdn_kernels/blackwell/`)
2. If cutedsl includes the step, default to including it in the FlyDSL megakernel
3. Deviation requires written reason in the checkpoint (one of: AMD primitive missing,
   measured zero perf benefit with profiling data, or different RTP-side constraint)
4. "I think we can skip this" / "this is small so not worth fusing" without checking
   cutedsl is **not a valid reason**

Past failure (codifies why C5 exists): plan rev 1.1 (2026-04-22) dropped both
l2_norm + cumsum from megakernel scope based on FLA timing data alone (l2_norm 49us,
cumsum 0.18ms). After actually checking cutedsl source, found:
- cutedsl agrees on l2_norm: not in megakernel (caller pre-normalizes) — drop was correct
- cutedsl disagrees on cumsum: warp 10 fuses cumsumlog inline; consumed by 4 downstream
  stages within the kernel — drop was wrong, cumsum reinstated in rev 1.2

The lesson: for architecture decisions, FLA timing tells us "what's expensive in FLA",
not "what cutedsl chose to fuse and why". Always check cutedsl first.

## 9. Test file inventory (11 files under `chunk_gdn_perf/`)

After 2026-04-22 cleanup. Full index with which session each is used by:
see `chunk_gdn_perf/README.md`.

Quick reference (validated FlyDSL building blocks):
- `test_flydsl_gemm{1_kk,12_kk_qk,3_ks,4567}.py` — individual GEMMs (Sessions 4-9)
- `test_flydsl_alu_mask_beta.py` — ALU primitives for mask/β/decay (Sessions 5-9)
- `test_flydsl_persistent_mfma_state.py` — register-resident state pattern (Session 4)
- `test_flydsl_persistent_gemm7.py` — dS GEMM persistent across chunks (Session 4)
- `test_flydsl_v9_runtime_multicta.py` — simplest runtime NT scf.for + multi-CTA (Sessions 2, 4)
- `test_flydsl_v11_full_multicta.py` — production-scale 16 atoms/warp (Sessions 4, 11)
- `test_flydsl_chunk_gdn_v2.py` — state-as-B-operand via bf16 LDS staging (Session 3)
- `test_flydsl_v6_inverse_stage1.py` — hierarchical inverse stage 1 (BLOCKED on `ds_bpermute`, Session 7 starting point)

## 10. Performance data points (MI308X)

### Validated baselines

| Workload | Source | Time | Status |
|---|---|---:|---|
| FLA full chunk_gdn (DK=DV=128, H=32, T=64K) | rocprof CSV in `chunk_gdn_perf/` | **29.33 ms/layer** | Production baseline |
| FLA full chunk_gdn (DK=DV=64, H=32, T=64K) | rocprof CSV | 16.96 ms/layer | Scaled-down baseline |
| FLA `fused_l2norm_qk` (T=122936 D=128) | commit `3c529f051` | **49us** | Near hardware peak (rsqrt+mul) |

### Building-block measurements (NOT speedup claims — see C2)

These are isolated FlyDSL primitive timings on incomplete algorithms.
**None of them is end-to-end precision-correct vs FLA**, so per C2 none can
become a speedup claim. Listed here only to size individual operations
relative to the FLA breakdown.

| Workload | Source | Time | What it is |
|---|---|---:|---|
| FlyDSL state-update only (DK=DV=64, H=32, NT=1024) | `test_flydsl_v11_full_multicta.py` | 6.88 ms | State accumulation primitive — does NOT include inverse, decay, or O. Cannot be compared to FLA's `chunk_h` (different algorithm coverage). |

### Retired claims (kept here as evidence for C2)

Per C2 (no unfair speedup claims), these are historical anti-examples:

| Past claim | Why retired |
|---|---|
| "1.92× MVP DK=DV=64" | MVP had simplified algorithm (no inverse / no decay / no intra) — not apples-to-apples |
| "4.08× / 4× state-only" | State update vs FLA `chunk_h` only — single op, not end-to-end |
| "6.1×" | Kernel-only vs FLA with launch overhead — ignored Python transpose |
| "113×" | Single-op micro-bench — not end-to-end |

A valid speedup number won't exist until **Session 6+** when both paths
produce the same output end-to-end at production shape (DK=DV=128).

---

**Last updated**: 2026-04-22
