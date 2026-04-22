# Chunk-GDN FlyDSL Megakernel — Status & Next Steps

> **Last updated**: 2026-04-22
> **Single-page status doc** — read this first. Pointers below for deeper dives.
>
> **Pointers**:
> - Architecture + constraints: `flydsl_chunk_gdn_checkpoint.md` (esp. §3 cutedsl baseline, §8.5 process constraints)
> - Session-by-session plan: `flydsl_chunk_gdn_plan.md` (rev 1.2)
> - Reference test code: `chunk_gdn_perf/README.md`
> - Full historical checkpoint: `flydsl_chunk_gdn_checkpoint_history.md` (1820-line archive)

---

## 1. Where we are (current state)

### Algorithm coverage vs cutedsl baseline

cutedsl `chunk_gated_delta_rule_sm100` fuses **7 of FLA's 8 kernels** into a
single megakernel (l2_norm stays in caller). Our FlyDSL port aims for the
same fusion boundary on AMD CDNA3/4 (cutedsl architecture in checkpoint §3,
AMD-specific deviations in §4). Current status:

| FLA kernel | Time (MI308X) | cutedsl | FlyDSL coverage | Plan session |
|---|---:|---|---|---|
| `chunk_local_cumsum_scalar_kernel` | 0.18 ms | ✓ fused (warp 10) | 0% | Session 10 |
| `l2norm_fwd_kernel1` (×2) | 49us (commit `3c529f051`) | ✗ external | n/a — kept Triton | **out of plan** |
| `chunk_scaled_dot_kkt_fwd_kernel` | 1.80 ms | ✓ fused | ~30% (`gemm1_kk`) | Session 9 |
| `solve_tril` + `merge_..._inverse` | 2.77 ms | ✓ fused (4-stage MFMA) | ~10% (stage-1 blocked on `ds_bpermute`) | Session 7 |
| `recompute_w_u_fwd_kernel` | 5.72 ms | ✓ fused | 0% (needs inverse) | Session 8 |
| `chunk_gated_delta_rule_fwd_kernel_h` | 11.43 ms | ✓ fused | ~25% (`v11_full_multicta`) | Sessions 4, 5 |
| `chunk_fwd_kernel_o` | 7.06 ms | ✓ fused | ~20% (`chunk_gdn_v2` state-as-B) | Sessions 3, 6 |

**Aggregate honest progress**: **~12% of the algorithm**, in the form of
isolated FlyDSL primitives. **Zero kernels are full drop-in replacements**.

### What's validated and reusable

✅ FlyDSL building blocks (in `chunk_gdn_perf/`, indexed in its README):
- All 7 GEMMs tested in isolation
- Runtime NT scf.for with state carry (true non-unrolled loop over 1024 chunks)
- Register-resident state accumulator (frag_C across chunks)
- State-as-B-operand pattern via bf16 LDS staging
- Multi-CTA grid for parallel per-head processing
- 16 atoms/warp at DK=DV=64

✅ RTP production tree clean:
- All experimental FlyDSL drop-in code reverted (`flydsl_megakernel.py`, `flydsl_state_update.py` deleted; `chunk.py`/`qwen3_next.py`/`block.py` back to FLA-only)
- Production path is pure FLA Triton — identical to upstream pre-FlyDSL state

✅ Design + planning artifacts:
- `flydsl_chunk_gdn_checkpoint.md` (~300 lines, condensed from 1820)
- `flydsl_chunk_gdn_plan.md` rev 1.2 (13-session breakdown, cutedsl-aligned)
- 5 process constraints in checkpoint §8.5
- 1820-line history archived

### What's blocked / risky

| Risk | Status | Mitigation |
|---|---|---|
| `ds_bpermute` API friction | **Blocking** Session 7 inverse | `test_flydsl_v6_inverse_stage1.py` left at the wall; need correct MLIR Value signature |
| `ds_bpermute` workaround unclear on AMD MI308X | Open | Possible `ds_swizzle` fallback or LDS broadcast |
| VGPR pressure at DK=DV=128 | Future Session 11 | Multi-warp tiled_mma to split state |
| Algorithm parity verification overhead | Ongoing | Per-stage test harness (Session 2 deliverable) |

---

## 2. Why progress is where it is

Cumulative cost so far: **~$140** for ~3 sessions (2 implementation + 1 housekeeping).
Honest cost attribution:

| Bucket | % | What it covered |
|---|---:|---|
| FlyDSL API friction | 40% | Each new API hit a 30-60min figure-out cycle (`fx.arith.constant`, `range_constexpr`, `scf.for` carry indexing, `ds_bpermute` signature, `mfma` cbsz/abid/blgp types, `apfloat` patchelf, `zipped_divide` runtime-slice limits) |
| Scope misestimation | 20% | Initial plan tried to fuse all 7 kernels in one go → got an algorithm-incorrect MVP (now deleted). Took several rounds of pushback to land on the 13-session split |
| Verification overhead | 20% | Each FlyDSL primitive needed precision test vs torch/FLA reference (40 test files originally, now pruned to 11 reference patterns) |
| Algorithm complexity | 20% | chunk_gdn is hairy: hierarchical inverse, state-as-B-operand, per-token + cross-chunk decay, intra-chunk masked attention with g cross-terms |

### Misleading speedup numbers — all retired

| Past claim | Why retired |
|---|---|
| "6.1×" | FlyDSL kernel-only vs FLA with launch overhead — apples to oranges |
| "4.08×" / "4×" | FlyDSL MVP missing 80% of algorithm; not real drop-in |
| "113×" | Single-op micro-bench (state update only vs FLA `chunk_h` only) |
| "1.92×" | MVP numerically wrong (rel err O(1.0)) |

A valid speedup number won't exist until **Session 6+** when both paths produce
the same output end-to-end. Constraint C2 in `flydsl_chunk_gdn_checkpoint.md` §8.5
locks the rules for future claims.

### Plan revision history (lessons)

- **rev 1.0** (2026-04-21): initial 13-session plan
- **rev 1.1** (2026-04-22): hastily dropped both `l2_norm` AND `cumsum` to 12 sessions WITHOUT checking cutedsl source. **Wrong.**
- **rev 1.2** (2026-04-22): course-corrected after grep'ing cutedsl. l2_norm stays out (cutedsl agrees), cumsum restored (cutedsl warp 10 fuses cumsumlog inline; consumed by 4+ downstream stages). Lesson codified as constraint **C5** in `flydsl_chunk_gdn_checkpoint.md` §8.5: "cutedsl is the baseline; deviate only with explicit reason after checking source first."

---

## 3. cutedsl baseline (concise — full detail in `flydsl_chunk_gdn_checkpoint.md` §3)

cutedsl `chunk_gated_delta_rule_sm100` is the reference architecture. Our
FlyDSL megakernel mirrors its structure with AMD-specific deviations.

**Inside the cutedsl megakernel** (per chunk):
1. cumsumlog + cumprod + T_pairwise (preprocessing in CG0)
2. 7 GEMMs: kk, qk, KS, QS, NV (=A_inv@V), qkv, dS
3. Hierarchical inverse (4-stage MFMA, replaces FLA's `solve_tril` + `merge`)
4. Epilogue: O = O_intra + T_col·QS, S_next = cumprod[-1]·S_prev + dS
5. Sparse checkpoints at block boundaries

**12-warp specialization**: warps 0-3 (CG0: T-pairwise/kk_epi/qk_epi/inverse), 4-7 (CG1: kv_decay_v/v-k*state/etc.), 8 (MMA), 9 (TMA q/k/v load), 10 (TMA gate/beta + cumsumlog), 11 (epilogue).

**SMEM 225.5 KB / TMEM 256 KB** on Blackwell sm100. AMD MI308 has only 64 KB LDS (160 KB on MI355) and no TMEM — state lives in VGPR frag_C with bf16 LDS staging for B-operand reuse.

**External to kernel**: l2_norm (caller pre-normalizes q,k). Zero `l2norm`/`rsqrt` matches in `blackwell/`.

---

## 4. What's next

### Immediate: Session 2 (skeleton + test harness)

Deliverables:
1. `flydsl_megakernel_v2.py` skeleton with all 13 algorithm stages as labeled placeholders (compiles even if stages produce garbage)
2. `test_megakernel_stages.py` — per-stage precision test harness:
   - Extract intermediate tensors from FLA (kk, A_inv, w, u, v_new, state history, O_inter, O_intra) — may need to instrument FLA Triton kernels to dump
   - Compare FlyDSL intermediate vs FLA golden at each stage with rel-err tolerance
3. Verify RTP production path remains pure-FLA (already cleaned; Session 2 is just a regression check)

Acceptance: skeleton compiles; harness reports per-stage rel err for any algorithm step; no regression in FLA default path.

### 13-session arc (rev 1.2)

| Session | Scope | Algo coverage |
|---|---|---|
| 2 | Skeleton + test harness | scaffold |
| 3 | state-as-B operand mechanics | state read path |
| 4 | v_new + state update (host w/u) | core recurrence |
| 5 | g_decay + final_state | + decay |
| 6 | O_inter + O_intra | + full O |
| 7 | Inverse on GPU | + inverse (replaces host inv) |
| 8 | w, u on GPU | + recompute_w_u fused |
| 9 | kk on GPU | + kk fused |
| 10 | cumsum on GPU (mirror cutedsl warp 10) | + cumsum fused |
| 11 | DK=DV=128 multi-warp tiled_mma | production shape |
| 12 | Sparse checkpoints + RTP integration | + reuse-cache |
| 13 | MI355X deployment + perf validation | production |

**RTP integration timing** (constraint C4): hands-off until Session 12. Current production tree stays pure FLA Triton.

### Performance + precision targets (cumulative)

| Session | Precision target | Performance target |
|---|---|---|
| 4 | final_state rel err < 1e-2 (no g) | not critical |
| 5 | final_state rel err < 1e-2 (with g) | — |
| 6 | O rel err < 5e-3 | ≥ parity with FLA |
| 10 | full megakernel (with external l2_norm) rel err < 5e-3 vs FLA end-to-end | — |
| 11 | DK=DV=128 rel err < 5e-3 | ≥ 1.3× FLA on MI308X |
| 13 | bit-exact or matching MI355X numerics | ≥ 1.5× FLA on MI355X |

---

## 5. Process constraints (locked-in, see `flydsl_chunk_gdn_checkpoint.md` §8.5)

These apply to every future session:

- **C1** — Honest per-session status: `✅ validated` / `⚠️ implemented but unvalidated` / `❌ not implemented` with cited tests + tolerances. No "basically done" / "mostly working" language.
- **C2** — No "X× speedup" claim unless apples-to-apples end-to-end (same algorithm, same shape DK=DV=128 H=32 BT=64, same Python overhead, named GPU). All prior numbers retired.
- **C3** — Stop-and-replan when scope is ≥1.5× original estimate. Write current state, split work, ask user before continuing.
- **C4** — Experimental code lives in `chunk_gdn_perf/`, NOT in `rtp_llm/models_py/`. RTP integration happens once at Session 12.
- **C5** — cutedsl is the baseline. Before skipping/merging/dropping any algorithm step, check `/root/flashinfer/flashinfer/gdn_kernels/blackwell/` first. "I think we can skip this" without checking cutedsl is not a valid reason.

---

## 6. Repository layout (after 2026-04-22 cleanup)

```
/root/rtp-llm/
├── flydsl_chunk_gdn_checkpoint.md          ← architecture + constraints (deep dive on cutedsl baseline)
├── flydsl_chunk_gdn_checkpoint_history.md  ← 1820-line archive (historical detail)
├── flydsl_chunk_gdn_plan.md                ← 13-session plan, rev 1.2
├── TODAY_RETROSPECTIVE.md                  ← THIS FILE — single-page status (read first)
├── flashinfer_gdn_porting.md               ← original algorithm analysis
└── chunk_gdn_perf/                         ← FlyDSL building-block reference
    ├── README.md                           ← test file index by session
    ├── .gitignore                          ← excludes coredumps + __pycache__
    ├── 11 × test_flydsl_*.py              ← validated reference patterns
    └── 6 × rocprof_*.csv                   ← FLA baseline kernel traces (TP=1, TP=2)

# Production (unchanged, pure FLA Triton):
/root/rtp-llm/rtp_llm/models_py/triton_kernels/fla/
└── chunk.py / chunk_delta_h.py / chunk_o.py / etc.
```

---

## 7. TL;DR for the next session

- **Where**: ~12% of algorithm ported as isolated primitives; 0 kernels complete
- **Stuck on**: `ds_bpermute` API for hierarchical inverse (Session 7)
- **Next**: Session 2 = skeleton + test harness; then build up stages 3-13
- **Don't**: claim speedups (none valid until Session 6+); add experimental code to `rtp_llm/`; deviate from cutedsl without checking source
- **Plan**: 13 sessions (~$3-5K), production-ready end of Session 13 on MI355X

---

**Authored**: 2026-04-21 housekeeping; comprehensively rewritten 2026-04-22
**Next action**: Session 2 (skeleton + test harness) — starts when user says go
