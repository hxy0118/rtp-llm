# chunk_gdn_perf — FlyDSL building-block reference

Validated FlyDSL primitives + perf data from prior sessions. These are
**reference patterns** for the 13-session megakernel implementation
(see `../flydsl_chunk_gdn_plan.md`), NOT production code.

Production code lives in `/root/rtp-llm/rtp_llm/models_py/triton_kernels/fla/`
(currently pure FLA Triton — see constraint C4 in `../flydsl_chunk_gdn_checkpoint.md` §8.5).

## Test files (kept for reference)

| File | Pattern | Used by Session |
|---|---|---|
| `test_flydsl_alu_mask_beta.py` | ALU primitives: mask, β scale, decay multiply | 5, 6, 9 |
| `test_flydsl_gemm1_kk.py` | GEMM 1: K @ K^T (kk) | 9 |
| `test_flydsl_gemm12_kk_qk.py` | GEMMs 1+2: kk + qk fused | 6, 9 |
| `test_flydsl_gemm3_ks.py` | GEMM 3: K @ state (state-as-B operand) | 3, 4 |
| `test_flydsl_gemm4567.py` | GEMMs 4-7: QS + NV + qkv + dS | 4, 6, 7 |
| `test_flydsl_chunk_gdn_v2.py` | State-as-B-operand pattern (bf16 LDS staging) | 3 |
| `test_flydsl_persistent_mfma_state.py` | Register-resident state (frag_C across NT chunks) | 4 |
| `test_flydsl_persistent_gemm7.py` | dS = K^T @ V (GEMM 7), persistent across chunks | 4 |
| `test_flydsl_v9_runtime_multicta.py` | Simplest production-scale runtime NT scf.for + multi-CTA | 2, 4 |
| `test_flydsl_v11_full_multicta.py` | Production-scale 16 atoms/warp at DK=DV=64 | 4, 11 |
| `test_flydsl_v6_inverse_stage1.py` | Hierarchical inverse stage-1 (BLOCKED on `ds_bpermute` API) | 7 |

## Reference data

- `rocprof_tp{1,2,2_v2}_baseline_*.csv` — FLA baseline kernel traces from rocprof on MI308X (TP=1, TP=2 variants). Source for the per-kernel timing in checkpoint §10.

## How to use

When implementing a new session:
1. Find the closest reference pattern from the table above
2. Copy + adapt — don't re-derive lane mappings / scf.for carry / byte offsets from scratch
3. Match precision against the same tolerance the reference used (see comments in each file)

## What was deleted (and why)

~28 files removed during 2026-04-22 cleanup:
- **MVPs / drop-ins** — algorithm-incorrect prototypes that would mislead future work (`*_mvp.py`, `*_final.py`, `*_real_dropin.py`, `test_end_to_end_dispatch.py`)
- **Iteration deadends** — `v3/v4/v5/v7/v8/v10*` superseded by `v9` and `v11`
- **FlyDSL hello-worlds** — `vecadd.py`, `lds_sanity.py`, `mfma_bf16.py` (one-time API smoke tests)
- **Tests of deleted production code** — `test_rtp_integration.py`, `test_prod_scale_perf.py`, `test_vs_fla_direct.py` referenced the now-deleted `flydsl_megakernel.py` drop-in
- **Stale benchmarks** — `bench_chunk_gdn.py`, `run_phase21_all.py`
- **Stale reports** — `FINAL_REPORT.md`, `FINAL_COMPARISON.md` contained retired "4× speedup" claims (per C2) and referenced now-deleted production code. Current status lives in `../TODAY_RETROSPECTIVE.md`.

`.gitignore` excludes Python coredumps + `__pycache__`.
