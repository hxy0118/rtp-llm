# Chunk-GDN Shape 泛化优化

Date: 2026-05-08

Source session record: `/tmp/kernel_opt_chunk_gdn_shape_generalization/memory.md`.

This document records the current FlyDSL Chunk-GDN shape-generalization and
small-H performance optimization state in the RTP repo. The canonical workspace
mirror is:

```text
/root/wenhua_code/flydsl/chunk_gdn_flydsl_workspace/megakernel/shape_generalization_optimization.md
```

## Scope

- Target framework: FlyDSL Chunk-GDN fused forward megakernel on MI308X/CDNA3.
- Production integration stays opt-in through `USE_FLYDSL=1`; default RTP
  remains Triton.
- Performance conclusions must come from `rocprofv3`. Do not use `do_bench`,
  manual timing, or `torch.cuda.Event` timing as evidence.
- Do not solve P0/P1 by removing shapes from FlyDSL routing. P0/P1 stay FlyDSL
  and are optimized at kernel level.
- Tiny suffixes `input_len=1/17` and Qwen3.5 0.8B/2B shapes were explicitly
  out of scope for this optimization pass.

## Supported Shape Set

All Qwen3.5/Qwen3.6 target runtime shapes with `K=V=128` are enabled:

```text
(16,16,128,128)
(8,8,128,128)
(16,32,128,128)
(8,16,128,128)
(16,48,128,128)
(8,24,128,128)
(16,64,128,128)
(8,32,128,128)
(4,16,128,128)
(2,8,128,128)
```

Future shapes outside this set still need correctness, prefix-cache, varlen, and
direct `ssm_states` validation before being enabled.

## Current Implementation

RTP files:

- `rtp_llm/models_py/triton_kernels/fla/flydsl_chunk_gdn_mi308x.py`
- `rtp_llm/models_py/triton_kernels/fla/flydsl_chunk_gdn_mi308x_fast.py`
- `rtp_llm/models_py/triton_kernels/fla/flydsl_chunk_gdn_mi308x_bdv32_fast.py`

The default/hot aligned path keeps `BLOCK_DV=64`, preserving the original
`(8,32,128,128)` path. The new BDV32 fast module is selected only for:

```text
(2,8,128,128)
(8,16,128,128)
```

The tail-safe path remains on the original V2 wrapper. BDV32 uses a distinct LDS
global symbol, `megakernel_mi300x_v2_bdv32_smem`, to avoid mixed-module LDS
symbol collisions with the BDV64 module.

## Profiling Evidence

`rocprofv3 --kernel-trace` showed that the old BDV64 megakernel used about
`62976` bytes LDS and `VGPR=92`. For long sequence length 200k, the workgroup
counts were:

| Shape | Old BDV64 workgroups |
|---|---:|
| `(2,8,128,128)` | 16 |
| `(8,16,128,128)` | 32 |
| `(8,32,128,128)` | 64 |

The BDV32 small-H module uses about `54784` bytes LDS and `VGPR=56`, and doubles
V-axis workgroups for P0/P1. This follows the small-grid profiling guidance:
increase grid parallelism first when the launch has too few workgroups.

## Performance Result

All numbers below are from `rocprofv3` records in
`/tmp/kernel_opt_chunk_gdn_shape_generalization/`.

| Target | Result |
|---|---|
| P0 `(2,8,128,128)` long sweep | `1.239-1.260x` over old FlyDSL; now `1.094-1.119x` vs current Triton |
| P0 200k | old FlyDSL `37.413 ms` -> BDV32 FlyDSL `29.730 ms` |
| P1 `(8,16,128,128)` long sweep | `1.237-1.250x` over old FlyDSL; now `1.355-1.385x` vs current Triton |
| P1 512 | `220.560 us` -> `197.201 us` |
| P1 2048 | `516.320 us` -> `428.640 us` |
| P1 8192 | `1742.204 us` -> `1400.903 us` |
| Hot `(8,32,128,128)` 200k | final `46.052 ms` vs old `46.061 ms`; no meaningful regression |

## Validation

- `python3 -m py_compile` passed for modified RTP and workspace files.
- Workspace correctness passed for `(2,8)`, `(8,16)`, and `(8,32)` at
  `T=129/512/2048`.
- RTP direct cache-store correctness passed for `(2,8)`, `(8,16)`, and
  `(8,32)` at `T=129/512`.
- Use stable `g=F.logsigmoid(...)` and normalized `k` in smoke tests; raw random
  `g` can create huge values and misleading correctness failures.

## Remaining Validation Before Wider Claims

- Prefix50 direct-store correctness for P0/P1.
- Varlen lens `[17,63,129]` if production enablement requires revalidation.
- `ssm_states` dtype `fp32` smoke for the BDV32 path.
- Any MI355/gfx950 claim must be reprofiled on MI355 with `rocprofv3`; do not
  assume MI308 BDV32 results transfer unchanged.
