# Kernel Opt Session Memory

## Target
- platform: MI308X / MI300X now, MI355X follow-up
- arch: CDNA3 now, CDNA4 follow-up
- framework: FlyDSL integrated into RTP-LLM
- dtype: bf16 inputs, fp32 hidden state accumulation
- kernel type: Qwen3-Next Chunk-GDN fused forward

## Inputs
- shapes: RTP production Chunk-GDN shape `(Hg,H,K,V)=(8,32,128,128)`
- reference: Triton FLA Chunk-GDN path in `rtp_llm/models_py/triton_kernels/fla`
- correctness threshold: `cos_o > 0.999`, `cos_ht > 0.999`; cache state must match default Triton store

## Hard Constraints
- default RTP path remains Triton
- FlyDSL path is opt-in via `USE_FLYDSL=1`
- support arbitrary `T`, including tail chunks where `T % 64 != 0`
- no pad-to-64 wrapper for production path
- direct `ssm_states` store must preserve prefix-cache/final-state semantics
- avoid adding per-token hot-loop overhead; cache writes only at chunk/block boundaries

## Tools
- correctness: local RTP tests / standalone comparison scripts
- profiling: `rocprofv3` for performance evidence when moving beyond correctness

## Stop Conditions
- direct FlyDSL cache-store path matches Triton `h + final_state -> ssm_states`
- tail cases and prefix-cache cases pass before MI355 warp-specialization work resumes

## Iteration Log
- V1 | RTP integration | `USE_FLYDSL=1` routes O/final_state through FlyDSL but still uses Triton `fwd_h` plus cache-store kernel | correct but leaves extra launches
- V2 | implemented | fused RTP `ssm_states` writes into FlyDSL h update path | correctness passed for tail, prefix-cache, varlen, and bf16/fp32 state-cache smoke cases
- V3 | measured | systematic `rocprofv3` RTP operator sweep vs Triton | direct-store p50 speedup vs Triton: `1.07x(T=128)`, `1.09x(T=1024)`, `0.95x(T=1025 whole-tail)`, `1.17x(T=4097 split-tail)`, `1.23x(T=16384)`; direct-store vs prior hybrid: `1.21-1.38x`
- V4 | optimized | split `chunk_gated_delta_rule_fwd_intra_a_only` so FlyDSL direct-store no longer computes unused `w/u` | correctness passed for tail/prefix/varlen/bf16-state cases; `rocprofv3` confirms `recompute_w_u_fwd_kernel` is gone and direct-store p50 speedup vs Triton is `1.22x(T=128)`, `1.43x(T=1024)`, `1.18x(T=1025)`, `1.59x(T=4097)`, `1.70x(T=16384)`
- V5 | measured | isolated cost of writing `h_acc` to RTP `ssm_states` inside the FlyDSL megakernel | direct-store adds `+10.9us/+9.5%(T=128)`, `+22.4us/+7.8%(T=1024)`, `+10.0us/+2.8%(T=1025)`, `+82.9us/+9.6%(T=4097)`, `+233.4us/+7.0%(T=16384)` versus the same A-only FlyDSL path without cache-state writes; profile confirms no Triton `fwd_h`, no `store_ssm_state_to_block_map`, and no `recompute_w_u`
