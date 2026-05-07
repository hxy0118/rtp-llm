# 实验 Checkpoint: RTP Chunk-GDN FlyDSL opt-in integration

## 环境
- framework: rtp-llm, local repo `/root/RTP-LLM/github-opensource`
- hardware: AMD MI308X/MI300X target for current FlyDSL megakernel; MI355 follow-up tracked separately
- model: Qwen3-Next Chunk-GDN path
- key env vars: `USE_FLYDSL=1` enables the FlyDSL Chunk-GDN path; default unset/0 keeps Triton

## 用户提供的标准命令
### 服务启动命令
TBD

### Benchmark 命令
TBD

## 基线数据
| 指标 | 值 | 测量条件 |
|------|---|---------|
| correctness | TBD | Triton default vs FlyDSL opt-in |
| performance | TBD | same RTP invocation, `USE_FLYDSL` toggled |

## 实验记录
### 实验 1: Chunk-GDN FlyDSL opt-in path
- 状态: 有效
- 改动: `rtp_llm/models_py/triton_kernels/fla/chunk.py` plus FlyDSL megakernel modules
- 结果: py_compile passed; RTP default-vs-FlyDSL correctness passed for B=1 `T=1/17/63/64/65/127/128/129`, dense `B=2,T=65`, varlen `cu=[0,17,80,209]`, and int32 varlen cu with `cos_o > 0.999`, `cos_ht > 0.999`, and matching `h` shape
- 结论: default remains Triton; `USE_FLYDSL=1` routes supported AMD Chunk-GDN shapes through FlyDSL for `O/final_state`, while Triton still produces `h` for RTP SSM block-state persistence

### 实验 2: FlyDSL direct `ssm_states` store
- 状态: 有效
- 改动: FlyDSL megakernel 增加 optional cache-store 变体；Qwen3Next prefill 在 `USE_FLYDSL=1` 且存在 `ssm_states` 时直接传入 `prefix_lengths/block_map/ssm_states/seq_size_per_block`，跳过额外 Triton `fwd_h` 和 `store_ssm_state_to_block_map`
- store 语义: final chunk always writes current `h_acc`; middle chunks only write when `(chunk + 1) * 64 % seq_size_per_block == 0`, matching RTP Triton store kernel; `block_idx <= 0` remains sentinel skip
- 正确性: direct-store vs default Triton store passed for `T=1/17/63/64/65/127/128/129/1000/1025`, prefix-cache case `prefix=128,T=65`, varlen `lens=[17,63,129], prefix=[128,0,256]`, and `ssm_states` dtype `bf16/fp32`; observed `cos_o/cos_ht/cos_ssm > 0.999`
- 性能 smoke: `rocprofv3 --kernel-trace --stats`, synthetic `T=1024,iters=6`, showed total kernel dispatch time `3.43ms -> 2.72ms` and kernel dispatch count `117 -> 99`; direct-store removes `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` and `store_ssm_state_to_block_map_kernel`, while megakernel avg increases `162us -> 175us` from the fused state writes

### 实验 3: Systematic RTP operator profile vs Triton
- 状态: 有效
- 命令形态: `rocprofv3 --kernel-trace --stats` over `/tmp/system_flydsl_vs_triton.py`, `warmup=10,iters=50`, p50 over 60 grouped calls
- 范围: RTP Python Chunk-GDN operator path with cache-state store; not full model/server E2E

| T | Triton p50 us | FlyDSL hybrid p50 us | FlyDSL direct p50 us | direct vs Triton | direct vs hybrid |
|---:|---:|---:|---:|---:|---:|
| 128 | 154.2 | 174.0 | 143.6 | 1.07x | 1.21x |
| 1024 | 437.4 | 520.5 | 401.3 | 1.09x | 1.30x |
| 1025 | 444.0 | 576.9 | 465.7 | 0.95x | 1.24x |
| 4097 | 1512.1 | 1698.1 | 1296.9 | 1.17x | 1.31x |
| 16384 | 6033.5 | 6751.3 | 4905.3 | 1.23x | 1.38x |

- 对齐结论: direct-store vs hybrid 的收益 `1.21-1.38x`，与之前单算子 smoke `~1.26x` 方向一致；主要来自移除 extra Triton `fwd_h` 和 `store_ssm_state_to_block_map` launches。
- 与 Triton 对比: 长序列和 aligned path 为正收益；`T=1025` whole-tail fallback 略慢，原因是 guarded tail-safe megakernel 更重且当前 direct helper 仍通过 `chunk_gated_delta_rule_fwd_intra` 计算了不再使用的 `w/u`。
- 下一步性能项: 增加 A-only KKT/solve path 或拆出不返回 `w/u` 的 intra helper，避免 FlyDSL direct-store 下重复/无用的 `recompute_w_u_fwd_kernel`。

### 实验 4: FlyDSL direct-store A-only front half
- 状态: 有效
- 改动: `chunk_gated_delta_rule_fwd_intra` 拆出 `chunk_gated_delta_rule_fwd_intra_a_only`；FlyDSL direct-store path 只运行 KKT/solve 生成 `A`，不再计算 megakernel 已内部重算的 `w/u`
- 正确性: direct-store vs default Triton store 重新通过 `T=1/17/63/64/65/127/128/129/1000/1025`，prefix-cache `prefix=128,T=65`，varlen `lens=[17,63,129], prefix=[128,0,256]`，`ssm_states` bf16/fp32，以及 split-tail `T=4097`
- profile 证据: `rocprofv3 --kernel-trace --stats` top kernels 中 `recompute_w_u_fwd_kernel` 不再出现；旧 direct 16k 该 kernel 为 `1368.8us/call`

| T | Triton p50 us | old direct p50 us | A-only direct p50 us | A-only vs Triton | A-only vs old direct |
|---:|---:|---:|---:|---:|---:|
| 128 | 154.2 | 143.6 | 126.7 | 1.22x | 1.13x |
| 1024 | 437.4 | 401.3 | 306.7 | 1.43x | 1.31x |
| 1025 | 444.0 | 465.7 | 376.4 | 1.18x | 1.24x |
| 4097 | 1512.1 | 1296.9 | 949.8 | 1.59x | 1.37x |
| 16384 | 6033.5 | 4905.3 | 3540.5 | 1.70x | 1.39x |

- 结论: 之前 16k 只有 `1.23x` 是 RTP direct path 仍支付了一个无用的 `recompute_w_u_fwd_kernel`；A-only 后同一系统 operator profile 回到 `1.70x`，高于早前单算子 `1.49x`，因为 direct-store 还额外移除了 RTP cache-state store 相关 launch

### 实验 5: Direct `ssm_states` write overhead
- 状态: 有效
- 命令形态: `rocprofv3 --kernel-trace --stats` over `/tmp/ssm_store_overhead_profile.py`, `warmup=10,iters=50`, p50 over 60 grouped calls
- 对比方式: 两条路径都走 `fused_l2norm_qk -> chunk_local_cumsum -> A-only kkt_solve -> FlyDSL megakernel`；唯一差别是 megakernel 是否传入 `prefix_lengths/block_map/ssm_states/seq_size_per_block`

| T | no-store p50 us | ssm-store p50 us | overhead us | overhead % | megakernel no/store avg us |
|---:|---:|---:|---:|---:|---:|
| 128 | 114.8 | 125.8 | +10.9 | +9.53% | 25.8/28.7 |
| 1024 | 288.2 | 310.6 | +22.4 | +7.77% | 158.6/173.0 |
| 1025 | 358.5 | 368.6 | +10.0 | +2.79% | 220.8/240.3 |
| 4097 | 865.6 | 948.5 | +82.9 | +9.58% | 313.9/356.2 |
| 16384 | 3314.9 | 3548.4 | +233.4 | +7.04% | 2477.5/2711.5 |

- profile 证据: store path top kernels 中无 `recompute_w_u_fwd_kernel`、无 Triton `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`、无 `store_ssm_state_to_block_map_kernel`
- 结论: direct `ssm_states` 写入确实增加 megakernel/整条 operator 时间，长序列约 `7-10%`；但这仍比生产路径重新引入外部 Triton `fwd_h + store_ssm_state_to_block_map` 更划算，且保持 cache-state 语义完整

### 实验 6: Long-context RTP operator profile to 200k
- 状态: 有效
- 命令形态: `rocprofv3 --kernel-trace --stats` over `tools/benchmarks/bench_flydsl_chunk_gdn_long.py`, `iters=5`, `seq_size_per_block=64`, `ssm_states=bf16`, B=1, `(Hg,H,K,V)=(8,32,128,128)`
- Triton 基线: `Current Triton` is this branch's optimized Triton; `507e Triton` is detached worktree at `507e404849065c2664d2440273cae30eb0393838`
- 路径定义: both paths include `load_initial_state_from_block_map`; Triton path runs `chunk_gated_delta_rule` plus `store_ssm_state_to_block_map`; FlyDSL path runs `chunk_gated_delta_rule_flydsl_with_cache_store` with direct `h_acc -> ssm_states` store
- prefix-cache 场景: total context length fixed; `prefix_len ~= total/2` and block-aligned to 64, current `input_len = total - prefix_len`

| Scenario | Total len | Prefix len | Input len | 507e Triton ms | Current Triton ms | FlyDSL ms | FlyDSL vs 507e | FlyDSL vs current |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 16,384 | 0 | 16,384 | 8.384 | 5.473 | 3.828 | 2.19x | 1.43x |
| normal | 65,536 | 0 | 65,536 | 33.477 | 21.742 | 15.131 | 2.21x | 1.44x |
| normal | 131,072 | 0 | 131,072 | 67.028 | 43.624 | 30.164 | 2.22x | 1.45x |
| normal | 200,000 | 0 | 200,000 | 102.441 | 67.489 | 45.940 | 2.23x | 1.47x |
| prefix50 | 16,384 | 8,192 | 8,192 | 4.202 | 2.830 | 1.966 | 2.14x | 1.44x |
| prefix50 | 65,536 | 32,768 | 32,768 | 16.816 | 10.985 | 7.568 | 2.22x | 1.45x |
| prefix50 | 131,072 | 65,536 | 65,536 | 33.505 | 21.765 | 15.239 | 2.20x | 1.43x |
| prefix50 | 200,000 | 99,968 | 100,032 | 51.146 | 33.369 | 22.924 | 2.23x | 1.46x |

- profile 证据: 200k normal Triton top kernels include `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` `21.98ms/call`, `recompute_w_u_fwd_kernel` `17.65ms/call`, `store_ssm_state_to_block_map_kernel` `3.36ms/call`; FlyDSL top kernels include `megakernel_fn_0` `36.44ms/call` and no external Triton `fwd_h/store_ssm_state_to_block_map`
- 原始 Triton profile 证据: 200k normal `507e` top kernels include `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` `39.70ms/call`, `chunk_fwd_kernel_o` `25.97ms/call`, `recompute_w_u_fwd_kernel` `18.05ms/call`, and `store_ssm_state_to_block_map_kernel` `3.33ms/call`
- 结论: with RTP cache-state writes enabled, long-context FlyDSL direct-store remains stable at about `1.43-1.47x` vs current optimized Triton and about `2.14-2.23x` vs `507e` original Triton through 200k total/context length; prefix-cache follows the same ratio because both paths compute only current suffix while loading prefix initial state once

### 实验 7: Short-sequence dispatch threshold sweep
- 状态: 有效
- 命令形态: `rocprofv3 --kernel-trace --stats` over `tools/benchmarks/bench_flydsl_chunk_gdn_long.py`; short inputs use `iters=30`, `2048/4096` use `iters=15`, `8192` uses `iters=8`; `seq_size_per_block=64`, `ssm_states=bf16`, B=1
- 场景: normal uses `prefix=0`; prefix-cache uses fixed `prefix=65536` and varies suffix `input_len`; `507e Triton` is detached worktree at `507e404849065c2664d2440273cae30eb0393838`

| Input len | 507e normal us | Current normal us | FlyDSL normal us | FlyDSL vs 507e | FlyDSL vs current | 507e prefix64k us | Current prefix64k us | FlyDSL prefix64k us | FlyDSL vs 507e | FlyDSL vs current |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 108.661 | 125.636 | 124.136 | 0.88x | 1.01x | 106.955 | 125.553 | 116.311 | 0.92x | 1.08x |
| 17 | 124.407 | 136.439 | 128.392 | 0.97x | 1.06x | 176.374 | 138.386 | 127.604 | 1.38x | 1.08x |
| 63 | 131.658 | 165.132 | 135.315 | 0.97x | 1.22x | 132.920 | 164.500 | 134.227 | 0.99x | 1.23x |
| 64 | 137.487 | 164.770 | 124.957 | 1.10x | 1.32x | 138.611 | 158.287 | 122.122 | 1.14x | 1.30x |
| 65 | 153.167 | 174.767 | 147.391 | 1.04x | 1.19x | 153.602 | 177.720 | 144.519 | 1.06x | 1.23x |
| 127 | 162.782 | 174.758 | 154.371 | 1.05x | 1.13x | 162.470 | 166.462 | 150.883 | 1.08x | 1.10x |
| 128 | 164.714 | 176.294 | 137.994 | 1.19x | 1.28x | 164.198 | 169.472 | 134.296 | 1.22x | 1.26x |
| 256 | 222.939 | 203.475 | 168.750 | 1.32x | 1.21x | 220.192 | 202.136 | 171.682 | 1.28x | 1.18x |
| 512 | 362.203 | 285.574 | 228.222 | 1.59x | 1.25x | 351.165 | 291.051 | 226.087 | 1.55x | 1.29x |
| 1024 | 590.561 | 433.394 | 339.649 | 1.74x | 1.28x | 591.773 | 435.945 | 343.219 | 1.72x | 1.27x |
| 2048 | 1083.682 | 769.450 | 575.830 | 1.88x | 1.34x | 1083.517 | 761.202 | 580.860 | 1.87x | 1.31x |
| 4096 | 2088.418 | 1431.894 | 1026.279 | 2.03x | 1.40x | 2268.362 | 1434.592 | 1026.546 | 2.21x | 1.40x |
| 8192 | 4206.173 | 2802.621 | 1953.469 | 2.15x | 1.43x | N/A | N/A | N/A | N/A | N/A |

- 结论: for the current optimized Triton fallback, no length threshold is needed; FlyDSL is at least break-even from `input_len=1`. Against `507e` original Triton, tiny suffixes (`input_len=1/17/63`) can still favor original Triton, while `input_len>=64` favors FlyDSL in both normal and prefix-cache rows except measurement noise. Production decision remains capability/shape gating under the current branch; if rolling back to `507e` Triton as fallback, use a conservative `input_len >= 64` FlyDSL cutoff.

## 已放弃的方向
- None

## 当前状态
- 当前 Phase: Phase 4 integration
- 当前 sub-skill: integration-validation
- 下一步: run rocprofv3 end-to-end comparison under the target deployment command if service commands are provided, then continue MI355 warp-specialization from the A-only direct-store baseline
