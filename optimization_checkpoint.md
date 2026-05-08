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

### 实验 8: Qwen3.5/Qwen3.6 Chunk-GDN shape inventory
- 状态: 有效
- 本机权重: `/root/Qwen3.5-27B/config.json` only; no local Qwen3.6 checkpoint/config found under `/root`
- 公开配置来源: Qwen3.5/Qwen3.6 public config snippets from Hugging Face/ModelScope search; values below are the linear-attention GDN shape fields relevant to FlyDSL
- RTP shape rule: global `(Hg,H,K,V)=(linear_num_key_heads, linear_num_value_heads, linear_key_head_dim, linear_value_head_dim)`; local runtime shape divides `Hg` and `H` by `attn_tp_size`
- Coverage requirement: Qwen3.5/Qwen3.6 models at 35B and below must support TP1 and TP2; larger models must support TP1, TP2, TP4, and TP8
- Coverage audit: after applying that TP policy, no additional runtime shapes were found beyond the 10-shape set below; current public Qwen3.6 coverage found is 27B and 35B-A3B, with no local or public Qwen3.6 `>35B` GDN shape found in this pass

| Size bucket | Family/model(s) | Global GDN `(Hg,H,K,V)` | group `H/Hg` | Required TP | Required runtime `(Hg,H,K,V)` | Current FlyDSL |
|---|---|---:|---:|---:|---:|---|
| <=35B | Qwen3.5-0.8B, Qwen3.5-2B | `(16,16,128,128)` | 1 | TP1 | `(16,16,128,128)` | yes |
| <=35B | Qwen3.5-0.8B, Qwen3.5-2B | `(16,16,128,128)` | 1 | TP2 | `(8,8,128,128)` | yes |
| <=35B | Qwen3.5-4B, Qwen3.5-9B, Qwen3.5-35B-A3B, Qwen3.6-35B-A3B | `(16,32,128,128)` | 2 | TP1 | `(16,32,128,128)` | yes |
| <=35B | Qwen3.5-4B, Qwen3.5-9B, Qwen3.5-35B-A3B, Qwen3.6-35B-A3B | `(16,32,128,128)` | 2 | TP2 | `(8,16,128,128)` | yes |
| <=35B | Qwen3.5-27B, Qwen3.6-27B | `(16,48,128,128)` | 3 | TP1 | `(16,48,128,128)` | yes |
| <=35B | Qwen3.5-27B, Qwen3.6-27B | `(16,48,128,128)` | 3 | TP2 | `(8,24,128,128)` | yes |
| >35B | Qwen3.5-122B-A10B, Qwen3.5-397B-A17B | `(16,64,128,128)` | 4 | TP1 | `(16,64,128,128)` | yes |
| >35B | Qwen3.5-122B-A10B, Qwen3.5-397B-A17B | `(16,64,128,128)` | 4 | TP2 | `(8,32,128,128)` | yes |
| >35B | Qwen3.5-122B-A10B, Qwen3.5-397B-A17B | `(16,64,128,128)` | 4 | TP4 | `(4,16,128,128)` | yes |
| >35B | Qwen3.5-122B-A10B, Qwen3.5-397B-A17B | `(16,64,128,128)` | 4 | TP8 | `(2,8,128,128)` | yes |

Shape-generalization implications:

- Unique runtime shape set to support: `(16,16,128,128)`, `(8,8,128,128)`, `(16,32,128,128)`, `(8,16,128,128)`, `(16,48,128,128)`, `(8,24,128,128)`, `(16,64,128,128)`, `(8,32,128,128)`, `(4,16,128,128)`, `(2,8,128,128)`.
- Current FlyDSL supports all 10 target runtime shapes after SG-V4; there is no unsupported shape left inside this Qwen3.5/Qwen3.6 target set.
- First kernel target should keep `K=V=128` and `BLOCK_DV=64`, but parameterize local `Hg`, local `H`, and `group=H//Hg` instead of hard-coding local `(Hg,H)=(8,32)`.
- Qwen3.5-27B and Qwen3.6-27B specifically require `(16,48,128,128)` under TP1 and `(8,24,128,128)` under TP2, so the existing FlyDSL validation must not route them to the current `(8,32)` specialization.
- RTP should keep shape-aware fallback to Triton for future shapes outside this target set until each FlyDSL specialization passes correctness with direct `ssm_states` store and prefix-cache.
- SG-V4 handoff for completed shape support lives in `/root/wenhua_code/flydsl/chunk_gdn_flydsl_workspace/megakernel/shape_generalization_next_handoff.md`.

### 实验 9: All-supported-shape performance sweep
- 状态: 有效；全矩阵已跑完，两个 Triton baseline 配置稳定 GPU memory fault，已记录为缺失 baseline。
- 目标: 对 SG-V4 已支持的 10 个 runtime shape，复用实验 6 和实验 7 的序列长度，分别比较 `507e Triton`、`Current normal` 和 `FlyDSL direct-store` 的 RTP operator 性能。
- Benchmark 改动: `tools/benchmarks/bench_flydsl_chunk_gdn_long.py` 增加 `--hg/--h/--k-dim/--v-dim`，按 shape 分配 `q/k=[1,input_len,Hg,128]`、`v=[1,input_len,H,128]`、`g/beta=[1,input_len,H]`、`ssm_states=[blocks,H,128,128]`、`initial_state=[1,H,128,128]`；输入改为稳定常数，避免 `torch.empty` 中异常值影响 profile。
- Profiling 规则: 只使用 `rocprofv3 --kernel-trace --stats` 产出性能结论；不使用 `do_bench`、手工 timing 或 `torch.cuda.Event` timing。
- 三条路径:
  - `507e Triton`: detached worktree at `507e404849065c2664d2440273cae30eb0393838`, `USE_FLYDSL` unset/0, runs `chunk_gated_delta_rule + store_ssm_state_to_block_map`;
  - `Current normal`: current branch, `USE_FLYDSL` unset/0, same Triton cache-state path;
  - `FlyDSL direct-store`: current branch, `USE_FLYDSL=1`, runs `chunk_gated_delta_rule_flydsl_with_cache_store`.
- 执行矩阵:
  - FlyDSL: 330 configs, 7740 operator groups parsed.
  - Current normal: 328 configs, 7726 operator groups parsed.
  - 507e Triton: 328 configs, 7726 operator groups parsed.
  - Missing baseline configs: `(16,48,128,128)` and `(16,64,128,128)` at long normal `input_len=200000`; both current and 507e Triton reproduce GPU memory access fault, while FlyDSL completes.
- Artifacts:
  - combined: `/tmp/kernel_opt_chunk_gdn_shape_generalization/exp9_profile_all_shapes/exp9_combined_results.csv`，包含 `model_tp` 列
  - FlyDSL: `/tmp/kernel_opt_chunk_gdn_shape_generalization/exp9_profile_all_shapes/flydsl/flydsl_all_results.csv`
  - Current normal: `/tmp/kernel_opt_chunk_gdn_shape_generalization/exp9_profile_all_shapes/current/current_all_results.csv`
  - 507e Triton: `/tmp/kernel_opt_chunk_gdn_shape_generalization/exp9_profile_all_shapes/triton_507e/triton_507e_all_results.csv`
  - parser/runner: `/tmp/kernel_opt_chunk_gdn_shape_generalization/exp9_all_shapes_driver.py`, `/tmp/kernel_opt_chunk_gdn_shape_generalization/exp9_parse_rocprof.py`, `/tmp/kernel_opt_chunk_gdn_shape_generalization/exp9_combine_results.py`
- Trace gate: FlyDSL parsed rows have `has_chunk_gated_delta_rule_fwd_kernel_h_blockdim64=no` and `has_store_ssm_state_to_block_map_kernel=no` for every measured config.
- Speedup summary, `baseline / FlyDSL`; `>1.0x` means FlyDSL faster:

| Shape | Model / TP | Long vs 507e | Long vs current | Short vs 507e | Short vs current | Missing baseline configs |
|---:|---|---:|---:|---:|---:|---|
| `(16,16,128,128)` | Qwen3.5 0.8B/2B TP1 | `1.208-1.262x` | `1.056-1.085x` | `0.697-1.209x` | `0.877-1.178x` | none |
| `(8,8,128,128)` | Qwen3.5 0.8B/2B TP2 | `0.848-0.933x` | `0.832-0.844x` | `0.678-0.919x` | `0.843-1.151x` | none |
| `(16,32,128,128)` | Qwen3.5 4B/9B/35B-A3B; Qwen3.6 35B-A3B TP1 | `2.129-2.201x` | `1.417-1.457x` | `0.775-2.130x` | `0.949-1.421x` | none |
| `(8,16,128,128)` | Qwen3.5 4B/9B/35B-A3B; Qwen3.6 35B-A3B TP2 | `1.225-1.284x` | `1.091-1.108x` | `0.685-1.223x` | `0.946-1.230x` | none |
| `(16,48,128,128)` | Qwen3.5/3.6 27B TP1 | `1.730-1.840x` | `1.245-1.264x` | `0.811-1.730x` | `0.972-1.243x` | normal `input_len=200000` |
| `(8,24,128,128)` | Qwen3.5/3.6 27B TP2 | `1.886-1.978x` | `1.234-1.283x` | `0.784-1.888x` | `0.932-1.234x` | none |
| `(16,64,128,128)` | Qwen3.5 122B-A10B/397B-A17B TP1 | `2.251-2.318x` | `1.461-1.487x` | `0.920-2.255x` | `0.908-1.471x` | normal `input_len=200000` |
| `(8,32,128,128)` | Qwen3.5 122B-A10B/397B-A17B TP2 | `2.148-2.229x` | `1.428-1.460x` | `0.772-2.146x` | `0.920-1.426x` | none |
| `(4,16,128,128)` | Qwen3.5 122B-A10B/397B-A17B TP4 | `1.215-1.300x` | `1.090-1.109x` | `0.681-1.215x` | `0.891-1.287x` | none |
| `(2,8,128,128)` | Qwen3.5 122B-A10B/397B-A17B TP8 | `0.876-0.932x` | `0.874-0.888x` | `0.650-0.914x` | `0.866-1.232x` | none |

- 长序列分类，使用实验 6 的 normal/prefix50 矩阵:
  - `507e Triton`: 原始 Triton baseline，长序列中通常最慢，尤其是 `H>=24` 的 model/TP 组合；仍保留外部 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 和 `store_ssm_state_to_block_map_kernel`。
  - `Current normal`: 当前分支 Triton 已明显优于 507e，但仍需要外部 `fwd_h + store_ssm_state_to_block_map`；对小 head-count 的 `(8,8)` 和 `(2,8)` 长序列比 FlyDSL 更快。
  - `FlyDSL direct-store`: 对 `H>=16` 的长序列基本为正收益，`(16,32)/(16,64)/(8,32)` 最明显；对 `(8,8)` 和 `(2,8)` 负收益，说明小 head-count 下 megakernel 固定开销超过 direct-store 融合收益。
- 短序列分类，使用实验 7 的 normal/prefix64k 矩阵:
  - `507e Triton`: 在很短 suffix，特别是 `input_len=1/17/63` 和小 head-count 时仍可能有优势；随着 input_len 增大，对中大 head-count 明显落后。
  - `Current normal`: 短序列最稳定，是小 head-count 的更稳 fallback；但对 `H>=24` 且 `input_len` 较大时，FlyDSL 通常追上或超过它。
  - `FlyDSL direct-store`: 消除了外部 `fwd_h/store`，但短序列 launch 和 megakernel 固定成本占比高；不适合无条件覆盖所有短序列，建议后续按 shape 和长度加阈值。

- Representative 200k rows, long-context units are ms:

| Shape | Model / TP | Scenario | Input len | 507e ms | Current ms | FlyDSL ms | vs 507e | vs current |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| `(16,16,128,128)` | Qwen3.5 0.8B/2B TP1 | normal | 200000 | 54.059 | 46.448 | 42.841 | `1.262x` | `1.084x` |
| `(16,16,128,128)` | Qwen3.5 0.8B/2B TP1 | prefix50 | 100032 | 27.009 | 23.232 | 21.512 | `1.255x` | `1.080x` |
| `(8,8,128,128)` | Qwen3.5 0.8B/2B TP2 | normal | 200000 | 36.316 | 34.022 | 40.692 | `0.892x` | `0.836x` |
| `(8,8,128,128)` | Qwen3.5 0.8B/2B TP2 | prefix50 | 100032 | 18.055 | 17.122 | 20.401 | `0.885x` | `0.839x` |
| `(16,32,128,128)` | Qwen3.5 4B/9B/35B-A3B; Qwen3.6 35B-A3B TP1 | normal | 200000 | 102.696 | 67.989 | 46.663 | `2.201x` | `1.457x` |
| `(16,32,128,128)` | Qwen3.5 4B/9B/35B-A3B; Qwen3.6 35B-A3B TP1 | prefix50 | 100032 | 50.978 | 33.669 | 23.398 | `2.179x` | `1.439x` |
| `(8,16,128,128)` | Qwen3.5 4B/9B/35B-A3B; Qwen3.6 35B-A3B TP2 | normal | 200000 | 52.629 | 45.457 | 41.044 | `1.282x` | `1.108x` |
| `(8,16,128,128)` | Qwen3.5 4B/9B/35B-A3B; Qwen3.6 35B-A3B TP2 | prefix50 | 100032 | 26.292 | 22.577 | 20.516 | `1.282x` | `1.100x` |
| `(16,48,128,128)` | Qwen3.5/3.6 27B TP1 | normal | 200000 | FAIL | FAIL | 88.371 | N/A | N/A |
| `(16,48,128,128)` | Qwen3.5/3.6 27B TP1 | prefix50 | 100032 | 78.217 | 55.941 | 44.269 | `1.767x` | `1.264x` |
| `(8,24,128,128)` | Qwen3.5/3.6 27B TP2 | normal | 200000 | 87.710 | 56.894 | 44.352 | `1.978x` | `1.283x` |
| `(8,24,128,128)` | Qwen3.5/3.6 27B TP2 | prefix50 | 100032 | 43.777 | 28.226 | 22.215 | `1.971x` | `1.271x` |
| `(16,64,128,128)` | Qwen3.5 122B-A10B/397B-A17B TP1 | normal | 200000 | FAIL | FAIL | 91.605 | N/A | N/A |
| `(16,64,128,128)` | Qwen3.5 122B-A10B/397B-A17B TP1 | prefix50 | 100032 | 106.494 | 68.106 | 45.950 | `2.318x` | `1.482x` |
| `(8,32,128,128)` | Qwen3.5 122B-A10B/397B-A17B TP2 | normal | 200000 | 102.632 | 67.206 | 46.036 | `2.229x` | `1.460x` |
| `(8,32,128,128)` | Qwen3.5 122B-A10B/397B-A17B TP2 | prefix50 | 100032 | 50.928 | 33.271 | 23.024 | `2.212x` | `1.445x` |
| `(4,16,128,128)` | Qwen3.5 122B-A10B/397B-A17B TP4 | normal | 200000 | 52.415 | 44.720 | 40.326 | `1.300x` | `1.109x` |
| `(4,16,128,128)` | Qwen3.5 122B-A10B/397B-A17B TP4 | prefix50 | 100032 | 26.035 | 22.278 | 20.237 | `1.286x` | `1.101x` |
| `(2,8,128,128)` | Qwen3.5 122B-A10B/397B-A17B TP8 | normal | 200000 | 34.830 | 33.148 | 37.353 | `0.932x` | `0.887x` |
| `(2,8,128,128)` | Qwen3.5 122B-A10B/397B-A17B TP8 | prefix50 | 100032 | 17.377 | 16.573 | 18.712 | `0.929x` | `0.886x` |

## 已放弃的方向
- None

## 当前状态
- 当前 Phase: Phase 5 validation/performance sweep
- 当前 sub-skill: final-validation
- 下一步: tiny suffix `input_len=1/17` 和 Qwen3.5 0.8B/2B shapes 暂不作为下一轮重点。P0 优化 Qwen3.5 122B-A10B/397B-A17B TP8 `(2,8,128,128)` long-context 负收益；P1 定位并优化 Qwen3.5 9B TP2 `(8,16,128,128)` 的长序列和非 tiny 短序列，短序列先看 `input_len=512`，再 spot-check `2048/8192`。另需单独调查 Triton baseline 在 `(16,48)/(16,64)` long normal `input_len=200000` 的 memory fault。
