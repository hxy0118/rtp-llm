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

## 已放弃的方向
- None

## 当前状态
- 当前 Phase: Phase 4 integration
- 当前 sub-skill: integration-validation
- 下一步: run rocprofv3 end-to-end comparison under the target deployment command, then continue MI355 warp-specialization from the A-only direct-store baseline
