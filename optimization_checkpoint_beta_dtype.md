# 实验 Checkpoint: GDN beta dtype 对齐 SGL (bf16 → fp32)

## 背景

RTP-LLM 的 `fused_gdn_gating` 把 `beta_output` 按 `b.dtype` 分配（实际 bf16），导致
`sigmoid(b)` 在 kernel 内 fp32 算后被 cast 回 bf16，进入下游 chunk / recurrent
kernel 前丢失精度（bf16 mantissa 7 bit，绝对精度 ~7.8e-3）。SGL 的
`fused_gdn_gating_prefill.py` 与 FlashQLA benchmark 都把 beta 保留为 fp32。

参考 `/root/sglang/python/sglang/srt/layers/attention/fla/fused_gdn_gating_prefill.py`
与 `/root/FlashQLA/benchmark/bench_gated_delta_rule.py:209-211`。

## 环境
- framework: rtp-llm, local repo `/root/RTP-LLM/github-opensource`
- hardware: AMD MI300X / MI308X target
- model: Qwen3-Next Chunk-GDN path (prefill + decode 共享 `fused_gdn_gating`)
- key env vars: `USE_FLYDSL=1` 启用 FlyDSL Chunk-GDN 路径；不设置走 Triton reference + recurrent

## 改动文件

| 文件 | 改动 |
|---|---|
| `rtp_llm/models_py/triton_kernels/fla/gdn_gating.py` | L78 `dtype=b.dtype` → `dtype=torch.float32`；kernel 内 store 自动跟 fp32，不再 cast |
| `rtp_llm/models_py/triton_kernels/fla/chunk.py` | `is_flydsl_chunk_gdn_shape_supported` 与 `_validate_flydsl_chunk_gdn_inputs`：接受 fp32 beta；`chunk_gated_delta_rule_flydsl_with_cache_store` 在 validate 前 cast fp32 → bf16，因为 FlyDSL megakernel 内部 lds_beta 写死 T.bf16 |

未改动（已天然支持 fp32 beta）：
- `chunk_gated_delta_rule` reference 路径：下游 `chunk_fwd_intra/h/o` 内部 `tl.load(beta).to(tl.float32)`
- `fused_recurrent_gated_delta_rule` decode 路径：`fused_recurrent.py:145/147` `b_beta = tl.load(p_beta).to(tl.float32)`

## 用户提供的标准命令

### 性能 bench 方式（待用户确认）
- 候选 A: 算子级 — `tools/benchmarks/bench_flydsl_chunk_gdn_long.py`，rocprofv3
- 候选 B: 服务级 — 拉起 Qwen3.5-9B server，sgl-bench / SGL benchmark

### Shape 选择（待用户确认）
- 默认 Qwen3.5-9B TP1 `(Hg=16, H=32, K=128, V=128)`
- Decode 1 token + Prefill T=4096/16384/65536

## 基线数据
| 指标 | 改前 (bf16-beta) | 改后 (fp32-beta) | 测量条件 |
|---|---|---|---|
| beta 精度 vs SGL | max diff ~7.8e-3 (bf16 ulp) | 0 (位级一致) | 同输入 |
| reference prefill us | TBD | TBD | rocprof, T=4k/16k/64k |
| FlyDSL prefill us | TBD | TBD | rocprof, T=4k/16k/64k |
| decode recurrent us | TBD | TBD | rocprof, T=1 |

## 实验记录

### 实验 1: 代码改动 + py_compile
- 状态: 有效
- 改动:
  - `gdn_gating.py:78` dtype b.dtype → torch.float32
  - `chunk.py is_flydsl_chunk_gdn_shape_supported`: 允许 fp32 beta
  - `chunk.py _validate_flydsl_chunk_gdn_inputs`: 允许 fp32 beta
  - `chunk.py chunk_gated_delta_rule_flydsl_with_cache_store` wrapper: validate 前 cast fp32 → bf16
- py_compile passed

### 实验 2: 三路径 sanity 跑通
- 状态: 有效
- 脚本: `/tmp/sanity_gdn_beta_dtype.py`
- 命令: `USE_FLYDSL=1 /opt/conda310/bin/python3 /tmp/sanity_gdn_beta_dtype.py`
- 结果:
  | 检查 | 结果 |
  |---|---|
  | `fused_gdn_gating` 输出 dtype | fp32 ✓ |
  | beta vs `b.sigmoid().float()` max diff | 5.96e-08（fp32 ulp 级，无 bf16 截断） |
  | reference Triton chunk: fp32 vs bf16 beta cos | 0.999994 |
  | decode recurrent: fp32 vs bf16 beta cos | 1.000000 |
  | FlyDSL wrapper: fp32-in 经 cast vs bf16-in max diff | 0.0（bit-exact） |
- 结论: 三路径全部接受 fp32-beta；FlyDSL wrapper cast 路径无副作用

### 实验 3: 数值对齐 SGL
- 状态: 有效
- 脚本: `/tmp/cross_check_rtp_vs_sgl.py`（importlib 直接 load SGL kernel 文件，绕过 SGL package init）
- 命令: `/opt/conda310/bin/python3 /tmp/cross_check_rtp_vs_sgl.py`
- 结果:

| 路径 | shape | beta max diff | g max diff | 结论 |
|---|---|---|---|---|
| Prefill (T=1, H=32) | RTP vs SGL | 0.000e+00 | 0.000e+00 | **bit-exact** |
| Prefill (T=64, H=32) | RTP vs SGL | 0.000e+00 | 0.000e+00 | **bit-exact** |
| Prefill (T=256, H=32) | RTP vs SGL | 0.000e+00 | 0.000e+00 | **bit-exact** |
| Prefill (T=4096, H=32) | RTP vs SGL | 0.000e+00 | 0.000e+00 | **bit-exact** |
| Decode (B=4, H=32) | RTP vs SGL | **1.92e-3** | 0.000e+00 | SGL decode 有 bug |

- **意外发现**: SGL 的 decode kernel `fused_gdn_gating.py:39` 写法是
  `tl.store(beta_output + off, blk_beta_output.to(b.dtype.element_ty))` — 分配是 fp32 但
  store 时 cast 到 `b.dtype`(bf16) 再隐式 promote 回 fp32，导致 sigmoid 输出被截断。
  RTP 改后的 fp32-beta 比 SGL decode 路径**更准确**。可考虑上游给 SGL 报这个 bug。

### 实验 4: 性能 bench (rocprofv3 算子级)

- shape: Qwen3.5-9B TP1 hot path `(Hg=16, H=32, K=128, V=128)`，B=1
- 脚本: `/tmp/bench_beta_dtype/run_bench.py`（rocprof wrapper `/tmp/bench_beta_dtype/run_rocprof.sh`）
- 命令: `USE_FLYDSL=1 rocprofv3 --kernel-trace --stats -f csv -- python run_bench.py --path {triton|flydsl|decode} --T <int> --beta-dtype {bf16|fp32} --warmup N --iters M`
- 配置: warmup=5 iters=20（decode 用 warmup=10 iters=60；flydsl T=4096 加跑 iters=60 验证噪声）
- bf16 配置模拟改前行为：`beta = fused_gdn_gating(...).to(torch.bfloat16)`；fp32 配置直接用改后输出
- 数据：每个 kernel 的 AverageNs（per-call）求和 = single-iter wall time

| Path | T | bf16 (us) | fp32 (us) | fp32/bf16 | delta us | 结论 |
|---|---:|---:|---:|---:|---:|---|
| triton  | 4096  | 1170.75   | 1150.04   | 0.982 | -20.71  | 无差异 (噪声) |
| triton  | 16384 | 4533.31   | 4477.96   | 0.988 | -55.35  | 无差异 (~1%) |
| triton  | 65536 | 18146.94  | 17935.80  | 0.988 | -211.14 | 无差异 (~1%) |
| flydsl  | 4096  | 733.59    | 939.87    | 1.281 | +206.28 | iter=20 噪声；iter=60 后 = -0.5% |
| flydsl  | 4096 (iter=60) | 1481.72 | 1474.77 | 0.995 | -6.95 | 无差异 |
| flydsl  | 16384 | 2754.55   | 2750.34   | 0.999 | -4.21   | 无差异 |
| flydsl  | 65536 | 10954.97  | 10947.47  | 0.999 | -7.50   | 无差异 |
| decode  | B=1   | 41.65     | 37.64     | 0.904 | -4.01   | T=1 太短，噪声范围 |
| decode  | B=4   | 72.90     | 72.43     | 0.994 | -0.47   | 无差异 |

- **FlyDSL T=4096 +28% 噪声诊断**：per-kernel breakdown 显示**完全不涉及 beta 的 kernel
  也全面变慢**——l2norm +36%、cumsum +71%、kkt_solve +25%、megakernel +29%。这是典型
  GPU thermal/DVFS run-to-run 噪声，不是 cast overhead。iter=60 重跑后差异降到 -0.5%
  确认。
- **wrapper fp32→bf16 cast 实际开销估算**: [1, T, H=32] fp32 → bf16，T=65536 时
  HBM 流量 8MB read + 4MB write = 12MB，按 1.5TB/s 算 ~8us。相对 megakernel ~10ms 完全
  可忽略 (~0.08%)。
- **结论**: fp32-beta 改动对所有路径 (Triton ref / FlyDSL / decode recurrent) **性能影响 < 1%**，
  全部落在 run-to-run 噪声范围内。可以安全合入。

## 已放弃的方向
- 直接修改 FlyDSL kernel 接受 fp32-beta：megakernel LDS pipeline 写死 T.bf16，改动 surface 太大且无收益（下游 dot/mul 仍 cast fp32 算）

## P1 Review 跟进

### P1.2 ht buffer guard (已修复，方案 A)

- 改动: 三个 FlyDSL 模块 (`flydsl_chunk_gdn_mi308x.py`, `_fast.py`, `_bdv32_fast.py`) 的
  `build_megakernel` 增加 `output_final_state=True` 参数；ht store 段用 Python 编译期
  `if output_final_state:` 包住，false 时整段 IR 不生成。`_launch_tail_safe_into` /
  `_launch_fast_into` 的 cache_key 加 output_final_state 维度。
- sanity 脚本: `/tmp/sanity_p12_ht_guard.py`
- 结果:
  | case | 验证 | 结果 |
  |---|---|---|
  | output_final_state=True | prod 路径仍工作 | finite + final_state 返回 ✓ |
  | output_final_state=False | 新路径返回 None | 无越界 store ✓ |
  | True vs False 的 out 对比 | **bit-exact (max diff = 0)** | 编译期 guard 只影响 ht，main attn_out 完全不变 |
  | cache_key 分流 | True/False 各 1 个变体 | 编译期 dispatch 正确 ✓ |

### P1.1 SSM cache 布局 (已验证不是 bug，**reviewer 误判**)

- 端到端 audit 脚本: `/tmp/p11_ssm_layout_audit.py`
- 测试: 同输入 prefill (T=256, Hg=16, H=32, K=V=128) 分别走 Triton path
  (chunk_gated_delta_rule + store_ssm_state_to_block_map) 与 FlyDSL direct-store，
  对比 ssm_states 物理 byte 内容；再用各自写入的 ssm_states 作为 decode 的 initial_state，
  跑 fused_recurrent_gated_delta_rule。
- 结果:
  | 项 | 结果 |
  |---|---|
  | attn_out (Triton vs FlyDSL) | cos = 0.999984 ✓ |
  | ssm_states 物理 byte (cos) | **1.000000** ✓ |
  | ssm_states max diff | 3.6e-2 (bf16 ulp 级，两边累积顺序略不同) |
  | FlyDSL vs transpose(Triton) | cos ≈ 0 — 不是转置关系 |
  | block 1 (sentinel) | 两边都没写 ✓ |
  | block 2/3/4 (中间+最后) | 两边都写，内容一致到 bf16 ulp |
  | decode 用 block 4 作为 initial_state | cos = 0.999983 ✓ |

- **结论**: Triton `store_ssm_state_to_block_map` 的 `(V, K) + strides (K, 1)` 视角与 FlyDSL
  `dk * V + dv` 在 K==V==128 下产生**同一组 byte 序列**，两路写入完全等价。Reviewer 建议
  改 `dv * K + dk` 不仅没必要，反而会让 FlyDSL 写出与 Triton 不同的物理 byte 序列。
- 实际依赖：RTP-LLM 中 `head_k_dim == head_v_dim == 128` 是 `qwen3_next.py:109` 的硬 assert；
  整个 GDN cache pipeline (prefill h 分配, Triton store, FlyDSL store, decode read) 都依赖
  这个约束。建议**只加显式注释/assert**，不动现有 layout。

## 当前状态
- 当前 Phase: Phase 5 验收完成
- 当前 sub-skill: integration-validation
- 改动总结:
  - `gdn_gating.py`: 1 行（dtype 从 `b.dtype` 改 `torch.float32`）
  - `chunk.py`: 3 处（shape gate / validate 接受 fp32；FlyDSL wrapper 入口加 fp32→bf16 cast）
- 验收: 精度 + 性能 双线通过
  - 精度: 三路径跑通；Prefill 与 SGL bit-exact；Decode 比 SGL 更准（SGL 有 truncation bug）
  - 性能: 全路径影响 < 1%，落在噪声范围
- 下一步:
  1. 由用户决定是否 commit + push
  2. 可选: 给 SGL 上游报 decode kernel `fused_gdn_gating.py:39` 的 truncation bug
