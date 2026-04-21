---
name: FlashInfer Blackwell chunk-GDN → RTP AMD/FlyDSL 移植
description: 把 flashinfer Blackwell SM100 的 CuTe DSL chunked Gated Delta Net megakernel 移植到 RTP-LLM AMD 后端 (FlyDSL/CDNA3-4)，对照 RTP 内置的老 FLA Triton 实现
type: project
originSessionId: 4091304f-6b8d-4a25-a978-d699e058b2a1
---
## 项目目标

把 flashinfer 在 PR #3001 (2026-04-14, commit `7c562d50`) 引入的 Blackwell SM100 chunk Gated Delta Net prefill **megakernel**（CuTe DSL 写）移植到 RTP-LLM 的 AMD 后端，使用 FlyDSL（CDNA3/CDNA4 上的 cutedsl 等价物）。RTP 当前用的应该是 `rtp-llm/3rdparty/flash_linear_attention` 子树里的老 FLA Triton 实现。

**Why:** RTP-LLM 在 AMD GPU 上跑 Qwen3.5 / 类似带 GDN 的混合架构模型，linear attention 段是性能瓶颈。flashinfer 这版 Blackwell kernel 相对 FLA Triton 给出 1.04x – 5.78x 加速（PR description 里实测，长 seq 越快越好），核心是 hierarchical block-wise inverse 把 (I+M_kk)^{-1} 映射到 ~8 次 Tensor Core MMA + persistent megakernel pipeline。

**How to apply:** 任何关于 RTP linear attention / GDN AMD kernel / chunk gated delta rule / FlyDSL 移植的讨论都参考这份 checkpoint。继续工作前先重读，避免遗忘已分析的内容。

## 关键文件路径

**FlashInfer Blackwell cutedsl 实现（源参考）**
- 主算法 (3733 行): `/Users/hxy/Desktop/hxy/flashinfer/flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py`
- Tile scheduler (263 行): `/Users/hxy/Desktop/hxy/flashinfer/flashinfer/gdn_kernels/blackwell/gated_delta_net_tile_scheduler.py`
- PyTorch adapter (295 行): `/Users/hxy/Desktop/hxy/flashinfer/flashinfer/gdn_kernels/blackwell/gdn_prefill.py`
- 顶层 dispatch: `/Users/hxy/Desktop/hxy/flashinfer/flashinfer/gdn_prefill.py` (line 291 处区分 SM100 vs SM90)

**FlashInfer 老 SM90 路径 (算法相同, 求逆参考)**
- C++ inverse 实现: `/Users/hxy/Desktop/hxy/flashinfer/include/flashinfer/flat/ampere/collective/flat_collective_inverse.hpp` (482 行, hierarchical 8→16→32→64 跟 cutedsl 完全同算法)
- Hopper kernel builder: `/Users/hxy/Desktop/hxy/flashinfer/include/flashinfer/flat/hopper/kernel/flat_kernel_builder_delta_rule.hpp`

**RTP-LLM 侧（移植目标）**
- RTP 主仓库: `/Users/hxy/Desktop/hxy/rtp-llm/`
- RTP/3rdparty/flash_linear_attention/ 只是 BUILD 文件 + patch，**不是源码**
- **真正的 FLA 源码（vendored from upstream）**: `/Users/hxy/Desktop/hxy/rtp-llm/rtp_llm/models_py/triton_kernels/fla/`
  - `chunk.py` - 主入口 chunk_gated_delta_rule_fwd（6 kernel 串行）
  - `chunk_delta_h.py` (343 行) - state 主循环 kernel，写 h 到 GMEM 是性能瓶颈
  - `solve_tril.py` (465 行) - inverse，3 个独立 Triton kernel 串行
  - `chunk_o.py` - 算 O
  - `chunk_scaled_dot_kkt.py` - K@K^T
  - `wy_fast.py` - WY representation
  - `cumsum.py`, `l2norm.py`, `index.py`, `gdn_gating.py`, `fused_recurrent.py`(decode)
- 各文件头注释 `Adapted from https://github.com/fla-org/flash-linear-attention/...`

**对照参考 (其他 FLA 副本)**
- SGLang FLA bench: `/Users/hxy/Desktop/hxy/sglang/benchmark/fla/`

**GPU 知识库 (gpu-kernel-optimizer skill 依赖, 服务器上必须克隆)**
- Git URL: `http://gitlab.alibaba-inc.com/tre-infra/gpu-wiki.git` (阿里内网)
- 目标路径: `/tmp/gpu-wiki`
- Clone 命令: `git clone http://gitlab.alibaba-inc.com/tre-infra/gpu-wiki.git /tmp/gpu-wiki`
- 用途: gpu-kernel-optimizer / gpu-kernel-baseline / gpu-kernel-profile-optimizer skill 在 profile 驱动优化阶段会查 `/tmp/gpu-wiki/docs/` 找 ISA / 硬件规格 / 框架 API，`/tmp/gpu-wiki/reference-kernels/` 找同类参考实现

## 核心算法回顾 (chunk_gated_delta_rule, BT=64)

每 chunk c (token [cC, (c+1)C)) 7 个 GEMM:
- GEMM1 W_kk = K@K^T,  GEMM2 W_qk = Q@K^T
- GEMM3 KS = K@S_prev, GEMM4 QS = Q@S_prev
- GEMM5 NV = A_inv @ V, A_inv = (I + T*β*W_kk)^{-1}  ← inverse 算 64x64 lower-tri
- GEMM6 O_intra = (T*β*W_qk) @ NV
- GEMM7 dS = K^T @ (V - KS),  S_next = cumprod[BT-1]*S_prev + dS

State `S[DK,DV] = [128,128] fp32` 全程留 TMEM；chunk-by-chunk 串行因依赖 S。

**Hierarchical block-wise inverse (核心 hot point)**: 把 (I+M_kk)^{-1} (64x64 unit-lower-tri) 拆 4 阶段:
- Stage1: 8 个 8x8 对角块 Gauss-Jordan (warp shuffle, ALU), `_invert_diagonal_NxN` line 2609
- Stage2: 4 个 16x16 块 -D⁻¹CA⁻¹ (SM80_16x8x8 MMA), `_blockwise_diagonal_8x8_to_16x16` line 2638
- Stage3: 2 个 32x32 块 (SM80_16x8x16), line 2745
- Stage4: 1 个 64x64 块 (4 warp 协同), line 2836

**这就是相对 FLA Triton 的核心算法优势** — FLA 里 (I+M)^{-1} 通常用 forward substitution 串行，本算法把求逆映射到 Tensor Core MMA。

## Megakernel 架构

12 warp / CTA 全部驻留:
- warps 0-3: compute_group_0 (T-pairwise, kk_epi, qk_epi, inverse, 224 reg/thread)
- warps 4-7: compute_group_1 (state ops, new_v_epi, qkv_epi, kv_update_epi, 256 reg/thread)
- warp 8: MMA warp (issue 全部 7 GEMM via tcgen05.MmaF16BF16Op)
- warp 9: TMA QKV warp
- warp 10: TMA gate/beta warp + prefix sum
- warp 11: epilogue (TMA store O)

Persistent grid: `(min(B*H, max_active_clusters), 1, 1)`，CTA round-robin 拉 (batch, head)。chunk 间串行(state 依赖)。

SMEM 225.5KB / TMEM 256KB (整个 state 留 TMEM 不落 GMEM)。

mbarrier-based 细粒度 producer-consumer pipeline (kk/qk acc 双缓冲让 MMA 与 epi 重叠；inverse 与 GEMM6 qkv 并行)。

## AMD/FlyDSL 移植关键差异

| 维度 | NVIDIA SM100 | AMD CDNA3/4 |
|---|---|---|
| Warp size | 32 | **64** (所有 lane mask、shuffle 重设计) |
| Tensor Core | tcgen05 + WGMMA + TMEM 256KB | MFMA, **无 TMEM** (state 必须落 LDS 或 GMEM) |
| Async copy | TMA | buffer_load_dwordx4 + s_waitcnt |
| 同步 | mbarrier (轻量) | s_barrier + LDS atomic (重) |
| Cluster | CGA | 无 (本实现 cluster=(1,1,1) 所以无影响) |

**最大坎: TMEM 缺失** — state [128,128] fp32 = 64KB 刚好等于 CDNA3 一个 CU 的 LDS 上限，意味着没有 SMEM staging 余量，可能要切 state 或换更大 LDS 的 CDNA4 (160KB/CU)。

## Megakernel 主循环结构 (kernel() line 847-1654)

```
kernel():                                          # 单个 device 函数
  分配 SMEM/TMEM
  创建 ~13 条 mbarrier pipeline (PipelineTmaUmma / PipelineUmmaAsync / PipelineAsyncUmma /
    PipelineTmaAsync / PipelineCpAsync / PipelineAsync)
  pipeline_init_arrive() / pipeline_init_wait()    # 全 CTA 同步初始化

  # 6 个 warp 角色，每个一段 if 分支，各自跑 persistent loop:
  if warp_idx in [0..3]:    compute_group_0:  while tile: for chunk: ...   # T-pair, kk_epi, qk_epi, inverse
  if warp_idx in [4..7]:    compute_group_1:  while tile: for chunk: ...   # state, decay_v, NV_epi, qkv_epi, KV_epi
  if warp_idx == 8:         mma_warp:          while tile: for chunk: ...   # issue 7 GEMM
  if warp_idx == 9:         tma_qkv_warp:      while tile: for chunk: ...   # TMA load Q/K/V
  if warp_idx == 10:        load_gate_beta_warp: while tile: for chunk:...  # ldgsts gate/beta + prefix sum
  if warp_idx == 11:        epilogue_warp:     while tile: for chunk: ...   # TMA store O
```

**关键点 1**：6 个角色各自独立 persistent loop，**全部驻留在同一个 CTA 里整段 kernel 执行期间**，这就是 megakernel 的"mega"——单 kernel launch 完成原本 6+ 个 kernel 的全部工作。

**关键点 2**：mma_warp (line 1949+) 一条流水线 issue 7 个 GEMM，每个 GEMM 之间靠 mbarrier acquire/commit 切换 TMEM 双 buffer，且严格按依赖顺序：
  GEMM1(kk) → kk_handle.commit → GEMM2(qk) → qk_handle.commit
  → 等 state_inp_ready → GEMM3(KS) + GEMM4(QS) → ks/qs.commit
  → 等 a_inv_ready + shared_inp_ready → GEMM5(NV)
  → 等 qk_ready → GEMM6(qkv)
  → 等 K^T → GEMM7(dS)

**关键点 3**：state S [DK,DV]=128x128 fp32 全程在 TMEM (`tmem_state_offset`)，**永不写 GMEM**（除 final/checkpoint）。chunk-by-chunk 在 TMEM 内 read-modify-write。

## FLA Triton vs CuteDSL Megakernel 数据流对比

**FLA** (chunk.py:28 chunk_gated_delta_rule_fwd) — 6 个独立 Triton kernel 串行：
```
kernel1: chunk_local_cumsum(g) → g(GMEM, 同 size)
kernel2: chunk_scaled_dot_kkt(k,β,g) → A=[BT,BT,B,T,H] (GMEM, 巨大!)
kernel3: solve_tril(A)  内部又是 3 个 sub-kernel:
   - solve_tril_16x16_kernel: 16x16 forward substitution → Ad
   - merge_16x16_to_32x32_inverse_kernel: -A22⁻¹·A21·A11⁻¹
   - merge_16x16_to_64x64_inverse_kernel: 类似但更大块
   → A_inv (GMEM, ~1GB for 8K seqlen)
kernel4: recompute_w_u_fwd → w, u (= new_v) 两个 [B,T,H,K/V] 张量到 GMEM
kernel5: chunk_gated_delta_rule_fwd_h → 输出 h[B,chunks,H,K,V] (GMEM, 100MB+) + v_new
kernel6: chunk_fwd_o → O
```

**CuteDSL megakernel**:
```
单 kernel:
  loop tile: loop chunk:
    [warp 9 TMA] Q/K/V 进 SMEM         [warp 10] gate/beta + prefix sum
    [warp 8 MMA] GEMM1 kk → TMEM       [CG0] T-pair + kk*β*T → SMEM_Ainv
    [warp 8 MMA] GEMM2 qk → TMEM       [CG0] qk*scale → SMEM_Qk
    [warp 8 MMA] GEMM3 KS → TMEM       [CG1] V-KS → TMEM_shared_inp
                 GEMM4 QS → TMEM       [CG0] hierarchical inverse SMEM in-place
    [warp 8 MMA] GEMM5 A_inv@V → TMEM  [CG1] NV → TMEM_shared_inp
    [warp 8 MMA] GEMM6 W_qkv@NV → TMEM [CG1] +T_col*QS → SMEM_O
    [warp 8 MMA] GEMM7 K^T@δ → TMEM_S(累加)  [warp 11] TMA store O
  S_final → GMEM (一次)
```

**性能 delta 来源**：
1. **GMEM 流量**: FLA 中间张量 A (8GB) + h (128MB+) 都是 GMEM round-trip; cutedsl 全在 TMEM/SMEM
2. **Kernel launch 开销**: 6 → 1
3. **Inverse**: 算法相同（hierarchical 8→16→32→64），FLA 用 3 sub-kernel + GMEM staging, cutedsl SMEM in-place
4. **State**: FLA 每 chunk 写 h 到 GMEM (chunk_delta_h.py:132 `tl.store(p_h1, b_h1...)`), cutedsl 始终 TMEM in-place

## 当前进度 (2026-04-21)

- [x] 算法+架构 已分析
- [x] hierarchical inverse 算法已读懂 (stage1-4 全部代码段)
- [x] tile scheduler 已读
- [x] 与老 SM90 C++ 实现对比 (同算法不同实现)
- [x] kernel() 主循环 + 12-warp 分工 + mbarrier 全图 已分析
- [x] mma_warp 7 个 GEMM issue 顺序 已分析
- [x] RTP vendored FLA 定位 + chunk.py + solve_tril.py + chunk_delta_h.py 已对比
- [ ] FlyDSL 基线实现规划 — 下一步
- [ ] reference correctness baseline (PyTorch) 在 AMD 上跑通
- [ ] FLA Triton on AMD baseline 性能数字

## 下一步建议

0. **环境准备 (服务器上做一次)**：`git clone http://gitlab.alibaba-inc.com/tre-infra/gpu-wiki.git /tmp/gpu-wiki` — 让 gpu-kernel-optimizer skill 可用
1. 跑通 RTP 现在 AMD 上 chunk_gated_delta_rule 的性能基线 (`rtp_llm/models_py/triton_kernels/fla/test/test_chunk_prefill.py`)
2. 在 FlyDSL 上实现 minimal megakernel：先单 warp group + 单 chunk + GMEM staging（验证算法）
3. 加入 hierarchical inverse (CDNA 上用 MFMA_16x16x16 替代 SM80_16x8x8)
4. 加入 K double-buffer + 多 warp pipeline
5. 加入 persistent grid + cross-tile state (在 LDS，不是 TMEM)
6. ROI 评估：对照 PR #3001 提供的 Blackwell 数字 1.04-5.78x，AMD 上预期 1.5-3x（受 LDS 容量限制）
