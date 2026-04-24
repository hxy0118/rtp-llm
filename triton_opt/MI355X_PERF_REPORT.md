# MI355X chunk-GDN (Gated Delta Net) 性能优化报告

> 日期: 2026-04-24
> GPU: AMD Instinct MI355X (gfx950, CDNA4, 304 CUs, 160KB LDS/CU)
> 环境: PyTorch 2.13.0.dev+rocm7.2, Triton 3.6.0, Gluon
> 模型: Qwen3.5-397B (DK=128, DV=128, BT=64)
> 测试: B=1, varlen=True, output_final_state=True (生产首次 prefill)

## 1. 综合性能 — TP2 (Hg=8, H=32) 生产配置

### 端到端 Pipeline (us)

| T | Triton orig | Triton new | Best-of-both | new vs orig | best vs new |
|---|------------|-----------|-------------|-------------|-------------|
| 4096 | 374 | 297 | **282** | 1.26x | **1.05x** |
| 8192 | 696 | 562 | **518** | 1.24x | **1.08x** |
| 16384 | 1371 | 1070 | **1037** | 1.28x | **1.03x** |
| 32768 | 2777 | 2152 | **2099** | 1.29x | **1.03x** |
| 65536 | 5630 | 4293 | **4162** | 1.31x | **1.03x** |
| 131072 | 11499 | **7587** | 8179 | 1.52x | 0.93x |
| 262144 | 21250 | 16835 | **16229** | 1.26x | **1.04x** |

### fwd_h 孤立性能 (us)

| T | Triton tuned | Gluon | 加速比 | 推荐 |
|---|-------------|-------|--------|------|
| 4096 | 160 | **143** | **1.12x** | Gluon ✓ |
| 8192 | 371 | **322** | **1.15x** | Gluon ✓ |
| 16384 | 692 | **650** | **1.07x** | Gluon ✓ |
| 32768 | 1317 | **1152** | **1.14x** | Gluon ✓ |
| 65536 | 2570 | **2276** | **1.13x** | Gluon ✓ |
| 131072 | **4197** | 4865 | 0.86x | Triton |
| 262144 | **8350** | 8737 | 0.96x | Triton |

**Gluon fwd_h 在 T ≤ 65536 稳定快 7-15%。T ≥ 128K 时 Triton 更优。**

## 2. TP1 (Hg=16, H=64) 性能

### 端到端 Pipeline (us)

| T | Triton orig | Triton new | Best-of-both |
|---|------------|-----------|-------------|
| 4096 | 573 | **489** | 499 |
| 8192 | 1104 | **931** | 935 |
| 16384 | 2200 | **1878** | 1917 |
| 32768 | 4319 | **3743** | 3762 |

### fwd_h 孤立 (us)

| T | Triton tuned | Gluon | 加速比 |
|---|-------------|-------|--------|
| 4096 | **264** | 278 | 0.95x |
| 8192 | **495** | 514 | 0.96x |
| 16384 | **1006** | 1052 | 0.96x |
| 32768 | **2108** | 2177 | 0.97x |

**TP1 (H=64) 下 Gluon 始终慢 3-5%。全部用 Triton。**

## 3. Dispatch 规则

```python
# 在 chunk_delta_h.py 的 wrapper 中自动 dispatch：
if H <= 32 and T <= 65536 and is_gfx950:
    gluon_fwd_h(...)    # +7-15%
else:
    triton_fwd_h(...)   # 原始路径
```

| 条件 | fwd_h 推荐 | 典型加速 |
|------|-----------|---------|
| **H ≤ 32, T ≤ 64K** | **Gluon** | **+7-15%** |
| H ≤ 32, T = 128K | Triton | (Gluon 慢 14%) |
| H ≤ 32, T = 256K | Triton | (Gluon 慢 4%) |
| H = 64, 所有 T | Triton | (Gluon 慢 3-5%) |

## 4. 各 Kernel 性能 (TP2, T=65536)

| # | Kernel | 时间 (us) | 占比 | 最优 |
|---|--------|----------|------|------|
| 1 | cumsum | 60 | 1.4% | Triton |
| 2-3 | fused_kkt_solve | 479 | 11.5% | Triton (vs 分离 2.39x) |
| 4 | recompute_w_u | 624 | 15.0% | Triton |
| 5 | **fwd_h** | **2276** | **54.7%** | **Gluon (+13%)** |
| 6 | fwd_o | 683 | 16.4% | Triton |

## 5. fwd_h 优化技术

| 步骤 | 提升 | 技术 | 发现方法 |
|------|------|------|---------|
| BV=16 | +8% | tile sweep | 参数搜索 |
| buffer_load 预取 | +33% | w/v/k^T 跨迭代 | 循环分析 |
| k_width=8 | +10% | 匹配 Triton kWidth | ISA dump |
| convert_layout(b_h/b_v) | +6% | 消除 ds_write_b16 | gpu-wiki 3.2 |
| blocked_v + convert_layout(v) | +8% | 匹配 TTGIR #blocked2 | TTGIR 分析 |

核心发现: **BV=16 窄 tile 用 smem 产生 ds_write_b16 (逐元素写入)，改 convert_layout 消除。**

## 6. RTP 集成

**新增**: `rtp_llm/models_py/triton_kernels/fla/chunk_delta_h_gluon.py`
**修改**: `rtp_llm/models_py/triton_kernels/fla/chunk_delta_h.py` (添加 Gluon dispatch)

- 零改动 fallback: Gluon 不可用或条件不满足时使用原始 Triton
- 位精确匹配: 所有输出 rel_err = 0.0
- 自动 dispatch: 根据 H, T, GPU arch 判断

## 7. 文件索引

| 文件 | 说明 |
|------|------|
| `triton_opt/bench_standalone.py` | Triton kernels + pipelines |
| `triton_opt/bench_gluon.py` | Gluon kernels (开发/测试) |
| `triton_opt/bench_comprehensive.py` | TP1/TP2 × T 综合测试 |
| `triton_opt/MI355X_PERF_REPORT.md` | 本报告 |
| `rtp_llm/.../fla/chunk_delta_h_gluon.py` | **Gluon fwd_h (生产)** |
| `rtp_llm/.../fla/chunk_delta_h.py` | **修改: 添加 dispatch** |
