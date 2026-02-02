# GatedDeltaNet Fused Recurrent Kernel 优化方案

本文档详细说明针对 rtp-llm 中 `fused_recurrent_gated_delta_rule_fwd_kernel` 的优化方案，基于 vLLM 相关实现的分析。

## 1. 优化概述

| 优化项 | 原实现 | 优化后 | 预期收益 |
|-------|--------|-------|---------|
| BV 块大小 | 8 | 32 (可配置) | 内存带宽利用率提升 4x |
| Speculative Decoding | 不支持 | 支持 | 推测解码场景可用 |
| KDA 模式 | 不支持 | 支持 | 扩展模型支持范围 |
| PAD_SLOT_ID 检查 | `<= 0` | `< 0` | 与 vLLM 保持一致 |
| 条件状态写入 | 总是写入 | 条件写入 | 减少无效内存操作 |

## 2. 详细优化说明

### 2.1 BV 块大小优化 (最重要)

**原代码：**
```python
# rtp-llm: fused_recurrent.py L195
BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
```

**vLLM 代码：**
```python
# vLLM: fused_recurrent.py L195
BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
```

**优化分析：**

1. **内存访问模式**：
   - 每个 warp (32 threads) 处理的 V 维度元素数从 8 增加到 32
   - 对于 V=128 的配置，原来需要 16 次迭代，现在只需要 4 次
   - 减少了循环开销和寄存器压力

2. **带宽利用率**：
   - GPU 内存事务通常以 32/64/128 字节为单位
   - BV=8 时，每次加载 8 × 2 = 16 字节 (bf16)，未充分利用
   - BV=32 时，每次加载 32 × 2 = 64 字节，更好地匹配内存事务大小

3. **Grid 配置影响**：
   ```python
   # BV=8 时
   NV = triton.cdiv(V, 8)  # V=128 -> NV=16
   grid = (NK, NV, N * HV)  # 更多 kernel 实例

   # BV=32 时
   NV = triton.cdiv(V, 32)  # V=128 -> NV=4
   grid = (NK, NV, N * HV)  # 更少 kernel 实例，每个做更多工作
   ```

**优化实现：**
```python
# 可配置 BV，通过环境变量或参数控制
DEFAULT_BV = int(os.getenv("RTP_LLM_FLA_BV", "32"))

def fused_recurrent_gated_delta_rule_fwd_optimized(
    ...,
    bv_size: Optional[int] = None,  # 新参数
):
    bv = bv_size if bv_size is not None else DEFAULT_BV
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), bv)
```

**使用方法：**
```python
# 方法 1：环境变量控制 (推荐用于全局配置)
export RTP_LLM_FLA_BV=32

# 方法 2：函数参数控制 (推荐用于精细调优)
o, state = fused_recurrent_gated_delta_rule_optimized(
    q, k, v, g, beta,
    initial_state=h0,
    bv_size=32,  # 或 16, 64 根据 V 维度调整
)
```

### 2.2 Speculative Decoding 支持

**vLLM 实现：**
```python
# Heuristics 中添加检测
"IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,

# Kernel 中使用
if IS_SPEC_DECODING:
    i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
else:
    i_t = 0
state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t)
```

**优化实现 (适配 rtp-llm block_map 架构)：**
```python
if IS_SPEC_DECODING:
    spec_offset = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
    load_block_offset = cal_block_idx(
        sequence_length - 1 + spec_offset, SEQ_SIZE_PER_BLOCK
    )
else:
    load_block_offset = cal_block_idx(sequence_length - 1, SEQ_SIZE_PER_BLOCK)
```

**使用场景：**
- 推测解码时，需要根据已接受的 token 数量确定正确的初始状态位置
- 避免在投机失败时读取错误的状态

### 2.3 KDA (Key Dimension Attention) 模式

**概念说明：**
- 标准模式：所有 K 维度共享一个标量门控 `g`
- KDA 模式：每个 K 维度有独立的门控 `g_k`

**vLLM 实现：**
```python
if not IS_KDA:
    p_g = g + bos * HV + i_hv
    ...
    b_g = tl.load(p_g).to(tl.float32)
    b_h *= exp(b_g)  # 标量广播
else:
    p_gk = g + (bos * HV + i_hv) * K + o_k
    ...
    b_gk = tl.load(p_gk).to(tl.float32)
    b_h *= exp(b_gk[:, None])  # 向量广播
```

**使用场景：**
- 某些模型架构需要更细粒度的遗忘门控制
- 例如：每个 key 维度代表不同的语义特征，需要独立衰减

**使用方法：**
```python
# 标准模式 (默认)
g = torch.randn(B, T, HV, device='cuda')  # shape: [B, T, HV]
o, state = fused_recurrent_gated_delta_rule_optimized(
    q, k, v, g, beta, initial_state=h0, is_kda=False
)

# KDA 模式
g_kda = torch.randn(B, T, HV, K, device='cuda')  # shape: [B, T, HV, K]
o, state = fused_recurrent_gated_delta_rule_optimized(
    q, k, v, g_kda, beta, initial_state=h0, is_kda=True
)
```

### 2.4 PAD_SLOT_ID 验证优化

**原代码：**
```python
if read_block_id <= 0:
    return
```

**优化代码：**
```python
# 与 vLLM PAD_SLOT_ID = -1 保持一致
if read_block_id < 0:
    return
```

**差异影响：**
- 原实现：`block_id = 0` 被视为无效
- 优化后：`block_id = 0` 是有效的（第一个块）
- 这修复了潜在的边界情况 bug

### 2.5 条件状态写入

**原代码：**
```python
if INPLACE_FINAL_STATE and IS_CONTINUOUS_BATCHING:
    write_block_offset = cal_block_idx(sequence_length, SEQ_SIZE_PER_BLOCK) + i_t
    write_block_id = tl.load(block_map + i_n * max_block_size + write_block_offset)
    p_ht = ht + write_block_id * stride_final_state_token
    # 总是写入，即使 block_id 无效
```

**优化代码：**
```python
if INPLACE_FINAL_STATE and IS_CONTINUOUS_BATCHING:
    write_block_offset = cal_block_idx(sequence_length, SEQ_SIZE_PER_BLOCK) + i_t
    write_block_id = tl.load(block_map + i_n * max_block_size + write_block_offset)
    # 只在 block_id 有效时写入
    if write_block_id >= 0:
        p_ht = ht + write_block_id * stride_final_state_token
        p_ht = p_ht + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
```

**优势：**
- 避免向无效地址写入
- 减少不必要的内存操作
- 提高安全性和稳定性

## 3. 性能预期

### 3.1 BV=32 性能提升估算

对于典型的 Qwen3-Next 配置 (K=128, V=128, HV=8):

| 配置 | BV | NV (Grid Y) | 每 kernel 工作量 | 预期加速比 |
|-----|----|----|----------------|-----------|
| 原始 | 8 | 16 | 8 elements/iter | 1.0x |
| 优化 | 32 | 4 | 32 elements/iter | ~1.3-1.5x |

**注意**：实际加速比取决于：
- GPU 型号和 SM 数量
- Batch size 和序列长度
- 内存带宽利用率

### 3.2 回归风险

| 风险项 | 概率 | 影响 | 缓解措施 |
|--------|-----|------|---------|
| 寄存器压力增加 | 中 | 可能降低 occupancy | 提供 BV 配置选项 |
| 数值差异 | 低 | 精度微小变化 | 添加单元测试验证 |
| 功能回归 | 低 | 某些场景不工作 | 保留原实现作为回退 |

## 4. 集成建议

### 4.1 渐进式集成

**阶段 1：测试验证**
```python
# 在测试环境中对比原实现和优化实现
from rtp_llm.models_py.triton_kernels.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule as original
)
from rtp_llm.models_py.triton_kernels.fla.fused_recurrent_optimized import (
    fused_recurrent_gated_delta_rule_optimized as optimized
)

# 验证数值一致性
o_orig, state_orig = original(q, k, v, g, beta, initial_state=h0)
o_opt, state_opt = optimized(q, k, v, g, beta, initial_state=h0, bv_size=8)  # 先用相同 BV
assert torch.allclose(o_orig, o_opt, rtol=1e-3, atol=1e-3)

# 验证 BV=32 的数值差异在可接受范围
o_opt32, state_opt32 = optimized(q, k, v, g, beta, initial_state=h0, bv_size=32)
assert torch.allclose(o_orig, o_opt32, rtol=1e-2, atol=1e-2)
```

**阶段 2：A/B 测试**
```python
# 通过环境变量控制使用哪个版本
USE_OPTIMIZED_KERNEL = os.getenv("RTP_LLM_USE_OPTIMIZED_FLA", "0") == "1"

if USE_OPTIMIZED_KERNEL:
    from .fused_recurrent_optimized import fused_recurrent_gated_delta_rule_optimized as fused_recurrent_gated_delta_rule
else:
    from .fused_recurrent import fused_recurrent_gated_delta_rule
```

**阶段 3：默认启用**
```python
# 将优化版本设为默认，保留原版本作为回退
from .fused_recurrent_optimized import fused_recurrent_gated_delta_rule_optimized as fused_recurrent_gated_delta_rule
```

### 4.2 配置推荐

| GPU 型号 | V 维度 | 推荐 BV | 说明 |
|---------|--------|--------|------|
| A100/H100 | 128 | 32 | 高带宽，大 BV 有利 |
| A10/L4 | 128 | 16-32 | 根据实测调整 |
| T4 | 128 | 8-16 | 寄存器有限，保守配置 |

### 4.3 模型层面集成

修改 `qwen3_next.py` 中的调用：

```python
# 在 Qwen3NextGatedDeltaNetDecode._fla 中
def _fla(self, q, k, v, g, beta, ssm_states, ...):
    # 原调用
    # o, final_state = fused_recurrent_gated_delta_rule(...)

    # 优化调用
    from rtp_llm.models_py.triton_kernels.fla.fused_recurrent_optimized import (
        fused_recurrent_gated_delta_rule_optimized
    )
    o, final_state = fused_recurrent_gated_delta_rule_optimized(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=ssm_states,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        block_map=block_map,
        seq_size_per_block=seq_size_per_block,
        sequence_lengths=sequence_lengths,
        use_qk_l2norm_in_kernel=True,
        # 新参数
        bv_size=32,  # 或通过配置传入
    )
    return o, final_state
```

## 5. 测试用例

```python
import torch
import pytest

def test_optimized_kernel_numerical_equivalence():
    """验证优化 kernel 与原 kernel 数值等价"""
    B, T, H, HV, K, V = 2, 16, 4, 8, 128, 128

    q = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(B, T, HV, V, device='cuda', dtype=torch.bfloat16)
    g = torch.randn(1, B * T, HV, device='cuda', dtype=torch.float32)
    beta = torch.rand(1, B * T, HV, device='cuda', dtype=torch.bfloat16).sigmoid()
    h0 = torch.randn(B, HV, K, V, device='cuda', dtype=torch.float32)

    from rtp_llm.models_py.triton_kernels.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule
    )
    from rtp_llm.models_py.triton_kernels.fla.fused_recurrent_optimized import (
        fused_recurrent_gated_delta_rule_optimized
    )

    o_orig, _ = fused_recurrent_gated_delta_rule(
        q.view(1, -1, H, K), k.view(1, -1, H, K), v.view(1, -1, HV, V),
        g, beta, initial_state=h0.clone()
    )

    o_opt, _ = fused_recurrent_gated_delta_rule_optimized(
        q.view(1, -1, H, K), k.view(1, -1, H, K), v.view(1, -1, HV, V),
        g, beta, initial_state=h0.clone(), bv_size=8  # 相同 BV
    )

    assert torch.allclose(o_orig, o_opt, rtol=1e-3, atol=1e-3)


def test_optimized_kernel_kda_mode():
    """验证 KDA 模式正确工作"""
    B, T, H, HV, K, V = 2, 16, 4, 8, 128, 128

    q = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(B, T, HV, V, device='cuda', dtype=torch.bfloat16)
    g_kda = torch.randn(1, B * T, HV, K, device='cuda', dtype=torch.float32)  # KDA shape
    beta = torch.rand(1, B * T, HV, device='cuda', dtype=torch.bfloat16).sigmoid()
    h0 = torch.randn(B, HV, K, V, device='cuda', dtype=torch.float32)

    from rtp_llm.models_py.triton_kernels.fla.fused_recurrent_optimized import (
        fused_recurrent_gated_delta_rule_optimized
    )

    o, final_state = fused_recurrent_gated_delta_rule_optimized(
        q.view(1, -1, H, K), k.view(1, -1, H, K), v.view(1, -1, HV, V),
        g_kda.view(1, -1, HV, K), beta, initial_state=h0, is_kda=True
    )

    assert o.shape == (1, B * T, HV, V)
    assert not torch.isnan(o).any()


def test_optimized_kernel_bv32_performance():
    """验证 BV=32 相比 BV=8 的性能提升"""
    import time

    B, T, H, HV, K, V = 8, 64, 4, 8, 128, 128

    q = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(B, T, HV, V, device='cuda', dtype=torch.bfloat16)
    g = torch.randn(1, B * T, HV, device='cuda', dtype=torch.float32)
    beta = torch.rand(1, B * T, HV, device='cuda', dtype=torch.bfloat16).sigmoid()
    h0 = torch.randn(B, HV, K, V, device='cuda', dtype=torch.float32)

    from rtp_llm.models_py.triton_kernels.fla.fused_recurrent_optimized import (
        fused_recurrent_gated_delta_rule_optimized
    )

    # Warmup
    for _ in range(10):
        fused_recurrent_gated_delta_rule_optimized(
            q.view(1, -1, H, K), k.view(1, -1, H, K), v.view(1, -1, HV, V),
            g, beta, initial_state=h0.clone(), bv_size=32
        )
    torch.cuda.synchronize()

    # BV=8
    start = time.time()
    for _ in range(100):
        fused_recurrent_gated_delta_rule_optimized(
            q.view(1, -1, H, K), k.view(1, -1, H, K), v.view(1, -1, HV, V),
            g, beta, initial_state=h0.clone(), bv_size=8
        )
    torch.cuda.synchronize()
    time_bv8 = time.time() - start

    # BV=32
    start = time.time()
    for _ in range(100):
        fused_recurrent_gated_delta_rule_optimized(
            q.view(1, -1, H, K), k.view(1, -1, H, K), v.view(1, -1, HV, V),
            g, beta, initial_state=h0.clone(), bv_size=32
        )
    torch.cuda.synchronize()
    time_bv32 = time.time() - start

    speedup = time_bv8 / time_bv32
    print(f"BV=8: {time_bv8:.3f}s, BV=32: {time_bv32:.3f}s, Speedup: {speedup:.2f}x")
    assert speedup > 1.0, f"Expected speedup > 1.0, got {speedup}"
```

## 6. 总结

本优化方案从 vLLM 的实现中借鉴了以下关键改进：

1. **BV 块大小提升**：从 8 增加到 32，提高内存带宽利用率
2. **Speculative Decoding 支持**：为推测解码场景提供正确的状态管理
3. **KDA 模式**：支持更细粒度的 key 维度门控
4. **安全性增强**：改进 PAD_SLOT_ID 检查和条件写入

通过渐进式集成策略，可以在保证稳定性的前提下逐步验证和部署这些优化。
