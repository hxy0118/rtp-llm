# GatedDeltaNet KV Cache 管理详解

## 1. 概述

GatedDeltaNet 的 KV Cache 管理与传统 Transformer 注意力机制有本质区别。传统注意力需要缓存所有历史 K/V 向量，而 GatedDeltaNet 只需维护**固定大小**的状态矩阵，这是其线性复杂度的关键。

### 1.1 核心区别

| 特性 | 传统 Attention KV Cache | GatedDeltaNet State Cache |
|------|------------------------|---------------------------|
| 缓存内容 | K, V 向量序列 | SSM 状态矩阵 + Conv 状态 |
| 空间复杂度 | O(序列长度 × 头数 × 维度) | O(头数 × K_dim × V_dim) 固定 |
| 增长方式 | 随序列线性增长 | 恒定不变 |
| 访问模式 | 全序列回顾 | 仅读写当前状态 |

### 1.2 状态组成

GatedDeltaNet 需要维护两类状态:

1. **SSM State (隐藏状态矩阵)**: 线性注意力的核心状态
2. **Conv State (卷积状态)**: 因果卷积的历史输入

## 2. 内存布局

### 2.1 KV Cache Block 结构

```
┌─────────────────────────────────────────────────────────────────┐
│                         KV Cache Block                          │
├─────────────────────────────────┬───────────────────────────────┤
│          SSM State              │         Conv State            │
│   [num_heads × V_dim × K_dim]   │   [(kernel_dim-1) × qkv_size] │
└─────────────────────────────────┴───────────────────────────────┘
```

### 2.2 尺寸计算

在 `qwen3_next.py` 中定义:

```python
class Qwen3NextGatedDeltaNetBase:
    def __init__(self, ...):
        # SSM 状态大小
        self.ssm_state_size = (
            self.local_num_v_heads * self.head_k_dim * self.head_v_dim
        )
        
        # QKV 合并向量大小
        self.qkv_size = (
            self.head_k_dim * self.local_num_k_heads * 2  # Q + K
            + self.head_v_dim * self.local_num_v_heads    # V
        )
        
        # Conv 状态大小 (kernel_dim 通常为 4)
        self.conv_state_size = (self.linear_conv_kernel_dim - 1) * self.qkv_size
```

**实例计算** (以 Qwen3-Next 典型配置为例):
- `num_heads = 32`, `head_k_dim = head_v_dim = 128`
- `kernel_dim = 4`
- SSM State: 32 × 128 × 128 = 524,288 elements
- Conv State: 3 × (128×32×2 + 128×32) = 3 × 12,288 = 36,864 elements

### 2.3 状态访问 (View 操作)

```python
def _get_ssm_states(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
    """从 KV cache block 中获取 SSM 状态视图"""
    ssm_states = torch.as_strided(
        kv_cache_tensor,
        size=(
            kv_cache_tensor.shape[0],  # num_blocks
            self.local_num_v_heads,     # 头数
            self.head_v_dim,            # V 维度
            self.head_k_dim,            # K 维度
        ),
        stride=(
            kv_cache_tensor.stride()[0],
            self.head_k_dim * self.head_v_dim,
            self.head_k_dim,
            1,
        ),
        storage_offset=kv_cache_tensor.storage_offset(),
    )
    return ssm_states

def _get_conv_states(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
    """从 KV cache block 中获取 Conv 状态视图"""
    conv_states = torch.as_strided(
        kv_cache_tensor,
        size=(
            kv_cache_tensor.shape[0],
            self.linear_conv_kernel_dim - 1,  # kernel_dim - 1 个历史
            self.qkv_size,
        ),
        stride=(
            kv_cache_tensor.stride()[0],
            self.qkv_size,
            1,
        ),
        storage_offset=self.ssm_state_size + kv_cache_tensor.storage_offset(),
    )
    return conv_states
```

## 3. Block Map 机制

### 3.1 Paged Attention 风格的内存管理

GatedDeltaNet 采用类似 vLLM 的 Paged Attention 内存管理:

```
block_map: [batch_size, max_num_blocks]

┌─────────────────────────────────────────────────────────┐
│  Sequence 0: [block_3, block_7, block_2, -1, -1]        │
│  Sequence 1: [block_1, block_5, -1, -1, -1]             │
│  Sequence 2: [block_0, block_4, block_8, block_6, -1]   │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│  Physical Blocks Pool:                                   │
│  [Block_0][Block_1][Block_2][Block_3][Block_4]...       │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Block 索引计算

```python
def cal_block_idx(position, seq_size_per_block):
    """计算位置对应的 block 索引"""
    return (position - 1) // seq_size_per_block

# 示例: seq_size_per_block = 128
# position = 1-128   → block 0
# position = 129-256 → block 1
# position = 257-384 → block 2
```

### 3.3 seq_size_per_block 参数

这是 GatedDeltaNet 特有的参数，定义了每个 block 覆盖的序列长度:

```python
# 在 kv_cache 中
seq_size_per_block = kv_cache.seq_size_per_block  # 通常为 128

# Block 分配策略:
# - 每 seq_size_per_block 个 token 共享一个 block 的状态快照
# - 状态只在 block 边界处保存
```

## 4. SSM 状态管理

### 4.1 状态加载 (`load_initial_state_from_block_map`)

用于 Prefill 阶段从 KV Cache 恢复之前的状态:

```python
def load_initial_state_from_block_map(
    prefix_lengths: torch.Tensor,    # [batch] 前缀长度
    block_map: torch.Tensor,         # [batch, max_blocks] block 映射
    conv_states: torch.Tensor,       # 物理 block 池 (包含 SSM state)
    initial_states: torch.Tensor,    # [batch, heads, V, K] 输出
    seq_size_per_block: int,
):
```

**Triton Kernel 核心逻辑**:

```python
@triton.jit
def load_initial_state_from_block_map_kernel(...):
    i_b, i_h, i_v = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    # 获取前缀长度
    prefix = tl.load(prefix_lengths + i_b)
    
    # 计算要读取的 block 位置
    is_zero = prefix == 0
    block_offset = tl.where(is_zero, 0, (prefix - 1) // SEQ_SIZE_PER_BLOCK)
    
    # 从 block_map 获取实际 block ID
    block_idx = tl.where(
        is_zero, 0, 
        tl.load(block_map + i_b * max_block_size + block_offset)
    )
    
    # 加载状态 (prefix=0 时填充零)
    b_in = tl.where(
        is_zero,
        tl.zeros([BLOCK_V, K], dtype=...),
        tl.load(p_in, boundary_check=(0, 1)),
    )
    
    tl.store(p_out, b_in, boundary_check=(0, 1))
```

### 4.2 状态保存 (`store_ssm_state_to_block_map`)

用于 Prefill 阶段保存中间状态和最终状态:

```python
def store_ssm_state_to_block_map(
    h: torch.Tensor,              # [num_chunks, heads, V, K] 中间状态
    final_states: torch.Tensor,   # [batch, heads, V, K] 最终状态
    prefix_lengths: torch.Tensor, # [batch] 前缀长度
    cu_seqlens: torch.Tensor,     # [batch+1] 累积序列长度
    block_map: torch.Tensor,      # [batch, max_blocks]
    ssm_states: torch.Tensor,     # 物理 block 池
    seq_size_per_block: int,
    chunk_size: int,              # 通常为 64
):
```

**保存策略**:

```python
@triton.jit
def store_ssm_state_to_block_map_kernel(...):
    i_c, i_h, i_v = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    batch = tl.load(chunk_indices + i_c * 2)
    chunk = tl.load(chunk_indices + i_c * 2 + 1)
    
    should_write = False
    
    # 情况1: 最后一个 chunk → 保存 final_state
    if (chunk + 1) * CHUNK_SIZE >= input_len:
        source_ptr = final_states + batch * SSM_PER_BATCH
        dest_block_pos = (prefix + input_len - 1) // SEQ_SIZE_PER_BLOCK
        should_write = True
    
    # 情况2: chunk 边界对齐 seq_size_per_block → 保存 h
    elif chunk > 0 and (chunk + 1) * CHUNK_SIZE % SEQ_SIZE_PER_BLOCK == 0:
        dest_block_pos = (prefix + chunk * CHUNK_SIZE + CHUNK_SIZE - 1) // SEQ_SIZE_PER_BLOCK
        source_ptr = h + (i_c + 1) * SSM_PER_BATCH
        should_write = True
    
    if should_write:
        block_idx = tl.load(block_map + batch * max_block_size + dest_block_pos)
        # 写入状态
        tl.store(dest_ptr, tl.load(source_ptr))
```

**保存时机图示**:

```
seq_size_per_block = 128, chunk_size = 64

Tokens:    [0...63][64...127][128...191][192...255][256...300]
Chunks:       C0       C1        C2         C3         C4
              ↓        ↓         ↓          ↓          ↓
保存点:              Block0              Block1     Block2(final)
                     边界                 边界        序列结束
```

### 4.3 Decode 阶段的 In-place 更新

在 Decode 阶段，状态直接原地更新:

```python
@triton.jit
def fused_recurrent_gated_delta_rule_fwd_kernel(...):
    # ...
    
    for i_t in range(T):
        # 计算新状态
        b_h = b_h * exp(b_g)
        b_v = b_v - tl.sum(b_h * b_k[:, None], 0)
        b_v = b_v * b_beta
        b_h = b_h + b_k[:, None] * b_v[None, :]
        
        # In-place 写回 (每个 token 都更新)
        if INPLACE_FINAL_STATE and IS_CONTINUOUS_BATCHING:
            write_block_offset = cal_block_idx(sequence_length, SEQ_SIZE_PER_BLOCK) + i_t
            write_block_id = tl.load(
                block_map + i_n * max_block_size + write_block_offset
            )
            p_ht = ht + write_block_id * stride_final_state_token
        
        tl.store(p_ht, b_h)
```

## 5. Conv State 管理

### 5.1 因果卷积状态结构

```
Conv State 用于保存最近 (kernel_width - 1) 个 token 的 QKV 向量

kernel_width = 4 时:
┌─────────────────────────────────────┐
│  Conv State [3, qkv_size]           │
├─────────────┬─────────────┬─────────┤
│   token[-3] │   token[-2] │ token[-1]│
└─────────────┴─────────────┴─────────┘
```

### 5.2 Prefill 阶段的 Conv State 处理

```python
# causal_conv1d.py
@triton.jit
def _causal_conv1d_fwd_kernel(...):
    # 首个 chunk 从 cache 加载历史状态
    if chunk_offset == 0:
        if HAS_CACHE and prefix_length > 0:
            # 计算读取位置
            init_state_block_pos = (prefix_length - 1) // SEQ_SIZE_PER_BLOCK
            init_state_block_idx = tl.load(
                block_map_ptr + idx_seq * max_block_size + init_state_block_pos
            )
            
            # 加载历史 token
            if KERNEL_WIDTH == 4:
                col2 = tl.load(prior_tokens)
                col1 = tl.load(prior_tokens - stride)
                col0 = tl.load(prior_tokens - 2 * stride)
        else:
            # 无前缀时用零填充
            col0 = col1 = col2 = zeros
    
    # 处理序列并更新状态
    for idx_token in range(segment_len):
        # 卷积计算
        acc = bias + col0 * w0 + col1 * w1 + col2 * w2 + x * w3
        
        # SiLU 激活
        if SILU_ACTIVATION:
            acc = acc / (1 + exp(-acc))
        
        # 滑动窗口更新
        col0, col1, col2 = col1, col2, x
        
        # 在 block 边界或序列结束时保存状态
        dest_idx = prefix_length + idx_token + token_offset
        write_to_block = (
            (dest_idx + 1) % SEQ_SIZE_PER_BLOCK == 0 or
            idx_token + token_offset + 1 == seqlen
        )
        
        if write_to_block:
            write_page_idx = tl.load(block_map + ...)
            tl.store(conv_state + write_page_idx * stride + 0, col0)
            tl.store(conv_state + write_page_idx * stride + 1, col1)
            tl.store(conv_state + write_page_idx * stride + 2, col2)
```

### 5.3 Decode 阶段的 Conv State 更新

```python
@triton.jit
def _causal_conv1d_update_kernel(...):
    # 读取之前的 conv state
    read_block_offset = cal_block_idx(sequence_length - 1, SEQ_SIZE_PER_BLOCK)
    read_block_id = tl.load(block_map + ...)
    
    conv_states_base = conv_state_ptr + read_block_id * stride
    col0 = tl.load(conv_states_base + 0)
    col1 = tl.load(conv_states_base + 1)
    col2 = tl.load(conv_states_base + 2)
    
    # 逐 token 处理 (支持多 token decode)
    for idx_token in range(seqlen):
        # 卷积计算
        acc = bias + col0 * w0 + col1 * w1 + col2 * w2 + x * w3
        
        # 滑动更新
        col0, col1, col2 = col1, col2, x
        
        # 每个 token 都写入新 block
        write_block_offset = cal_block_idx(sequence_length, SEQ_SIZE_PER_BLOCK) + idx_token
        write_block_id = tl.load(block_map + ...)
        tl.store(conv_state + write_block_id * stride, new_state)
```

## 6. 连续批处理 (Continuous Batching)

### 6.1 变长序列支持

```python
# cu_seqlens: 累积序列长度 [0, len1, len1+len2, ...]
cu_seqlens = attention_inputs.cu_seqlens

# 示例:
# 3 个序列长度分别为 100, 50, 75
# cu_seqlens = [0, 100, 150, 225]
```

### 6.2 Block Map 与变长序列

```python
# block_map 索引方式
# block_map[batch_idx, block_offset] → physical_block_id

# 每个序列独立管理自己的 block 分配
for i_n in range(num_sequences):
    prefix = prefix_lengths[i_n]
    input_len = cu_seqlens[i_n + 1] - cu_seqlens[i_n]
    
    # 计算需要的 block 数量
    total_tokens = prefix + input_len
    num_blocks = ceil(total_tokens / seq_size_per_block)
    
    # 从 block_map 获取分配的 blocks
    blocks = block_map[i_n, :num_blocks]
```

### 6.3 Speculative Decoding 支持

多 token 预测时的状态管理:

```python
# sequence_lengths: 每个序列在 decode 前的长度
# S: 同时预测的 token 数

# 需要为每个预测 token 分配独立的 block
# 状态更新: 每个 token 后都保存一次
for s in range(S):
    write_block_offset = (sequence_length - 1) // seq_size_per_block + s
    write_block_id = block_map[batch, write_block_offset]
    # 写入状态到 write_block_id
```

## 7. Prefill 与 Decode 流程对比

### 7.1 Prefill 流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        Prefill Flow                              │
├─────────────────────────────────────────────────────────────────┤
│  1. 加载初始状态                                                 │
│     load_initial_state_from_block_map(prefix, block_map, ...)   │
│                                                                  │
│  2. 分块处理 (chunk_size=64)                                     │
│     for chunk in chunks:                                        │
│         h = update_state(h, q, k, v, g, beta)                   │
│         if chunk 在 block 边界:                                  │
│             save_state_to_block(h, block_map)                   │
│                                                                  │
│  3. 保存最终状态                                                 │
│     store_ssm_state_to_block_map(h, final_state, ...)           │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Decode 流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        Decode Flow                               │
├─────────────────────────────────────────────────────────────────┤
│  1. 读取状态 (通过 block_map 定位)                               │
│     h = load_state(block_map[seq_len // seq_size_per_block])    │
│                                                                  │
│  2. 逐 token 更新                                                │
│     for token in new_tokens:                                    │
│         h = update_state(h, q, k, v, g, beta)                   │
│         write_state(h, block_map[(seq_len + i) // block_size])  │
│                                                                  │
│  3. In-place 更新 (状态直接覆写到 block)                         │
└─────────────────────────────────────────────────────────────────┘
```

## 8. 内存效率分析

### 8.1 传统 Attention vs GatedDeltaNet

假设: `num_heads=32`, `head_dim=128`, `seq_len=8192`

**传统 Attention KV Cache**:
```
每层内存 = 2 × seq_len × num_heads × head_dim × dtype_size
        = 2 × 8192 × 32 × 128 × 2 (bf16)
        = 134 MB
```

**GatedDeltaNet State Cache**:
```
SSM State = num_heads × head_k_dim × head_v_dim × dtype_size
         = 32 × 128 × 128 × 4 (fp32)
         = 2 MB

Conv State = (kernel_dim - 1) × qkv_size × dtype_size
          = 3 × 12288 × 2
          = 0.07 MB

总计 ≈ 2 MB (恒定)
```

**内存节省**: 对于 8K 序列，约 **67x** 内存节省

### 8.2 Block 数量规划

```python
# 总 block 数量计算
def estimate_blocks(max_seq_len, batch_size, seq_size_per_block):
    blocks_per_seq = ceil(max_seq_len / seq_size_per_block)
    total_blocks = blocks_per_seq * batch_size * safety_factor
    return total_blocks

# 示例: max_seq=128K, batch=8, block_size=128
# blocks = ceil(128K / 128) * 8 * 1.2 ≈ 9600 blocks
```

## 9. 总结

GatedDeltaNet 的 KV Cache 管理特点:

1. **固定大小状态**: SSM 状态矩阵大小与序列长度无关
2. **分块存储**: 使用 `seq_size_per_block` 控制保存粒度
3. **Paged 管理**: 通过 `block_map` 实现灵活的内存分配
4. **双状态设计**: 同时维护 SSM 和 Conv 两种状态
5. **增量更新**: Decode 阶段原地更新，无需复制
6. **批处理友好**: 支持变长序列和连续批处理

这种设计使得 GatedDeltaNet 能够处理超长序列，同时保持高效的内存使用和推理速度。
