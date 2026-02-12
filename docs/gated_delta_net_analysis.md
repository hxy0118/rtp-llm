# GatedDeltaNet 计算逻辑详解

## 1. 概述

GatedDeltaNet 是一种结合了门控机制的线性注意力变体，源自 Flash Linear Attention (FLA) 项目。它是 Qwen3-Next 等新一代模型中使用的高效注意力机制，具有线性时间复杂度和常数空间复杂度的优势。

### 1.1 核心特点

- **线性时间复杂度**: O(n) 而非标准注意力的 O(n²)
- **门控机制**: 通过可学习的遗忘门控制信息流动
- **Delta Rule 更新**: 基于增量规则更新隐藏状态
- **分块处理**: 采用 Chunk-wise 计算提升并行效率

## 2. 数学原理

### 2.1 Delta Rule 基础

传统的 Delta Rule 更新公式为:

```
h_t = h_{t-1} + k_t ⊗ (v_t - k_t^T h_{t-1})
```

其中:
- `h_t` 是时刻 t 的隐藏状态矩阵 [K, V]
- `k_t` 是 key 向量 [K]
- `v_t` 是 value 向量 [V]
- `⊗` 表示外积

### 2.2 Gated Delta Rule

GatedDeltaNet 在 Delta Rule 基础上引入了门控机制:

```
g_t = -exp(A_log) * softplus(a_t + dt_bias)
β_t = sigmoid(b_t)

h_t = exp(g_t) * h_{t-1} + k_t ⊗ β_t * (v_t - k_t^T h_{t-1})
o_t = q_t^T h_t
```

其中:
- `g_t` 是遗忘门(在 log 空间)
- `β_t` 是输入门
- `A_log` 是可学习参数
- `a_t, b_t` 是从输入投影得到的门控信号

## 3. 代码架构

### 3.1 整体流程

```
qwen3_next.py
├── Qwen3NextGatedDeltaNet (主模块)
│   ├── in_proj_qkvz    # 投影得到 Q, K, V, Z
│   ├── in_proj_ba      # 投影得到门控信号 b, a
│   ├── prefill_gdn     # Prefill 阶段处理
│   │   ├── _conv1d()   # 因果卷积
│   │   └── _fla()      # 分块线性注意力
│   ├── decode_gdn      # Decode 阶段处理
│   │   ├── _conv1d()   # 因果卷积更新
│   │   └── _fla()      # 递归更新
│   ├── norm            # RMS Norm with Gating
│   └── out_proj        # 输出投影
```

### 3.2 关键文件说明

| 文件 | 功能 |
|------|------|
| `qwen3_next.py` | 模型定义和前向传播 |
| `chunk.py` | Prefill 阶段分块处理入口 |
| `fused_recurrent.py` | Decode 阶段递归处理 |
| `gdn_gating.py` | 门控信号计算 |
| `chunk_delta_h.py` | 分块隐藏状态更新 |
| `chunk_o.py` | 分块输出计算 |
| `wy_fast.py` | WY 表示重计算 |
| `l2norm.py` | L2 归一化 |
| `cumsum.py` | 分块累加和 |
| `solve_tril.py` | 下三角矩阵求逆 |
| `block.py` | KV Cache 状态管理 |

## 4. Prefill 阶段计算

Prefill 使用分块(Chunk-wise)并行计算，chunk_size 默认为 64。

### 4.1 门控信号计算 (`gdn_gating.py`)

```python
def fused_gdn_gating(A_log, a, b, dt_bias):
    """
    计算门控信号:
    g = -exp(A_log) * softplus(a + dt_bias)  # 遗忘门(log空间)
    beta = sigmoid(b)                         # 输入门
    """
```

Triton Kernel 实现:
```python
@triton.jit
def fused_gdn_gating_kernel(...):
    # softplus 计算: log(1 + exp(β * x)) / β
    x = blk_a + blk_bias
    softplus_x = where(β * x <= threshold, 
                       (1/β) * log(1 + exp(β * x)), x)
    
    # 遗忘门
    blk_g = -exp(blk_A_log) * softplus_x
    
    # 输入门
    blk_beta = sigmoid(blk_b)
```

### 4.2 L2 归一化 (`l2norm.py`)

对 Q, K 进行 L2 归一化:
```python
def l2norm_fwd(x, eps=1e-6):
    """
    y = x / sqrt(sum(x^2) + eps)
    """
```

### 4.3 分块累加和 (`cumsum.py`)

对遗忘门信号进行块内累加:
```python
def chunk_local_cumsum(g, chunk_size=64):
    """
    在每个 chunk 内计算累加和
    g_cumsum[i] = sum(g[0:i]) for i in [0, chunk_size)
    """
```

### 4.4 WY 表示计算

#### 4.4.1 计算 A 矩阵 (`chunk_scaled_dot_kkt.py`)

```python
def chunk_scaled_dot_kkt_fwd(k, beta, g_cumsum):
    """
    计算: A = β * K * K^T * exp(g_diff)
    
    其中 g_diff[i,j] = g_cumsum[i] - g_cumsum[j]
    
    结果是一个严格下三角矩阵 (i > j 的部分)
    """
```

#### 4.4.2 下三角矩阵求逆 (`solve_tril.py`)

```python
def solve_tril(A):
    """
    计算 (I + A)^{-1}
    
    使用 16x16 分块递归方法:
    1. 先求解每个 16x16 块的逆
    2. 合并成 32x32 或 64x64
    """
```

#### 4.4.3 重计算 W 和 U (`wy_fast.py`)

```python
def recompute_w_u_fwd(k, v, beta, A, g_cumsum):
    """
    计算 WY 表示:
    u = A @ (v * β)           # 变换后的 value
    w = A @ (k * β * exp(g))  # 变换后的 key
    """
```

### 4.5 隐藏状态更新 (`chunk_delta_h.py`)

```python
def chunk_gated_delta_rule_fwd_h(k, w, u, g, initial_state):
    """
    分块更新隐藏状态:
    
    对每个 chunk:
        1. 保存当前 h 到 h_buffer
        2. 计算 v_new = v - w @ h  (delta 修正)
        3. h = h * exp(g_last) + k^T @ v_new
    
    返回: h_buffer (中间状态), v_new (修正后的 value), final_state
    """
```

核心 Triton Kernel:
```python
@triton.jit
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(...):
    for i_t in range(NT):
        # 保存当前状态
        tl.store(h, b_h)
        
        # 计算 delta 修正
        b_v = tl.load(v) - tl.dot(b_w, b_h)
        
        # 应用遗忘门
        b_h = b_h * exp(b_g_last)
        
        # 更新状态
        b_h += tl.dot(b_k, b_v)
```

### 4.6 输出计算 (`chunk_o.py`)

```python
def chunk_fwd_o(q, k, v_new, h, g, scale):
    """
    计算输出:
    
    对每个 chunk:
        # 跨 chunk 贡献
        o = Q @ H * exp(g) * scale
        
        # chunk 内贡献 (下三角 attention)
        A_local = Q @ K^T * exp(g_diff)
        A_local = tril(A_local)  # 保持因果性
        o += A_local @ V_new * scale
    """
```

### 4.7 完整 Prefill 流程

```python
def chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale, initial_state):
    # 1. 块内累加遗忘门
    g = chunk_local_cumsum(g, chunk_size=64)
    
    # 2. 计算 WY 表示的 A 矩阵
    A = chunk_scaled_dot_kkt_fwd(k, beta, g)
    
    # 3. 求解下三角矩阵的逆
    A = solve_tril(A)
    
    # 4. 重计算 w, u
    w, u = recompute_w_u_fwd(k, v, beta, A, g)
    
    # 5. 更新隐藏状态
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k, w, u, g, initial_state)
    
    # 6. 计算输出
    o = chunk_fwd_o(q, k, v_new, h, g, scale)
    
    return o, h, final_state
```

## 5. Decode 阶段计算

Decode 使用逐 token 递归更新，更加高效。

### 5.1 递归更新公式

```python
def fused_recurrent_gated_delta_rule_fwd(q, k, v, g, beta, initial_state):
    """
    逐 token 递归更新:
    
    for t in range(T):
        # L2 归一化
        q = q / ||q||
        k = k / ||k||
        
        # 应用遗忘门
        h = h * exp(g)
        
        # Delta 修正
        v = v - sum(h * k)  # 相当于 k^T @ h
        v = v * beta
        
        # 更新状态
        h = h + outer(k, v)
        
        # 计算输出
        o = sum(h * q)  # 相当于 q^T @ h
    """
```

### 5.2 Triton Kernel 实现

```python
@triton.jit
def fused_recurrent_gated_delta_rule_fwd_kernel(...):
    # 加载初始状态
    b_h = tl.load(h0)
    
    for i_t in range(T):
        b_q = tl.load(p_q)
        b_k = tl.load(p_k)
        b_v = tl.load(p_v)
        b_g = tl.load(p_g)
        
        # L2 归一化
        if USE_QK_L2NORM:
            b_q = b_q / sqrt(sum(b_q * b_q) + eps)
            b_k = b_k / sqrt(sum(b_k * b_k) + eps)
        
        b_q = b_q * scale
        
        # 遗忘门衰减
        b_h = b_h * exp(b_g)
        
        # Delta 修正: v -= h @ k
        b_v = b_v - tl.sum(b_h * b_k[:, None], 0)
        b_v = b_v * b_beta
        
        # 状态更新: h += outer(k, v)
        b_h = b_h + b_k[:, None] * b_v[None, :]
        
        # 输出: o = q @ h
        b_o = tl.sum(b_h * b_q[:, None], 0)
        
        tl.store(p_o, b_o)
```

## 6. 状态管理 (KV Cache)

GatedDeltaNet 需要维护两种状态:
1. **Conv1D 状态**: 因果卷积的历史输入
2. **SSM 状态**: 线性注意力的隐藏状态矩阵

### 6.1 状态布局

```python
# KV Cache 块布局:
# [SSM_STATE | CONV_STATE | ...]
# 
# SSM_STATE: [num_heads, head_v_dim, head_k_dim]
# CONV_STATE: [conv_kernel_dim - 1, qkv_size]
```

### 6.2 状态加载 (`block.py`)

```python
def load_initial_state_from_block_map(prefix_lengths, block_map, 
                                       conv_states, initial_states):
    """
    从 block_map 指向的 KV cache 块加载初始状态
    用于 prefill 时继续之前的计算
    """
```

### 6.3 状态保存 (`block.py`)

```python
def store_ssm_state_to_block_map(h, final_states, prefix_lengths, 
                                  cu_seqlens, block_map, ssm_states):
    """
    将中间状态和最终状态保存到 KV cache
    
    保存时机:
    1. 每个完整的 seq_size_per_block 边界
    2. 序列结束时的最终状态
    """
```

## 7. 完整前向传播

### 7.1 Qwen3NextGatedDeltaNet.forward

```python
def forward(self, hidden_states, fmha_impl, kv_cache, attention_inputs, attn_meta):
    # 1. 投影得到 Q, K, V, Z 和门控信号
    qkvz = self.in_proj_qkvz(hidden_states)
    ba = self.in_proj_ba(hidden_states)
    mixed_qkv, z, b, a = self.fix_query_key_value_ordering(qkvz, ba)
    
    # 2. 根据阶段选择处理方式
    if is_prefill:
        attn_output = self.prefill_gdn(mixed_qkv, b, a, ...)
    else:
        attn_output = self.decode_gdn(mixed_qkv, b, a, ...)
    
    # 3. Gated RMS Norm
    attn_output = self.norm(attn_output, z)
    
    # 4. 输出投影
    attn_output = self.out_proj(attn_output)
    
    return attn_output
```

### 7.2 Prefill 流程

```python
class Qwen3NextGatedDeltaNetPrefill:
    def forward(self, mixed_qkv, b, a, attn_inputs, kv_cache):
        # 1. 因果卷积 (SiLU 激活)
        mixed_qkv = self._conv1d(mixed_qkv, ...)
        
        # 2. 计算门控信号
        g, beta = fused_gdn_gating(self.alog, a, b, self.dt_bias)
        
        # 3. 加载初始状态
        initial_states = load_initial_state_from_block_map(...)
        
        # 4. 分块线性注意力
        attn_out, h, final_state = chunk_gated_delta_rule(
            q, k, v, g, beta, 
            initial_state=initial_states,
            use_qk_l2norm_in_kernel=True
        )
        
        # 5. 保存状态
        store_ssm_state_to_block_map(h, final_state, ...)
        
        return attn_out
```

### 7.3 Decode 流程

```python
class Qwen3NextGatedDeltaNetDecode:
    def forward(self, mixed_qkv, b, a, attn_inputs, kv_cache):
        # 1. 因果卷积更新
        mixed_qkv = self._conv1d(mixed_qkv, ...)
        
        # 2. 计算门控信号
        g, beta = fused_gdn_gating(self.alog, a, b, self.dt_bias)
        
        # 3. 递归更新
        attn_out, _ = fused_recurrent_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=ssm_states,
            inplace_final_state=True,
            use_qk_l2norm_in_kernel=True
        )
        
        return attn_out
```

## 8. 性能优化

### 8.1 Triton Kernel 优化

1. **分块处理**: 充分利用 GPU 共享内存
2. **融合计算**: 减少内存访问次数
3. **自动调优**: 使用 `triton.autotune` 选择最优配置

### 8.2 内存优化

1. **In-place 状态更新**: `inplace_final_state=True`
2. **分块状态缓存**: 只保存必要的中间状态
3. **连续批处理支持**: `block_map` 机制

### 8.3 计算优化

1. **L2 归一化融合**: 在 kernel 内完成
2. **门控融合**: `fused_gdn_gating` 单 kernel 完成
3. **WY 表示**: 避免显式矩阵求逆

## 9. 与标准 Attention 对比

| 特性 | Standard Attention | GatedDeltaNet |
|------|-------------------|---------------|
| 时间复杂度 | O(n²) | O(n) |
| 空间复杂度 | O(n²) | O(1) (固定状态大小) |
| 长序列支持 | 受限于显存 | 无限长度理论支持 |
| KV Cache | [n, h, d] | [h, k, v] (固定) |
| 推理延迟 | 随序列增长 | 恒定 |

## 10. 总结

GatedDeltaNet 通过以下机制实现高效的线性注意力:

1. **门控 Delta Rule**: 结合遗忘门和输入门控制信息流动
2. **分块并行计算**: Prefill 阶段利用分块结构并行处理
3. **递归更新**: Decode 阶段逐 token 高效递归
4. **WY 表示**: 通过矩阵分解避免显式求逆
5. **状态压缩**: 固定大小的隐藏状态支持无限长度序列

这种设计使得 GatedDeltaNet 特别适合长序列生成任务，在保持良好建模能力的同时显著降低了计算和存储开销。
