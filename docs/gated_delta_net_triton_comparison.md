# GatedDeltaNet 核心算子 Triton 实现对比：rtp-llm vs vLLM

本文档详细分析 GatedDeltaNet（Gated Delta Rule Linear Attention）在 rtp-llm 和 vLLM 两个推理框架中核心 Triton 算子的实现差异。

## 1. 概述

GatedDeltaNet 是一种高效的线性注意力机制，主要用于 Qwen3-Next 等新一代大语言模型。两个框架的实现都基于 [flash-linear-attention (FLA)](https://github.com/fla-org/flash-linear-attention) 项目，但针对各自的推理场景做了不同的优化。

### 核心算子列表

| 算子名称 | 功能 | rtp-llm 路径 | vLLM 路径 |
|---------|------|-------------|-----------|
| `fused_recurrent` | Decode 阶段逐 token 递归计算 | `fla/fused_recurrent.py` | `fla/ops/fused_recurrent.py` |
| `chunk` | Prefill 阶段分块并行计算入口 | `fla/chunk.py` | `fla/ops/chunk.py` |
| `chunk_delta_h` | 分块隐藏状态更新 | `fla/chunk_delta_h.py` | `fla/ops/chunk_delta_h.py` |
| `chunk_o` | 分块输出计算 | `fla/chunk_o.py` | `fla/ops/chunk_o.py` |
| `gdn_gating` | 门控信号 (g, beta) 融合计算 | `fla/gdn_gating.py` | 内联在模型文件中 |
| `l2norm` | L2 归一化 | `fla/l2norm.py` | `fla/ops/l2norm.py` |
| `solve_tril` | 下三角矩阵求逆 | `fla/solve_tril.py` | `fla/ops/solve_tril.py` |
| `wy_fast` | WY 表示重计算 | `fla/wy_fast.py` | `fla/ops/wy_fast.py` |
| `cumsum` | 块内累积求和 | `fla/cumsum.py` | `fla/ops/cumsum.py` |
| `chunk_scaled_dot_kkt` | K*K^T 缩放点积 | `fla/chunk_scaled_dot_kkt.py` | `fla/ops/chunk_scaled_dot_kkt.py` |

## 2. 核心差异分析

### 2.1 Fused Recurrent Kernel (Decode 阶段核心算子)

这是 Decode 阶段最核心的算子，负责逐 token 更新隐藏状态。

#### 2.1.1 状态管理方式差异

**rtp-llm 实现：使用 Block Map 分页管理**

```python
# rtp-llm: fused_recurrent.py
@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["block_map"] is not None,  # 关键差异
    }
)
```

```python
# rtp-llm 状态加载逻辑
if USE_INITIAL_STATE:
    if IS_CONTINUOUS_BATCHING:
        load_block_offset = cal_block_idx(sequence_length - 1, SEQ_SIZE_PER_BLOCK)
        read_block_id = tl.load(
            block_map + i_n * max_block_size + load_block_offset
        ).to(tl.int64)
        if read_block_id <= 0:
            return
        p_h0 = h0 + read_block_id * stride_init_state_token
```

**vLLM 实现：使用 SSM State Indices 索引管理**

```python
# vLLM: fused_recurrent.py
@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,  # 关键差异
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,  # vLLM 特有
    }
)
```

```python
# vLLM 状态加载逻辑
if USE_INITIAL_STATE:
    if IS_CONTINUOUS_BATCHING:
        if IS_SPEC_DECODING:
            i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
        else:
            i_t = 0
        state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(tl.int64)
        if state_idx < 0:  # PAD_SLOT_ID 检查
            return
        p_h0 = h0 + state_idx * stride_init_state_token
```

**差异总结：**

| 特性 | rtp-llm | vLLM |
|-----|---------|------|
| 状态索引方式 | `block_map` 分页表 + `seq_size_per_block` | `ssm_state_indices` 直接索引 |
| 块计算公式 | `(seq_len - 1) // seq_size_per_block` | 直接索引查表 |
| Speculative Decoding | 不支持 | 原生支持 (`num_accepted_tokens`) |
| 无效槽位检查 | `read_block_id <= 0` | `state_idx < 0` (PAD_SLOT_ID) |

#### 2.1.2 Tensor Stride 支持差异

**rtp-llm：显式 Stride 参数**

```python
# rtp-llm 显式传递所有 stride
stride_qb: tl.constexpr,  # q stride for batch/token dimension
stride_qs: tl.constexpr,  # q stride for head dimension
stride_qh: tl.constexpr,  # q stride for K dimension
stride_kb: tl.constexpr,
stride_ks: tl.constexpr,
stride_kh: tl.constexpr,
stride_vb: tl.constexpr,
stride_vs: tl.constexpr,
stride_vh: tl.constexpr,

# 使用 stride 访问数据
p_q = q + bos * stride_qs + i_h * stride_qh + o_k
p_k = k + bos * stride_ks + i_h * stride_kh + o_k
p_v = v + bos * stride_vs + i_hv * stride_vh + o_v
```

**vLLM：硬编码 Stride**

```python
# vLLM 使用硬编码的 stride 计算
p_q = q + (bos * H + i_h) * K + o_k
p_k = k + (bos * H + i_h) * K + o_k
p_v = v + (bos * HV + i_hv) * V + o_v
```

**差异影响：**
- rtp-llm 支持非连续 (non-contiguous) 张量，更灵活
- vLLM 假设连续内存布局，代码更简洁但限制更大

#### 2.1.3 Block Size 差异

```python
# rtp-llm
BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)  # BV=8

# vLLM
BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)  # BV=32
```

vLLM 使用更大的 `BV` 块，可能在某些配置下有更好的性能，但消耗更多寄存器。

#### 2.1.4 KDA (Key Dimension Attention) 支持

**vLLM 独有功能：**

```python
# vLLM 支持 IS_KDA 模式
IS_KDA: tl.constexpr,

if not IS_KDA:
    p_g = g + bos * HV + i_hv
else:
    p_gk = g + (bos * HV + i_hv) * K + o_k

# 状态衰减
if not IS_KDA:
    b_g = tl.load(p_g).to(tl.float32)
    b_h *= exp(b_g)
else:
    b_gk = tl.load(p_gk).to(tl.float32)
    b_h *= exp(b_gk[:, None])  # 每个 K 维度单独衰减
```

rtp-llm 不支持 KDA 模式，始终使用标量门控。

### 2.2 Chunk Prefill Kernel

#### 2.2.1 函数入口差异

**rtp-llm：返回额外的中间结果**

```python
# rtp-llm: chunk.py
def chunk_gated_delta_rule_fwd(...):
    ...
    return g, o, A, final_state, w, h, v_new  # 返回 w, h, v_new

# ChunkGatedDeltaRuleFunction.forward 返回
return o.to(q.dtype), h, final_state  # 返回 h
```

**vLLM：条件性返回**

```python
# vLLM: chunk.py
def chunk_gated_delta_rule_fwd(...):
    ...
    if SUPPRESS_LEVEL < 3:
        return g, o, A, final_state, None, None, None  # 不返回中间结果
    elif SUPPRESS_LEVEL >= 3:
        return g, o, A, final_state, w, h, v_new

# ChunkGatedDeltaRuleFunction.forward 返回
return o.to(q.dtype), final_state  # 不返回 h
```

**差异影响：**
- rtp-llm 始终保留中间结果 `h`，可能用于后续调试或特殊计算
- vLLM 通过 `SUPPRESS_LEVEL` 环境变量控制，默认不返回以节省内存

### 2.3 Chunk Delta H Kernel (隐藏状态更新)

#### 2.3.1 AutoTune 配置差异

**vLLM：启用 AutoTune**

```python
# vLLM: chunk_delta_h.py
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BV in [32, 64]
    ],
    key=["H", "K", "V", "BT"],
    use_cuda_graph=use_cuda_graph,
)
```

**rtp-llm：禁用 AutoTune，使用固定配置**

```python
# rtp-llm: chunk_delta_h.py
# @triton.autotune(...)  # 注释掉
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(...):
    ...

# 调用时使用固定参数
chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
    ...
    BV=32,          # 固定值
    num_warps=4,    # 固定值
    num_stages=2,   # 固定值
)
```

**差异原因：**
- vLLM 优先吞吐量，通过 AutoTune 自动选择最优配置
- rtp-llm 优先延迟稳定性，避免 AutoTune 带来的编译开销

#### 2.3.2 安全 Exp 函数

**rtp-llm：使用 safe_exp**

```python
# rtp-llm: chunk_delta_h.py
from rtp_llm.models_py.triton_kernels.fla.op import exp, safe_exp

if USE_G:
    b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
    ...
    b_v = b_v * safe_exp(b_g_last - b_g)[:, None]  # safe_exp
```

**vLLM：使用普通 exp 加掩码**

```python
# vLLM: chunk_delta_h.py
from .op import exp

if USE_G:
    m_t = (i_t * BT + tl.arange(0, BT)) < T
    b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
    ...
    b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]  # 掩码保护
```

#### 2.3.3 Hopper 架构优化

**rtp-llm：专门的 Hopper 优化**

```python
# rtp-llm: chunk_delta_h.py
from rtp_llm.models_py.triton_kernels.fla.utils import is_nvidia_hopper

NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8, 16]
```

vLLM 使用固定的 `NUM_WARPS = [2, 4, 8, 16]`，不区分架构。

### 2.4 GDN Gating Kernel

这是计算门控信号 `g` (forget gate) 和 `beta` (input gate) 的融合算子。

#### 2.4.1 代码组织差异

**rtp-llm：独立模块**

```python
# rtp-llm: fla/gdn_gating.py
@triton.jit
def fused_gdn_gating_kernel(
    g, beta_output, A_log, a, b, dt_bias, seq_len,
    NUM_HEADS: tl.constexpr,
    stride_ab: tl.constexpr,  # 支持非连续张量
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    ...
```

**vLLM：内联在模型文件中**

```python
# vLLM: models/qwen3_next.py
@triton.jit
def fused_gdn_gating_kernel(
    g, beta_output, A_log, a, b, dt_bias, seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,  # 无 stride_ab
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    ...
```

#### 2.4.2 Stride 支持差异

**rtp-llm：支持非连续张量**

```python
# rtp-llm
stride_ab = a.stride(0)  # 获取实际 stride
stride_ah = a.stride(1)
assert stride_ah == 1 and stride_bh == 1
assert stride_ab == stride_bb

ba_off = i_b * seq_len * stride_ab + i_s * stride_ab + head_off
blk_a = tl.load(a + ba_off, mask=mask)
```

**vLLM：假设连续**

```python
# vLLM
off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
blk_a = tl.load(a + off, mask=mask)
```

### 2.5 L2 Norm Kernel

#### 2.5.1 实现策略差异

**vLLM：三种 Kernel + 环境变量控制**

```python
# vLLM: l2norm.py
USE_DEFAULT_FLA_NORM = int(os.getenv("USE_DEFAULT_FLA_NORM", "0"))

def l2norm_fwd(x, eps=1e-6, output_dtype=None):
    if not USE_DEFAULT_FLA_NORM:
        # 默认使用优化的 kernel2
        l2norm_fwd_kernel2[(triton.cdiv(T, MBLOCK),)](...)
    else:
        if D <= 512:
            l2norm_fwd_kernel[grid](...)  # AutoTune 版本
        else:
            l2norm_fwd_kernel1[(T,)](...)  # 大维度版本
```

**rtp-llm：根据 T 和 D 选择**

```python
# rtp-llm: l2norm.py
def l2norm_fwd(x, eps=1e-6, output_dtype=None):
    # 不使用 AutoTune 以避免编译开销
    if D <= 512 and T <= 128:
        l2norm_fwd_kernel[grid](
            ...,
            BT=16,           # 固定配置
            num_warps=8,
            num_stages=3,
        )
    else:
        l2norm_fwd_kernel1[(T,)](
            ...,
            num_warps=8,
            num_stages=3,
        )
```

**差异原因：**
- vLLM 提供 `USE_DEFAULT_FLA_NORM` 环境变量切换实现
- rtp-llm 注释掉 AutoTune，使用固定配置避免编译开销

### 2.6 Custom Op 集成差异

#### vLLM：使用 `torch.compile` + Custom Op

```python
# vLLM: qwen3_next.py
def forward(self, hidden_states, output):
    ...
    torch.ops.vllm.gdn_attention_core(
        mixed_qkv, b, a, core_attn_out, self.prefix,
    )
    ...

direct_register_custom_op(
    op_name="gdn_attention_core",
    op_func=gdn_attention_core,
    mutates_args=["core_attn_out"],
    fake_impl=gdn_attention_core_fake,
)
```

#### rtp-llm：直接 Python 调用

```python
# rtp-llm: qwen3_next.py
def forward(self, hidden_states, ...):
    ...
    # 直接调用 Triton 函数
    if is_prefill:
        o, h, final_state = chunk_gated_delta_rule(...)
    else:
        o, final_state = fused_recurrent_gated_delta_rule(...)
    ...
```

**差异影响：**
- vLLM 通过 Custom Op 支持 `torch.compile`，可以与图编译器集成
- rtp-llm 使用直接调用，更简单但不支持图编译

## 3. 性能优化策略对比

### 3.1 AutoTune 策略

| 框架 | 策略 | 优点 | 缺点 |
|-----|------|-----|------|
| vLLM | 启用 AutoTune | 自动选择最优配置 | 首次运行有编译开销 |
| rtp-llm | 禁用 AutoTune | 延迟稳定，无编译抖动 | 可能非最优配置 |

### 3.2 CUDA Graph 支持

**vLLM：**
```python
# vLLM: chunk_delta_h.py
from .utils import use_cuda_graph

@triton.autotune(
    ...,
    use_cuda_graph=use_cuda_graph,  # 支持 CUDA Graph
)
```

**rtp-llm：**
未在 Triton kernel 层面直接支持 CUDA Graph，但在更上层的推理流程中可能有集成。

### 3.3 Speculative Decoding 支持

| 框架 | 支持程度 |
|-----|---------|
| vLLM | 原生支持 (`num_accepted_tokens`, `spec_query_start_loc`) |
| rtp-llm | 不支持 (无相关参数) |

## 4. 数值精度差异

### 4.1 safe_exp 实现

**rtp-llm：**
```python
# 使用 safe_exp 防止数值溢出
b_v = b_v * safe_exp(b_g_last - b_g)[:, None]
```

**vLLM：**
```python
# 使用掩码处理边界
b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
```

### 4.2 状态更新精度

两者都使用 `tl.float32` 进行中间计算：
```python
b_h = tl.zeros([BK, BV], dtype=tl.float32)
```

## 5. 代码可维护性对比

| 方面 | rtp-llm | vLLM |
|-----|---------|------|
| 模块化 | `gdn_gating` 独立模块 | 内联在模型文件 |
| 配置管理 | 代码内固定 | 环境变量 + AutoTune |
| Stride 支持 | 显式参数，更灵活 | 硬编码，更简洁 |
| 代码复用 | 算子可独立使用 | 与模型紧耦合 |

## 6. 总结

### 6.1 主要差异汇总

| 特性 | rtp-llm | vLLM |
|-----|---------|------|
| **状态管理** | Block Map 分页 | SSM State Indices |
| **Speculative Decoding** | 不支持 | 原生支持 |
| **KDA 模式** | 不支持 | 支持 |
| **AutoTune** | 禁用 | 启用 |
| **torch.compile** | 不支持 | 支持 (Custom Op) |
| **非连续张量** | 支持 | 不支持 |
| **Hopper 优化** | 有 | 无 |
| **数值安全** | safe_exp | 掩码保护 |

### 6.2 适用场景

**rtp-llm 更适合：**
- 低延迟敏感场景（避免 AutoTune 编译抖动）
- 需要灵活内存布局的场景
- 稳定性优先的生产环境

**vLLM 更适合：**
- 高吞吐量批处理场景
- 需要 Speculative Decoding 的场景
- 与 `torch.compile` 集成的场景

### 6.3 共同点

- 都基于 flash-linear-attention (FLA) 项目
- 核心计算逻辑相同（Delta Rule 公式一致）
- 都支持 Variable Length (cu_seqlens)
- 都使用 64 的 chunk size
- 都支持最大 256 维的 head dimension
