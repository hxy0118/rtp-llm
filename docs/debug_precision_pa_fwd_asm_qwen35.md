# ROCm PA 精度问题定位经验总结：qwen3.5 hybrid attention decode 阶段

## 问题背景

- **现象**：rtp-llm 在 qwen3.5（hybrid attention 网络，包含 full attention + linear attention）上，decode 阶段的 full attention 输出与 sglang 标杆存在较大精度差异
- **对比基线**：sglang 同模型、同输入、同配置下精度正确
- **已知条件**：prefill 阶段结果一致，仅 decode 阶段不一致；qwen3（纯 full attention，无 linear attention）在相同配置下精度正确

## 定位过程

### 第一步：缩小问题范围

1. **确认 prefill vs decode**：prefill 结果一致 → 问题仅在 decode 阶段
2. **确认模型差异**：qwen3（无 linear attn）精度 OK，qwen3.5（hybrid）精度不对 → 初步怀疑 hybrid cache 架构引入问题
3. **确认配置一致性**：`seq_size_per_block=1024`、`kernel_seq_size_per_block=16`、`block_table shape=[1,64]` 且值一致

### 第二步：排查 hybrid cache 架构是否污染 kv cache

**怀疑方向**：linear attention 的 kv cache 是否污染了 full attention 的 kv cache

**排查方法**：深入分析代码架构

- `HybridTypeKVCacheAllocator` 的 per-layer tensor 映射逻辑
- `BlockPoolConfigHelper` 中 `group_layer_num = gcd(linear_count, full_count)`
- `MemoryLayoutStrategy` 中 per-layer tensor 的创建方式
- `OpDefs.h` 中 `getLayerCache` 的 reshape 逻辑

**结论**：linear attention 不会直接污染 full attention 的 kv cache 数据。每层有独立的 per-layer tensor，不同 group 通过不同 block id 索引。

### 第三步：dump kv cache 数据对比

在 `AiterDecodeAttnOpAsm.forward` 中添加 dump 代码，逐层对比：

1. **dump prefill 写入的 kv cache** → 与 sglang 一致 ✅
2. **dump vectorized layout 提取的 k/v cache** → 确认 reshape 后数据正确 ✅
3. **dump kv_cache_base tensor 属性**（shape、stride、data_ptr、is_contiguous、storage_offset）→ 排查 reshape 导致的 tensor 不连续问题

### 第四步：对比 sglang 使用的算子

**关键发现**：sglang 和 rtp-llm 使用的是**完全不同的 PA 算子**！

| 维度 | sglang (qwen3.5) | rtp-llm (qwen3.5) |
|---|---|---|
| **算子** | `paged_attention_ragged` | `pa_fwd_asm` (ASM PA) |
| **page_size** | 1 | 16 (kernel_seq_size_per_block) |
| **KV cache layout** | NHD，原始 layout | vectorized layout (vs=8) |
| **k_cache view** | `(-1, 1, num_kv_heads, head_dim)` | `(num_blocks, num_kv_heads, head_dim/vs, block_size, vs)` |
| **来源** | `from aiter import paged_attention_ragged` | `from aiter import pa_fwd_asm` |

### 第五步：最终定位

**根因**：`pa_fwd_asm` 算子本身在 qwen3.5 网络上存在精度问题，与 hybrid cache 架构无关。qwen3 上该算子精度正常，说明是特定网络配置（如 head 数量、head_dim 等）触发了算子 bug。

## AMD 后端 KV Cache 特殊排布注意事项

### Vectorized Layout（ROCm ASM PA 专用）

AMD ROCm 后端的 `pa_fwd_asm` 算子要求 KV Cache 使用 **vectorized layout**，与 NVIDIA 后端和 `paged_attention_ragged` 使用的标准 NHD layout 完全不同。在 dump 和对比 KV Cache 数据时必须注意这一点，否则会看到"数据不一致"的假象。

#### 标准 layout vs vectorized layout

| 维度 | 标准 NHD layout | ROCm vectorized layout |
|---|---|---|
| **K cache** | `[num_blocks, num_kv_heads, block_size, head_dim]` | `[num_blocks, num_kv_heads, head_dim/vs, block_size, vs]` |
| **V cache** | `[num_blocks, num_kv_heads, block_size, head_dim]` | `[num_blocks, num_kv_heads, block_size/vs, head_dim, vs]` |
| **vs 值** | 不适用 | bf16: vs=8, fp16: vs=8, fp8: vs=16 |

其中 `vs`（vector size）是 AMD GPU 的 SIMD 宽度对齐参数，定义在 `rtp_llm/cpp/kernels/kv_cache/kv_cache_utils.h` 中。

#### K 和 V 的 vectorized 方式不同

**这是最容易踩坑的地方**：K cache 和 V cache 的 vectorize 维度不同！

- **K cache**：沿 `head_dim` 维度做 vectorize → `[numHeads, head_dim/vs, block_size, vs]`
- **V cache**：沿 `block_size`（token）维度做 vectorize → `[numHeads, block_size/vs, head_dim, vs]`

#### 从 vectorized layout 还原为标准 layout

```python
vs = 8  # bf16 的 vector size
block_idx = block_table[0, 0].item()

# K cache 还原：permute(0, 2, 1, 3) 把 block_size 移到 head_dim/vs 前面
k_block = key_cache[block_idx]  # [numHeads, head_dim/vs, block_size, vs]
k_restored = k_block.permute(0, 2, 1, 3).reshape(num_kv_heads, block_size, head_dim)

# V cache 还原：permute(0, 1, 3, 2) 把 vs 移到 head_dim 前面
v_block = value_cache[block_idx]  # [numHeads, block_size/vs, head_dim, vs]
v_restored = v_block.permute(0, 1, 3, 2).reshape(num_kv_heads, block_size, head_dim)
```

#### Hybrid Cache 的额外 reshape

对于 hybrid attention 网络（如 qwen3.5），full attention 层的 KV Cache 还会经过一次额外的 reshape（在 `OpDefs.h` 的 `getLayerCache` 中）：

```cpp
// 原始 shape: [physical_block_num, 2, num_kv_heads, seq_size_per_block, head_dim]
// full attention 层 reshape 为:
//   [kernel_block_num, 2, num_kv_heads, kernel_seq_size_per_block, head_dim]
// 其中 kernel_block_num = physical_block_num * (seq_size_per_block / kernel_seq_size_per_block)
```

这个 reshape 是通过 `torch.Tensor::reshape` 实现的，**不涉及数据拷贝**，只改变 stride。但需要注意 reshape 后的 tensor 可能不再是 contiguous 的，dump 时应检查 `is_contiguous()` 和 `stride()`。

### 不同算子对 KV Cache layout 的要求

| 算子 | 来源 | KV Cache Layout | Page Size |
|---|---|---|---|
| `pa_fwd_asm` | `aiter` (ASM 汇编) | vectorized layout | 通常 16 |
| `paged_attention_ragged` | `aiter` (flashinfer 风格) | 标准 NHD layout | 通常 1 |
| `unified_attention` | `aiter` (triton) | 标准 NHD layout | 可配置 |

**在对比不同框架的 KV Cache 数据时，必须先确认双方使用的 layout 是否一致，否则直接对比原始 tensor 数据是无意义的。**

## Dump 代码编写指南

### 方式一：使用项目内置的 `debug.py` 工具（推荐）

项目提供了通用的 tensor dump 工具 `rtp_llm/models_py/utils/debug.py`，支持环境变量控制、按层/步过滤、多 GPU rank 隔离、自动保存 `.pt` 文件，适合系统性的精度对比。

#### 环境变量配置

| 环境变量 | 说明 | 默认值 |
|---|---|---|
| `RTP_DUMP_TENSOR=1` | 启用 tensor dump | `0`（关闭） |
| `RTP_DUMP_TENSOR_DIR=/path` | `.pt` 文件保存目录 | `/tmp/rtp_dump_tensors` |
| `RTP_DUMP_TENSOR_LAYER=0,5` | 只 dump 指定层（逗号分隔） | 空（dump 所有层） |
| `RTP_DUMP_TENSOR_STEPS=0,1` | 只 dump 指定 step（逗号分隔） | `0`（只 dump 第一个 step） |
| `RTP_DUMP_TENSOR_SKIP_STEPS=10` | 跳过前 N 个 step（如 warmup） | `0` |

#### 启动示例

```bash
# 启用 dump，只看 layer 0 和 layer 5，跳过前 2 个 warmup step，dump step 0 和 step 1
RTP_DUMP_TENSOR=1 \
RTP_DUMP_TENSOR_LAYER=0,5 \
RTP_DUMP_TENSOR_STEPS=0,1 \
RTP_DUMP_TENSOR_SKIP_STEPS=2 \
RTP_DUMP_TENSOR_DIR=/tmp/my_debug_dump \
python -m rtp_llm.start_server ...
```

#### 在代码中使用

```python
from rtp_llm.models_py.utils.debug import dump_tensor, dump_tensor_enabled, dump_tensor_step_begin

# 在每个 forward step 开始时调用（用于 step 计数）
dump_tensor_step_begin()

# 在需要 dump 的位置
if dump_tensor_enabled():
    dump_tensor(query, "layer0.pa_fwd_asm.query", layer_idx=0)
    dump_tensor(output, "layer0.pa_fwd_asm.output", layer_idx=0)
```

#### `dump_tensor` 输出内容

每次调用会同时输出日志和保存文件：

- **日志**（stdout + logging）：`shape`、`dtype`、`mean`、`std`、`min`、`max`、`abs_mean`、`has_nan`、`has_inf`
- **文件**：保存到 `{RTP_DUMP_TENSOR_DIR}/rank{N}/step{S}_{name}.pt`，可用 `torch.load()` 加载后与标杆做逐元素对比

#### 对比两个框架的 dump 数据

```python
import torch

# 加载 rtp-llm 和 sglang 的 dump 数据
rtp_tensor = torch.load("/tmp/rtp_dump/rank0/step0_layer0.pa_output.pt")
sglang_tensor = torch.load("/tmp/sglang_dump/rank0/step0_layer0.pa_output.pt")

# 逐元素对比
diff = (rtp_tensor.float() - sglang_tensor.float()).abs()
print(f"max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")
print(f"relative_diff={diff.max() / sglang_tensor.float().abs().max():.6e}")
```

### 方式二：手动添加临时 dump 代码（灵活但需清理）

适合快速定位特定算子的问题，直接在算子调用前后添加 logging。

#### 在哪里加 dump

在 attention 算子调用的**前后**添加 dump，关键位置：

```python
# 文件：rtp_llm/models_py/modules/factory/attention/rocm_impl/aiter.py
# 类：AiterDecodeAttnOpAsm
# 方法：forward
```

#### dump 什么信息

**1. KV Cache Tensor 属性（排查 reshape/view 问题）**

```python
import logging

_base = kv_cache.kv_cache_base
logging.info(
    f"[DUMP] layer{layer_id} kv_cache_base tensor info:\n"
    f"  shape={list(_base.shape)}, dtype={_base.dtype}, "
    f"stride={list(_base.stride())}, contiguous={_base.is_contiguous()}, "
    f"data_ptr=0x{_base.data_ptr():x}, storage_offset={_base.storage_offset()}, "
    f"nbytes={_base.nelement() * _base.element_size()}\n"
    f"  seq_size_per_block={kv_cache.seq_size_per_block}, "
    f"layer_id={kv_cache.layer_id}"
)
```

**2. KV Cache 数据值（注意 vectorized layout 还原）**

```python
# 必须先从 vectorized layout 还原为标准 layout 再对比！
vs = 8  # bf16 的 vector size
block_idx = block_table[0, 0].item()

# K cache 还原
k_block = key_cache[block_idx]  # [numHeads, head_dim/vs, block_size, vs]
k_restored = k_block.permute(0, 2, 1, 3).reshape(num_kv_heads, block_size, head_dim)

# V cache 还原（注意 V 的 vectorize 维度与 K 不同！）
v_block = value_cache[block_idx]  # [numHeads, block_size/vs, head_dim, vs]
v_restored = v_block.permute(0, 1, 3, 2).reshape(num_kv_heads, block_size, head_dim)

seq_len = common.context_lengths_host[0].item()
logging.info(f"[DUMP] layer{layer_id} K[0,:5]: {k_restored[0, :seq_len, :5]}")
logging.info(f"[DUMP] layer{layer_id} V[0,:5]: {v_restored[0, :seq_len, :5]}")
```

**3. Attention 输入输出（定位算子问题）**

```python
# 算子调用前
logging.info(
    f"[DUMP] layer{layer_id} pa_fwd_asm inputs:\n"
    f"  query: shape={list(query.shape)}, dtype={query.dtype}\n"
    f"  key_cache: shape={list(key_cache.shape)}, data_ptr=0x{key_cache.data_ptr():x}\n"
    f"  value_cache: shape={list(value_cache.shape)}, data_ptr=0x{value_cache.data_ptr():x}\n"
    f"  block_table: shape={list(block_table.shape)}, values={block_table}\n"
    f"  context_lens: {context_lens}\n"
    f"  scale: {scale}"
)

# 算子调用
pa_fwd_asm(query, key_cache, value_cache, ...)

# 算子调用后
logging.info(
    f"[DUMP] layer{layer_id} pa_fwd_asm output:\n"
    f"  output: shape={list(output.shape)}, dtype={output.dtype}\n"
    f"  output[0,:5]: {output[0, :5]}\n"
    f"  output stats: min={output.min():.6f}, max={output.max():.6f}, "
    f"mean={output.float().mean():.6f}"
)
```

### 手动 dump 的最佳实践

```python
# 1. 使用环境变量控制 dump 开关，避免影响性能
import os
_dump = os.environ.get("DUMP_ATTN_DEBUG", "0") == "1"

# 2. 只 dump 前几个 step，避免日志爆炸
_dump_count = 0
_dump_max = 3

# 3. 在 forward 方法中
if _dump:
    global _dump_count
    if _dump_count < _dump_max:
        # ... dump 代码 ...
        if layer_id == last_layer_id:
            _dump_count += 1
```

## 经验总结

### 精度问题定位方法论

1. **先缩小范围**：prefill vs decode、哪些层、哪个模型配置
2. **与标杆逐层对比**：从输入到输出，逐步缩小差异出现的位置
3. **排查数据流**：kv cache 写入 → kv cache 读取 → attention 计算 → attention 输出
4. **确认算子一致性**：对比标杆使用的算子和自己使用的算子是否相同
5. **控制变量**：用已知精度正确的模型（如 qwen3）作为对照组

### 关键教训

- **不要假设算子本身没问题**：即使算子在某些模型上精度正确，也可能在特定配置下有 bug
- **先确认标杆用的是什么算子**：sglang 用的是 `paged_attention_ragged`，rtp-llm 用的是 `pa_fwd_asm`，这是两个完全不同的实现
- **hybrid cache 架构增加了复杂度**：reshape、stride、per-layer tensor 映射等都可能引入问题，需要逐一排查
- **dump tensor 属性比 dump 数据值更重要**：`stride`、`is_contiguous`、`data_ptr`、`storage_offset` 这些属性能快速定位 reshape/view 导致的内存布局问题

### 排查清单（Checklist）

- [ ] 确认 prefill 和 decode 哪个阶段有问题
- [ ] 确认是所有层都有问题还是特定层
- [ ] dump kv cache 写入数据，与标杆对比
- [ ] dump kv cache 读取数据（注意 vectorized layout 的还原）
- [ ] dump attention 算子的所有输入参数（shape、dtype、stride、data_ptr）
- [ ] dump attention 算子的输出，与标杆对比
- [ ] 确认标杆使用的算子和自己使用的算子是否一致
- [ ] 如果算子不同，尝试换成标杆的算子验证
- [ ] 对比不同模型配置（如 qwen3 vs qwen3.5）找出触发条件
