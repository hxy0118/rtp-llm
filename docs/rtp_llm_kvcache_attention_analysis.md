# RTP-LLM KVCache 管理与 Attention 实现技术分析

本文档详细分析 rtp-llm 的 KVCache 管理机制、Attention 实现方式，以及对并行解码、公共前缀等高级特性的支持。

## 1. KVCache 架构概述

### 1.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              KVCache 管理架构                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   KVCacheConfig │    │   CacheConfig   │    │  BlockPoolConfig│         │
│  │   (Python)      │ -> │   (C++)         │ -> │  (C++)          │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│           │                      │                      │                   │
│           v                      v                      v                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         KVCacheManager                               │   │
│  │  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐   │   │
│  │  │  KVCacheAllocator│   │    BlockPool    │   │   BlockCache    │   │   │
│  │  │  (分配策略)      │   │  (物理块管理)   │   │  (LRU 缓存)     │   │   │
│  │  └─────────────────┘   └─────────────────┘   └─────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    v                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          KVCache (Tensor)                            │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │ kv_cache_base: [num_blocks, seq_per_block, 2, heads, head_dim] │ │   │
│  │  │ kv_scale_base: [num_blocks, seq_per_block, 2, heads] (可选)    │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 核心数据结构

#### Python 层

```python
# rtp_llm/ops/librtp_compute_ops/__init__.pyi
class KVCache:
    def __init__(self) -> None: ...
    def get_layer_cache(self, layer_id: int) -> KVCache: ...

    @property
    def kv_cache_base(self) -> torch.Tensor:
        """Key-Value 缓存主体张量"""

    @property
    def kv_scale_base(self) -> torch.Tensor:
        """量化缩放因子张量 (INT8/FP8)"""

    @property
    def layer_id(self) -> int:
        """当前层 ID"""

    @property
    def seq_size_per_block(self) -> int:
        """每个物理块容纳的 token 数量"""
```

#### C++ 层

```cpp
// rtp_llm/cpp/cache/CacheConfig.h
enum KVCacheType {
    MultiHeadAttention,       // 标准 MHA
    MultiHeadLatentAttention, // MLA (DeepSeek-V2)
    LinearAttention,          // 线性注意力 (GatedDeltaNet)
};

struct KVCacheSpec {
    uint32_t layer_num;
    uint32_t local_head_num_kv;
    uint32_t seq_size_per_block = 1;
    KVCacheType type;
    DataType dtype;

    virtual size_t block_size() const = 0;
    virtual size_t k_block_size() const = 0;
    virtual size_t v_block_size() const = 0;
};
```

### 1.3 支持的缓存类型

| 缓存类型 | 描述 | 块大小计算 |
|---------|------|-----------|
| **MHAKVCacheSpec** | 标准多头注意力 | `2 × heads × head_dim × seq_per_block` |
| **MLAKVCacheSpec** | 多头潜在注意力 (MLA) | `heads × (kv_lora_rank + rope_head_dim) × seq_per_block` |
| **LinearKVCacheSpec** | 线性注意力 (GatedDeltaNet) | `(conv_state_size + temporal_state_size) × seq_per_block` |

## 2. Paged Attention 实现

### 2.1 分页机制

rtp-llm 实现了类似 vLLM 的 Paged Attention 机制：

```
逻辑序列:     [tok0, tok1, tok2, tok3, tok4, tok5, tok6, tok7, ...]
                  ↓           ↓           ↓           ↓
物理块:       Block 0    Block 1    Block 2    Block 3    (seq_size_per_block=2)
                  ↓           ↓           ↓           ↓
Block Map:     [5,        3,        7,        1, ...]   (物理块索引)
```

#### 关键参数

```python
# 配置项 (kv_cache_group_args.py)
--seq_size_per_block    # 每块 token 数量，默认 64
--kv_cache_mem_mb       # KVCache 总内存大小
--reserve_block_ratio   # 预留块百分比 (防止运行时分配失败)
```

### 2.2 Block Map 管理

```python
# PyAttentionInputs 中的关键字段
class PyAttentionInputs:
    kv_cache_block_id_device: torch.Tensor  # 设备端块索引 [batch, max_blocks]
    kv_cache_block_id_host: torch.Tensor    # 主机端块索引 [batch, max_blocks]
    sequence_lengths: torch.Tensor          # 每个序列的当前长度
    input_lengths: torch.Tensor             # 每个序列的输入长度
    prefix_lengths: torch.Tensor            # 前缀长度 (用于缓存复用)
    cu_seqlens: torch.Tensor                # 累积序列长度 (用于 varlen)
```

### 2.3 Block 分配与回收

```cpp
// rtp_llm/cpp/cache/BlockPool.h
class BlockPool {
public:
    std::vector<BlockIdxType> malloc(int num_blocks);  // 分配块
    void requestFree(BlockIdxType block_idx);          // 释放块
    void requestReference(BlockIdxType block_idx);     // 增加引用计数

    BlockCachePtr blockCache();  // 获取 LRU 缓存
    size_t freeBlocksNum() const;
    size_t availableBlocksNum() const;
};
```

## 3. Attention 实现机制

### 3.1 FMHA 后端选择

rtp-llm 支持多种 FMHA (Fused Multi-Head Attention) 后端：

```python
# rtp_llm/ops/__init__.py
class FMHAType(Enum):
    NONE = 0
    FLASH_INFER = 1       # FlashInfer
    TRT = 2               # TensorRT
    XQA = 3               # XQA (自研)
    # ... 更多后端
```

### 3.2 FMHA 实现基类

```python
# rtp_llm/models_py/modules/factory/attention/fmha_impl_base.py
class FMHAImplBase:
    def __init__(self, fmha_impl, rope_kvcache_impl, attn_inputs):
        self.fmha_impl = fmha_impl
        self.rope_kvcache_impl = rope_kvcache_impl
        self.attn_inputs = attn_inputs

    def forward(self, qkv: torch.Tensor, kv_cache: KVCache, need_rope_kv_cache=True):
        # 1. RoPE + KVCache 写入
        if need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, ...)
        # 2. Attention 计算
        res = self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)
        # 3. Cache Store (可选)
        if self.write_cache_store_impl:
            self.write_cache_store_impl(kv_cache)
        return res
```

### 3.3 FlashInfer 实现

```python
# rtp_llm/models_py/modules/factory/attention/cuda_impl/flash_infer.py
class FlashInferPrefillImpl(FMHAPrefillImplBase):
    def __init__(self, attn_configs, attn_inputs):
        super().__init__(
            FlashInferPrefillOp(attn_configs),
            FusedRopeKVCachePrefillOp(attn_configs),
            attn_inputs,
        )

    def support_cuda_graph(self) -> bool:
        return True

class FlashInferDecodeImpl(FMHADecodeImplBase):
    def prepare(self, attn_inputs):
        batch_size = attn_inputs.input_lengths.size(0)
        self.fmha_params.fill_params(
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_block_id_host,
            batch_size,
            self.seq_size_per_block,
        )
```

### 3.4 Attention 工厂模式

```python
# rtp_llm/models_py/modules/__init__.py
class AttnImplFactory:
    @staticmethod
    def get_fmha_impl(config, parallelism_config, weight, attn_inputs, fmha_config, is_cuda_graph):
        # 根据配置选择最优的 FMHA 实现
        # 优先级: FlashInfer > TRT > XQA > ...
        pass
```

## 4. 公共前缀缓存 (Prefix Cache / Reuse Cache)

### 4.1 功能概述

rtp-llm 支持公共前缀缓存，允许多个请求共享相同的前缀 KV Cache：

```
请求 1: [System Prompt] + [User Query 1]
请求 2: [System Prompt] + [User Query 2]
                 ↓
        共享 System Prompt 的 KV Cache
```

### 4.2 配置方式

```bash
# 启用缓存复用
export REUSE_CACHE=true

# 配置多任务提示
export MULTI_TASK_PROMPT=/path/to/multi_task_prompt.json
# 或使用字符串
export MULTI_TASK_PROMPT_STR='[{"task_id": "1", "prompt": "You are a helpful assistant."}]'

# Memory Block Cache (可选)
export MEMORY_BLOCK_CACHE_SIZE_MB=1024
```

### 4.3 缓存键计算

```python
# rtp_llm/ops/__init__.py
def get_block_cache_keys(token_ids: List[int], block_size: int) -> List[int]:
    """
    将 token_ids 分块并计算每块的缓存键。
    只保留完整块的缓存键。
    """
    token_ids_list: List[List[int]] = []
    for i in range(0, len(token_ids), block_size):
        chunk = token_ids[i : i + block_size]
        if len(chunk) == block_size:  # 只保留完整块
            token_ids_list.append(chunk)
    return cpp_get_block_cache_keys(token_ids_list)
```

### 4.4 BlockCache (LRU 缓存)

```cpp
// rtp_llm/cpp/cache/BlockCache.h
class BlockCache {
public:
    struct CacheItem {
        CacheKeyType cache_key;
        GroupIdType  group_id;
        BlockIdxType block_index;
        bool         is_resident = false;  // 是否常驻
    };

    bool put(CacheItem& cache_item);
    MatchResult match(CacheKeyType cache_key, int group_id = 0);
    BlockIndicesType pop(int n);  // 淘汰 n 个 LRU 块

private:
    LRUCache<CacheKeyGroupPair, CacheItem> lru_cache_;
};
```

### 4.5 使用场景

| 场景 | 描述 | 配置 |
|-----|------|-----|
| **System Prompt 缓存** | 多请求共享系统提示词 | `REUSE_CACHE=true` |
| **多任务提示** | 不同任务共享各自的提示词 | `MULTI_TASK_PROMPT` |
| **常驻缓存** | 某些块永不淘汰 | `is_resident=true` |

## 5. 并行解码支持

### 5.1 Speculative Decoding (投机解码)

rtp-llm 支持基于 MTP (Multi-Token Prediction) 的投机解码：

```python
# 配置项 (speculative_decoding_group_args.py)
--sp_type              # 投机类型: "vanilla" (不启用), "mtp" (启用)
--sp_model_type        # 草稿模型类型: "mixtbstars-mtp", "deepseek-v3-mtp"
--sp_min_token_match   # 最小匹配 token 数 (默认 2)
--sp_max_token_match   # 最大匹配 token 数 (默认 2)
--gen_num_per_cycle    # 每轮生成 token 数 (默认 1)
```

#### 执行流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    Speculative Decoding 流程                     │
├─────────────────────────────────────────────────────────────────┤
│  1. Propose 阶段 (草稿模型)                                       │
│     - 使用小模型生成 N 个候选 token                                │
│     - 快速但精度较低                                              │
│                                                                  │
│  2. Score 阶段 (目标模型)                                         │
│     - 使用大模型验证候选 token                                     │
│     - 一次前向传播验证多个 token                                   │
│                                                                  │
│  3. Accept/Reject                                                │
│     - 接受匹配的 token                                            │
│     - 从第一个不匹配处重新生成                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Tree Decoding (树形解码)

```python
--tree_decode_config   # 树形解码配置文件路径
```

### 5.3 MTP 模块 KVCache

```cpp
// CacheConfig 支持 MTP 子配置
struct CacheConfig {
    std::vector<std::shared_ptr<CacheConfig>> mtp_sub_configs;
    // ...
};

// KVCacheManager 支持获取 MTP 模块的 KVCache
class KVCacheManager {
    KVCacheBuffer getMTPModuleKVCacheBuffer(int mtp_module_id) const;
    const CacheConfig& getMTPModuleCacheConfig(int mtp_module_id) const;
};
```

## 6. 量化 KVCache

### 6.1 支持的量化类型

| 类型 | 配置 | 内存节省 |
|-----|------|---------|
| **INT8** | `--int8_kv_cache=1` | ~50% |
| **FP8** | `--fp8_kv_cache=1` | ~50% |

### 6.2 实现细节

```cpp
// MemoryLayoutConfig 中的 Scale 支持
struct MemoryLayoutConfig {
    bool enable_kv_scale = false;

    size_t kv_cache_offset_bytes = 0;   // KV 缓存偏移
    size_t kv_scale_offset_bytes = 0;   // Scale 偏移

    size_t kv_block_pool_size_bytes = 0;
    size_t kv_scale_pool_size_bytes = 0;

    bool hasScale() const {
        return enable_kv_scale && kv_scale_pool_size_bytes > 0;
    }
};
```

```python
# KVCache 类的 scale 属性
class KVCache:
    @property
    def kv_scale_base(self) -> torch.Tensor:
        """量化缩放因子"""
```

## 7. 混合注意力模型支持

### 7.1 Hybrid Attention (标准 Attention + Linear Attention)

rtp-llm 支持混合注意力模型，如 Qwen3-Next：

```python
# Qwen3-Next 层类型配置
layer_types = [
    "linear_attention",  # GatedDeltaNet
    "linear_attention",
    "full_attention",    # 标准 Attention
    "linear_attention",
    ...
]
```

### 7.2 Linear Attention KVCache 布局

对于 GatedDeltaNet 等线性注意力：

```cpp
struct LinearKVCacheSpec : public KVCacheSpec {
    uint32_t conv_state_size;      // Conv1D 状态大小
    uint32_t temporal_state_size;  // SSM 隐藏状态大小

    size_t k_block_size() const override {
        return conv_state_size * seq_size_per_block;
    }
    size_t v_block_size() const override {
        return temporal_state_size * seq_size_per_block;
    }
};
```

### 7.3 MLA (Multi-head Latent Attention)

支持 DeepSeek-V2 风格的 MLA：

```cpp
struct MLAKVCacheSpec : public KVCacheSpec {
    uint32_t kv_lora_rank;   // 压缩后的 KV 维度
    uint32_t rope_head_dim;  // RoPE 部分维度

    size_t k_block_size() const override {
        return local_head_num_kv * kv_lora_rank * seq_size_per_block;
    }
    size_t v_block_size() const override {
        return local_head_num_kv * rope_head_dim * seq_size_per_block;
    }
};
```

## 8. CUDA Graph 支持

### 8.1 概述

rtp-llm 支持 CUDA Graph 以减少 kernel launch 开销：

```python
class FMHAImplBase:
    def support_cuda_graph(self) -> bool:
        return False  # 默认不支持

class FlashInferDecodeImpl(FMHADecodeImplBase):
    def support_cuda_graph(self) -> bool:
        return True  # FlashInfer 支持
```

### 8.2 参数更新

```python
# CUDA Graph 模式下的参数填充
class GptModelBase:
    def fill_params(self, sequence_lengths, input_lengths, kv_cache_block_id_host,
                   replay_batch_size, capture_batch_size, seq_size_per_block):
        params_ptr = self.params_dict[capture_batch_size]
        params_ptr.fillParams(
            sequence_lengths, input_lengths, kv_cache_block_id_host,
            replay_batch_size, seq_size_per_block
        )
```

## 9. 分布式 KVCache

### 9.1 Tensor Parallelism

每个 TP rank 持有部分 KV heads：

```cpp
struct KVCacheSpec {
    uint32_t local_head_num_kv;  // 本地 KV head 数量 = total_heads / tp_size
};
```

### 9.2 Memory Block Cache 同步

```python
--memory_block_cache_size_mb        # 每个 RANK 的缓存大小
--memory_block_cache_sync_timeout_ms # 多 TP 同步超时时间 (默认 10000ms)
```

## 10. 功能支持矩阵

| 功能 | 是否支持 | 备注 |
|-----|---------|------|
| **Paged Attention** | ✅ | 类似 vLLM 的分页机制 |
| **公共前缀缓存** | ✅ | `REUSE_CACHE=true` |
| **多任务提示缓存** | ✅ | `MULTI_TASK_PROMPT` |
| **INT8 KVCache** | ✅ | `--int8_kv_cache=1` |
| **FP8 KVCache** | ✅ | `--fp8_kv_cache=1` |
| **Speculative Decoding** | ✅ | MTP 模式 |
| **Tree Decoding** | ✅ | 配置文件方式 |
| **Continuous Batching** | ✅ | 默认启用 |
| **CUDA Graph** | ✅ | FlashInfer 后端 |
| **MLA** | ✅ | DeepSeek-V2 |
| **Linear Attention** | ✅ | GatedDeltaNet |
| **Tensor Parallelism** | ✅ | 自动分片 |
| **Pipeline Parallelism** | ✅ | 需配置 |

## 11. 性能调优建议

### 11.1 内存配置

```bash
# 根据模型大小调整 KVCache 内存
export KV_CACHE_MEM_MB=8192

# 调整块大小 (权衡内存碎片 vs 复用效率)
export SEQ_SIZE_PER_BLOCK=64  # 较小值减少碎片
export SEQ_SIZE_PER_BLOCK=128 # 较大值提高复用率
```

### 11.2 缓存策略

```bash
# 启用缓存复用
export REUSE_CACHE=true

# 配置预留块比例 (防止分配失败)
export RESERVE_BLOCK_RATIO=5  # 5%
```

### 11.3 量化建议

| 场景 | 推荐配置 |
|-----|---------|
| 长上下文 | FP8 KVCache |
| 内存受限 | INT8 KVCache |
| 精度敏感 | FP16/BF16 (默认) |

## 12. 总结

rtp-llm 的 KVCache 管理具有以下特点：

1. **灵活的分页管理**：支持动态分配和释放，减少内存碎片
2. **丰富的缓存类型**：支持 MHA、MLA、Linear Attention 等多种架构
3. **高效的前缀复用**：LRU 缓存 + 多任务提示支持
4. **完善的量化支持**：INT8/FP8 量化减少内存占用
5. **并行解码能力**：Speculative Decoding、Tree Decoding
6. **分布式友好**：TP/PP 并行支持
7. **CUDA Graph 优化**：减少 kernel launch 开销

通过合理配置，rtp-llm 可以在保证推理质量的前提下，最大化吞吐量和资源利用率。
