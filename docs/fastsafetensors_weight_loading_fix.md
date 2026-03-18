# fastsafetensors 权重加载 OOM 修复

## 问题背景

使用 fastsafetensors 加载 Qwen3.5-397B（TP8, MI300X 192GB）时，权重加载阶段 OOM：

```
HIP out of memory. Tried to allocate 8.00 GiB. GPU 5 has a total capacity of 191.98 GiB
of which 5.43 GiB is free. Of the allocated memory 184.00 GiB is allocated by PyTorch.
```

模型权重总计 752GB，TP8 split 后每卡仅需 ~94GB，理论上 192GB 完全够用。

## 根因分析

### fastsafetensors 加载流程

fastsafetensors 按批加载 safetensors 文件到 GPU（每批 `pg.size()` 个文件），然后 yield 出每个 tensor。rtp-llm 通过 `TensorCollector` 收集 tensor，当一个 `WeightModule` 所需的全部 tensor 到齐后触发 `weight.load()`（包含 TP split），split 后的小 tensor 存入 `model_weights`，原始大 tensor 引用释放。

### OOM 原因：Collector 积压

Qwen3.5-397B 使用 stacked MoE 权重格式，每层的 `experts.gate_up_proj` 是一个 8GB 的 tensor（512 experts × 2048 × 4096 × bf16）。

**原始实现中**，整层的所有权重（包括 MoE expert 权重、gate、shared expert、attention 等）共用一个 `TensorCollector`。Collector 需要等所有子权重的 tensor 全部到齐才触发 load+split：

```
Collector 等待:
  ├── experts.gate_up_proj  (8GB)  ← 第 2 批到达，存入 collector
  ├── experts.down_proj     (4GB)  ← 第 5 批才到达
  ├── gate.weight           (tiny) ← 第 8 批才到达  
  ├── shared_expert.*       (tiny) ← 第 10 批才到达
  └── attention.*           (tiny) ← 第 12 批才到达
```

在 `down_proj` 到达之前，`gate_up_proj`（8GB）一直被 collector 持有在 GPU 上。60 层的 `gate_up_proj` 逐步积压：

```
第 20 个 tensor 到达时:
  - completed = 0（没有任何 collector complete）
  - pending = 20 × 8GB = 160GB（全部积压在 GPU 上）
  - GPU allocated = 168GB → 下一个 8GB tensor 分配时 OOM
```

### 为什么 Qwen3-235B（分离 expert）不受影响

Qwen3-235B 的 expert 权重是分离格式（`experts.0.gate_proj`, `experts.1.gate_proj`...），每个 tensor 仅 ~12MB。即使 collector 积压了 256 个 expert tensor，也只有 ~3GB，不会 OOM。

### 为什么 CUDA EP8 不受影响

CUDA 使用 A100 80GB 卡，`_is_memory_enough_for_fastsafetensor` 检查 `(78GB - 94GB) > 27GB` 不通过，被迫走了 scratch（逐 tensor 从 CPU 加载）路径，不存在 collector 积压问题。

## 修复方案

### 核心修改：拆分 Collector 粒度

将 `CompositeWeight`（如 `MoeWithSharedWeight`）拆分为独立的子 `AtomicWeight`，每个子权重拥有自己的 `TensorCollector`：

```python
# 修复前: 一个 CompositeWeight → 一个 Collector，等所有子权重到齐
names = weight.get_tensor_names(layer_id, load_config)  # 返回所有子权重的 tensor name
collector = TensorCollector(names, ...)  # 等全部到齐

# 修复后: 每个子 AtomicWeight → 独立 Collector，收到即 complete
for component in weight.get_components():  # 拆分为子 AtomicWeight
    names = component.get_tensor_names(layer_id, load_config)  # 每个只需 1 个 tensor
    collector = TensorCollector(names, ...)  # 收到即 complete
```

修复后，每个 8GB 的 `gate_up_proj` 到达后立即 complete → split 成 1GB → 释放原始 8GB。

### 修复效果（Qwen3.5-397B, TP8）

| | 修复前 | 修复后 |
|---|---|---|
| 20 个 tensor 后 completed | 0 | 20 |
| 20 个 tensor 后 pending | 160GB | 0GB |
| 20 个 tensor 后 GPU alloc | 168GB → **OOM** | 36GB |
| 最终 GPU alloc | OOM | ~94GB ✓ |

### MoeAtomicWeight 独立 postprocess

由于子权重现在独立 load，`MoeWithSharedWeight._postprocess` 中的 `shuffle_moe_weight` 逻辑不再被执行。因此在 `MoeAtomicWeight` 中覆写了 `_postprocess`，确保 `moe_w1`/`moe_w2` 在独立加载路径中也执行 shuffle：

```python
class MoeAtomicWeight(AtomicWeight):
    def _postprocess(self, tensor, device, load_config):
        raw_tensor = tensor.get(self.name) if isinstance(tensor, dict) else tensor
        if self.name in [W.moe_w1, W.moe_w2]:
            raw_tensor = load_config.exported_device.shuffle_moe_weight(
                raw_tensor, load_config.compute_dtype, self.name
            )
        return {self.name: load_config.exported_device.maybe_rewrite_weight_by_key(
            self.name, raw_tensor
        )}
```

### 对非 stacked MoE（Qwen3-235B 等）的影响

无负面影响。分离 expert 格式的 `MoeAtomicWeight.get_components()` 返回 `[self]`，行为与修改前完全一致。每个 expert tensor 仅 ~12MB，即使 collector 积压也只有 ~3GB。

## 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `rtp_llm/model_loader/loader.py` | `_generate_weight_info` 中对 CompositeWeight 使用 `get_components()` 拆分 collector |
| `rtp_llm/model_loader/ffn_weight.py` | `MoeAtomicWeight` 覆写 `_postprocess`，支持独立加载时执行 shuffle |
| `rtp_llm/device/device_impl.py` | `shuffle_moe_weight` 去掉 512 padding，改用 `aiter.ops.shuffle.shuffle_weight`（与 sglang 对齐） |
