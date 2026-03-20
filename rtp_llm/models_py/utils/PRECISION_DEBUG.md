# RTP-LLM 精度调试 Tensor Dump 使用说明

## 快速开始

```bash
# 最简用法：dump 所有层的第一个 forward step
RTP_DUMP_TENSOR=1 python -m rtp_llm.server ...

# 只 dump 第 0、5 层（减少 IO 开销）
RTP_DUMP_TENSOR=1 RTP_DUMP_TENSOR_LAYER=0,5 python -m rtp_llm.server ...

# 自定义保存路径
RTP_DUMP_TENSOR=1 RTP_DUMP_TENSOR_DIR=/data/debug_tensors python -m rtp_llm.server ...

# dump 多个 step
RTP_DUMP_TENSOR=1 RTP_DUMP_TENSOR_STEPS=0,1 python -m rtp_llm.server ...
```

## 环境变量

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `RTP_DUMP_TENSOR` | `0` | 设为 `1` 启用 |
| `RTP_DUMP_TENSOR_DIR` | `/tmp/rtp_dump_tensors` | `.pt` 文件保存目录 |
| `RTP_DUMP_TENSOR_LAYER` | 空（全部层） | 逗号分隔的层号，如 `0,5,10` |
| `RTP_DUMP_TENSOR_STEPS` | `0`（仅第一步） | 逗号分隔的 step 号 |

## Dump 点覆盖

### Qwen3NextModel（`qwen3_next.py`）

| Dump 名称 | 位置 |
|-----------|------|
| `embedding_out` | embedding 输出 |
| `final_norm_out` | 最终 RMSNorm 输出 |

### Qwen3NextDecoderLayer（每层）

| Dump 名称 | 位置 |
|-----------|------|
| `layer{i}.input` | 层输入 hidden_states |
| `layer{i}.input_layernorm_out` | input RMSNorm 之后 |
| `layer{i}.attn_out` | attention 输出（allreduce 之后） |
| `layer{i}.post_attn_layernorm_out` | post-attention RMSNorm 之后 |
| `layer{i}.mlp_out` | MLP/MoE 输出 |

### Full Attention 层（`CausalAttention` + `Qwen3NextAttention`）

| Dump 名称 | 位置 |
|-----------|------|
| `layer{i}.full_attn.gate` | Qwen3Next gate 投影输出 |
| `layer{i}.full_attn.qkv_proj_out` | QKV 线性投影输出 |
| `layer{i}.full_attn.qk_norm_out` | FusedQKRMSNorm 之后 |
| `layer{i}.full_attn.fmha_out` | Flash Attention 输出 |
| `layer{i}.full_attn.o_proj_out` | O 投影输出（allreduce 前） |
| `layer{i}.full_attn.allreduce_out` | AllReduce 之后（TP>1 时） |
| `layer{i}.full_attn.out` | gate sigmoid 乘法后最终输出 |

### Linear Attention 层（`Qwen3NextGatedDeltaNet`）

| Dump 名称 | 位置 |
|-----------|------|
| `layer{i}.linear_attn.mixed_qkv` | QKV 投影 + 拆分后 |
| `layer{i}.linear_attn.z` | z gate 向量 |
| `layer{i}.linear_attn.b` | beta 向量 |
| `layer{i}.linear_attn.a` | alpha 向量 |
| `layer{i}.linear_attn.fla_out` | chunk_gated_delta_rule / fused_recurrent 输出 |
| `layer{i}.linear_attn.norm_out` | RmsNormGated 之后 |
| `layer{i}.linear_attn.out_proj` | 输出投影后（allreduce 前） |

## 输出示例

**日志输出**（stdout）：
```
[DUMP] step=1 embedding_out: shape=[128, 4096], dtype=torch.bfloat16, mean=0.001234, std=0.045678, min=-0.234567, max=0.345678, abs_mean=0.034567, has_nan=False, has_inf=False
[DUMP] step=1 layer0.input_layernorm_out: shape=[128, 4096], ...
[DUMP] step=1 layer0.full_attn.qkv_proj_out: shape=[128, 6144], ...
```

**文件输出**（`/tmp/rtp_dump_tensors/`）：
```
step1_embedding_out.pt
step1_layer0.input.pt
step1_layer0.input_layernorm_out.pt
step1_layer0.full_attn.qkv_proj_out.pt
step1_layer0.full_attn.fmha_out.pt
step1_layer1.linear_attn.mixed_qkv.pt
step1_layer1.linear_attn.fla_out.pt
step1_final_norm_out.pt
...
```

## CUDA vs ROCm 精度对比脚本

分别在 CUDA 和 ROCm 环境下用相同输入跑一次，然后对比：

```python
import os
import glob
import torch

cuda_dir = "/path/to/cuda_dump"
rocm_dir = "/path/to/rocm_dump"

for f in sorted(glob.glob(f"{cuda_dir}/*.pt")):
    name = os.path.basename(f)
    rocm_f = os.path.join(rocm_dir, name)
    if not os.path.exists(rocm_f):
        print(f"MISSING in ROCm: {name}")
        continue
    a = torch.load(f, weights_only=True).float()
    b = torch.load(rocm_f, weights_only=True).float()
    cos = torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
    ).item()
    max_diff = (a - b).abs().max().item()
    status = "✅" if cos > 0.999 else "❌"
    print(f"{status} {name}: cos_sim={cos:.8f}, max_abs_diff={max_diff:.6f}")
```

第一个出现 `❌` 的位置就是精度发散点。

## 改动文件

| 文件 | 改动 |
|------|------|
| `rtp_llm/models_py/utils/debug.py` | 新增 `dump_tensor()` / `dump_tensor_enabled()` / `dump_tensor_step_begin()` |
| `rtp_llm/models_py/model_desc/qwen3_next.py` | Model / DecoderLayer / Attention / GatedDeltaNet forward 添加 dump |
| `rtp_llm/models_py/modules/hybrid/causal_attention.py` | CausalAttention forward 添加 qkv / qk_norm / fmha / o_proj / allreduce dump |
