# RTP-LLM 整体架构详解

## 1. 项目概述

RTP-LLM (Real-Time Production LLM) 是阿里巴巴开源的高性能大语言模型推理引擎。它采用 Python + C++ 混合架构，支持多种主流 LLM 模型，并提供高效的分布式推理能力。

### 1.1 核心特性

- **高性能推理**: 基于 CUDA/ROCm 的优化 kernel
- **多模型支持**: Qwen、LLaMA、ChatGLM、DeepSeek 等
- **分布式部署**: 支持 TP/DP/PP 并行策略
- **Continuous Batching**: 连续批处理最大化吞吐
- **KV Cache 优化**: Paged Attention, Prefix Caching
- **推测解码**: Multi-Token Prediction, Eagle 等
- **多模态**: 图像/视频理解能力

## 2. 目录结构

```
rtp_llm/
├── cpp/                      # C++ 核心实现
│   ├── api_server/           # HTTP/gRPC API 服务
│   ├── cache/                # KV Cache 管理
│   ├── config/               # 配置模块
│   ├── cuda/                 # CUDA kernels
│   ├── devices/              # 设备抽象层
│   ├── engine_base/          # 引擎基础设施
│   ├── kernels/              # 高性能计算 kernel
│   ├── models/               # C++ 模型实现
│   ├── normal_engine/        # 普通推理引擎
│   ├── speculative_engine/   # 推测解码引擎
│   └── pybind/               # Python 绑定
│
├── models/                   # Python 模型定义
│   ├── base_model.py         # 模型基类
│   ├── llama.py              # LLaMA 系列
│   ├── qwen*.py              # Qwen 系列
│   ├── qwen3_next/           # Qwen3-Next (GatedDeltaNet)
│   └── ...                   # 其他模型
│
├── models_py/                # 纯 Python 模型实现
│   ├── model_desc/           # 模型描述 (PyTorch)
│   │   ├── module_base.py    # 基类 GptModelBase
│   │   ├── qwen3_next.py     # Qwen3-Next 实现
│   │   └── ...
│   ├── modules/              # 可复用模块
│   ├── triton_kernels/       # Triton 自定义 kernel
│   │   ├── fla/              # Flash Linear Attention
│   │   ├── causal_conv1d/    # 因果卷积
│   │   └── moe/              # MoE 相关
│   └── bindings/             # Op 绑定
│
├── frontend/                 # 前端处理
│   ├── frontend_worker.py    # 前端 Worker
│   └── tokenizer_factory/    # Tokenizer 工厂
│
├── pipeline/                 # 推理 Pipeline
│   └── pipeline.py           # 核心 Pipeline
│
├── async_decoder_engine/     # 异步解码引擎
│   ├── engine_creator.py     # 引擎创建
│   ├── rpc_engine.py         # RPC 引擎
│   └── base_engine.py        # 引擎基类
│
├── server/                   # 服务器模块
│   ├── server_args/          # 服务参数
│   └── backend_rpc_server_visitor.py
│
├── config/                   # 配置管理
│   ├── model_config.py       # 模型配置
│   ├── engine_config.py      # 引擎配置
│   └── generate_config.py    # 生成配置
│
├── model_loader/             # 模型加载
│   ├── loader.py             # 权重加载器
│   └── model_weight_info.py  # 权重信息
│
├── ops/                      # 算子接口
│   └── *.pyi                 # Python 类型定义
│
└── utils/                    # 工具函数
```

## 3. 核心架构

### 3.1 分层架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                           API Layer                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │ OpenAI API  │  │  gRPC API   │  │   HTTP API  │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
├─────────────────────────────────────────────────────────────────────┤
│                        Frontend Layer                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐     │
│  │ FrontendWorker  │  │    Tokenizer    │  │ Request Renderer │     │
│  └─────────────────┘  └─────────────────┘  └──────────────────┘     │
├─────────────────────────────────────────────────────────────────────┤
│                        Pipeline Layer                                │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │                      Pipeline                              │      │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │      │
│  │  │ GenerateInput│→│BackendVisitor│→│ GenerateOutput   │  │      │
│  │  └──────────────┘  └──────────────┘  └─────────────────┘  │      │
│  └───────────────────────────────────────────────────────────┘      │
├─────────────────────────────────────────────────────────────────────┤
│                        Engine Layer                                  │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   AsyncDecoderEngine                         │    │
│  │  ┌────────────────┐      ┌────────────────────────────────┐ │    │
│  │  │  NormalEngine  │      │    SpeculativeEngine           │ │    │
│  │  │  (C++)         │      │  ┌──────────┐ ┌─────────────┐  │ │    │
│  │  │                │      │  │ Proposer │→│   Scorer    │  │ │    │
│  │  └────────────────┘      │  └──────────┘ └─────────────┘  │ │    │
│  │                          └────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│                        Model Layer                                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  BaseModel                                                      │ │
│  │  ┌─────────────┐  ┌────────────────┐  ┌──────────────────────┐ │ │
│  │  │ ModelConfig │  │  ModelWeights  │  │ GptModelBase (PyTorch)│ │ │
│  │  └─────────────┘  └────────────────┘  └──────────────────────┘ │ │
│  │                                        ↓                        │ │
│  │                           ┌────────────────────────┐            │ │
│  │                           │ Qwen3NextModel         │            │ │
│  │                           │ (layers, attention,...)│            │ │
│  │                           └────────────────────────┘            │ │
│  └────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                        Kernel Layer                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────────┐ │
│  │ CUDA Kernels │ │Triton Kernels│ │ CUTLASS GEMM │ │ FlashAttn   │ │
│  │ (cuBLAS,     │ │ (FLA, Conv1D,│ │ (FP8, MoE)   │ │ FlashInfer  │ │
│  │  custom ops) │ │  MoE, Norm)  │ │              │ │             │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 请求处理流程

```
User Request
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. API Server (HTTP/gRPC)                                       │
│    - 接收请求, 解析参数                                          │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. FrontendWorker                                               │
│    - Tokenize 输入文本                                           │
│    - 创建 GenerateConfig                                         │
│    - 处理多模态输入 (如有)                                        │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Pipeline                                                      │
│    - 封装 GenerateInput                                          │
│    - 通过 BackendRPCServerVisitor 发送请求                       │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. AsyncDecoderEngine (C++)                                     │
│    - Scheduler 调度请求                                          │
│    - Continuous Batching                                         │
│    - KV Cache 分配                                               │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Model Forward                                                │
│    ┌────────────────────┐    ┌────────────────────────────────┐ │
│    │ Prefill Phase      │    │ Decode Phase                   │ │
│    │ - 处理输入 tokens   │    │ - 逐 token 生成                │ │
│    │ - 构建 KV Cache    │ → │ - 增量更新 KV Cache            │ │
│    │ - 首次输出         │    │ - 采样下一个 token            │ │
│    └────────────────────┘    └────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. Post-processing                                              │
│    - Detokenize 输出                                             │
│    - Stop words 处理                                             │
│    - Streaming 输出                                              │
└─────────────────────────────────────────────────────────────────┘
```

## 4. 核心组件详解

### 4.1 模型注册机制

通过装饰器模式注册模型:

```python
# rtp_llm/model_factory_register.py
_model_factory: Dict[str, Type[Any]] = {}

def register_model(name: str, model_type: Any, 
                   support_architectures: List[str] = []):
    _model_factory[name] = model_type
    for arch in support_architectures:
        register_hf_architecture(arch, name)

# 使用示例 (qwen3_next.py)
register_model("qwen3_next", Qwen3Next, ["Qwen3NextForCausalLM"])
```

### 4.2 BaseModel 模型基类

```python
class BaseModel:
    # 配置对象
    model_config: ModelConfig           # 模型结构配置
    parallelism_config: ParallelismConfig  # 并行策略
    hw_kernel_config: HWKernelConfig    # 硬件 kernel 配置
    kv_cache_config: KVCacheConfig      # KV Cache 配置
    fmha_config: FMHAConfig             # Flash Attention 配置
    moe_config: MoeConfig               # MoE 配置
    
    # 核心属性
    weight: ModelWeights                # 模型权重
    tokenizer: BaseTokenizer            # Tokenizer
    py_model: GptModelBase              # Python 模型 (可选)
    
    def load(self):
        """加载模型的主入口"""
        self._may_init_multimodal()
        self.model_weights_loader = self.create_model_loader()
        self._load(device_str)
        if self.load_python_model:
            self._create_python_model()
    
    def _create_python_model(self) -> GptModelBase:
        """创建 Python 模型实例 (子类实现)"""
        raise NotImplementedError
```

### 4.3 GptModelBase (PyTorch 模型基类)

```python
class GptModelBase(nn.Module):
    def __init__(self, config, parallelism_config, weight, ...):
        self.config = config
        self.weight = weight
        self.kv_cache: Optional[KVCache] = None
        
    def initialize(self, init_resource: PyModelInitResources) -> bool:
        """引擎初始化时调用"""
        self.kv_cache = init_resource.kv_cache
        return True
    
    def prepare_fmha_impl(self, inputs) -> Any:
        """准备 Flash Attention 实现"""
        return AttnImplFactory.get_fmha_impl(...)
    
    def forward(self, inputs: PyModelInputs) -> PyModelOutputs:
        """前向传播 (子类实现)"""
        raise NotImplementedError
```

### 4.4 Pipeline 推理管道

```python
class Pipeline:
    def __init__(self, special_tokens, addresses, tokenizer, ...):
        self.tokenizer = tokenizer
        self.backend_rpc_server_visitor = BackendRPCServerVisitor(
            max_seq_len=max_seq_len,
            addresses=addresses,
            ...
        )
    
    async def pipeline_async(self, prompt, **kwargs):
        # 1. 创建生成配置
        generate_config = self.create_generate_config(...)
        
        # 2. Tokenize
        token_ids = self.tokenizer.encode(prompt)
        
        # 3. 调用后端生成
        return self.generate_stream(request_id, token_ids, ...)
    
    async def generate_stream(self, request_id, token_ids, ...):
        # 封装输入
        input = GenerateInput(
            request_id=request_id,
            token_ids=token_ids,
            generate_config=generate_config,
        )
        
        # 发送到后端
        stream = await self.backend_rpc_server_visitor.enqueue(input)
        
        # 流式处理输出
        async for outputs in stream:
            texts = self.decode_incremental_tokens(...)
            yield GenerateResponse(generate_outputs=outputs, generate_texts=texts)
```

### 4.5 AsyncDecoderEngine (C++)

```cpp
// cpp/normal_engine/NormalEngine.h
class NormalEngine : public EngineBase {
public:
    // 初始化
    void init(const EngineInitParams& params);
    
    // 添加请求
    void enqueue(GenerateStreamPtr stream);
    
    // 主循环
    void step();
    
private:
    // 调度器
    std::unique_ptr<Scheduler> scheduler_;
    
    // 批处理器
    std::unique_ptr<BatchStreamProcessor> batch_processor_;
    
    // KV Cache 管理器
    std::unique_ptr<KVCacheManager> kv_cache_manager_;
    
    // 模型执行
    std::unique_ptr<ModelExecutor> executor_;
};
```

## 5. Qwen3-Next 运行示例

### 5.1 模型架构特点

Qwen3-Next 是混合架构模型:
- **Hybrid Attention**: 标准 Attention + GatedDeltaNet 交替
- **MoE**: 稀疏 MoE 层 (shared expert + routed experts)
- **Linear Attention**: 线性时间复杂度的 GatedDeltaNet

```python
# 层类型分布 (full_attention_interval=8)
Layer 0: Linear Attention + MoE
Layer 1: Linear Attention + Dense
Layer 2: Linear Attention + MoE
...
Layer 7: Standard Attention + MoE  # 每 8 层一个标准 Attention
Layer 8: Linear Attention + MoE
...
```

### 5.2 模型类结构

```
Qwen3Next (BaseModel)
│
├── Qwen3NextWeight (权重管理)
│   ├── ModelWeightInfo (层级权重定义)
│   └── AtomicWeight (原子权重加载)
│
└── Qwen3NextModel (GptModelBase)
    ├── embed_tokens (Embedding)
    ├── layers (ModuleList)
    │   └── Qwen3NextDecoderLayer
    │       ├── input_layernorm (RMSNorm)
    │       ├── self_attn
    │       │   ├── Qwen3NextAttention (标准层)
    │       │   └── Qwen3NextGatedDeltaNet (线性层)
    │       ├── post_attention_layernorm (RMSNorm)
    │       └── mlp (GenericMoeLayer)
    └── norm (RMSNorm)
```

### 5.3 前向传播流程

```python
class Qwen3NextModel(GptModelBase):
    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        # 1. 输入嵌入
        input_ids = inputs.input_ids
        hidden_states = self.embed_tokens(input_ids)
        
        # 2. 准备元数据
        attention_inputs = inputs.attention_inputs
        if attention_inputs.is_prefill:
            prefill_conv1d_meta = prepare_causal_conv1d_metadata(...)
        attn_meta = Qwen3NextMetadata(prefill_conv1d_meta, is_target_verify)
        
        # 3. 准备 FMHA 实现 (用于标准 Attention 层)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
            fmha_impl.prepare(inputs.attention_inputs)
        
        # 4. 逐层前向传播
        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                attention_inputs=attention_inputs,
                attn_meta=attn_meta,
            )
        
        # 5. 最终层归一化
        hidden_states = self.norm(hidden_states)
        
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
```

### 5.4 Decoder Layer 详解

```python
class Qwen3NextDecoderLayer(nn.Module):
    def __init__(self, config, weights, layer_idx, ...):
        # 根据层类型选择注意力实现
        if self.layer_type == HybridAttentionType.LINEAR:
            self.self_attn = Qwen3NextGatedDeltaNet(...)  # 线性注意力
        else:
            self.self_attn = Qwen3NextAttention(...)      # 标准注意力
        
        # MoE 或 Dense FFN
        self.mlp = GenericMoeLayer(...)
        
        # 层归一化
        self.input_layernorm = RMSNorm(...)
        self.post_attention_layernorm = RMSNorm(...)
    
    def forward(self, hidden_states, fmha_impl, kv_cache, attention_inputs, attn_meta):
        # Pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            fmha_impl=fmha_impl,
            kv_cache=kv_cache,
            attention_inputs=attention_inputs,
            attn_meta=attn_meta,
        )
        hidden_states = residual + hidden_states
        
        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
```

### 5.5 GatedDeltaNet 注意力

```python
class Qwen3NextGatedDeltaNet(nn.Module):
    def forward(self, hidden_states, fmha_impl, kv_cache, attention_inputs, attn_meta):
        # 1. 投影得到 Q, K, V, Z 和门控信号
        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        mixed_qkv, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )
        
        # 2. 根据阶段选择处理方式
        if attention_inputs.is_prefill and not attn_meta.is_target_verify:
            # Prefill: 分块并行处理
            attn_output = self.prefill_gdn(
                mixed_qkv, b, a, attention_inputs, kv_cache, attn_meta
            )
        else:
            # Decode: 逐 token 递归
            attn_output = self.decode_gdn(
                mixed_qkv, b, a, attention_inputs, kv_cache, attn_meta
            )
        
        # 3. Gated RMS Norm
        attn_output = self.norm(
            attn_output.reshape(-1, self.head_v_dim),
            z.reshape(-1, self.head_v_dim)
        )
        
        # 4. 输出投影
        attn_output = attn_output.reshape(-1, self.local_num_v_heads * self.head_v_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output
```

### 5.6 启动服务示例

```bash
# 方式 1: 使用 CLI
python -m rtp_llm.cli.server \
    --model_path /path/to/qwen3-next \
    --tp_size 4 \
    --max_seq_len 32768 \
    --enable_cuda_graph

# 方式 2: 使用 AutoModel (独立模式)
python -c "
from rtp_llm.models_py.standalone.auto_model import AutoModel

model = AutoModel.from_pretrained('/path/to/qwen3-next')
response = model.generate('你好，请问你是谁？')
print(response)
"

# 方式 3: Docker 部署
docker run -d --gpus all \
    -v /path/to/model:/model \
    -p 8080:8080 \
    rtp-llm:latest \
    --model_path /model --tp_size 4
```

### 5.7 配置参数说明

```python
# ModelConfig 关键配置
model_config.num_layers = 48
model_config.hidden_size = 8192
model_config.attn_config.head_num = 64
model_config.attn_config.kv_head_num = 8

# Hybrid Attention 配置
model_config.hybrid_attention_config.enable_hybrid_attention = True
model_config.hybrid_attention_config.hybrid_attention_types = [
    HybridAttentionType.LINEAR,  # 层 0: GatedDeltaNet
    HybridAttentionType.LINEAR,  # 层 1
    ...
    HybridAttentionType.NONE,    # 层 7: 标准 Attention
    ...
]

# Linear Attention 配置
model_config.linear_attention_config.linear_conv_kernel_dim = 4
model_config.linear_attention_config.linear_key_head_dim = 128
model_config.linear_attention_config.linear_num_key_heads = 8
model_config.linear_attention_config.linear_value_head_dim = 128
model_config.linear_attention_config.linear_num_value_heads = 32

# MoE 配置
model_config.expert_num = 128
model_config.moe_k = 8
model_config.moe_layer_index = [0, 2, 4, ...]  # MoE 层索引
```

## 6. 关键优化技术

### 6.1 Continuous Batching

```
时间 T1: [Req1-Prefill, Req2-Decode, Req3-Decode]
时间 T2: [Req1-Decode,  Req2-Decode, Req4-Prefill]  # Req3 完成, Req4 加入
时间 T3: [Req1-Decode,  Req4-Decode, Req5-Decode]  # Req2 完成, Req5 加入
```

### 6.2 Paged Attention (KV Cache)

```
┌─────────────────────────────────────────────────────────────┐
│                    Physical Block Pool                       │
│  [Block0][Block1][Block2][Block3][Block4][Block5]...        │
└─────────────────────────────────────────────────────────────┘
        ↑           ↑           ↑
        │           │           │
┌───────┴───────────┴───────────┴─────────────────────────────┐
│  Seq1 block_map: [0, 2, 5]                                  │
│  Seq2 block_map: [1, 3, 4]                                  │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 推测解码 (Speculative Decoding)

```
┌─────────────────────────────────────────────────────────────┐
│                    SpeculativeEngine                         │
│                                                              │
│  ┌──────────────────┐     ┌──────────────────────────┐      │
│  │  Propose Model   │ →→→ │    Score Model (Main)    │      │
│  │  (小模型快速生成) │     │   (大模型验证接受)       │      │
│  │                  │     │                          │      │
│  │  生成 k 个 tokens │     │  验证并接受 m ≤ k 个    │      │
│  └──────────────────┘     └──────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 6.4 CUDA Graph

```python
# 预捕获不同 batch size 的计算图
for batch_size in [1, 2, 4, 8, 16, 32]:
    with torch.cuda.graph(cuda_graph):
        output = model.forward(inputs[batch_size])
    graphs[batch_size] = cuda_graph

# 推理时直接 replay
graphs[current_batch_size].replay()
```

## 7. 分布式部署

### 7.1 并行策略

```
┌─────────────────────────────────────────────────────────────┐
│                  Tensor Parallelism (TP=4)                   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │ GPU 0   │ │ GPU 1   │ │ GPU 2   │ │ GPU 3   │           │
│  │ Head    │ │ Head    │ │ Head    │ │ Head    │           │
│  │ 0-15    │ │ 16-31   │ │ 32-47   │ │ 48-63   │           │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘           │
│      ↓           ↓           ↓           ↓                  │
│                   AllReduce                                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  Data Parallelism (DP=2)                     │
│  ┌───────────────────────┐ ┌───────────────────────┐        │
│  │ Instance 0 (TP=4)     │ │ Instance 1 (TP=4)     │        │
│  │ Batch 0, 2, 4, ...    │ │ Batch 1, 3, 5, ...    │        │
│  └───────────────────────┘ └───────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 WorldInfo 配置

```python
class WorldInfo:
    members: List[MemberInfo]  # 所有 worker 信息
    
class MemberInfo:
    ip: str
    world_rank: int
    rpc_server_port: int
```

## 8. 总结

RTP-LLM 的架构设计特点:

1. **分层清晰**: API → Frontend → Pipeline → Engine → Model → Kernel
2. **Python/C++ 混合**: Python 处理配置和高层逻辑，C++ 处理性能关键路径
3. **模型抽象**: 统一的 BaseModel 接口支持多种模型架构
4. **可扩展性**: 通过注册机制轻松添加新模型
5. **高性能**: Triton/CUDA kernels, Continuous Batching, Paged Attention
6. **分布式**: 完善的 TP/DP 支持

Qwen3-Next 作为混合架构模型的示例，展示了 RTP-LLM 如何支持:
- 标准 Attention 与线性 Attention 混合
- MoE 稀疏专家网络
- 高效的 Triton kernel 实现
- 灵活的 KV Cache 管理
