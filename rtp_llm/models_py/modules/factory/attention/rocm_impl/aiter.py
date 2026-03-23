import logging
from typing import Any, List, Optional

import aiter
import torch

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, FMHAType, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOpAsm,
    FusedRopeKVCacheDecodeOpNonAsm,
    FusedRopeKVCachePrefillOpAsm,
    FusedRopeKVCachePrefillOpNonAsm,
    LayerKVCache,
    ParamsBase,
    PyAttentionInputs,
    paged_attention_atrex,
)


# Pure Python implementation of FMHAParams
class FMHAParams(ParamsBase):
    """Python implementation of FMHAParams for Aiter attention operations."""

    def __init__(
        self,
        attn_inputs: PyAttentionInputs,
        is_prefill: bool = True,
        enable_cuda_graph: bool = True,
    ):
        super().__init__()
        self.enable_cuda_graph = enable_cuda_graph

        # Prefill mode
        if is_prefill:
            input_lengths = attn_inputs.input_lengths
            prefix_lengths = (
                attn_inputs.prefix_lengths
                if hasattr(attn_inputs, "prefix_lengths")
                else None
            )

            self.max_seq_len = input_lengths.max().item()
            batch_size = input_lengths.size(0)

            # Create cu_seqlens_q for query (based on input_lengths only)
            self.cu_seqlens_q = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=input_lengths.device
            )
            self.cu_seqlens_q[1:] = torch.cumsum(input_lengths, 0)

            # Create cu_seqlens_k for key/value (includes prefix_lengths)
            if prefix_lengths is not None and prefix_lengths.numel() > 0:
                kv_lengths = input_lengths + prefix_lengths
                self.cu_seqlens_k = torch.zeros(
                    batch_size + 1, dtype=torch.int32, device=input_lengths.device
                )
                self.cu_seqlens_k[1:] = torch.cumsum(kv_lengths, 0)
                # Calculate max sequence length including prefix
                max_prefix_length = (
                    prefix_lengths.max().item() if prefix_lengths.numel() > 0 else 0
                )
                self.max_seqlen_k = self.max_seq_len + max_prefix_length
            else:
                # No prefix, kv_lengths equals input_lengths
                kv_lengths = input_lengths
                self.cu_seqlens_k = self.cu_seqlens_q.clone()
                self.max_seqlen_k = self.max_seq_len

            self.max_seqlen_q = self.max_seq_len
            self.seq_lens = None
            self.kv_cache_block_id_device = getattr(
                attn_inputs, "kv_cache_block_id_device", None
            )
            self.prefix_lengths = prefix_lengths
            self.token_q_num = input_lengths.sum().item()
            self.token_kv_num = kv_lengths.sum().item()
        # Decode mode
        else:
            input_lengths = attn_inputs.input_lengths
            sequence_lengths = getattr(attn_inputs, "sequence_lengths", None)
            kv_cache_block_id_device = getattr(
                attn_inputs, "kv_cache_block_id_device", None
            )

            self.sequence_lengths = sequence_lengths
            self.kv_cache_block_id_device = kv_cache_block_id_device

            if self.enable_cuda_graph:
                self.max_seq_len = 8192
            else:
                self.max_seq_len = input_lengths.max().item() + 1

            self.max_seqlen_k = self.max_seq_len
            self.max_seqlen_q = 0
            self.cu_seqlens_q = None
            self.cu_seqlens_k = None

            # Create seq_lens on CUDA
            if sequence_lengths is not None:
                self.seq_lens = (sequence_lengths + 1).to(torch.device("cuda"))
            else:
                self.seq_lens = None

    def fillParams(
        self,
        sequence_lengths,
        input_lengths,
        kv_cache_block_id_host=None,
        kv_cache_block_id_device=None,
    ):
        self.sequence_lengths = sequence_lengths
        self.input_lengths = input_lengths
        self.kv_cache_block_id_host = kv_cache_block_id_host
        if kv_cache_block_id_device is not None:
            self.kv_cache_block_id_device = kv_cache_block_id_device
        if self.seq_lens is not None and self.sequence_lengths is not None:
            self.seq_lens.copy_((self.sequence_lengths + 1).to(torch.device("cuda")))
            self.max_seq_len = 8192 if self.enable_cuda_graph else self.max_seq_len

    def check_recycle(self) -> bool:
        """Check whether the params can be recycled automatically."""
        return True


class AiterPrefillAttnOp:
    def __init__(self, attn_configs: AttentionConfigs):
        self.head_num = attn_configs.head_num
        self.head_dim = attn_configs.size_per_head
        self.head_num_kv = attn_configs.kv_head_num
        self.is_causal = attn_configs.is_causal

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        self.fmha_params = FMHAParams(
            attn_inputs=attn_inputs,
            is_prefill=True,
        )
        return self.fmha_params

    def reshape_qkv(self, qkv):
        """Reshape qkv tensor(s) to the format expected by flash attention.
        Returns:
            Tuple of (q, k, v) tensors, each with shape (total_tokens, num_heads, head_dim).
        """
        if isinstance(qkv, (tuple, list)) and len(qkv) == 3 and qkv[0].dim() == 3:

            # 3D case: (head_num, tokens, head_dim) - need to permute
            q_contiguous = qkv[0].permute(1, 0, 2).contiguous()
            k_contiguous = qkv[1].permute(1, 0, 2).contiguous()
            v_contiguous = qkv[2].permute(1, 0, 2).contiguous()

            # Apply slicing based on fmha_params
            q_contiguous = q_contiguous[: self.fmha_params.token_q_num]
            k_contiguous = k_contiguous[: self.fmha_params.token_kv_num]
            v_contiguous = v_contiguous[: self.fmha_params.token_kv_num]

            return q_contiguous, k_contiguous, v_contiguous

        if isinstance(qkv, (tuple, list)) and len(qkv) == 3 and qkv[0].dim() == 2:
            qkv = qkv[0]  # specific for fp8 attention

        tokens = qkv.size(0)
        q_size = self.head_num * self.head_dim
        kv_size = self.head_num_kv * self.head_dim
        # Split qkv into q, k, v
        q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
        # Reshape to (tokens, num_heads, head_dim)
        q = q.view(tokens, self.head_num, self.head_dim)
        k = k.view(tokens, self.head_num_kv, self.head_dim)
        v = v.view(tokens, self.head_num_kv, self.head_dim)
        # Apply slicing based on fmha_params
        q = q[: self.fmha_params.token_q_num]
        k = k[: self.fmha_params.token_kv_num]
        v = v[: self.fmha_params.token_kv_num]
        return q.contiguous(), k.contiguous(), v.contiguous()

    def forward(self, qkv, kv_cache, fmha_params):
        q_tensor, k_tensor, v_tensor = self.reshape_qkv(qkv)

        cu_seqlens_q = fmha_params.cu_seqlens_q.to(q_tensor.device)
        cu_seqlens_k = fmha_params.cu_seqlens_k.to(k_tensor.device)
        max_seqlen_q = fmha_params.max_seqlen_q
        max_seqlen_k = fmha_params.max_seqlen_k

        if (
            q_tensor.dtype == torch.float8_e4m3fnuz
            and k_tensor.dtype == torch.float8_e4m3fnuz
            and v_tensor.dtype == torch.float8_e4m3fnuz
        ):
            res = aiter.flash_attn_varlen_fp8_pertensor_func(
                q_tensor,
                k_tensor,
                v_tensor,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                causal=self.is_causal,
            )
        else:
            res = aiter.flash_attn_varlen_func(
                q_tensor,  # Query张量: (total_q, nheads, headdim_q) - 批次中所有query token的总数
                k_tensor,  # Key张量: (total_k, nheads_k, headdim_q) - 批次中所有key token的总数
                v_tensor,  # Value张量: (total_k, nheads_k, headdim_v) - 批次中所有value token的总数
                cu_seqlens_q,  # Query累积序列长度: (batch_size + 1,) dtype=int32 - 用于索引q张量
                cu_seqlens_k,  # Key累积序列长度: (batch_size + 1,) dtype=int32 - 用于索引k/v张量
                max_seqlen_q,  # 批次中最大query序列长度
                max_seqlen_k,  # 批次中最大key序列长度
                dropout_p=0.0,  # Dropout概率 - 评估时应设为0.0
                causal=self.is_causal,  # 因果注意力掩码 - 用于自回归建模，每个位置只能关注自己和之前的位置
            )
        token_num = fmha_params.token_q_num
        final_result = res.reshape(token_num, self.head_num * self.head_dim)
        return final_result


class AiterDecodeAttnOpBase:
    """Base class for Aiter decode attention operations."""

    def __init__(self, attn_configs: AttentionConfigs):
        self.head_num = attn_configs.head_num
        self.head_dim = attn_configs.size_per_head
        self.head_num_kv = attn_configs.kv_head_num
        self.tokens_per_block = attn_configs.kernel_tokens_per_block
        self.enable_cuda_graph = True

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        # Create decode parameters using pure Python implementation
        fmha_params = FMHAParams(
            attn_inputs=attn_inputs,
            is_prefill=False,
            enable_cuda_graph=self.enable_cuda_graph,
        )
        return fmha_params


class AiterDecodeAttnOpAsm(AiterDecodeAttnOpBase):
    """Aiter decode attention operation using ASM paged attention."""

    def forward(
        self, query: torch.Tensor, kv_cache: Optional[LayerKVCache], fmha_params
    ) -> torch.Tensor:
        from rtp_llm.models_py.utils.debug import dump_tensor, dump_tensor_enabled
        _li = getattr(self, '_dump_layer_idx', -1)
        _dump = dump_tensor_enabled()

        seq_lens = fmha_params.seq_lens

        key_cache = kv_cache.kv_cache_base.select(1, 0)
        value_cache = kv_cache.kv_cache_base.select(1, 1)
        block_tables_id_device = fmha_params.kv_cache_block_id_device
        max_num_blocks = block_tables_id_device.shape[1]

        if _dump:
            import logging
            _seq_lengths = getattr(fmha_params, 'sequence_lengths', None)
            _seq_lengths_val = _seq_lengths.cpu().tolist() if _seq_lengths is not None else None
            logging.info(
                f"[DUMP] layer{_li} pa_fwd_asm ALL params:\n"
                f"  --- pa_fwd_asm positional args ---\n"
                f"  [0] query: shape={list(query.shape)}, dtype={query.dtype}, "
                f"stride={list(query.stride())}, contiguous={query.is_contiguous()}\n"
                f"  [1] key_cache: shape={list(key_cache.shape)}, dtype={key_cache.dtype}, "
                f"stride={list(key_cache.stride())}, contiguous={key_cache.is_contiguous()}\n"
                f"  [2] value_cache: shape={list(value_cache.shape)}, dtype={value_cache.dtype}, "
                f"stride={list(value_cache.stride())}, contiguous={value_cache.is_contiguous()}\n"
                f"  [3] block_tables: shape={list(block_tables_id_device.shape)}, dtype={block_tables_id_device.dtype}, "
                f"values={block_tables_id_device.cpu().tolist()}\n"
                f"  [4] seq_lens: shape={list(seq_lens.shape)}, dtype={seq_lens.dtype}, "
                f"values={seq_lens.cpu().tolist()}\n"
                f"  [5] max_num_blocks={max_num_blocks}\n"
                f"  [6] num_kv_splits=1\n"
                f"  [7] K_QScale=None\n"
                f"  [8] V_QScale=None\n"
                f"  [9] out_: shape={list(query.shape)}, dtype={query.dtype}\n"
                f"  [10] alibi_slopes=None\n"
                f"  [11] logits_soft_cap=0\n"
                f"  --- fmha_params extra fields ---\n"
                f"  sequence_lengths(raw)={_seq_lengths_val}\n"
                f"  max_seq_len={getattr(fmha_params, 'max_seq_len', 'N/A')}\n"
                f"  enable_cuda_graph={getattr(fmha_params, 'enable_cuda_graph', 'N/A')}\n"
                f"  --- attn op config ---\n"
                f"  head_num={self.head_num}, head_dim={self.head_dim}, "
                f"head_num_kv={self.head_num_kv}, tokens_per_block={self.tokens_per_block}"
            )
            dump_tensor(query, f"layer{_li}.full_attn.pa_fwd_asm_query", _li)

        K_QScale = None
        V_QScale = None
        if (
            key_cache.dtype == torch.float8_e4m3fnuz
            and value_cache.dtype == torch.float8_e4m3fnuz
        ):
            K_QScale = kv_cache.kv_scale_base.select(1, 0)
            V_QScale = kv_cache.kv_scale_base.select(1, 1)
        out_ = torch.empty_like(query)
        output = aiter.pa_fwd_asm(
            query,  # [num_seqs, num_heads, head_size]
            key_cache,  # [num_blocks, num_kv_heads, block_size, head_size]
            value_cache,  # [num_blocks, num_kv_heads, block_size, head_size]
            block_tables_id_device,
            seq_lens,
            max_num_blocks,
            1,
            K_QScale,
            V_QScale,
            out_,
            None,
            0,
        )

        if _dump:
            dump_tensor(output, f"layer{_li}.full_attn.pa_fwd_asm_out", _li)

        output_reshaped = output.view(output.shape[0], -1)
        return output_reshaped


class AiterDecodeAttnOpNonAsm(AiterDecodeAttnOpBase):
    """Aiter decode attention operation using non-ASM paged attention."""

    def forward(
        self, query: torch.Tensor, kv_cache: Optional[LayerKVCache], fmha_params
    ) -> torch.Tensor:
        seq_lens = fmha_params.seq_lens
        key_cache = kv_cache.kv_cache_base.select(1, 0)
        value_cache = kv_cache.kv_cache_base.select(1, 1)

        K_QScale = None
        V_QScale = None
        using_fp8_kvcache = False
        if (
            key_cache.dtype == torch.float8_e4m3fnuz
            and value_cache.dtype == torch.float8_e4m3fnuz
        ):
            K_QScale = kv_cache.kv_scale_base.select(1, 0)
            V_QScale = kv_cache.kv_scale_base.select(1, 1)
            using_fp8_kvcache = True

        block_tables_id_device = fmha_params.kv_cache_block_id_device

        max_seq_len = fmha_params.max_seq_len
        scale = 1.0 / (self.head_dim**0.5)
        alibi_slopes = None
        num_kv_heads = self.head_num_kv
        num_seqs, num_heads, head_size = query.shape
        block_size = value_cache.shape[2]
        # TODO(wenhua): avoid asd pa accuracy in qwen35
        # if max_seq_len <= 16384 and (not using_fp8_kvcache):
        output = torch.empty_like(query).view((num_seqs, num_heads, head_size))
        if False:
            _PARTITION_SIZE_ROCM = 512
            max_num_partitions = (
                max_seq_len + _PARTITION_SIZE_ROCM - 1
            ) // _PARTITION_SIZE_ROCM
            x = 16 // key_cache.element_size()
            grp_size = num_heads // num_kv_heads
            kv_sizes = value_cache.shape
            # init output
            output = torch.empty_like(query).view((num_seqs, num_heads, head_size))
            exp_sums = torch.empty(
                size=(num_seqs, num_kv_heads, max_num_partitions, grp_size),
                dtype=torch.float32,
                device=output.device,
            )
            max_logits = torch.empty_like(exp_sums)
            # init tmp_output
            tmp_output = torch.empty(
                size=(num_seqs, num_kv_heads, max_num_partitions, grp_size, head_size),
                dtype=output.dtype,
                device=output.device,
            )
            query = query.view((num_seqs, num_heads, head_size))
            key_cache = key_cache.view(
                (kv_sizes[0], kv_sizes[1], kv_sizes[3] // x, kv_sizes[2], x)
            )
            value_cache = value_cache.view(
                (kv_sizes[0], kv_sizes[1], kv_sizes[3], kv_sizes[2])
            )
            paged_attention_atrex(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                seq_lens,
                block_tables_id_device,
                scale,
                max_seq_len,
                alibi_slopes,
            )
        else:
            _PARTITION_SIZE_ROCM = 256

            max_num_partitions = (
                max_seq_len + _PARTITION_SIZE_ROCM - 1
            ) // _PARTITION_SIZE_ROCM
            assert _PARTITION_SIZE_ROCM % block_size == 0
            # init tmp_output
            tmp_output = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions, head_size),
                dtype=output.dtype,
                device=output.device,
            )

            # init exp_sums
            exp_sums = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions),
                dtype=torch.float32,
                device=output.device,
            )
            fp8_out_scale = None
            cpa_fp8_out = False
            # init max_logits
            max_logits = torch.ones_like(exp_sums)

            kv_cache_dtype = "auto"
            k_scale = (
                K_QScale
                if kv_cache and K_QScale is not None
                else torch.tensor(1.0, device=query.device, dtype=query.dtype)
            )
            v_scale = (
                V_QScale
                if kv_cache and V_QScale is not None
                else torch.tensor(1.0, device=query.device, dtype=query.dtype)
            )
            aiter.paged_attention_rocm(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                float(scale),
                block_tables_id_device,
                seq_lens,
                block_size,
                max_seq_len,
                alibi_slopes,
                kv_cache_dtype,  # kv_cache_dtype
                k_scale,
                v_scale,
                fp8_out_scale if cpa_fp8_out else None,
                _PARTITION_SIZE_ROCM,
            )

        output_reshaped = output.view(output.shape[0], -1)
        return output_reshaped


class AiterPrefillImplAsm(FMHAImplBase):
    """Aiter prefill attention implementation using ASM."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = AiterPrefillAttnOp(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpAsm(attn_configs)

        # Store input info
        self.attn_inputs = attn_inputs

        # Create params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
    ) -> torch.Tensor:
        from rtp_llm.models_py.utils.debug import dump_tensor, dump_tensor_enabled
        _li = getattr(self, '_dump_layer_idx', -1)
        _dump = dump_tensor_enabled()

        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # --- dump kv cache after prefill write for prefill/decode consistency check ---
        if _dump and kv_cache is not None and kv_cache.kv_cache_base is not None:
            try:
                cache = kv_cache.kv_cache_base
                block_size = cache.shape[3] if cache.dim() == 5 else 0
                block_table = self.fmha_params.kv_cache_block_id_device
                input_lengths = self.fmha_params.cu_seqlens_q  # cu_seqlens: [batch+1]
                prefix_lengths = self.fmha_params.prefix_lengths

                import logging
                logging.info(
                    f"[DUMP] layer{_li} prefill: kv_cache_base.shape={list(cache.shape)}, "
                    f"block_table.shape={list(block_table.shape) if block_table is not None else 'None'}, "
                    f"kv_cache.seq_size_per_block={kv_cache.seq_size_per_block}"
                )

                if block_table is not None and cache.dim() == 5:
                    # Reconstruct per-batch seq_lens from cu_seqlens
                    batch_size = input_lengths.shape[0] - 1
                    for batch_idx in range(batch_size):
                        input_len = (input_lengths[batch_idx + 1] - input_lengths[batch_idx]).item()
                        prefix_len = prefix_lengths[batch_idx].item() if (
                            prefix_lengths is not None and prefix_lengths.numel() > 0
                        ) else 0
                        total_seq_len = input_len + prefix_len
                        num_blocks_needed = (total_seq_len + block_size - 1) // block_size
                        block_ids = block_table[batch_idx, :num_blocks_needed]

                        # --- naive layout extraction (for reference) ---
                        k_blocks = cache[block_ids, 0]
                        v_blocks = cache[block_ids, 1]
                        k_blocks = k_blocks.permute(0, 2, 1, 3)
                        v_blocks = v_blocks.permute(0, 2, 1, 3)
                        k_continuous = k_blocks.reshape(-1, k_blocks.shape[2], k_blocks.shape[3])
                        v_continuous = v_blocks.reshape(-1, v_blocks.shape[2], v_blocks.shape[3])
                        k_continuous = k_continuous[:total_seq_len]
                        v_continuous = v_continuous[:total_seq_len]
                        dump_tensor(k_continuous, f"layer{_li}.full_attn.prefill_k_cache_naive_b{batch_idx}", _li)
                        dump_tensor(v_continuous, f"layer{_li}.full_attn.prefill_v_cache_naive_b{batch_idx}", _li)

                        # --- vectorized layout extraction ---
                        # ROCm ASM stores K and V caches in DIFFERENT vectorized layouts:
                        #   K: [numHeads, dimsPerHead/vs, mTokensPerBlock, vs]
                        #   V: [numHeads, mTokensPerBlock/vs, dimsPerHead, vs]
                        # where vs = 16/element_size = 8 for bf16
                        num_kv_heads = cache.shape[2]
                        head_dim = cache.shape[4]
                        vec_size = 8  # bf16 vectorization size

                        # K cache: reshape [block_size, head_dim] as [head_dim/vs, block_size, vs]
                        k_raw = cache[block_ids, 0]  # [nblk, num_kv_heads, block_size, head_dim]
                        k_vec = k_raw.reshape(num_blocks_needed, num_kv_heads, head_dim // vec_size, block_size, vec_size)
                        # permute to [nblk, block_size, num_kv_heads, head_dim/vs, vs]
                        k_vec = k_vec.permute(0, 3, 1, 2, 4).reshape(num_blocks_needed * block_size, num_kv_heads, head_dim)
                        k_vec = k_vec[:total_seq_len]
                        dump_tensor(k_vec, f"layer{_li}.full_attn.prefill_k_cache_b{batch_idx}", _li)

                        # V cache: reshape [block_size, head_dim] as [block_size/vs, head_dim, vs]
                        v_raw = cache[block_ids, 1]  # [nblk, num_kv_heads, block_size, head_dim]
                        v_vec = v_raw.reshape(num_blocks_needed, num_kv_heads, block_size // vec_size, head_dim, vec_size)
                        # permute to [nblk, num_kv_heads, block_size/vs, vs, head_dim]
                        # then reshape to [nblk * block_size, num_kv_heads, head_dim]
                        v_vec = v_vec.permute(0, 2, 4, 1, 3).reshape(num_blocks_needed * block_size, num_kv_heads, head_dim)
                        v_vec = v_vec[:total_seq_len]
                        dump_tensor(v_vec, f"layer{_li}.full_attn.prefill_v_cache_b{batch_idx}", _li)

                        # Also dump raw block data for offline analysis
                        dump_tensor(cache[block_ids[0], 0, 0], f"layer{_li}.full_attn.prefill_raw_k_block0_head0", _li)
                        dump_tensor(block_table, f"layer{_li}.full_attn.prefill_block_table", _li)
            except Exception as e:
                import logging
                logging.warning(f"Failed to dump prefill kv cache: {e}")

        # Execute FMHA forward
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)


class AiterPrefillImplNonAsm(FMHAImplBase):
    """Aiter prefill attention implementation using non-ASM."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = AiterPrefillAttnOp(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpNonAsm(attn_configs)

        # Store input info
        self.attn_inputs = attn_inputs

        # Create params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
    ) -> torch.Tensor:
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)


class AiterDecodeImplAsm(FMHAImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = AiterDecodeAttnOpAsm(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCacheDecodeOpAsm(attn_configs)
        self.head_num = attn_configs.head_num
        self.kv_head_num = attn_configs.kv_head_num
        self.size_per_head = attn_configs.size_per_head

        # Store input info
        self.attn_inputs = attn_inputs

        # Create params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
    ) -> torch.Tensor:
        from rtp_llm.models_py.utils.debug import dump_tensor, dump_tensor_enabled
        _li = getattr(self, '_dump_layer_idx', -1)
        _dump = dump_tensor_enabled()

        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        if _dump:
            # fmha_input is query after rope: [batch, num_heads, head_dim]
            dump_tensor(fmha_input, f"layer{_li}.full_attn.q_rope_out", _li)

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        if _dump and kv_cache is not None and kv_cache.kv_cache_base is not None:
            try:
                # --- dump block table, kv cache shape, seq_lens for consistency check ---
                cache = kv_cache.kv_cache_base
                block_size = cache.shape[3]
                seq_lens = self.fmha_params.seq_lens  # [batch], already +1 in decode
                block_table = self.fmha_params.kv_cache_block_id_device  # [batch, max_blocks]

                import logging
                logging.info(
                    f"[DUMP] layer{_li} decode: kv_cache_base.shape={list(cache.shape)}, "
                    f"block_table.shape={list(block_table.shape)}, "
                    f"seq_lens={seq_lens.cpu().tolist()}, "
                    f"block_table[0,:8]={block_table[0, :min(8, block_table.shape[1])].cpu().tolist()}, "
                    f"kv_cache.seq_size_per_block={kv_cache.seq_size_per_block}"
                )

                # dump block table and seq_lens as tensors for offline comparison
                dump_tensor(block_table, f"layer{_li}.full_attn.decode_block_table", _li)
                dump_tensor(seq_lens, f"layer{_li}.full_attn.decode_seq_lens", _li)

                for batch_idx in range(seq_lens.shape[0]):
                    seq_len = seq_lens[batch_idx].item()
                    num_blocks_needed = (seq_len + block_size - 1) // block_size
                    block_ids = block_table[batch_idx, :num_blocks_needed]
                    num_kv_heads = cache.shape[2]
                    head_dim = cache.shape[4]
                    vec_size = 8  # bf16 vectorization size

                    # --- naive layout extraction (for reference) ---
                    k_blocks = cache[block_ids, 0]
                    v_blocks = cache[block_ids, 1]
                    k_blocks_p = k_blocks.permute(0, 2, 1, 3)
                    v_blocks_p = v_blocks.permute(0, 2, 1, 3)
                    k_naive = k_blocks_p.reshape(-1, k_blocks_p.shape[2], k_blocks_p.shape[3])[:seq_len]
                    v_naive = v_blocks_p.reshape(-1, v_blocks_p.shape[2], v_blocks_p.shape[3])[:seq_len]
                    dump_tensor(k_naive, f"layer{_li}.full_attn.k_cache_naive_b{batch_idx}", _li)
                    dump_tensor(v_naive, f"layer{_li}.full_attn.v_cache_naive_b{batch_idx}", _li)

                    # --- vectorized layout extraction ---
                    # ROCm ASM stores K and V caches in DIFFERENT vectorized layouts:
                    #   K: [numHeads, dimsPerHead/vs, mTokensPerBlock, vs]
                    #   V: [numHeads, mTokensPerBlock/vs, dimsPerHead, vs]
                    # where vs = 16/element_size = 8 for bf16

                    # K cache: reshape [block_size, head_dim] as [head_dim/vs, block_size, vs]
                    k_raw = cache[block_ids, 0]  # [nblk, num_kv_heads, block_size, head_dim]
                    k_vec = k_raw.reshape(num_blocks_needed, num_kv_heads, head_dim // vec_size, block_size, vec_size)
                    k_continuous = k_vec.permute(0, 3, 1, 2, 4).reshape(num_blocks_needed * block_size, num_kv_heads, head_dim)[:seq_len]

                    # V cache: reshape [block_size, head_dim] as [block_size/vs, head_dim, vs]
                    v_raw = cache[block_ids, 1]  # [nblk, num_kv_heads, block_size, head_dim]
                    v_vec = v_raw.reshape(num_blocks_needed, num_kv_heads, block_size // vec_size, head_dim, vec_size)
                    v_continuous = v_vec.permute(0, 2, 4, 1, 3).reshape(num_blocks_needed * block_size, num_kv_heads, head_dim)[:seq_len]
                    dump_tensor(k_continuous, f"layer{_li}.full_attn.k_cache_b{batch_idx}", _li)
                    dump_tensor(v_continuous, f"layer{_li}.full_attn.v_cache_b{batch_idx}", _li)

                    # Extract the last written token (current decode step) as k_rope_out and v
                    last_token_pos = seq_len - 1
                    last_k = k_continuous[last_token_pos:last_token_pos + 1]
                    last_v = v_continuous[last_token_pos:last_token_pos + 1]
                    dump_tensor(last_k, f"layer{_li}.full_attn.k_rope_out", _li)
                    dump_tensor(last_v, f"layer{_li}.full_attn.v", _li)

                    # Also dump raw block data for offline analysis
                    dump_tensor(cache[block_ids[0], 0, 0], f"layer{_li}.full_attn.decode_raw_k_block0_head0", _li)

                    # --- dump prefill-written portion (tokens 0..seq_len-2) for prefill/decode consistency check ---
                    if seq_len > 1:
                        prefill_k = k_continuous[:seq_len - 1]
                        prefill_v = v_continuous[:seq_len - 1]
                        dump_tensor(prefill_k, f"layer{_li}.full_attn.prefill_k_in_decode_b{batch_idx}", _li)
                        dump_tensor(prefill_v, f"layer{_li}.full_attn.prefill_v_in_decode_b{batch_idx}", _li)
            except Exception as e:
                import logging
                logging.warning(f"Failed to extract kv cache for dump: {e}")
        # Execute FMHA forward
        self.fmha_impl._dump_layer_idx = _li
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        # Replay path must reuse capture-time FMHA params object to keep graph memory stable.
        self.fmha_params.fillParams(
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_block_id_host,
            attn_inputs.kv_cache_block_id_device,
        )
        if hasattr(self.rope_params, "update_kv_cache_offset"):
            self.rope_params.update_kv_cache_offset(
                attn_inputs.kv_cache_block_id_device
            )


class AiterDecodeImplNonAsm(FMHAImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = AiterDecodeAttnOpNonAsm(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCacheDecodeOpNonAsm(attn_configs)

        # Store input info
        self.attn_inputs = attn_inputs

        # Create params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
    ) -> torch.Tensor:
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        # Replay path must reuse capture-time FMHA params object to keep graph memory stable.
        self.fmha_params.fillParams(
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_block_id_host,
            attn_inputs.kv_cache_block_id_device,
        )
        if hasattr(self.rope_params, "update_kv_cache_offset"):
            self.rope_params.update_kv_cache_offset(
                attn_inputs.kv_cache_block_id_device
            )
