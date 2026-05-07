import argparse
import math
import os
from typing import Tuple

import torch

from rtp_llm.models_py.triton_kernels.fla.block import (
    load_initial_state_from_block_map,
    store_ssm_state_to_block_map,
)
from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule


def _state_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def _make_block_map(
    total_len: int, seq_size_per_block: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_count = max(1, math.ceil(total_len / seq_size_per_block))
    # RTP reserves block id 0 as an invalid/sentinel block. Use ids [1, block_count].
    block_map = torch.arange(1, block_count + 1, device="cuda", dtype=torch.int32).view(
        1, -1
    )
    return block_map, torch.tensor([block_count + 1], device="cuda", dtype=torch.int32)


def make_inputs(
    input_len: int, prefix_len: int, seq_size_per_block: int, state_dtype: torch.dtype
):
    device = torch.device("cuda")
    q = torch.empty(1, input_len, 8, 128, device=device, dtype=torch.bfloat16)
    k = torch.empty(1, input_len, 8, 128, device=device, dtype=torch.bfloat16)
    v = torch.empty(1, input_len, 32, 128, device=device, dtype=torch.bfloat16)
    g = torch.empty(1, input_len, 32, device=device, dtype=torch.bfloat16)
    beta = torch.empty(1, input_len, 32, device=device, dtype=torch.bfloat16)
    cu = torch.tensor([0, input_len], device=device, dtype=torch.long)
    prefix = torch.tensor([prefix_len], device=device, dtype=torch.int32)
    block_map, block_shape = _make_block_map(prefix_len + input_len, seq_size_per_block)
    ssm_states = torch.empty(
        int(block_shape.item()), 32, 128, 128, device=device, dtype=state_dtype
    )
    initial_state = torch.empty(1, 32, 128, 128, device=device, dtype=state_dtype)
    return q, k, v, g, beta, cu, prefix, block_map, ssm_states, initial_state


def run_once(mode: str, tensors, seq_size_per_block: int) -> None:
    q, k, v, g, beta, cu, prefix, block_map, ssm_states, initial_state = tensors
    load_initial_state_from_block_map(
        prefix,
        block_map,
        ssm_states,
        initial_state,
        seq_size_per_block,
    )
    if mode == "triton":
        os.environ.pop("USE_FLYDSL", None)
        _, h, final_state = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True,
        )
        store_ssm_state_to_block_map(
            h.float(),
            final_state,
            prefix,
            cu,
            block_map,
            ssm_states,
            seq_size_per_block,
            chunk_size=64,
        )
    elif mode == "flydsl":
        from rtp_llm.models_py.triton_kernels.fla.chunk import (
            chunk_gated_delta_rule_flydsl_with_cache_store,
        )

        os.environ["USE_FLYDSL"] = "1"
        chunk_gated_delta_rule_flydsl_with_cache_store(
            q,
            k,
            v,
            g,
            beta,
            prefix_lengths=prefix,
            block_map=block_map,
            ssm_states=ssm_states,
            seq_size_per_block=seq_size_per_block,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True,
        )
    else:
        raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["triton", "flydsl"], required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--prefix-len", type=int, default=0)
    parser.add_argument("--seq-size-per-block", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--state-dtype", choices=["bf16", "fp32"], default="bf16")
    args = parser.parse_args()

    if args.input_len <= 0:
        raise ValueError("--input-len must be positive")
    if args.prefix_len < 0:
        raise ValueError("--prefix-len must be non-negative")

    tensors = make_inputs(
        args.input_len,
        args.prefix_len,
        args.seq_size_per_block,
        _state_dtype(args.state_dtype),
    )

    for _ in range(args.warmup):
        run_once(args.mode, tensors, args.seq_size_per_block)
    torch.cuda.synchronize()

    for _ in range(args.iters):
        run_once(args.mode, tensors, args.seq_size_per_block)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
