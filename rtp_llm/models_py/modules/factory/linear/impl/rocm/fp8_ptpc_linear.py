"""ROCm FP8 PTPC (Per-Token Per-Channel) quantized Linear implementation"""

import logging
import os
from typing import Dict, Optional, Tuple

import aiter
import torch
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle_cktile

from rtp_llm.models_py.modules.factory.linear import LinearBase

logger = logging.getLogger(__name__)

from rtp_llm.models_py.kernels.rocm.fp8_kernel import rocm_per_token_quant_fp8
from rtp_llm.ops import HWKernelConfig

_USE_FLYDSL = os.environ.get("USE_FLYDSL", "0") == "1"

# ── FlyDSL Preshuffle GEMM FP8 ──────────────────────────────────────────────
# Lazy-initialized; only imported when USE_FLYDSL=1.
_flydsl_initialized = False
_flydsl_compile_fn = None  # compile_preshuffle_gemm_a8
_flyc_compile = None  # flydsl.compiler.compile
_flydsl_arch = ""
_flydsl_fp8_dtype = None

# Cache of compiled FlyDSL kernels keyed by (M, N, K).
_flydsl_gemm_cache: Dict[Tuple[int, int, int], object] = {}

# Optimized tile configs keyed by (M, K, N) → (tile_m, tile_n, tile_k, xcd_swizzle, use_async_copy).
# Benchmarked on MI308X (gfx942, 80 CUs), 2026-05-18.
# Fallback: if exact (M,K,N) not found, match by M with K/N heuristics.
_FLYDSL_TILE_TABLE_FULL: Dict[Tuple[int, int, int], Tuple[int, int, int, int, bool]] = {
    # 397B shapes (K, N from Qwen3.5-397B TP4/TP8)
    # M=16
    (16, 4096, 3072): (16, 64, 512, 0, True),
    (16, 4096, 4608): (16, 64, 512, 0, True),
    (16, 4096, 2048): (16, 64, 512, 0, True),
    (16, 4096, 2304): (16, 64, 512, 0, True),
    (16, 2048, 4096): (16, 64, 512, 4, True),
    (16, 1024, 4096): (16, 128, 256, 2, False),
    (16, 4096, 512): (16, 64, 512, 0, True),
    (16, 4096, 256): (16, 64, 512, 0, True),
    (16, 256, 4096): (16, 64, 256, 0, False),
    (16, 128, 4096): (16, 64, 128, 0, False),
    # M=32
    (32, 4096, 3072): (16, 64, 256, 2, False),
    (32, 4096, 4608): (16, 64, 256, 2, False),
    (32, 4096, 2048): (16, 64, 512, 2, True),
    (32, 4096, 2304): (16, 64, 512, 2, True),
    (32, 2048, 4096): (16, 128, 256, 2, False),
    (32, 1024, 4096): (16, 64, 256, 2, False),
    (32, 4096, 512): (16, 64, 512, 0, True),
    (32, 4096, 256): (16, 64, 256, 0, False),
    (32, 256, 4096): (32, 64, 256, 4, False),
    (32, 128, 4096): (32, 64, 128, 2, False),
    # M=64
    (64, 4096, 3072): (16, 64, 256, 4, False),
    (64, 4096, 4608): (32, 64, 512, 2, True),
    (64, 4096, 2048): (16, 128, 256, 2, False),
    (64, 4096, 2304): (16, 64, 256, 2, False),
    (64, 2048, 4096): (32, 128, 256, 2, False),
    (64, 1024, 4096): (16, 128, 256, 4, False),
    (64, 4096, 512): (16, 64, 512, 0, True),
    (64, 4096, 256): (16, 64, 512, 0, True),
    (64, 256, 4096): (32, 128, 128, 2, False),
    (64, 128, 4096): (32, 64, 128, 2, False),
    # M=128
    (128, 4096, 3072): (32, 128, 256, 2, False),
    (128, 4096, 4608): (64, 128, 256, 2, False),
    (128, 4096, 2048): (32, 64, 256, 2, False),
    (128, 4096, 2304): (32, 64, 256, 2, False),
    (128, 2048, 4096): (64, 128, 256, 2, False),
    (128, 1024, 4096): (64, 128, 128, 2, False),
    (128, 4096, 512): (32, 64, 256, 2, False),
    (128, 4096, 256): (32, 64, 256, 2, False),
    (128, 256, 4096): (32, 128, 128, 2, False),
    (128, 128, 4096): (32, 128, 128, 2, False),
    # M=256
    (256, 4096, 3072): (32, 128, 256, 2, False),
    (256, 4096, 4608): (64, 128, 256, 2, False),
    (256, 4096, 2048): (32, 128, 256, 2, False),
    (256, 4096, 2304): (32, 128, 256, 2, False),
    (256, 2048, 4096): (64, 128, 128, 2, False),
    (256, 1024, 4096): (64, 128, 128, 2, False),
    (256, 4096, 512): (32, 64, 256, 2, False),
    (256, 4096, 256): (32, 64, 256, 2, False),
    (256, 256, 4096): (64, 128, 128, 2, False),
    (256, 128, 4096): (64, 128, 128, 2, False),
    # M=512
    (512, 4096, 3072): (64, 128, 128, 2, False),
    (512, 4096, 4608): (64, 256, 128, 2, False),
    (512, 4096, 2048): (64, 128, 128, 2, False),
    (512, 4096, 2304): (64, 128, 128, 2, False),
    (512, 2048, 4096): (64, 256, 128, 2, False),
    (512, 1024, 4096): (64, 256, 128, 2, False),
    (512, 4096, 512): (64, 64, 256, 2, False),
    (512, 4096, 256): (64, 64, 256, 2, False),
    (512, 256, 4096): (64, 128, 128, 2, False),
    (512, 128, 4096): (64, 128, 128, 2, False),
}

# Heuristic fallback by M when exact (M, K, N) not in table.
_FLYDSL_TILE_FALLBACK: Dict[int, Tuple[int, int, int, int, bool]] = {
    16: (16, 64, 256, 2, False),
    32: (16, 64, 256, 2, False),
    64: (32, 64, 256, 2, False),
    128: (64, 128, 128, 2, False),
    256: (64, 128, 128, 2, False),
    512: (64, 128, 128, 2, False),
}


def _select_flydsl_config(M: int, K: int, N: int) -> Tuple[int, int, int, int, bool]:
    """Select best (tile_m, tile_n, tile_k, xcd_swizzle, use_async_copy) for given shape."""
    cfg = _FLYDSL_TILE_TABLE_FULL.get((M, K, N))
    if cfg is not None:
        return cfg
    # Fallback: use M-based heuristic
    fb = _FLYDSL_TILE_FALLBACK.get(M)
    if fb is not None:
        tile_m, tile_n, tile_k, xcd, async_c = fb
        # Ensure tile_n divides N and tile_k divides K
        if N % tile_n != 0:
            tile_n = 64
        if K % tile_k != 0:
            tile_k = 128 if K % 128 == 0 else 64
        return (tile_m, tile_n, tile_k, xcd, async_c)
    # Last resort
    return (16, 64, 256, 0, False)


_flydsl_compile_v2_fn = None  # compile_preshuffle_gemm_v2


def _init_flydsl():
    global _flydsl_initialized, _flydsl_compile_fn, _flydsl_compile_v2_fn, _flyc_compile
    global _flydsl_arch, _flydsl_fp8_dtype
    if _flydsl_initialized:
        return
    import sys

    flydsl_home = os.environ.get("FLYDSL_HOME", "/root/FlyDSL")
    sys.path.insert(0, flydsl_home)
    sys.path.insert(0, os.path.join(flydsl_home, "build-fly/python_packages"))
    os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "1")

    import flydsl.compiler as flyc
    from flydsl.runtime.device import get_rocm_arch
    from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8
    from kernels.preshuffle_gemm_v2 import compile_preshuffle_gemm_v2

    _flydsl_compile_fn = compile_preshuffle_gemm_a8
    _flydsl_compile_v2_fn = compile_preshuffle_gemm_v2
    _flyc_compile = flyc.compile
    _flydsl_arch = str(get_rocm_arch())
    _flydsl_fp8_dtype = (
        torch.float8_e4m3fn if "gfx95" in _flydsl_arch else torch.float8_e4m3fnuz
    )
    _flydsl_initialized = True
    logger.info("FlyDSL Preshuffle GEMM FP8 initialized (arch=%s)", _flydsl_arch)


def _flydsl_preshuffle_gemm(
    input_fp8: torch.Tensor,
    weight: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    M: int,
    N: int,
    K: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Run FlyDSL preshuffle GEMM FP8, with JIT compile caching."""
    _init_flydsl()

    # M < 16: pad to 16 and use v2 kernel
    if M < 16:
        return _flydsl_preshuffle_gemm_small_m(
            input_fp8, weight, x_scales, w_scales, M, N, K, out_dtype
        )

    cache_key = (M, N, K)
    compiled = _flydsl_gemm_cache.get(cache_key)

    tile_m, tile_n, tile_k, xcd_swizzle, use_async = _select_flydsl_config(M, K, N)

    output = torch.zeros(M, N, device=input_fp8.device, dtype=out_dtype)
    dummy_bias = torch.empty(0, dtype=out_dtype, device=input_fp8.device)

    a_i8 = input_fp8.view(torch.int8).contiguous().view(-1)
    b_i8 = weight.view(torch.int8).contiguous().view(-1)
    sa_flat = x_scales.contiguous().view(-1)
    sb_flat = w_scales.contiguous().view(-1)
    stream = torch.cuda.current_stream()

    out_dtype_str = "bf16" if out_dtype == torch.bfloat16 else "fp16"

    args = (
        output.contiguous().view(-1),
        a_i8,
        b_i8,
        sa_flat,
        sb_flat,
        dummy_bias,
        M,
        N,
        stream,
    )

    if compiled is None:
        launch_fn = _flydsl_compile_fn(
            N=N,
            K=K,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            in_dtype="fp8",
            out_dtype=out_dtype_str,
            lds_stage=2,
            use_async_copy=use_async,
            xcd_swizzle=xcd_swizzle,
        )
        compiled = _flyc_compile(launch_fn, *args)
        _flydsl_gemm_cache[cache_key] = compiled
        logger.info(
            "FlyDSL GEMM compiled: M=%d K=%d N=%d tile=(%d,%d,%d) xcd=%d async=%s",
            M,
            K,
            N,
            tile_m,
            tile_n,
            tile_k,
            xcd_swizzle,
            use_async,
        )

    compiled(*args)
    return output


def _flydsl_preshuffle_gemm_small_m(
    input_fp8: torch.Tensor,
    weight: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    M: int,
    N: int,
    K: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """M < 16 path: pad A/scale to 16 rows, run v2 kernel, slice output."""
    M_pad = 16
    cache_key = (M_pad, N, K, "v2")
    compiled = _flydsl_gemm_cache.get(cache_key)

    # Pad input and scales
    a_padded = torch.zeros(M_pad, K, device=input_fp8.device, dtype=input_fp8.dtype)
    a_padded[:M] = input_fp8
    sa_padded = torch.zeros(
        M_pad, x_scales.shape[-1], device=x_scales.device, dtype=x_scales.dtype
    )
    sa_padded[:M] = x_scales

    output = torch.zeros(M_pad, N, device=input_fp8.device, dtype=out_dtype)

    a_i8 = a_padded.view(torch.int8).contiguous().view(-1)
    b_i8 = weight.view(torch.int8).contiguous().view(-1)
    sa_flat = sa_padded.contiguous().view(-1)
    sb_flat = w_scales.contiguous().view(-1)
    stream = torch.cuda.current_stream()

    out_dtype_str = "bf16" if out_dtype == torch.bfloat16 else "fp16"

    args = (
        output.contiguous().view(-1),
        a_i8,
        b_i8,
        sa_flat,
        sb_flat,
        M_pad,
        N,
        stream,
    )

    if compiled is None:
        tile_n = 64 if N % 64 == 0 else 128
        tile_k = min(K, 256) if K % 256 == 0 else (128 if K % 128 == 0 else 64)
        launch_fn = _flydsl_compile_v2_fn(
            N=N,
            K=K,
            tile_m=16,
            tile_n=tile_n,
            tile_k=tile_k,
            in_dtype="fp8",
            out_dtype=out_dtype_str,
        )
        compiled = _flyc_compile(launch_fn, *args)
        _flydsl_gemm_cache[cache_key] = compiled
        logger.info(
            "FlyDSL v2 GEMM compiled: M_pad=%d K=%d N=%d tile=(16,%d,%d)",
            M_pad,
            K,
            N,
            tile_n,
            tile_k,
        )

    compiled(*args)
    return output[:M]


class RocmFp8PTPCLinear(LinearBase):
    """ROCm FP8 PTPC (Per-Token Per-Channel) quantized Linear"""

    @classmethod
    def can_handle(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        """Handle FP8_PER_CHANNEL_COMPRESSED and FP8_PER_CHANNEL_QUARK"""
        if weight_scales is None or quant_config is None:
            return False

        # Check if weight is FP8 format
        if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            return False

        # Check quantization method
        quant_method = quant_config.get_method()
        return quant_method in ("FP8_PER_CHANNEL_COMPRESSED", "FP8_PER_CHANNEL_QUARK")

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        quant_config: object = None,
        weight_scale_2: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            weight, weight_scales, input_scales, bias, quant_config, weight_scale_2
        )
        self.hidden_size = weight.shape[0]  # k
        self.output_size = weight.shape[1]  # n
        # Reshape weight from [k, n] to [n, k] as done in C++ code
        self.weight = weight.reshape([weight.shape[1], weight.shape[0]])
        self.weight_scales = weight_scales.reshape(
            [weight_scales.shape[1], weight_scales.shape[0]]
        )
        self.bias = bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        original_dtype = input.dtype
        # Convert to BF16 if needed
        if input.dtype != torch.bfloat16:
            input_bf16 = input.to(torch.bfloat16)
        else:
            input_bf16 = input

        # Get input dimensions
        M = input_bf16.shape[0]
        N = self.output_size

        quantization_eps = 1e-10
        # Use per-token quantization (not per-token-block)
        input_fp8, input_scales = rocm_per_token_quant_fp8(
            input_bf16,
            eps=quantization_eps,
        )

        input_scales = input_scales.to(torch.float32)

        # Use per-token scales (M, 1)
        x_scales = input_scales
        w_scales = self.weight_scales

        K = input_fp8.shape[-1]

        # FlyDSL path: decode batches M=1~512 where FlyDSL has 1.1x-4.6x advantage.
        # Also handles K=128 which aiter doesn't support.
        if _USE_FLYDSL and 1 <= M <= 512 and K >= 128:
            output = _flydsl_preshuffle_gemm(
                input_fp8,
                self.weight,
                x_scales,
                w_scales,
                M,
                N,
                K,
                input_bf16.dtype,
            )
        elif K < 192:
            # 192 is aiter's empirical threshold: small-K uses bpreshuffle_cktile
            # (caller-allocated output); large-K uses gemm_a8w8_bpreshuffle (returns new).
            output = torch.empty(
                (M, N), dtype=input_bf16.dtype, device=input_bf16.device
            )
            gemm_a8w8_bpreshuffle_cktile(
                input_fp8, self.weight, x_scales, w_scales, output
            )
        else:
            output = aiter.gemm_a8w8_bpreshuffle(
                input_fp8,
                self.weight,
                x_scales,
                w_scales,
                None,
                input_bf16.dtype,
            )

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)

        # Convert back to original dtype if needed
        if output.dtype != original_dtype:
            output = output.to(original_dtype)

        return output
