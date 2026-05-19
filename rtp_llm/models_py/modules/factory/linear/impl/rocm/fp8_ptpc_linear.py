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

# Optimized tile configs keyed by (M, K, N) → (tile_m, tile_n, tile_k, xcd_swizzle, use_async_copy, scheduler_mode).
# Benchmarked on MI308X (gfx942, 80 CUs), 2026-05-18.
# Fallback: if exact (M,K,N) not found, match by M with K/N heuristics.
_FLYDSL_TILE_TABLE_FULL: Dict[
    Tuple[int, int, int], Tuple[int, int, int, int, bool, int]
] = {
    # 397B shapes (K, N from Qwen3.5-397B TP4/TP8)
    # M=16
    (16, 4096, 3072): (16, 64, 512, 0, True, 0),
    (16, 4096, 4608): (16, 64, 512, 0, True, 0),
    (16, 4096, 2048): (16, 64, 512, 0, True, 0),
    (16, 4096, 2304): (16, 64, 512, 0, True, 0),
    (16, 2048, 4096): (16, 64, 512, 4, True, 0),
    (16, 1024, 4096): (16, 128, 256, 2, False, 0),
    (16, 4096, 512): (16, 64, 512, 0, True, 0),
    (16, 4096, 256): (16, 64, 512, 0, True, 0),
    (16, 256, 4096): (16, 64, 256, 0, False, 0),
    (16, 128, 4096): (16, 64, 128, 0, False, 0),
    # M=32
    (32, 4096, 3072): (16, 64, 256, 2, False, 0),
    (32, 4096, 4608): (16, 64, 256, 2, False, 0),
    (32, 4096, 2048): (16, 64, 512, 2, True, 0),
    (32, 4096, 2304): (16, 64, 512, 2, True, 0),
    (32, 2048, 4096): (16, 128, 256, 2, False, 0),
    (32, 1024, 4096): (16, 64, 256, 2, False, 0),
    (32, 4096, 512): (16, 64, 512, 0, True, 0),
    (32, 4096, 256): (16, 64, 256, 0, False, 0),
    (32, 256, 4096): (32, 64, 256, 4, False, 0),
    (32, 128, 4096): (32, 64, 128, 2, False, 0),
    # M=64
    (64, 4096, 3072): (16, 64, 256, 4, False, 0),
    (64, 4096, 4608): (32, 64, 512, 2, True, 0),
    (64, 4096, 2048): (16, 128, 256, 2, False, 0),
    (64, 4096, 2304): (16, 64, 256, 2, False, 0),
    (64, 2048, 4096): (32, 128, 256, 2, False, 0),
    (64, 1024, 4096): (16, 128, 256, 4, False, 0),
    (64, 4096, 512): (16, 64, 512, 0, True, 0),
    (64, 4096, 256): (16, 64, 512, 0, True, 0),
    (64, 256, 4096): (32, 128, 128, 2, False, 0),
    (64, 128, 4096): (32, 64, 128, 2, False, 0),
    # M=128
    (128, 4096, 3072): (32, 128, 256, 2, False, 0),
    (128, 4096, 4608): (64, 128, 256, 2, False, 0),
    (128, 4096, 2048): (32, 64, 256, 2, False, 0),
    (128, 4096, 2304): (32, 64, 256, 2, False, 0),
    (128, 2048, 4096): (64, 128, 256, 2, False, 0),
    (128, 1024, 4096): (64, 128, 128, 2, False, 0),
    (128, 4096, 512): (32, 64, 256, 2, False, 0),
    (128, 4096, 256): (32, 64, 256, 2, False, 0),
    (128, 256, 4096): (32, 128, 128, 2, False, 0),
    (128, 128, 4096): (32, 128, 128, 2, False, 0),
    # M=256
    (256, 4096, 3072): (32, 128, 256, 2, False, 0),
    (256, 4096, 4608): (64, 128, 256, 2, False, 0),
    (256, 4096, 2048): (32, 128, 256, 2, False, 0),
    (256, 4096, 2304): (32, 128, 256, 2, False, 0),
    (256, 2048, 4096): (64, 128, 128, 2, False, 0),
    (256, 1024, 4096): (64, 128, 128, 2, False, 0),
    (256, 4096, 512): (32, 64, 256, 2, False, 0),
    (256, 4096, 256): (32, 64, 256, 2, False, 0),
    (256, 256, 4096): (64, 128, 128, 2, False, 0),
    (256, 128, 4096): (64, 128, 128, 2, False, 0),
    # M=512
    (512, 4096, 3072): (64, 128, 128, 2, False, 0),
    (512, 4096, 4608): (64, 256, 128, 2, False, 0),
    (512, 4096, 2048): (64, 128, 128, 2, False, 0),
    (512, 4096, 2304): (64, 128, 128, 2, False, 0),
    (512, 2048, 4096): (64, 256, 128, 2, False, 0),
    (512, 1024, 4096): (64, 256, 128, 2, False, 0),
    (512, 4096, 512): (64, 64, 256, 2, False, 0),
    (512, 4096, 256): (64, 64, 256, 2, False, 0),
    (512, 256, 4096): (64, 128, 128, 2, False, 0),
    (512, 128, 4096): (64, 128, 128, 2, False, 0),
    # ── Qwen3.5-27B TP2 shapes (benchmarked 2026-05-18, +28% overall vs fallback) ──
    # 5120×8192 (lin_qkv/TP2, P1 V5)
    (16, 5120, 8192): (16, 64, 256, 0, False, 5),
    # 5120×4096 (full_qkv/TP2, P1 V5: xcd 4→0, sm 2→0)
    (16, 5120, 4096): (16, 64, 512, 0, False, 0),
    (24, 5120, 4096): (16, 64, 256, 4, False, 1),
    (32, 5120, 4096): (32, 64, 512, 0, False, 2),
    (48, 5120, 4096): (32, 128, 256, 2, False, 1),
    (64, 5120, 4096): (32, 128, 256, 4, False, 1),
    (80, 5120, 4096): (32, 128, 256, 2, False, 0),
    (96, 5120, 4096): (32, 128, 256, 4, False, 0),
    (112, 5120, 4096): (32, 128, 256, 4, False, 0),
    (128, 5120, 4096): (32, 128, 256, 4, False, 0),
    # 3072×5120 (full_o+lin_o/TP2)
    (16, 3072, 5120): (16, 64, 512, 0, False, 0),
    (24, 3072, 5120): (16, 64, 256, 2, False, 1),
    (32, 3072, 5120): (32, 64, 512, 0, False, 2),
    (48, 3072, 5120): (32, 128, 256, 2, False, 1),
    (64, 3072, 5120): (32, 128, 256, 2, False, 1),
    (80, 3072, 5120): (32, 128, 256, 4, False, 0),
    (96, 3072, 5120): (32, 128, 256, 4, False, 0),
    (112, 3072, 5120): (32, 128, 256, 2, False, 0),
    (128, 3072, 5120): (32, 128, 256, 2, False, 0),
    # 5120×7168 (lin_qkv/TP2)
    (16, 5120, 7168): (16, 64, 256, 0, False, 2),
    (24, 5120, 7168): (16, 64, 256, 2, False, 0),
    (32, 5120, 7168): (32, 128, 256, 0, False, 1),
    (48, 5120, 7168): (32, 128, 256, 0, False, 0),
    (64, 5120, 7168): (64, 128, 256, 0, False, 0),
    (80, 5120, 7168): (32, 128, 256, 4, False, 0),
    (96, 5120, 7168): (32, 128, 256, 4, False, 0),
    (112, 5120, 7168): (32, 128, 256, 2, False, 0),
    (128, 5120, 7168): (32, 128, 256, 2, False, 0),
    # 5120×3072 (lin_o/TP2, newly discovered independent O proj)
    (16, 5120, 3072): (16, 64, 512, 0, False, 0),
    # 5120×17408 (gate_up/TP2)
    (16, 5120, 17408): (16, 128, 256, 0, False, 0),
    (24, 5120, 17408): (16, 64, 512, 4, False, 0),
    (32, 5120, 17408): (32, 128, 256, 2, False, 0),
    (48, 5120, 17408): (32, 128, 256, 4, False, 1),
    (64, 5120, 17408): (64, 128, 256, 2, False, 0),
    (80, 5120, 17408): (32, 128, 256, 4, False, 0),
    (96, 5120, 17408): (32, 128, 256, 4, False, 1),
    (112, 5120, 17408): (64, 256, 128, 4, False, 0),
    (128, 5120, 17408): (64, 256, 128, 2, False, 0),
    # 8704×5120 (down/TP2)
    (16, 8704, 5120): (16, 64, 512, 0, False, 5),
    (24, 8704, 5120): (16, 64, 512, 4, False, 0),
    (32, 8704, 5120): (32, 64, 512, 0, False, 2),
    (48, 8704, 5120): (32, 128, 256, 4, False, 1),
    (64, 8704, 5120): (32, 128, 256, 4, False, 1),
    (80, 8704, 5120): (64, 128, 256, 4, False, 0),
    (96, 8704, 5120): (64, 128, 256, 2, False, 0),
    (112, 8704, 5120): (64, 128, 256, 4, False, 0),
    (128, 8704, 5120): (64, 128, 256, 2, False, 0),
}

# Per-shape LDS stage override (default is 2 = ping-pong double buffer).
# lds_stage=1 benefits shapes with short K-loops (fewer iterations → less double-buffer benefit).
_FLYDSL_LDS_STAGE_TABLE: Dict[Tuple[int, int, int], int] = {
    (16, 5120, 3072): 1,  # K/tile_k=5120/512=10 iters, lds_stage=1 is ~30% faster
    (16, 3072, 5120): 1,  # K/tile_k=3072/512=6 iters, lds_stage=1 beats aiter by 6%
    # P1 V5 additions: lds_stage=1 better than 2 for these shapes
    (16, 5120, 17408): 1,  # gate_up: tile_k=256, K/tk=20 iters
    (16, 8704, 5120): 1,  # down: tile_k=512, K/tk=17 iters
    (16, 5120, 4096): 1,  # full_qkv: tile_k=512, K/tk=10 iters
}

# Per-shape preload override: (dsrd_preload, dvmem_preload). Default is (-1, -1) = auto.
_FLYDSL_PRELOAD_TABLE: Dict[Tuple[int, int, int], Tuple[int, int]] = {
    # P1 V5: down tuned to (4,2) (was (8,2)) for better K-loop pipeline
    (16, 8704, 5120): (4, 2),
    # P1 V5 additions
    (16, 5120, 8192): (4, 1),  # lin_qkv
    (16, 3072, 5120): (4, 1),  # full_o
    (16, 5120, 3072): (4, 2),  # lin_o
}

# Heuristic fallback by M when exact (M, K, N) not in table.
_FLYDSL_TILE_FALLBACK: Dict[int, Tuple[int, int, int, int, bool, int]] = {
    16: (16, 64, 256, 2, False, 0),
    24: (16, 64, 256, 2, False, 1),
    32: (32, 64, 256, 2, False, 0),
    48: (32, 128, 256, 2, False, 1),
    64: (32, 128, 256, 2, False, 0),
    80: (32, 128, 256, 4, False, 0),
    96: (32, 128, 256, 4, False, 0),
    112: (32, 128, 256, 4, False, 0),
    128: (64, 128, 256, 2, False, 0),
    256: (64, 128, 128, 2, False, 0),
    512: (64, 128, 128, 2, False, 0),
}


def _select_flydsl_config(
    M: int, K: int, N: int
) -> Tuple[int, int, int, int, bool, int]:
    """Select best (tile_m, tile_n, tile_k, xcd_swizzle, use_async_copy, scheduler_mode) for given shape."""
    cfg = _FLYDSL_TILE_TABLE_FULL.get((M, K, N))
    if cfg is not None:
        return cfg
    # Fallback: use M-based heuristic
    fb = _FLYDSL_TILE_FALLBACK.get(M)
    if fb is not None:
        tile_m, tile_n, tile_k, xcd, async_c, sm = fb
        if N % tile_n != 0:
            tile_n = 64
        if K % tile_k != 0:
            tile_k = 128 if K % 128 == 0 else 64
        return (tile_m, tile_n, tile_k, xcd, async_c, sm)
    # Last resort
    return (16, 64, 256, 0, False, 0)


_flydsl_compile_v2_fn = None  # compile_preshuffle_gemm_v2


def _init_flydsl():
    global _flydsl_initialized, _flydsl_compile_fn, _flydsl_compile_v2_fn, _flyc_compile
    global _flydsl_arch, _flydsl_fp8_dtype
    if _flydsl_initialized:
        return
    import sys

    flydsl_home = os.environ.get("FLYDSL_HOME", "/root/FlyDSL")
    # Only add flydsl_home for kernel imports; flydsl package comes from site-packages
    if flydsl_home not in sys.path:
        sys.path.insert(0, flydsl_home)
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


_flydsl_dispatch_log_count = {"main": 0, "small_m": 0}


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

    # P1 V5 debug: log first 50 dispatches to verify whether FlyDSL is actually invoked
    if _flydsl_dispatch_log_count["main"] + _flydsl_dispatch_log_count["small_m"] < 50:
        graph_state = "?"
        try:
            graph_state = (
                "capturing" if torch.cuda.is_current_stream_capturing() else "normal"
            )
        except Exception:
            pass
        path = "small_m" if M < 16 else "main"
        _flydsl_dispatch_log_count[path] += 1
        logger.info(
            "FlyDSL dispatch [%s #%d]: M=%d K=%d N=%d (graph=%s, init_done=%s)",
            path,
            _flydsl_dispatch_log_count[path],
            M,
            K,
            N,
            graph_state,
            _flydsl_initialized,
        )

    # M < 16: pad to 16 and use v1 kernel with M=16 tuned config (P1 V5)
    if M < 16:
        return _flydsl_preshuffle_gemm_small_m(
            input_fp8, weight, x_scales, w_scales, M, N, K, out_dtype
        )

    cache_key = (M, N, K)
    compiled = _flydsl_gemm_cache.get(cache_key)

    tile_m, tile_n, tile_k, xcd_swizzle, use_async, scheduler_mode = (
        _select_flydsl_config(M, K, N)
    )

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
        lds_stage = _FLYDSL_LDS_STAGE_TABLE.get((M, K, N), 2)
        dsrd_pl, dvmem_pl = _FLYDSL_PRELOAD_TABLE.get((M, K, N), (-1, -1))
        launch_fn = _flydsl_compile_fn(
            N=N,
            K=K,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            in_dtype="fp8",
            out_dtype=out_dtype_str,
            lds_stage=lds_stage,
            use_async_copy=use_async,
            xcd_swizzle=xcd_swizzle,
            scheduler_mode=scheduler_mode,
            dsrd_preload=dsrd_pl,
            dvmem_preload=dvmem_pl,
        )
        compiled = _flyc_compile(launch_fn, *args)
        _flydsl_gemm_cache[cache_key] = compiled
        logger.info(
            "FlyDSL GEMM compiled: M=%d K=%d N=%d tile=(%d,%d,%d) xcd=%d sm=%d",
            M,
            K,
            N,
            tile_m,
            tile_n,
            tile_k,
            xcd_swizzle,
            scheduler_mode,
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
    """M < 16 path: directly call v1 kernel with actual M (no padding, no copy).

    P1 V7: FlyDSL v1 kernel already supports M < tile_m=16 thanks to AMD buffer_load
    hardware OOB protection:
    - kernel reads `c_m = i32_m` runtime arg (not tile_m)
    - buffer resource num_records = c_m × K × elem_bytes (matches actual buffer size)
    - BUFFER_LOAD_DWORDX4 returns 0 for OOB rows (no segfault)
    - MFMA computes 16 rows (m_repeat=1), but rows ≥ M see all-zero A → output 0
    - BUFFER_STORE drops OOB writes → only first M rows of C get written
    No padding, no copy, no graph overhead. Reuses M=16 tuned config (kernel timing
    is M-independent at tile_m=16).

    Earlier versions:
    - V5: 3× torch.zeros + 2× copy_, +9.4ms regression (memset overhead in graph)
    - V6: persistent buffers, 2× copy_, still +1ms regression (copy in graph)
    - V7: zero copy, expects ~30% kernel speedup (single-op verified) to translate
      directly to TPOT improvement
    """
    cache_key = (16, N, K, "v1_small_m")
    compiled = _flydsl_gemm_cache.get(cache_key)

    # Output buffer must have only M rows (kernel store will be OOB-clipped to M rows
    # via c_rsrc num_records = M × N × 2 bytes).
    output = torch.empty(M, N, device=input_fp8.device, dtype=out_dtype)
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
        M,  # ← actual M, not 16. kernel uses this for buffer resource num_records.
        N,
        stream,
    )

    if compiled is None:
        # Reuse M=16 tuned config (kernel timing is M-independent at tile_m=16)
        tile_m, tile_n, tile_k, xcd_swizzle, use_async, scheduler_mode = (
            _select_flydsl_config(16, K, N)
        )
        lds_stage = _FLYDSL_LDS_STAGE_TABLE.get((16, K, N), 2)
        dsrd_pl, dvmem_pl = _FLYDSL_PRELOAD_TABLE.get((16, K, N), (-1, -1))
        launch_fn = _flydsl_compile_fn(
            N=N,
            K=K,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            in_dtype="fp8",
            out_dtype=out_dtype_str,
            lds_stage=lds_stage,
            use_async_copy=use_async,
            xcd_swizzle=xcd_swizzle,
            scheduler_mode=scheduler_mode,
            dsrd_preload=dsrd_pl,
            dvmem_preload=dvmem_pl,
        )
        compiled = _flyc_compile(launch_fn, *args)
        _flydsl_gemm_cache[cache_key] = compiled
        logger.info(
            "FlyDSL v1 small-M GEMM compiled (V7 zero-copy): actual_M=%d K=%d N=%d tile=(%d,%d,%d) xcd=%d sm=%d lds=%d pl=(%d,%d)",
            M,
            K,
            N,
            tile_m,
            tile_n,
            tile_k,
            xcd_swizzle,
            scheduler_mode,
            lds_stage,
            dsrd_pl,
            dvmem_pl,
        )

    compiled(*args)
    return output


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

        # FlyDSL path: CUDA graph decode M values (1-128) where per-shape tuned kernels beat aiter.
        # P1 V5: M ∈ [1,15] routed through small_m (pad-to-16 + v1 with M=16 tuned config),
        # rocprofv3 confirms 90/90 combos beat aiter ≥10% on 27B TP2 shapes.
        # M > 128: prefill, aiter is better (FlyDSL tile table only covers decode sizes).
        # Safety: small-M only uses FlyDSL when (16, K, N) is in _FLYDSL_TILE_TABLE_FULL,
        # so non-27B shapes fall back to aiter (avoids untuned fallback config slowdown).
        _small_m_safe = M >= 16 or (16, K, N) in _FLYDSL_TILE_TABLE_FULL

        # P1 V5 debug: log first 30 forward calls to verify dispatch decision
        if not hasattr(self.__class__, "_dbg_fwd_count"):
            self.__class__._dbg_fwd_count = 0
        if self.__class__._dbg_fwd_count < 30:
            self.__class__._dbg_fwd_count += 1
            graph_state = "?"
            try:
                graph_state = (
                    "capturing"
                    if torch.cuda.is_current_stream_capturing()
                    else "normal"
                )
            except Exception:
                pass
            will_use_flydsl = bool(
                _USE_FLYDSL and 1 <= M <= 128 and K >= 128 and _small_m_safe
            )
            logger.info(
                "RocmFp8PTPCLinear.forward [#%d]: M=%d K=%d N=%d, USE_FLYDSL=%s, gating: 1<=M<=128=%s K>=128=%s small_m_safe=%s → use_flydsl=%s (graph=%s)",
                self.__class__._dbg_fwd_count,
                M,
                K,
                N,
                _USE_FLYDSL,
                1 <= M <= 128,
                K >= 128,
                _small_m_safe,
                will_use_flydsl,
                graph_state,
            )

        if _USE_FLYDSL and 1 <= M <= 128 and K >= 128 and _small_m_safe:
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
