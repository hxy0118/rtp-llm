# -*- coding: utf-8 -*-
"""
Standalone chunk-GDN kernel pipeline: exp vs exp2 precision & perf comparison.

All Triton kernels extracted from rtp-llm, no rtp-llm imports.
Two code paths run on identical bf16 inputs through the real Triton kernels,
differing only in the gating domain:
  - OLD: cumsum(g), then exp()  everywhere
  - NEW: cumsum(g) * RCP_LN2, then exp2() everywhere

Usage:
    /opt/conda310/bin/python3 test_exp_vs_exp2_kernels.py
"""

import logging
import os
import time
from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

os.environ["TRITON_F32_DEFAULT"] = "ieee"
logging.basicConfig(
    level="INFO", format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

RCP_LN2 = 1.0 / 0.6931471805599453
DEVICE = "cuda"


# ============================================================================
# Utility: index helpers (from fla/index.py)
# ============================================================================
def prepare_chunk_indices(cu_seqlens, chunk_size):
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    indices = torch.cat(
        [torch.arange(n) for n in triton.cdiv(lens, chunk_size).tolist()]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_chunk_offsets(cu_seqlens, chunk_size):
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(lens, chunk_size)]
    ).cumsum(-1)


# ============================================================================
# Kernel 1: chunk_local_cumsum  (parameterized via scale)
# ============================================================================
@triton.heuristics(
    {
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def _cumsum_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T
    p_s = tl.make_block_ptr(s + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


def chunk_local_cumsum(g, chunk_size=64, scale=None, cu_seqlens=None):
    B, T, H = g.shape
    BT = chunk_size
    ci = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(ci)
    out = torch.empty_like(g, dtype=torch.float32)
    _cumsum_kernel[(NT, B * H)](
        g, out, scale, cu_seqlens, ci, T, B=B, H=H, BT=BT, num_warps=8, num_stages=3
    )
    return out


# ============================================================================
# Kernel 2: chunk_scaled_dot_kkt  (USE_EXP2 parameterized)
# ============================================================================
@triton.heuristics(
    {
        "IS_VARLEN": lambda a: a["cu_seqlens"] is not None,
        "USE_G": lambda a: a["g_cumsum"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def _kkt_kernel(
    k,
    beta,
    g_cumsum,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T
    o_t = tl.arange(0, BT)
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, tl.trans(b_k))
    if USE_G:
        p_g = tl.make_block_ptr(
            g_cumsum + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
        )
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_diff = b_g[:, None] - b_g[None, :]
        if USE_EXP2:
            b_A = b_A * tl.math.exp2(b_g_diff)
        else:
            b_A = b_A * tl.where(b_g_diff <= 0, tl.exp(b_g_diff), 0.0)
    b_A *= b_beta[:, None]
    b_A = tl.where(o_t[:, None] > o_t[None, :], b_A, 0)
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (BT * H, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


def chunk_kkt(k, beta, g, cu_seqlens=None, use_exp2=True, chunk_size=64):
    B, T, H, K = k.shape
    BT = chunk_size
    ci = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(ci)
    A = torch.empty(B, T, H, BT, device=k.device, dtype=torch.float32)
    _kkt_kernel[(NT, B * H)](
        k,
        beta,
        g,
        A,
        cu_seqlens,
        ci,
        T,
        H=H,
        K=K,
        BT=BT,
        BK=64,
        USE_EXP2=use_exp2,
        num_warps=8,
        num_stages=3,
    )
    return A


# ============================================================================
# Kernel 3: solve_tril (identical for both paths, no exp involved)
# ============================================================================
@triton.heuristics({"IS_VARLEN": lambda a: a["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def _solve_16x16(
    A,
    Ad,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T
    A = A + (bos * H + i_h) * BT
    Ad = Ad + (bos * H + i_h) * 16
    offset = (i_t * 16) % BT
    p_A = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * 16, offset), (16, 16), (1, 0)
    )
    p_Ai = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (i_t * 16, 0), (16, 16), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
    b_A = -tl.where(tl.arange(0, 16)[:, None] > tl.arange(0, 16)[None, :], b_A, 0)
    o_i = tl.arange(0, 16)
    for i in range(1, min(16, T - i_t * 16)):
        b_a = -tl.load(A + (i_t * 16 + i) * H * BT + o_i + offset)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        b_A = tl.where((o_i == i)[:, None], b_a, b_A)
    b_A += o_i[:, None] == o_i[None, :]
    tl.store(
        p_Ai,
        b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


@triton.heuristics({"IS_VARLEN": lambda a: a["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def _merge_64(
    A,
    Ad,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T
    A += (bos * H + i_h) * 64
    Ad += (bos * H + i_h) * 16
    Ai += (bos * H + i_h) * 64
    stA, stD = H * 64, H * 16
    A21 = tl.load(
        tl.make_block_ptr(A, (T, 64), (stA, 1), (i_t * 64 + 16, 0), (16, 16), (1, 0)),
        boundary_check=(0, 1),
    ).to(tl.float32)
    A32 = tl.load(
        tl.make_block_ptr(A, (T, 64), (stA, 1), (i_t * 64 + 32, 16), (16, 16), (1, 0)),
        boundary_check=(0, 1),
    ).to(tl.float32)
    A31 = tl.load(
        tl.make_block_ptr(A, (T, 64), (stA, 1), (i_t * 64 + 32, 0), (16, 16), (1, 0)),
        boundary_check=(0, 1),
    ).to(tl.float32)
    A43 = tl.load(
        tl.make_block_ptr(A, (T, 64), (stA, 1), (i_t * 64 + 48, 32), (16, 16), (1, 0)),
        boundary_check=(0, 1),
    ).to(tl.float32)
    A42 = tl.load(
        tl.make_block_ptr(A, (T, 64), (stA, 1), (i_t * 64 + 48, 16), (16, 16), (1, 0)),
        boundary_check=(0, 1),
    ).to(tl.float32)
    A41 = tl.load(
        tl.make_block_ptr(A, (T, 64), (stA, 1), (i_t * 64 + 48, 0), (16, 16), (1, 0)),
        boundary_check=(0, 1),
    ).to(tl.float32)
    Ai11 = tl.load(
        tl.make_block_ptr(Ad, (T, 16), (stD, 1), (i_t * 64, 0), (16, 16), (1, 0)),
        boundary_check=(0, 1),
    ).to(tl.float32)
    Ai22 = tl.load(
        tl.make_block_ptr(Ad, (T, 16), (stD, 1), (i_t * 64 + 16, 0), (16, 16), (1, 0)),
        boundary_check=(0, 1),
    ).to(tl.float32)
    Ai33 = tl.load(
        tl.make_block_ptr(Ad, (T, 16), (stD, 1), (i_t * 64 + 32, 0), (16, 16), (1, 0)),
        boundary_check=(0, 1),
    ).to(tl.float32)
    Ai44 = tl.load(
        tl.make_block_ptr(Ad, (T, 16), (stD, 1), (i_t * 64 + 48, 0), (16, 16), (1, 0)),
        boundary_check=(0, 1),
    ).to(tl.float32)
    Ai21 = -tl.dot(
        tl.dot(Ai22, A21, input_precision="ieee"), Ai11, input_precision="ieee"
    )
    Ai32 = -tl.dot(
        tl.dot(Ai33, A32, input_precision="ieee"), Ai22, input_precision="ieee"
    )
    Ai43 = -tl.dot(
        tl.dot(Ai44, A43, input_precision="ieee"), Ai33, input_precision="ieee"
    )
    Ai31 = -tl.dot(
        Ai33,
        tl.dot(A31, Ai11, input_precision="ieee")
        + tl.dot(A32, Ai21, input_precision="ieee"),
        input_precision="ieee",
    )
    Ai42 = -tl.dot(
        Ai44,
        tl.dot(A42, Ai22, input_precision="ieee")
        + tl.dot(A43, Ai32, input_precision="ieee"),
        input_precision="ieee",
    )
    Ai41 = -tl.dot(
        Ai44,
        tl.dot(A41, Ai11, input_precision="ieee")
        + tl.dot(A42, Ai21, input_precision="ieee")
        + tl.dot(A43, Ai31, input_precision="ieee"),
        input_precision="ieee",
    )
    z = tl.zeros((16, 16), dtype=tl.float32)
    p11 = tl.make_block_ptr(Ai, (T, 64), (stA, 1), (i_t * 64, 0), (16, 16), (1, 0))
    p22 = tl.make_block_ptr(
        Ai, (T, 64), (stA, 1), (i_t * 64 + 16, 16), (16, 16), (1, 0)
    )
    p33 = tl.make_block_ptr(
        Ai, (T, 64), (stA, 1), (i_t * 64 + 32, 32), (16, 16), (1, 0)
    )
    p44 = tl.make_block_ptr(
        Ai, (T, 64), (stA, 1), (i_t * 64 + 48, 48), (16, 16), (1, 0)
    )
    p21 = tl.make_block_ptr(Ai, (T, 64), (stA, 1), (i_t * 64 + 16, 0), (16, 16), (1, 0))
    p31 = tl.make_block_ptr(Ai, (T, 64), (stA, 1), (i_t * 64 + 32, 0), (16, 16), (1, 0))
    p32 = tl.make_block_ptr(
        Ai, (T, 64), (stA, 1), (i_t * 64 + 32, 16), (16, 16), (1, 0)
    )
    p41 = tl.make_block_ptr(Ai, (T, 64), (stA, 1), (i_t * 64 + 48, 0), (16, 16), (1, 0))
    p42 = tl.make_block_ptr(
        Ai, (T, 64), (stA, 1), (i_t * 64 + 48, 16), (16, 16), (1, 0)
    )
    p43 = tl.make_block_ptr(
        Ai, (T, 64), (stA, 1), (i_t * 64 + 48, 32), (16, 16), (1, 0)
    )
    tl.store(
        p11,
        Ai11.to(p11.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p22,
        Ai22.to(p22.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p33,
        Ai33.to(p33.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p44,
        Ai44.to(p44.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p21,
        Ai21.to(p21.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p31,
        Ai31.to(p31.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p32,
        Ai32.to(p32.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p41,
        Ai41.to(p41.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p42,
        Ai42.to(p42.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p43,
        Ai43.to(p43.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    pz12 = tl.make_block_ptr(Ai, (T, 64), (stA, 1), (i_t * 64, 16), (16, 16), (1, 0))
    pz13 = tl.make_block_ptr(Ai, (T, 64), (stA, 1), (i_t * 64, 32), (16, 16), (1, 0))
    pz14 = tl.make_block_ptr(Ai, (T, 64), (stA, 1), (i_t * 64, 48), (16, 16), (1, 0))
    pz23 = tl.make_block_ptr(
        Ai, (T, 64), (stA, 1), (i_t * 64 + 16, 32), (16, 16), (1, 0)
    )
    pz24 = tl.make_block_ptr(
        Ai, (T, 64), (stA, 1), (i_t * 64 + 16, 48), (16, 16), (1, 0)
    )
    pz34 = tl.make_block_ptr(
        Ai, (T, 64), (stA, 1), (i_t * 64 + 32, 48), (16, 16), (1, 0)
    )
    tl.store(pz12, z.to(pz12.dtype.element_ty), boundary_check=(0, 1))
    tl.store(pz13, z.to(pz13.dtype.element_ty), boundary_check=(0, 1))
    tl.store(pz14, z.to(pz14.dtype.element_ty), boundary_check=(0, 1))
    tl.store(pz23, z.to(pz23.dtype.element_ty), boundary_check=(0, 1))
    tl.store(pz24, z.to(pz24.dtype.element_ty), boundary_check=(0, 1))
    tl.store(pz34, z.to(pz34.dtype.element_ty), boundary_check=(0, 1))


def solve_tril(A, cu_seqlens=None, output_dtype=torch.bfloat16, chunk_size=64):
    B, T, H, BT = A.shape
    Ad = torch.empty(B, T, H, 16, device=A.device, dtype=torch.float32)
    ci16 = prepare_chunk_indices(cu_seqlens, 16) if cu_seqlens is not None else None
    NT16 = len(ci16) if ci16 is not None else triton.cdiv(T, 16)
    _solve_16x16[NT16, B * H](
        A, Ad, cu_seqlens, ci16, T, H=H, BT=BT, num_warps=1, num_stages=4
    )
    Ai = torch.empty(B, T, H, BT, device=A.device, dtype=output_dtype)
    ci64 = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT64 = len(ci64) if ci64 is not None else triton.cdiv(T, BT)
    _merge_64[NT64, B * H](
        A, Ad, Ai, cu_seqlens, ci64, T, H=H, BT=BT, num_warps=4, num_stages=3
    )
    return Ai


# ============================================================================
# Kernel 4: recompute_w_u  (USE_EXP2 parameterized)
# ============================================================================
@triton.heuristics({"IS_VARLEN": lambda a: a["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def _wy_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    p_g = tl.make_block_ptr(g + (bos * H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_g_raw = tl.load(p_g, boundary_check=(0,))
    if USE_EXP2:
        b_g = tl.math.exp2(b_g_raw)
    else:
        b_g = tl.exp(b_g_raw)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def recompute_w_u(k, v, beta, A, g, cu_seqlens=None, use_exp2=True):
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = A.shape[-1]
    ci = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(ci)
    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    _wy_kernel[(NT, B * H)](
        k,
        v,
        beta,
        w,
        u,
        A,
        g,
        cu_seqlens,
        ci,
        T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=64,
        BV=min(128, V),
        USE_EXP2=use_exp2,
        num_warps=4,
        num_stages=3,
    )
    return w, u


# ============================================================================
# Kernel 5: fwd_h  (USE_EXP2 parameterized)
# ============================================================================
@triton.heuristics(
    {
        "USE_G": lambda a: a["g"] is not None,
        "USE_INITIAL_STATE": lambda a: a["h0"] is not None,
        "STORE_FINAL_STATE": lambda a: a["ht"] is not None,
        "IS_VARLEN": lambda a: a["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def _fwd_h_kernel(
    k,
    v,
    w,
    v_new,
    g,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos = i_n * T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)

    h += ((boh * H + i_h) * K * V).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * H + i_h) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v, stride_h, stride_k, stride_w = H * V, H * K * V, H * K, H * K

    if USE_INITIAL_STATE:
        p = tl.make_block_ptr(
            h0 + i_nh * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
        )
        b_h1 += tl.load(p, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p2 = tl.make_block_ptr(
                h0 + i_nh * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            )
            b_h2 += tl.load(p2, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(
            h + i_t * stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(
                h + i_t * stride_h, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v_sub = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w2 = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_v_sub += tl.dot(tl.load(p_w2, boundary_check=(0, 1)), b_h2.to(b_w.dtype))
        p_v = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v_val = tl.load(p_v, boundary_check=(0, 1)) - b_v_sub

        p_vn = tl.make_block_ptr(
            v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        tl.store(p_vn, b_v_val.to(p_vn.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + (bos * H + last_idx * H + i_h).to(tl.int64)).to(
                tl.float32
            )
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            if USE_EXP2:
                b_v_val = (
                    b_v_val * tl.where(m_t, tl.math.exp2(b_g_last - b_g), 0)[:, None]
                )
                b_g_last_exp = tl.math.exp2(b_g_last)
            else:
                b_v_val = b_v_val * tl.where(m_t, tl.exp(b_g_last - b_g), 0)[:, None]
                b_g_last_exp = tl.exp(b_g_last)
            b_h1 = b_h1 * b_g_last_exp
            if K > 64:
                b_h2 = b_h2 * b_g_last_exp
        b_v_val = b_v_val.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_h1 += tl.dot(tl.load(p_k, boundary_check=(0, 1)), b_v_val)
        if K > 64:
            p_k2 = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_h2 += tl.dot(tl.load(p_k2, boundary_check=(0, 1)), b_v_val)

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht2 = tl.make_block_ptr(
                ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_ht2, b_h2.to(p_ht2.dtype.element_ty), boundary_check=(0, 1))


def fwd_h(
    k, w, u, g, h0, output_final_state, cu_seqlens=None, use_exp2=True, chunk_size=64
):
    B, T, H, K = k.shape
    V = u.shape[-1]
    BT = chunk_size
    if cu_seqlens is None:
        N, NT, co = B, triton.cdiv(T, BT), None
    else:
        N = len(cu_seqlens) - 1
        co = prepare_chunk_offsets(cu_seqlens, BT)
        NT = int(co[-1])
    h = k.new_empty(B, NT, H, K, V, dtype=torch.float32)
    ht = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u)
    grid = lambda meta: (triton.cdiv(V, meta["BV"]), N * H)
    _fwd_h_kernel[grid](
        k,
        u,
        w,
        v_new,
        g,
        h,
        h0,
        ht,
        cu_seqlens,
        co,
        T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BV=64,
        USE_EXP2=use_exp2,
        num_warps=4,
        num_stages=2,
    )
    return h, v_new, ht


# ============================================================================
# Kernel 6: fwd_o  (USE_EXP2 parameterized)
# ============================================================================
@triton.heuristics(
    {
        "USE_G": lambda a: a["g"] is not None,
        "IS_VARLEN": lambda a: a["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def _fwd_o_kernel(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos = i_b * T
    q2 = q + (bos * H + i_h) * K
    k2 = k + (bos * H + i_h) * K
    v2 = v + (bos * H + i_h) * V
    o2 = o + (bos * H + i_h) * V
    h2 = h + (i_tg * H + i_h).to(tl.int64) * K * V
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q2, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k2, (K, T), (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1)
        )
        p_h = tl.make_block_ptr(
            h2, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
        )
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_h.to(b_q.dtype))
        b_A += tl.dot(b_q, b_k)
    if USE_G:
        p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        if USE_EXP2:
            b_o = b_o * tl.math.exp2(b_g)[:, None]
            b_A = b_A * tl.math.exp2(b_g[:, None] - b_g[None, :])
        else:
            b_o = b_o * tl.exp(b_g)[:, None]
            b_A = b_A * tl.where(
                b_g[:, None] - b_g[None, :] <= 0,
                tl.exp(b_g[:, None] - b_g[None, :]),
                0.0,
            )
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_v = tl.make_block_ptr(
        v2, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_o = tl.make_block_ptr(
        o2, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def fwd_o(q, k, v, h, g, scale, cu_seqlens=None, use_exp2=True, chunk_size=64):
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    ci = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(ci)
    o = torch.zeros_like(v)
    grid = lambda meta: (triton.cdiv(V, meta["BV"]), NT, B * H)
    _fwd_o_kernel[grid](
        q,
        k,
        v,
        h,
        g,
        o,
        cu_seqlens,
        ci,
        scale,
        T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=64,
        BV=128,
        USE_EXP2=use_exp2,
        num_warps=4,
        num_stages=1,
    )
    return o


# ============================================================================
# Full pipeline: parameterized on use_exp2
# ============================================================================
def chunk_gdn_pipeline(q, k, v, g_raw, beta, scale, h0, cu_seqlens=None, use_exp2=True):
    """Run the full chunk-GDN Triton kernel pipeline."""
    g_scale = RCP_LN2 if use_exp2 else None
    g = chunk_local_cumsum(g_raw, chunk_size=64, scale=g_scale, cu_seqlens=cu_seqlens)
    A_raw = chunk_kkt(k, beta, g, cu_seqlens=cu_seqlens, use_exp2=use_exp2)
    A = solve_tril(A_raw, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u = recompute_w_u(k, v, beta, A, g, cu_seqlens=cu_seqlens, use_exp2=use_exp2)
    h, v_new, ht = fwd_h(k, w, u, g, h0, True, cu_seqlens=cu_seqlens, use_exp2=use_exp2)
    o = fwd_o(q, k, v_new, h, g, scale, cu_seqlens=cu_seqlens, use_exp2=use_exp2)
    return o, ht


# ============================================================================
# Recurrent reference
# ============================================================================
def recurrent_ref(q, k, v, beta, g, scale, h0=None):
    B, T, H, K = q.shape
    V = v.shape[-1]
    q2, k2, v2, beta2, g2 = [x.float() for x in (q, k, v, beta, g)]
    o = torch.zeros(B, T, H, V, device=q.device)
    h = torch.zeros(B, H, K, V, device=q.device)
    if h0 is not None:
        h = h0.float().clone()
    for i in range(T):
        h = h * g2[:, i, :, None, None].exp()
        vi = v2[:, i].clone() - (h * k2[:, i, :, :, None]).sum(-2)
        vi = vi * beta2[:, i, :, None]
        h = h + k2[:, i, :, :, None] * vi[:, :, None, :]
        o[:, i] = torch.einsum("bhd,bhdv->bhv", q2[:, i] * scale, h)
    return o, h


# ============================================================================
# Metrics
# ============================================================================
def compare(name, a, b):
    a, b = a.float().flatten(), b.float().flatten()
    d = a - b
    base = a.square().mean().sqrt().item()
    abs_max = d.abs().max().item()
    rel = d.square().mean().sqrt().item() / (base + 1e-12)
    cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return name, abs_max, rel, cos


def log_cmp(name, abs_max, rel, cos):
    logger.info(
        f"  {name:25s}  abs_max={abs_max:.6e}  rel_err={rel:.6e}  cos={cos:.8f}"
    )


# ============================================================================
# Main
# ============================================================================
def run_test(B, T, H, K, V=None, cu_seqlens=None, seed=42):
    if V is None:
        V = K
    logger.info("=" * 65)
    logger.info(f"B={B} T={T} H={H} K={K} V={V} cu_seqlens={cu_seqlens}")
    logger.info("=" * 65)
    torch.manual_seed(seed)
    N = len(cu_seqlens) - 1 if cu_seqlens else B
    B_eff = 1 if cu_seqlens else B
    T_total = cu_seqlens[-1] if cu_seqlens else T

    q = torch.randn(B_eff, T_total, H, K, device=DEVICE, dtype=torch.bfloat16)
    k = F.normalize(torch.randn(B_eff, T_total, H, K, device=DEVICE), dim=-1).bfloat16()
    v = torch.randn(B_eff, T_total, H, V, device=DEVICE, dtype=torch.bfloat16)
    g = F.logsigmoid(torch.rand(B_eff, T_total, H, device=DEVICE, dtype=torch.bfloat16))
    beta = torch.rand(B_eff, T_total, H, device=DEVICE, dtype=torch.bfloat16).sigmoid()
    h0 = torch.randn(N, H, K, V, device=DEVICE, dtype=torch.bfloat16)
    scale = K**-0.5
    cu = torch.LongTensor(cu_seqlens).to(DEVICE) if cu_seqlens else None

    # Run both pipelines
    o_exp, ht_exp = chunk_gdn_pipeline(q, k, v, g, beta, scale, h0, cu, use_exp2=False)
    o_exp2, ht_exp2 = chunk_gdn_pipeline(q, k, v, g, beta, scale, h0, cu, use_exp2=True)

    # Run recurrent reference
    if cu_seqlens:
        ro, rh = [], []
        for i in range(N):
            s, e = cu_seqlens[i], cu_seqlens[i + 1]
            oi, hi = recurrent_ref(
                q[:, s:e],
                k[:, s:e],
                v[:, s:e],
                beta[:, s:e],
                g[:, s:e],
                scale,
                h0[i : i + 1],
            )
            ro.append(oi)
            rh.append(hi)
        ref_o, ref_h = torch.cat(ro, 1), torch.cat(rh, 0)
    else:
        ref_o, ref_h = recurrent_ref(q, k, v, beta, g, scale, h0)

    logger.info("--- exp (old) vs recurrent ref ---")
    log_cmp(*compare("o_exp_vs_ref", ref_o, o_exp))
    log_cmp(*compare("ht_exp_vs_ref", ref_h, ht_exp))

    logger.info("--- exp2 (new) vs recurrent ref ---")
    log_cmp(*compare("o_exp2_vs_ref", ref_o, o_exp2))
    log_cmp(*compare("ht_exp2_vs_ref", ref_h, ht_exp2))

    logger.info("--- exp vs exp2 (the delta) ---")
    log_cmp(*compare("o_exp_vs_exp2", o_exp, o_exp2))
    log_cmp(*compare("ht_exp_vs_exp2", ht_exp, ht_exp2))

    # Benchmark
    torch.cuda.synchronize()
    for _ in range(3):
        chunk_gdn_pipeline(q, k, v, g, beta, scale, h0, cu, use_exp2=False)
        chunk_gdn_pipeline(q, k, v, g, beta, scale, h0, cu, use_exp2=True)
    torch.cuda.synchronize()

    n_iter = 20
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        chunk_gdn_pipeline(q, k, v, g, beta, scale, h0, cu, use_exp2=False)
    torch.cuda.synchronize()
    ms_exp = (time.perf_counter() - t0) / n_iter * 1e3

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        chunk_gdn_pipeline(q, k, v, g, beta, scale, h0, cu, use_exp2=True)
    torch.cuda.synchronize()
    ms_exp2 = (time.perf_counter() - t0) / n_iter * 1e3

    logger.info(
        f"--- perf: exp={ms_exp:.3f} ms, exp2={ms_exp2:.3f} ms, "
        f"speedup={ms_exp/ms_exp2:.3f}x ---"
    )


def main():
    run_test(B=1, T=256, H=4, K=64, cu_seqlens=[0, 256])
    run_test(B=1, T=512, H=4, K=128, cu_seqlens=[0, 512])
    run_test(B=1, T=1000, H=4, K=64, cu_seqlens=[0, 256, 500, 1000])
    run_test(B=1, T=2048, H=32, K=128, cu_seqlens=[0, 2048])
    run_test(B=1, T=4096, H=4, K=128, cu_seqlens=[0, 4096])
    run_test(B=2, T=256, H=4, K=64)


if __name__ == "__main__":
    main()
