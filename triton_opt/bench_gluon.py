#!/usr/bin/env python3
"""Gluon chunk-GDN benchmark: Gluon fwd_o vs original Triton baseline.

Default = production first-prefill mode (varlen=True, output_final_state=True, initial_state=None).

Usage:
  python bench_gluon.py                         # TP2 64K, production mode
  python bench_gluon.py --T 4096                # shorter sequence
  python bench_gluon.py --Hg 16 --H 64          # TP1
  python bench_gluon.py --no-prod               # raw kernel (no varlen/state)
"""

import argparse
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy

# ═══════════════════════════════════════════════════════════════════
# Import Triton baseline from bench_standalone.py
# ═══════════════════════════════════════════════════════════════════

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_standalone import (
    chunk_local_cumsum,
    kkt_fwd,
    solve_tril,
    recompute_w_u_fwd,
    fused_kkt_solve,
    fwd_h_orig,
    fwd_h as fwd_h_tuned,
    fwd_o_orig,
    fwd_o as fwd_o_tuned,
    prepare_chunk_indices,
    prepare_chunk_offsets,
    new_pipeline,
    old_pipeline as triton_old_pipeline,
    RCP_LN2,
)


def old_pipeline(q, k, v, g, beta, scale, output_final_state=False,
                 cu_seqlens=None, chunk_indices=None, chunk_offsets=None):
    """RTP original Triton baseline."""
    g_cum = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    A = kkt_fwd(k, beta, g_cum)
    A = solve_tril(A, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k, v, beta, A, g_cum, use_exp2=False)
    h, v_new, ht = fwd_h_orig(k, w, u, g_cum, use_exp2=False,
                               output_final_state=output_final_state,
                               cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets)
    o = fwd_o_orig(q, k, v_new, h, g_cum, scale, use_exp2=False)
    return o, ht


# ═══════════════════════════════════════════════════════════════════
# Gluon fwd_o kernel
# ═══════════════════════════════════════════════════════════════════

@triton.heuristics({
    "USE_G": lambda args: args["g"] is not None,
})
@gluon.jit
def gluon_chunk_fwd_kernel_o(
    q, k, v, h, g, o,
    scale,
    T,
    H: gl.constexpr,
    Hg: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BT: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    USE_G: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    i_v = gl.program_id(0)
    i_t = gl.program_id(1)
    i_bh = gl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H

    NT = gl.cdiv(T, BT)
    i_tg = i_b * NT + i_t
    bos = i_b * T

    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[32, 32, 16],
        transposed=True, warps_per_cta=[NUM_WARPS, 1],
    )
    dot_op0: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=8)
    dot_op1: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=8)

    # Shared layouts matching Triton TTGIR
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[1, 0])
    shared_h: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, order=[1, 0])
    shared_layout_t: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[0, 1])

    blocked_64: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[8, 8],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0],
    )
    blocked_2d_t: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1], threads_per_warp=[8, 8],
        warps_per_cta=[NUM_WARPS, 1], order=[0, 1],
    )

    # blocked_hv: for [64,128] h/v tiles — works with any NUM_WARPS
    blocked_hv: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[4, 16],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0])

    q_base = q + (bos * Hg + i_h // (H // Hg)) * K
    k_base = k + (bos * Hg + i_h // (H // Hg)) * K
    v_base = v + (bos * H + i_h) * V
    o_base = o + (bos * H + i_h) * V
    h_base = h + gl.cast(i_tg * H + i_h, gl.int64) * K * V

    q_row = gl.arange(0, BT, layout=gl.SliceLayout(1, blocked_64))
    q_col = gl.arange(0, BK, layout=gl.SliceLayout(0, blocked_64))
    k_row = gl.arange(0, BK, layout=gl.SliceLayout(1, blocked_2d_t))
    k_col = gl.arange(0, BT, layout=gl.SliceLayout(0, blocked_2d_t))
    h_row = gl.arange(0, BK, layout=gl.SliceLayout(1, blocked_hv))
    h_col = gl.arange(0, BV, layout=gl.SliceLayout(0, blocked_hv))
    t_mask = (i_t * BT + q_row[:, None]) < T
    t_mask_col = (i_t * BT + k_col[None, :]) < T

    q_offs = gl.cast((i_t * BT + q_row[:, None]) * (Hg * K) + q_col[None, :], gl.int32)
    k_offs = gl.cast(k_row[:, None] + (i_t * BT + k_col[None, :]) * (Hg * K), gl.int32)
    h_offs = gl.cast(h_row[:, None] * V + (i_v * BV + h_col[None, :]), gl.int32)

    b_q = gl.amd.cdna4.buffer_load(ptr=q_base, offsets=q_offs, mask=t_mask, other=0.0)
    b_k = gl.amd.cdna4.buffer_load(ptr=k_base, offsets=k_offs, mask=t_mask_col, other=0.0)
    b_h = gl.amd.cdna4.buffer_load(ptr=h_base, offsets=h_offs)

    q_dot = gl.convert_layout(b_q, dot_op0)
    h_smem = gl.allocate_shared_memory(gl.bfloat16, [BK, BV], shared_h, value=b_h)
    k_smem = gl.allocate_shared_memory(gl.bfloat16, [BK, BT], shared_layout_t, value=b_k)
    h_dot = h_smem.load(dot_op1)
    k_dot = k_smem.load(dot_op1)
    b_o = gl.zeros((BT, BV), dtype=gl.float32, layout=mma)
    b_A = gl.zeros((BT, BT), dtype=gl.float32, layout=mma)
    b_o = gl.amd.cdna4.mfma(q_dot, h_dot, b_o)
    b_A = gl.amd.cdna4.mfma(q_dot, k_dot, b_A)

    if K > BK:
        q_offs2 = gl.cast((i_t * BT + q_row[:, None]) * (Hg * K) + (BK + q_col[None, :]), gl.int32)
        k_offs2 = gl.cast((BK + k_row[:, None]) + (i_t * BT + k_col[None, :]) * (Hg * K), gl.int32)
        h_offs2 = gl.cast((BK + h_row[:, None]) * V + (i_v * BV + h_col[None, :]), gl.int32)

        b_q2 = gl.amd.cdna4.buffer_load(ptr=q_base, offsets=q_offs2, mask=t_mask, other=0.0)
        b_k2 = gl.amd.cdna4.buffer_load(ptr=k_base, offsets=k_offs2, mask=t_mask_col, other=0.0)
        b_h2 = gl.amd.cdna4.buffer_load(ptr=h_base, offsets=h_offs2)

        q_dot2 = gl.convert_layout(b_q2, dot_op0)
        h_smem2 = gl.allocate_shared_memory(gl.bfloat16, [BK, BV], shared_h, value=b_h2)
        k_smem2 = gl.allocate_shared_memory(gl.bfloat16, [BK, BT], shared_layout_t, value=b_k2)
        h_dot2 = h_smem2.load(dot_op1)
        k_dot2 = k_smem2.load(dot_op1)
        b_o = gl.amd.cdna4.mfma(q_dot2, h_dot2, b_o)
        b_A = gl.amd.cdna4.mfma(q_dot2, k_dot2, b_A)

    if USE_G:
        g_base = g + bos * H + i_h
        g_row_idx = gl.arange(0, BT, layout=gl.SliceLayout(1, mma))
        g_col_idx = gl.arange(0, BT, layout=gl.SliceLayout(0, mma))
        g_row_offs = gl.cast((i_t * BT + g_row_idx) * H, gl.int32)
        g_col_offs = gl.cast((i_t * BT + g_col_idx) * H, gl.int32)
        g_row_mask = (i_t * BT + g_row_idx) < T
        g_col_mask = (i_t * BT + g_col_idx) < T
        b_g_row = gl.amd.cdna4.buffer_load(ptr=g_base, offsets=g_row_offs, mask=g_row_mask, other=0.0).to(gl.float32)
        b_g_col = gl.amd.cdna4.buffer_load(ptr=g_base, offsets=g_col_offs, mask=g_col_mask, other=0.0).to(gl.float32)
        b_o = b_o * gl.exp(b_g_row)[:, None]
        b_A = b_A * gl.exp(b_g_row[:, None] - b_g_col[None, :])

    o_t_row = i_t * BT + gl.arange(0, BT, layout=gl.SliceLayout(1, mma))
    o_t_col = i_t * BT + gl.arange(0, BT, layout=gl.SliceLayout(0, mma))
    m_t_row = o_t_row < T
    m_t_col = o_t_col < T
    m_A = (o_t_row[:, None] >= o_t_col[None, :]) & (m_t_row[:, None] & m_t_col[None, :])
    b_A = gl.where(m_A, b_A, 0.0)

    v_row = gl.arange(0, BT, layout=gl.SliceLayout(1, blocked_hv))
    v_col = gl.arange(0, BV, layout=gl.SliceLayout(0, blocked_hv))
    v_offs = gl.cast((i_t * BT + v_row[:, None]) * (H * V) + (i_v * BV + v_col[None, :]), gl.int32)
    v_mask = (i_t * BT + v_row[:, None]) < T
    b_v = gl.amd.cdna4.buffer_load(ptr=v_base, offsets=v_offs, mask=v_mask, other=0.0)

    A_smem = gl.allocate_shared_memory(gl.bfloat16, [BT, BT], shared_layout, value=b_A.to(gl.bfloat16))
    v_smem = gl.allocate_shared_memory(gl.bfloat16, [BT, BV], shared_h, value=b_v)
    A_dot = A_smem.load(dot_op0)
    v_dot = v_smem.load(dot_op1)
    b_intra = gl.zeros((BT, BV), dtype=gl.float32, layout=mma)
    b_intra = gl.amd.cdna4.mfma(A_dot, v_dot, b_intra)

    b_o = b_o * scale + b_intra * scale

    o_out = b_o.to(gl.bfloat16)
    o_row = gl.arange(0, BT, layout=gl.SliceLayout(1, mma))
    o_col = gl.arange(0, BV, layout=gl.SliceLayout(0, mma))
    o_offs = gl.cast((i_t * BT + o_row[:, None]) * (H * V) + (i_v * BV + o_col[None, :]), gl.int32)
    o_mask = (i_t * BT + o_row[:, None]) < T
    gl.amd.cdna4.buffer_store(stored_value=o_out, ptr=o_base, offsets=o_offs, mask=o_mask)


def gluon_fwd_o(q, k, v_new, h, g, scale):
    B, T, Hg, K = q.shape
    V = v_new.shape[-1]
    H = v_new.shape[-2]
    BT = min(64, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT)
    o = torch.zeros_like(v_new)

    def grid(meta):
        return (triton.cdiv(V, 128), NT, B * H)

    gluon_chunk_fwd_kernel_o[grid](
        q, k, v_new, h, g, o, scale,
        T=T, H=H, Hg=Hg, K=K, V=V,
        BT=BT, BK=64, BV=128,
        NUM_WARPS=1,
        num_warps=1, num_stages=1,
    )
    return o


def gluon_pipeline(q, k, v, g, beta, scale, output_final_state=False,
                   cu_seqlens=None, chunk_indices=None, chunk_offsets=None):
    """Gluon fwd_o + Triton for other kernels (hybrid pipeline)."""
    g_cum = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    A = kkt_fwd(k, beta, g_cum)
    A = solve_tril(A, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k, v, beta, A, g_cum, use_exp2=False)
    h, v_new, ht = fwd_h_orig(k, w, u, g_cum, use_exp2=False,
                               output_final_state=output_final_state,
                               cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets)
    o = gluon_fwd_o(q, k, v_new, h, g_cum, scale)
    return o, ht


# ═══════════════════════════════════════════════════════════════════
# Gluon fwd_o v2: async_copy (DMA bypass registers)
# ═══════════════════════════════════════════════════════════════════

@triton.heuristics({
    "USE_G": lambda args: args["g"] is not None,
})
@gluon.jit
def gluon_chunk_fwd_kernel_o_v2(
    q, k, v, h, g, o,
    scale,
    T,
    H: gl.constexpr,
    Hg: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BT: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    USE_G: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    i_v = gl.program_id(0)
    i_t = gl.program_id(1)
    i_bh = gl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H

    NT = gl.cdiv(T, BT)
    i_tg = i_b * NT + i_t
    bos = i_b * T

    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[32, 32, 16],
        transposed=True, warps_per_cta=[NUM_WARPS, 1],
    )
    dot_op0: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=8)
    dot_op1: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=8)

    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[1, 0])
    shared_layout_t: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[0, 1])

    # blocked_64: for async_copy of [*, 64] tiles (q). spt[1]*16=128 ✓
    blocked_64: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[8, 8],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0],
    )
    # blocked_128: for async_copy of [*, 128] tiles (h, v). tpw=[4,16] keeps spt[1]=8 → 128 bits
    blocked_128: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[4, 16],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0],
    )
    # blocked_2d_t: for k (buffer_load path, order=[0,1] not supported by async_copy)
    blocked_2d_t: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1], threads_per_warp=[8, 8],
        warps_per_cta=[NUM_WARPS, 1], order=[0, 1],
    )

    q_base = q + (bos * Hg + i_h // (H // Hg)) * K
    k_base = k + (bos * Hg + i_h // (H // Hg)) * K
    v_base = v + (bos * H + i_h) * V
    o_base = o + (bos * H + i_h) * V
    h_base = h + gl.cast(i_tg * H + i_h, gl.int64) * K * V
    h_tile_off = (i_tg * H + i_h) * (K * V)

    # q indices (blocked_64, order=[1,0])
    q_row = gl.arange(0, BT, layout=gl.SliceLayout(1, blocked_64))
    q_col = gl.arange(0, BK, layout=gl.SliceLayout(0, blocked_64))
    t_mask = (i_t * BT + q_row[:, None]) < T
    q_offs = gl.cast((i_t * BT + q_row[:, None]) * (Hg * K) + q_col[None, :], gl.int32)

    # k indices (blocked_2d_t, order=[0,1] — buffer_load path)
    k_row = gl.arange(0, BK, layout=gl.SliceLayout(1, blocked_2d_t))
    k_col = gl.arange(0, BT, layout=gl.SliceLayout(0, blocked_2d_t))
    t_mask_col = (i_t * BT + k_col[None, :]) < T
    k_offs = gl.cast(k_row[:, None] + (i_t * BT + k_col[None, :]) * (Hg * K), gl.int32)

    # h indices (blocked_128, order=[1,0] — async_copy path, V contiguous)
    h_row = gl.arange(0, BK, layout=gl.SliceLayout(1, blocked_128))
    h_col = gl.arange(0, BV, layout=gl.SliceLayout(0, blocked_128))
    h_offs = gl.cast(h_tile_off + h_row[:, None] * V + (i_v * BV + h_col[None, :]), gl.int32)

    b_o = gl.zeros((BT, BV), dtype=gl.float32, layout=mma)
    b_A = gl.zeros((BT, BT), dtype=gl.float32, layout=mma)

    # === Async copy q, h (DMA, bypass registers) ===
    q_smem = gl.allocate_shared_memory(gl.bfloat16, [BT, BK], shared_layout)
    async_copy.buffer_load_to_shared(dest=q_smem, ptr=q_base, offsets=q_offs, mask=t_mask, other=0.0)

    h_smem = gl.allocate_shared_memory(gl.bfloat16, [BK, BV], shared_layout)
    async_copy.buffer_load_to_shared(dest=h_smem, ptr=h, offsets=h_offs)

    # === k: buffer_load path (order=[0,1] not supported by async_copy) ===
    b_k = gl.amd.cdna4.buffer_load(ptr=k_base, offsets=k_offs, mask=t_mask_col, other=0.0)
    k_smem = gl.allocate_shared_memory(gl.bfloat16, [BK, BT], shared_layout_t, value=b_k)

    if K > BK:
        q_offs2 = gl.cast((i_t * BT + q_row[:, None]) * (Hg * K) + (BK + q_col[None, :]), gl.int32)
        k_offs2 = gl.cast((BK + k_row[:, None]) + (i_t * BT + k_col[None, :]) * (Hg * K), gl.int32)
        h_offs2 = gl.cast(h_tile_off + (BK + h_row[:, None]) * V + (i_v * BV + h_col[None, :]), gl.int32)

        q_smem2 = gl.allocate_shared_memory(gl.bfloat16, [BT, BK], shared_layout)
        async_copy.buffer_load_to_shared(dest=q_smem2, ptr=q_base, offsets=q_offs2, mask=t_mask, other=0.0)

        h_smem2 = gl.allocate_shared_memory(gl.bfloat16, [BK, BV], shared_layout)
        async_copy.buffer_load_to_shared(dest=h_smem2, ptr=h, offsets=h_offs2)

        b_k2 = gl.amd.cdna4.buffer_load(ptr=k_base, offsets=k_offs2, mask=t_mask_col, other=0.0)
        k_smem2 = gl.allocate_shared_memory(gl.bfloat16, [BK, BT], shared_layout_t, value=b_k2)

    # === Prefetch v (async_copy, [BT, BV] order=[1,0]) ===
    v_row = gl.arange(0, BT, layout=gl.SliceLayout(1, blocked_128))
    v_col = gl.arange(0, BV, layout=gl.SliceLayout(0, blocked_128))
    v_offs = gl.cast((i_t * BT + v_row[:, None]) * (H * V) + (i_v * BV + v_col[None, :]), gl.int32)
    v_mask = (i_t * BT + v_row[:, None]) < T
    v_smem = gl.allocate_shared_memory(gl.bfloat16, [BT, BV], shared_layout)
    async_copy.buffer_load_to_shared(dest=v_smem, ptr=v_base, offsets=v_offs, mask=v_mask, other=0.0)

    async_copy.commit_group()
    async_copy.wait_group(0)

    # === Consume chunk 1 ===
    q_dot = q_smem.load(dot_op0)
    h_dot = h_smem.load(dot_op1)
    k_dot = k_smem.load(dot_op1)
    b_o = gl.amd.cdna4.mfma(q_dot, h_dot, b_o)
    b_A = gl.amd.cdna4.mfma(q_dot, k_dot, b_A)

    if K > BK:
        q_dot2 = q_smem2.load(dot_op0)
        h_dot2 = h_smem2.load(dot_op1)
        k_dot2 = k_smem2.load(dot_op1)
        b_o = gl.amd.cdna4.mfma(q_dot2, h_dot2, b_o)
        b_A = gl.amd.cdna4.mfma(q_dot2, k_dot2, b_A)

    # === Gating ===
    if USE_G:
        g_base = g + bos * H + i_h
        g_row_idx = gl.arange(0, BT, layout=gl.SliceLayout(1, mma))
        g_col_idx = gl.arange(0, BT, layout=gl.SliceLayout(0, mma))
        g_row_offs = gl.cast((i_t * BT + g_row_idx) * H, gl.int32)
        g_col_offs = gl.cast((i_t * BT + g_col_idx) * H, gl.int32)
        g_row_mask = (i_t * BT + g_row_idx) < T
        g_col_mask = (i_t * BT + g_col_idx) < T
        b_g_row = gl.amd.cdna4.buffer_load(ptr=g_base, offsets=g_row_offs, mask=g_row_mask, other=0.0).to(gl.float32)
        b_g_col = gl.amd.cdna4.buffer_load(ptr=g_base, offsets=g_col_offs, mask=g_col_mask, other=0.0).to(gl.float32)
        b_o = b_o * gl.exp(b_g_row)[:, None]
        b_A = b_A * gl.exp(b_g_row[:, None] - b_g_col[None, :])

    # === Causal mask + A@v ===
    o_t_row = i_t * BT + gl.arange(0, BT, layout=gl.SliceLayout(1, mma))
    o_t_col = i_t * BT + gl.arange(0, BT, layout=gl.SliceLayout(0, mma))
    m_t_row = o_t_row < T
    m_t_col = o_t_col < T
    m_A = (o_t_row[:, None] >= o_t_col[None, :]) & (m_t_row[:, None] & m_t_col[None, :])
    b_A = gl.where(m_A, b_A, 0.0)

    A_smem = gl.allocate_shared_memory(gl.bfloat16, [BT, BT], shared_layout, value=b_A.to(gl.bfloat16))
    A_dot = A_smem.load(dot_op0)
    v_dot = v_smem.load(dot_op1)
    b_intra = gl.zeros((BT, BV), dtype=gl.float32, layout=mma)
    b_intra = gl.amd.cdna4.mfma(A_dot, v_dot, b_intra)

    b_o = b_o * scale + b_intra * scale

    # === Store output ===
    o_out = b_o.to(gl.bfloat16)
    o_row = gl.arange(0, BT, layout=gl.SliceLayout(1, mma))
    o_col = gl.arange(0, BV, layout=gl.SliceLayout(0, mma))
    o_offs = gl.cast((i_t * BT + o_row[:, None]) * (H * V) + (i_v * BV + o_col[None, :]), gl.int32)
    o_mask = (i_t * BT + o_row[:, None]) < T
    gl.amd.cdna4.buffer_store(stored_value=o_out, ptr=o_base, offsets=o_offs, mask=o_mask)


def gluon_fwd_o_v2(q, k, v_new, h, g, scale):
    B, T, Hg, K = q.shape
    V = v_new.shape[-1]
    H = v_new.shape[-2]
    BT = min(64, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT)
    o = torch.zeros_like(v_new)

    def grid(meta):
        return (triton.cdiv(V, 128), NT, B * H)

    gluon_chunk_fwd_kernel_o_v2[grid](
        q, k, v_new, h, g, o, scale,
        T=T, H=H, Hg=Hg, K=K, V=V,
        BT=BT, BK=64, BV=128,
        NUM_WARPS=1,
        num_warps=1, num_stages=1,
    )
    return o


def gluon_pipeline_v2(q, k, v, g, beta, scale, output_final_state=False,
                      cu_seqlens=None, chunk_indices=None, chunk_offsets=None):
    """Gluon fwd_o v2 (async_copy) + Triton for other kernels."""
    g_cum = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    A = kkt_fwd(k, beta, g_cum)
    A = solve_tril(A, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k, v, beta, A, g_cum, use_exp2=False)
    h, v_new, ht = fwd_h_orig(k, w, u, g_cum, use_exp2=False,
                               output_final_state=output_final_state,
                               cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets)
    o = gluon_fwd_o_v2(q, k, v_new, h, g_cum, scale)
    return o, ht


# ═══════════════════════════════════════════════════════════════════
# Gluon fwd_o v3: pure buffer_load + prefetch + optimal smem layouts
# ═══════════════════════════════════════════════════════════════════

@triton.heuristics({
    "USE_G": lambda args: args["g"] is not None,
})
@gluon.jit
def gluon_chunk_fwd_kernel_o_v3(
    q, k, v, h, g, o,
    scale,
    T,
    H: gl.constexpr,
    Hg: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BT: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    USE_G: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    i_v = gl.program_id(0)
    i_t = gl.program_id(1)
    i_bh = gl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H

    NT = gl.cdiv(T, BT)
    i_tg = i_b * NT + i_t
    bos = i_b * T

    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[32, 32, 16],
        transposed=True, warps_per_cta=[NUM_WARPS, 1],
    )
    dot_op0: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=8)
    dot_op1: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=8)

    # Optimal shared layouts: order=[1,0] for dot_op0, order=[0,1] for dot_op1
    shared_A: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[1, 0])
    shared_B: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[0, 1])

    # Load layouts
    blocked_row: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[8, 8],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0],
    )
    blocked_col: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1], threads_per_warp=[8, 8],
        warps_per_cta=[NUM_WARPS, 1], order=[0, 1],
    )

    # Base pointers
    q_base = q + (bos * Hg + i_h // (H // Hg)) * K
    k_base = k + (bos * Hg + i_h // (H // Hg)) * K
    v_base = v + (bos * H + i_h) * V
    o_base = o + (bos * H + i_h) * V
    h_base = h + gl.cast(i_tg * H + i_h, gl.int64) * K * V

    # Index arrays
    q_row = gl.arange(0, BT, layout=gl.SliceLayout(1, blocked_row))
    q_col = gl.arange(0, BK, layout=gl.SliceLayout(0, blocked_row))
    k_row = gl.arange(0, BK, layout=gl.SliceLayout(1, blocked_col))
    k_col = gl.arange(0, BT, layout=gl.SliceLayout(0, blocked_col))
    h_col = gl.arange(0, BV, layout=gl.SliceLayout(0, blocked_col))
    v_row = gl.arange(0, BT, layout=gl.SliceLayout(1, blocked_row))
    v_col = gl.arange(0, BV, layout=gl.SliceLayout(0, blocked_row))

    t_mask = (i_t * BT + q_row[:, None]) < T
    t_mask_col = (i_t * BT + k_col[None, :]) < T

    # ── Pre-allocate reusable smem (avoids allocate_shared_memory(value=) overhead) ──
    smem_A = gl.allocate_shared_memory(gl.bfloat16, [1, BT, BK], shared_A)
    smem_B = gl.allocate_shared_memory(gl.bfloat16, [1, BK, BV], shared_B)
    smem_Bk = gl.allocate_shared_memory(gl.bfloat16, [1, BK, BT], shared_B)

    # ── Phase 1: Issue ALL loads upfront (hardware OOO overlaps them) ──
    q_offs1 = gl.cast((i_t * BT + q_row[:, None]) * (Hg * K) + q_col[None, :], gl.int32)
    k_offs1 = gl.cast(k_row[:, None] + (i_t * BT + k_col[None, :]) * (Hg * K), gl.int32)
    h_offs1 = gl.cast(k_row[:, None] * V + (i_v * BV + h_col[None, :]), gl.int32)

    b_q1 = gl.amd.cdna4.buffer_load(ptr=q_base, offsets=q_offs1, mask=t_mask, other=0.0)
    b_k1 = gl.amd.cdna4.buffer_load(ptr=k_base, offsets=k_offs1, mask=t_mask_col, other=0.0)
    b_h1 = gl.amd.cdna4.buffer_load(ptr=h_base, offsets=h_offs1)

    if K > BK:
        q_offs2 = gl.cast((i_t * BT + q_row[:, None]) * (Hg * K) + (BK + q_col[None, :]), gl.int32)
        k_offs2 = gl.cast((BK + k_row[:, None]) + (i_t * BT + k_col[None, :]) * (Hg * K), gl.int32)
        h_offs2 = gl.cast((BK + k_row[:, None]) * V + (i_v * BV + h_col[None, :]), gl.int32)
        b_q2 = gl.amd.cdna4.buffer_load(ptr=q_base, offsets=q_offs2, mask=t_mask, other=0.0)
        b_k2 = gl.amd.cdna4.buffer_load(ptr=k_base, offsets=k_offs2, mask=t_mask_col, other=0.0)
        b_h2 = gl.amd.cdna4.buffer_load(ptr=h_base, offsets=h_offs2)

    v_offs = gl.cast((i_t * BT + v_row[:, None]) * (H * V) + (i_v * BV + v_col[None, :]), gl.int32)
    v_mask = (i_t * BT + v_row[:, None]) < T
    b_v = gl.amd.cdna4.buffer_load(ptr=v_base, offsets=v_offs, mask=v_mask, other=0.0)

    if USE_G:
        g_base = g + bos * H + i_h
        g_row_idx = gl.arange(0, BT, layout=gl.SliceLayout(1, mma))
        g_col_idx = gl.arange(0, BT, layout=gl.SliceLayout(0, mma))
        g_row_offs = gl.cast((i_t * BT + g_row_idx) * H, gl.int32)
        g_col_offs = gl.cast((i_t * BT + g_col_idx) * H, gl.int32)
        g_row_mask = (i_t * BT + g_row_idx) < T
        g_col_mask = (i_t * BT + g_col_idx) < T
        b_g_row = gl.amd.cdna4.buffer_load(ptr=g_base, offsets=g_row_offs, mask=g_row_mask, other=0.0)
        b_g_col = gl.amd.cdna4.buffer_load(ptr=g_base, offsets=g_col_offs, mask=g_col_mask, other=0.0)

    # ── Phase 2: K-block 1 (reuse pre-allocated smem) ──
    smem_A.index(0).store(b_q1)
    smem_B.index(0).store(b_h1)
    smem_Bk.index(0).store(b_k1)
    q_dot = smem_A.index(0).load(dot_op0)
    h_dot = smem_B.index(0).load(dot_op1)
    k_dot = smem_Bk.index(0).load(dot_op1)
    b_o = gl.zeros((BT, BV), dtype=gl.float32, layout=mma)
    b_A = gl.zeros((BT, BT), dtype=gl.float32, layout=mma)
    b_o = gl.amd.cdna4.mfma(q_dot, h_dot, b_o)
    b_A = gl.amd.cdna4.mfma(q_dot, k_dot, b_A)

    # ── Phase 3: K-block 2 (reuse same smem) ──
    if K > BK:
        smem_A.index(0).store(b_q2)
        smem_B.index(0).store(b_h2)
        smem_Bk.index(0).store(b_k2)
        q_dot2 = smem_A.index(0).load(dot_op0)
        h_dot2 = smem_B.index(0).load(dot_op1)
        k_dot2 = smem_Bk.index(0).load(dot_op1)
        b_o = gl.amd.cdna4.mfma(q_dot2, h_dot2, b_o)
        b_A = gl.amd.cdna4.mfma(q_dot2, k_dot2, b_A)

    # ── Phase 4: Gating (g data already loaded) ──
    if USE_G:
        b_g_r = b_g_row.to(gl.float32)
        b_g_c = b_g_col.to(gl.float32)
        b_o = b_o * gl.exp(b_g_r)[:, None]
        b_A = b_A * gl.exp(b_g_r[:, None] - b_g_c[None, :])

    # ── Phase 5: Causal mask ──
    o_t_row = i_t * BT + gl.arange(0, BT, layout=gl.SliceLayout(1, mma))
    o_t_col = i_t * BT + gl.arange(0, BT, layout=gl.SliceLayout(0, mma))
    m_t_row = o_t_row < T
    m_t_col = o_t_col < T
    m_A = (o_t_row[:, None] >= o_t_col[None, :]) & (m_t_row[:, None] & m_t_col[None, :])
    b_A = gl.where(m_A, b_A, 0.0)

    # ── Phase 6: A@v (reuse smem, optimal dot_op1 layout for v) ──
    smem_A.index(0).store(b_A.to(gl.bfloat16))
    smem_B.index(0).store(b_v)
    A_dot = smem_A.index(0).load(dot_op0)
    v_dot = smem_B.index(0).load(dot_op1)
    b_intra = gl.zeros((BT, BV), dtype=gl.float32, layout=mma)
    b_intra = gl.amd.cdna4.mfma(A_dot, v_dot, b_intra)

    # ── Phase 7: Output ──
    b_o = b_o * scale + b_intra * scale
    o_out = b_o.to(gl.bfloat16)
    o_row = gl.arange(0, BT, layout=gl.SliceLayout(1, mma))
    o_col = gl.arange(0, BV, layout=gl.SliceLayout(0, mma))
    o_offs = gl.cast((i_t * BT + o_row[:, None]) * (H * V) + (i_v * BV + o_col[None, :]), gl.int32)
    o_mask = (i_t * BT + o_row[:, None]) < T
    gl.amd.cdna4.buffer_store(stored_value=o_out, ptr=o_base, offsets=o_offs, mask=o_mask)


def gluon_fwd_o_v3(q, k, v_new, h, g, scale):
    B, T, Hg, K = q.shape
    V = v_new.shape[-1]
    H = v_new.shape[-2]
    BT = min(64, max(16, triton.next_power_of_2(T)))
    o = torch.zeros_like(v_new)
    def grid(meta):
        return (triton.cdiv(V, 128), triton.cdiv(T, BT), B * H)
    gluon_chunk_fwd_kernel_o_v3[grid](
        q, k, v_new, h, g, o, scale,
        T=T, H=H, Hg=Hg, K=K, V=V,
        BT=BT, BK=64, BV=128,
        NUM_WARPS=1,
        num_warps=1, num_stages=1,
    )
    return o


# ═══════════════════════════════════════════════════════════════════
# Gluon fwd_h: state accumulation kernel
# ═══════════════════════════════════════════════════════════════════

@triton.heuristics({
    "USE_G": lambda args: args["g"] is not None,
    "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
    "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
    "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@gluon.jit
def gluon_chunk_fwd_kernel_h(
    k, v, w, v_new, g, h, h0, ht,
    cu_seqlens, chunk_offsets,
    T,
    H: gl.constexpr,
    Hg: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BT: gl.constexpr,
    BV: gl.constexpr,
    USE_G: gl.constexpr,
    USE_INITIAL_STATE: gl.constexpr,
    STORE_FINAL_STATE: gl.constexpr,
    SAVE_NEW_VALUE: gl.constexpr,
    USE_EXP2: gl.constexpr,
    IS_VARLEN: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    i_v = gl.program_id(0)
    i_nh = gl.program_id(1)
    i_n = i_nh // H
    i_h = i_nh % H

    if IS_VARLEN:
        bos = gl.load(cu_seqlens + i_n).to(gl.int32)
        eos = gl.load(cu_seqlens + i_n + 1).to(gl.int32)
        T = eos - bos
        NT = gl.cdiv(T, BT)
        boh = gl.load(chunk_offsets + i_n).to(gl.int32)
    else:
        bos = i_n * T
        NT = gl.cdiv(T, BT)
        boh = i_n * NT

    # ── Layouts ──
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32],
        transposed=True, warps_per_cta=[NUM_WARPS, 1],
    )
    dot_op0: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=8)
    dot_op1: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=8)

    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[1, 0])
    shared_layout_t: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[0, 1])

    # blocked_wk: for loading w [BT,64] and k^T [64,BT] (both [64,64])
    blocked_wk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[8, 8],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0],
    )
    # blocked_kt: for loading k^T with K(dim0) contiguous
    blocked_kt: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1], threads_per_warp=[8, 8],
        warps_per_cta=[1, NUM_WARPS], order=[0, 1],
    )

    # ── Base pointers & strides ──
    h_base = h + (boh * H + i_h) * K * V
    v_base = v + (bos * H + i_h) * V
    k_base = k + (bos * Hg + i_h // (H // Hg)) * K
    w_base = w + (bos * H + i_h) * K

    stride_v = H * V
    stride_h = H * K * V
    stride_k = Hg * K
    stride_w = H * K

    # ── State accumulators (register-resident across all iterations) ──
    b_h1 = gl.zeros((64, BV), dtype=gl.float32, layout=mma)
    if K > 64:
        b_h2 = gl.zeros((64, BV), dtype=gl.float32, layout=mma)

    # ── Load initial state ──
    if USE_INITIAL_STATE:
        h0_base = h0 + i_nh * K * V
        h0_row = gl.arange(0, 64, layout=gl.SliceLayout(1, mma))
        h0_col = gl.arange(0, BV, layout=gl.SliceLayout(0, mma))
        h0_offs = gl.cast(h0_row[:, None] * V + (i_v * BV + h0_col[None, :]), gl.int32)
        b_h1 = b_h1 + gl.amd.cdna4.buffer_load(ptr=h0_base, offsets=h0_offs).to(gl.float32)
        if K > 64:
            h0_offs2 = gl.cast((64 + h0_row[:, None]) * V + (i_v * BV + h0_col[None, :]), gl.int32)
            b_h2 = b_h2 + gl.amd.cdna4.buffer_load(ptr=h0_base, offsets=h0_offs2).to(gl.float32)

    # ── Pre-allocate double-buffered shared memory ──
    smem_a = gl.allocate_shared_memory(gl.bfloat16, [2, 64, 64], shared_layout)

    # ── Index arrays ──
    m_row = gl.arange(0, 64, layout=gl.SliceLayout(1, mma))
    m_col_bv = gl.arange(0, BV, layout=gl.SliceLayout(0, mma))
    m_row_bt = gl.arange(0, BT, layout=gl.SliceLayout(1, mma))
    # Triton #blocked2 for v loading (BV=16): spt=[1,4] tpw=[16,4] wpc=[4,1]
    blocked_v: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4], threads_per_warp=[16, 4],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0]
    )
    v_row_b2 = gl.arange(0, BT, layout=gl.SliceLayout(1, blocked_v))
    v_col_b2 = gl.arange(0, BV, layout=gl.SliceLayout(0, blocked_v))

    w_row = gl.arange(0, BT, layout=gl.SliceLayout(1, blocked_wk))
    w_col = gl.arange(0, 64, layout=gl.SliceLayout(0, blocked_wk))
    kt_row = gl.arange(0, 64, layout=gl.SliceLayout(1, blocked_kt))
    kt_col = gl.arange(0, BT, layout=gl.SliceLayout(0, blocked_kt))

    h_offs_base1 = gl.cast(m_row[:, None] * V + (i_v * BV + m_col_bv[None, :]), gl.int32)
    if K > 64:
        h_offs_base2 = gl.cast(64 * V + m_row[:, None] * V + (i_v * BV + m_col_bv[None, :]), gl.int32)
    w_offs_base1 = gl.cast(w_row[:, None] * stride_w + w_col[None, :], gl.int32)
    if K > 64:
        w_offs_base2 = gl.cast(w_row[:, None] * stride_w + (64 + w_col[None, :]), gl.int32)
    v_offs_base = gl.cast(v_row_b2[:, None] * stride_v + (i_v * BV + v_col_b2[None, :]), gl.int32)
    kt_offs_base1 = gl.cast(kt_row[:, None] + kt_col[None, :] * stride_k, gl.int32)
    if K > 64:
        kt_offs_base2 = gl.cast(64 + kt_row[:, None] + kt_col[None, :] * stride_k, gl.int32)

    if USE_G:
        g_base = g + bos * H + i_h
        g_offs_base = gl.cast(m_row_bt * H, gl.int32)
    if SAVE_NEW_VALUE:
        vn_base = v_new + (bos * H + i_h) * V

    # ── Prologue: prefetch w[0] and v[0] ──
    b_w1_pre = gl.amd.cdna4.buffer_load(ptr=w_base, offsets=w_offs_base1)
    if K > 64:
        b_w2_pre = gl.amd.cdna4.buffer_load(ptr=w_base, offsets=w_offs_base2)
    b_v_pre = gl.amd.cdna4.buffer_load(ptr=v_base, offsets=v_offs_base)

    # ── Main loop ──
    for i_t in range(NT):
        i_t_bt = gl.cast(i_t * BT, gl.int32)

        # ──── 1. Store h[i_t] ────
        h_off_iter = gl.cast(i_t * stride_h, gl.int32)
        gl.amd.cdna4.buffer_store(
            stored_value=b_h1.to(gl.bfloat16), ptr=h_base, offsets=h_offs_base1 + h_off_iter)
        if K > 64:
            gl.amd.cdna4.buffer_store(
                stored_value=b_h2.to(gl.bfloat16), ptr=h_base, offsets=h_offs_base2 + h_off_iter)

        # ──── 2. w@h: w→smem(dot_op0), b_h→convert_layout(dot_op1) ────
        smem_a.index(0).store(b_w1_pre)
        if K > 64:
            smem_a.index(1).store(b_w2_pre)
        w_dot = smem_a.index(0).load(dot_op0)
        h_dot = gl.convert_layout(b_h1.to(gl.bfloat16), dot_op1)
        b_wh = gl.zeros((BT, BV), dtype=gl.float32, layout=mma)
        b_wh = gl.amd.cdna4.mfma(w_dot, h_dot, b_wh)
        if K > 64:
            w_dot2 = smem_a.index(1).load(dot_op0)
            h_dot2 = gl.convert_layout(b_h2.to(gl.bfloat16), dot_op1)
            b_wh = gl.amd.cdna4.mfma(w_dot2, h_dot2, b_wh)

        # ──── Prefetch k^T[i_t] (overlap with v_sub + v_new + gating) ────
        b_kt1_pre = gl.amd.cdna4.buffer_load(ptr=k_base, offsets=kt_offs_base1 + i_t_bt * stride_k)
        if K > 64:
            b_kt2_pre = gl.amd.cdna4.buffer_load(ptr=k_base, offsets=kt_offs_base2 + i_t_bt * stride_k)

        # ──── 3. b_v = v - w@h (convert v from blocked_v to mma layout) ────
        b_v = gl.convert_layout(b_v_pre, mma).to(gl.float32) - b_wh

        # ──── Prefetch w[i_t+1] + v[i_t+1] ────
        if i_t + 1 < NT:
            next_bt = i_t_bt + gl.cast(BT, gl.int32)
            b_w1_pre = gl.amd.cdna4.buffer_load(ptr=w_base, offsets=w_offs_base1 + next_bt * stride_w)
            if K > 64:
                b_w2_pre = gl.amd.cdna4.buffer_load(ptr=w_base, offsets=w_offs_base2 + next_bt * stride_w)
            b_v_pre = gl.amd.cdna4.buffer_load(ptr=v_base, offsets=v_offs_base + next_bt * stride_v)

        # ──── 4. Store v_new ────
        if SAVE_NEW_VALUE:
            vn_offs = gl.cast((i_t * BT + m_row_bt[:, None]) * stride_v + (i_v * BV + m_col_bv[None, :]), gl.int32)
            gl.amd.cdna4.buffer_store(
                stored_value=b_v.to(gl.bfloat16), ptr=vn_base, offsets=vn_offs)

        # ──── 5. Gating ────
        if USE_G:
            last_idx = (i_t + 1) * BT - 1
            m_t = (i_t * BT + m_row_bt) < T
            b_g_last = gl.load(g_base + last_idx * H).to(gl.float32)
            b_g = gl.amd.cdna4.buffer_load(
                ptr=g_base, offsets=g_offs_base + i_t_bt * H, mask=m_t, other=0.0).to(gl.float32)
            if USE_EXP2:
                b_scale = gl.where(m_t, gl.exp2(b_g_last - b_g), 0.0)
                b_g_last_val = gl.exp2(b_g_last)
            else:
                b_scale = gl.where(m_t, gl.exp(b_g_last - b_g), 0.0)
                b_g_last_val = gl.exp(b_g_last)
            b_v = b_v * b_scale[:, None]
            b_h1 = b_h1 * b_g_last_val
            if K > 64:
                b_h2 = b_h2 * b_g_last_val

        # ──── 6. k^T @ b_v: b_v→convert_layout(dot_op1), k→smem(dot_op0) ────
        smem_a.index(0).store(b_kt1_pre)
        if K > 64:
            smem_a.index(1).store(b_kt2_pre)
        v_dot = gl.convert_layout(b_v.to(gl.bfloat16), dot_op1)
        k_dot = smem_a.index(0).load(dot_op0)
        b_h1 = gl.amd.cdna4.mfma(k_dot, v_dot, b_h1)
        if K > 64:
            k_dot2 = smem_a.index(1).load(dot_op0)
            b_h2 = gl.amd.cdna4.mfma(k_dot2, v_dot, b_h2)

    # ── Store final state ──
    if STORE_FINAL_STATE:
        ht_base = ht + i_nh * K * V
        ht_offs1 = gl.cast(m_row[:, None] * V + (i_v * BV + m_col_bv[None, :]), gl.int32)
        gl.amd.cdna4.buffer_store(
            stored_value=b_h1, ptr=ht_base, offsets=ht_offs1)
        if K > 64:
            ht_offs2 = gl.cast((64 + m_row[:, None]) * V + (i_v * BV + m_col_bv[None, :]), gl.int32)
            gl.amd.cdna4.buffer_store(
                stored_value=b_h2, ptr=ht_base, offsets=ht_offs2)


def gluon_fwd_h(k, w, u, g, use_exp2=False, initial_state=None, output_final_state=False,
                cu_seqlens=None, chunk_offsets=None, BV=32, NUM_WARPS=4):
    B, T, Hg, K = k.shape
    V = u.shape[-1]
    H = u.shape[-2]
    BT = 64
    NT = triton.cdiv(T, BT)
    N = B
    if cu_seqlens is not None:
        N = len(cu_seqlens) - 1
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
    h = k.new_empty(B, NT, H, K, V)
    v_new = torch.empty_like(u)
    ht = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    gluon_chunk_fwd_kernel_h[grid](
        k=k, v=u, w=w, v_new=v_new, g=g,
        h=h, h0=initial_state, ht=ht,
        cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets,
        T=T, H=H, Hg=Hg, K=K, V=V, BT=BT, BV=BV,
        USE_EXP2=use_exp2,
        NUM_WARPS=NUM_WARPS,
        num_warps=NUM_WARPS, num_stages=1,
    )
    return h, v_new, ht


def gluon_pipeline_h(q, k, v, g, beta, scale, output_final_state=False,
                     cu_seqlens=None, chunk_indices=None, chunk_offsets=None):
    """Hybrid: Gluon fwd_h + Triton tuned fwd_o."""
    g_cum = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    A = kkt_fwd(k, beta, g_cum)
    A = solve_tril(A, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k, v, beta, A, g_cum, use_exp2=False)
    h, v_new, ht = gluon_fwd_h(k, w, u, g_cum, use_exp2=False,
                                output_final_state=output_final_state,
                                cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets)
    o = fwd_o_tuned(q, k, v_new, h, g_cum, scale, use_exp2=False)
    return o, ht


def best_pipeline(q, k, v, g, beta, scale, output_final_state=False,
                  cu_seqlens=None, chunk_indices=None, chunk_offsets=None):
    """Best-of-both: fused_kkt_solve + exp2 + Gluon fwd_h + Triton tuned fwd_o."""
    g_cum = chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    A = fused_kkt_solve(k, g_cum, beta, use_exp2=True)
    w, u = recompute_w_u_fwd(k, v, beta, A, g_cum, use_exp2=True)
    h, v_new, ht = gluon_fwd_h(k, w, u, g_cum, use_exp2=True,
                                output_final_state=output_final_state,
                                cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets,
                                BV=16)
    o = fwd_o_tuned(q, k, v_new, h, g_cum, scale, use_exp2=True)
    return o, ht


# ═══════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════

def bench_fn(fn, warmup=5, repeat=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    return sum(times) / len(times), min(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=65536)
    parser.add_argument("--Hg", type=int, default=8)
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--DK", type=int, default=128)
    parser.add_argument("--DV", type=int, default=128)
    parser.add_argument("--no-prod", action="store_true",
                        help="disable production mode (no varlen, no final_state)")
    args = parser.parse_args()
    use_prod = not args.no_prod

    B, T, Hg, H, DK, DV = 1, args.T, args.Hg, args.H, args.DK, args.DV
    SCALE = DK**-0.5

    print(f"Shape: B={B} T={T} Hg={Hg} H={H} DK={DK} DV={DV}  chunks={T // 64}")
    if use_prod:
        print(f"  production mode: varlen=True, output_final_state=True, initial_state=None")
    else:
        print(f"  raw kernel mode: no varlen, no state")
    print()

    torch.manual_seed(42)
    q = torch.randn(B, T, Hg, DK, device="cuda", dtype=torch.bfloat16)
    k = F.normalize(
        torch.randn(B, T, Hg, DK, device="cuda", dtype=torch.bfloat16), p=2, dim=-1
    )
    v = torch.randn(B, T, H, DV, device="cuda", dtype=torch.bfloat16)
    g = F.logsigmoid(torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16))
    beta = torch.rand(B, T, H, device="cuda", dtype=torch.bfloat16).sigmoid()

    cu_seqlens = None
    output_final_state = False
    chunk_indices = None
    chunk_offsets = None
    if use_prod:
        cu_seqlens = torch.tensor([0, T], device="cuda", dtype=torch.long)
        output_final_state = True
        chunk_indices = prepare_chunk_indices(cu_seqlens, 64)
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, 64)

    pipe_kw = dict(output_final_state=output_final_state,
                   cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets)

    # ── Precision: v1 (buffer_load) ──
    print("=== Precision: Gluon v1 (buffer_load) vs Triton baseline ===")
    torch.cuda.synchronize()
    o_old, ht_old = old_pipeline(q, k, v, g, beta, SCALE, **pipe_kw)
    torch.cuda.synchronize()
    o_gluon, ht_gluon = gluon_pipeline(q, k, v, g, beta, SCALE, **pipe_kw)
    torch.cuda.synchronize()

    diff = (o_old.float() - o_gluon.float()).abs()
    rel = diff / (o_old.float().abs() + 1e-8)
    print(f"  max_abs_diff:  {diff.max().item():.6e}")
    print(f"  mean_abs_diff: {diff.mean().item():.6e}")
    print(f"  mean_rel_err:  {rel.mean().item():.6e}")
    print(f"  → {'PASS' if rel.mean().item() < 1e-2 else 'FAIL'}")
    if output_final_state and ht_old is not None and ht_gluon is not None:
        ht_rel = (ht_old.float() - ht_gluon.float()).abs() / (ht_old.float().abs() + 1e-8)
        print(f"  final_state mean_rel_err: {ht_rel.mean().item():.6e} → {'PASS' if ht_rel.mean().item() < 1e-2 else 'FAIL'}")
    print()

    # ── Precision: v2 (async_copy) ──
    print("=== Precision: Gluon v2 (async_copy) vs Triton baseline ===")
    torch.cuda.synchronize()
    o_v2, ht_v2 = gluon_pipeline_v2(q, k, v, g, beta, SCALE, **pipe_kw)
    torch.cuda.synchronize()

    diff2 = (o_old.float() - o_v2.float()).abs()
    rel2 = diff2 / (o_old.float().abs() + 1e-8)
    print(f"  max_abs_diff:  {diff2.max().item():.6e}")
    print(f"  mean_abs_diff: {diff2.mean().item():.6e}")
    print(f"  mean_rel_err:  {rel2.mean().item():.6e}")
    print(f"  → {'PASS' if rel2.mean().item() < 1e-2 else 'FAIL'}")
    if output_final_state and ht_old is not None and ht_v2 is not None:
        ht_rel2 = (ht_old.float() - ht_v2.float()).abs() / (ht_old.float().abs() + 1e-8)
        print(f"  final_state mean_rel_err: {ht_rel2.mean().item():.6e} → {'PASS' if ht_rel2.mean().item() < 1e-2 else 'FAIL'}")
    print()

    # ── End-to-End ──
    print("=== End-to-End: Triton vs Gluon v1 vs Gluon v2 ===")
    avg_old, min_old = bench_fn(lambda: old_pipeline(q, k, v, g, beta, SCALE, **pipe_kw))
    avg_gluon, min_gluon = bench_fn(lambda: gluon_pipeline(q, k, v, g, beta, SCALE, **pipe_kw))
    avg_v2, min_v2 = bench_fn(lambda: gluon_pipeline_v2(q, k, v, g, beta, SCALE, **pipe_kw))
    print(f"  Triton baseline:     avg={avg_old:8.0f} us  min={min_old:8.0f} us")
    print(f"  Gluon v1 (buf_load): avg={avg_gluon:8.0f} us  min={min_gluon:8.0f} us  ({avg_old/avg_gluon:.3f}x)")
    print(f"  Gluon v2 (async_cp): avg={avg_v2:8.0f} us  min={min_v2:8.0f} us  ({avg_old/avg_v2:.3f}x)")
    print()

    # ── fwd_o isolated comparison ──
    print("=== fwd_o isolated: Triton orig vs tuned vs Gluon v1 vs v2 ===")
    g_cum = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    A = kkt_fwd(k, beta, g_cum)
    A = solve_tril(A, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k, v, beta, A, g_cum, use_exp2=False)
    h, v_new, _ = fwd_h_orig(k, w, u, g_cum, use_exp2=False,
                              output_final_state=output_final_state,
                              cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets)
    torch.cuda.synchronize()

    avg_orig, _ = bench_fn(lambda: fwd_o_orig(q, k, v_new, h, g_cum, SCALE, use_exp2=False))
    avg_tuned, _ = bench_fn(lambda: fwd_o_tuned(q, k, v_new, h, g_cum, SCALE, use_exp2=False))
    avg_gl, _ = bench_fn(lambda: gluon_fwd_o(q, k, v_new, h, g_cum, SCALE))
    avg_gl2, _ = bench_fn(lambda: gluon_fwd_o_v2(q, k, v_new, h, g_cum, SCALE))
    try:
        o_v3 = gluon_fwd_o_v3(q, k, v_new, h, g_cum, SCALE)
        torch.cuda.synchronize()
        v3_diff = (o_v3.float() - (fwd_o_tuned(q, k, v_new, h, g_cum, SCALE, use_exp2=False)).float()).abs()
        v3_rel = v3_diff / (o_v3.float().abs() + 1e-8)
        print(f"  v3 precision: mean_rel_err={v3_rel.mean().item():.6e} → {'PASS' if v3_rel.mean().item() < 1e-2 else 'FAIL'}")
        avg_gl3, _ = bench_fn(lambda: gluon_fwd_o_v3(q, k, v_new, h, g_cum, SCALE))
    except Exception as e:
        avg_gl3 = float('inf')
        print(f"  v3 ERROR: {str(e)[:200]}")
    print(f"  Triton orig:   avg={avg_orig:8.0f} us  (BK=128 BV=64 warps=4)")
    print(f"  Triton tuned:  avg={avg_tuned:8.0f} us  (BK=64  BV=128 warps=1)")
    print(f"  Gluon v1:      avg={avg_gl:8.0f} us  (buffer_load)")
    print(f"  Gluon v2:      avg={avg_gl2:8.0f} us  (async_copy)")
    print(f"  Gluon v3:      avg={avg_gl3:8.0f} us  (prefetch+layout)")
    print(f"  v3 vs tuned: {avg_tuned/avg_gl3:.3f}x")
    print()

    # ── fwd_h: Gluon vs Triton ──
    print("=== fwd_h: Gluon vs Triton ===")
    g_cum2 = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    A2 = kkt_fwd(k, beta, g_cum2)
    A2 = solve_tril(A2, output_dtype=k.dtype)
    w2, u2 = recompute_w_u_fwd(k, v, beta, A2, g_cum2, use_exp2=False)

    h_tri, vn_tri, ht_tri = fwd_h_orig(k, w2, u2, g_cum2, use_exp2=False,
                                         output_final_state=output_final_state,
                                         cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets)
    torch.cuda.synchronize()
    try:
        h_gl, vn_gl, ht_gl = gluon_fwd_h(k, w2, u2, g_cum2, use_exp2=False,
                                           output_final_state=output_final_state,
                                           cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets)
        torch.cuda.synchronize()

        h_diff = (h_tri.float() - h_gl.float()).abs()
        h_rel = h_diff / (h_tri.float().abs() + 1e-8)
        print(f"  h  mean_rel_err: {h_rel.mean().item():.6e} → {'PASS' if h_rel.mean().item() < 1e-2 else 'FAIL'}")

        vn_diff = (vn_tri.float() - vn_gl.float()).abs()
        vn_rel = vn_diff / (vn_tri.float().abs() + 1e-8)
        print(f"  vn mean_rel_err: {vn_rel.mean().item():.6e} → {'PASS' if vn_rel.mean().item() < 1e-2 else 'FAIL'}")

        if output_final_state and ht_tri is not None and ht_gl is not None:
            ht_rel = (ht_tri.float() - ht_gl.float()).abs() / (ht_tri.float().abs() + 1e-8)
            print(f"  ht mean_rel_err: {ht_rel.mean().item():.6e} → {'PASS' if ht_rel.mean().item() < 1e-2 else 'FAIL'}")

        avg_h_orig, _ = bench_fn(lambda: fwd_h_orig(k, w2, u2, g_cum2, use_exp2=False,
                                                      output_final_state=output_final_state,
                                                      cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets))
        avg_h_tuned, _ = bench_fn(lambda: fwd_h_tuned(k, w2, u2, g_cum2, use_exp2=False,
                                                       output_final_state=output_final_state,
                                                       cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets))
        # Triton BV=16 no-pipeline (num_stages=1) for comparison
        from bench_standalone import chunk_gated_delta_rule_fwd_kernel_h as triton_fwd_h_kernel
        def triton_fwd_h_nopipe():
            B2, T2, Hg2, K2 = k.shape
            V2, H2 = u2.shape[-1], u2.shape[-2]
            NT2 = triton.cdiv(T2, 64)
            N2 = B2
            if cu_seqlens is not None:
                N2 = len(cu_seqlens) - 1
            h_tmp = k.new_empty(B2, NT2, H2, K2, V2)
            vn_tmp = torch.empty_like(u2)
            ht_tmp = k.new_empty(N2, H2, K2, V2, dtype=torch.float32) if output_final_state else None
            triton_fwd_h_kernel[(triton.cdiv(V2, 16), N2 * H2)](
                k=k, v=u2, w=w2, v_new=vn_tmp, g=g_cum2, gk=None,
                h=h_tmp, h0=None, ht=ht_tmp,
                cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets,
                T=T2, H=H2, Hg=Hg2, K=K2, V=V2, BT=64, BV=16,
                USE_EXP2=False, num_warps=4, num_stages=1)
            return h_tmp, vn_tmp, ht_tmp
        avg_h_nopipe, _ = bench_fn(triton_fwd_h_nopipe)
        print(f"  Triton orig  (BV=32 stg=2): avg={avg_h_orig:8.0f} us")
        print(f"  Triton tuned (BV=16 stg=2): avg={avg_h_tuned:8.0f} us")
        print(f"  Triton nopip (BV=16 stg=1): avg={avg_h_nopipe:8.0f} us")

        configs = [(16, 4), (16, 2), (32, 4), (32, 2)]
        for bv, nw in configs:
            try:
                avg_h_gl, _ = bench_fn(lambda bv=bv, nw=nw: gluon_fwd_h(k, w2, u2, g_cum2, use_exp2=False,
                                                         output_final_state=output_final_state,
                                                         cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets,
                                                         BV=bv, NUM_WARPS=nw))
                print(f"  Gluon BV={bv:2d} W={nw}: avg={avg_h_gl:8.0f} us  (vs orig {avg_h_orig/avg_h_gl:.3f}x, vs tuned {avg_h_tuned/avg_h_gl:.3f}x)")
            except Exception as e:
                print(f"  Gluon BV={bv:2d} W={nw}: ERROR: {str(e)[:120]}")

        avg_e2e_gl_h, _ = bench_fn(lambda: gluon_pipeline_h(q, k, v, g, beta, SCALE, **pipe_kw))
        print(f"  End-to-end Gluon fwd_h pipeline: avg={avg_e2e_gl_h:8.0f} us ({avg_old/avg_e2e_gl_h:.3f}x vs Triton)")
    except Exception as e:
        print(f"  Gluon fwd_h COMPILE ERROR: {e}")
    print()

    # ── Pipeline comparison: all combos ──
    print("=== Pipeline comparison ===")
    avg_orig_p, _ = bench_fn(lambda: triton_old_pipeline(q, k, v, g, beta, SCALE, **pipe_kw))
    try:
        avg_new_p, _ = bench_fn(lambda: new_pipeline(q, k, v, g, beta, SCALE, **pipe_kw))
    except Exception as e:
        avg_new_p = float('inf')
        print(f"  new_pipeline error: {str(e)[:100]}")
    try:
        avg_best, _ = bench_fn(lambda: best_pipeline(q, k, v, g, beta, SCALE, **pipe_kw))
    except Exception as e:
        avg_best = float('inf')
        print(f"  best_pipeline error: {str(e)[:100]}")
    print(f"  Triton orig  (6-kernel, no exp2):     avg={avg_orig_p:8.0f} us")
    print(f"  Triton new   (fused+exp2+tuned):      avg={avg_new_p:8.0f} us")
    print(f"  Best-of-both (fused+exp2+Gluon fwd_h): avg={avg_best:8.0f} us")
    if avg_new_p < float('inf'):
        print(f"  Best vs new:  {avg_new_p/avg_best:.3f}x")
    print(f"  Best vs orig: {avg_orig_p/avg_best:.3f}x")
    print()

    print("Done.")


if __name__ == "__main__":
    main()
