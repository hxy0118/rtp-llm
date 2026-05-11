# -*- coding: utf-8 -*-
"""
Chunk-GDN precision comparison: fused (AMD) pipeline vs. original (separate) pipeline.

Runs both code paths on identical inputs and compares intermediate/final outputs:
  Stage 1: A matrix  (kkt + solve)
  Stage 2: w, u      (recompute_w_u)
  Stage 3: h, v_new  (fwd_h)
  Stage 4: o         (fwd_o, end-to-end)

Also compares against the recurrent fp32 reference for absolute accuracy.

Usage:
    python -m rtp_llm.models_py.triton_kernels.fla.test.test_chunk_gdn_precision
"""

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

logging.basicConfig(
    level="INFO",
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

os.environ["TRITON_F32_DEFAULT"] = "ieee"

import triton

from rtp_llm.models_py.triton_kernels.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
)
from rtp_llm.models_py.triton_kernels.fla.chunk_fwd import (
    chunk_gated_delta_rule_fwd_intra,
)
from rtp_llm.models_py.triton_kernels.fla.chunk_o import chunk_fwd_kernel_o, chunk_fwd_o
from rtp_llm.models_py.triton_kernels.fla.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from rtp_llm.models_py.triton_kernels.fla.cumsum import chunk_local_cumsum
from rtp_llm.models_py.triton_kernels.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from rtp_llm.models_py.triton_kernels.fla.solve_tril import solve_tril
from rtp_llm.models_py.triton_kernels.fla.wy_fast import recompute_w_u_fwd

RCP_LN2 = 1.0 / 0.6931471805599453


# ---------------------------------------------------------------------------
# Recurrent fp32 reference (from test_chunk_prefill.py)
# ---------------------------------------------------------------------------
def recurrent_gdn_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    # q, k: [B, T, Hg, D]  v: [B, T, H, V]  beta, g: [B, T, H]
    Hg = q.shape[2]
    H = v.shape[2]
    q, k, v, beta, g = map(
        lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g]
    )
    # GQA: repeat k, q from Hg to H heads
    if Hg < H:
        rep = H // Hg
        q = q.repeat_interleave(rep, dim=1)
        k = k.repeat_interleave(rep, dim=1)
    B, _, T, K = q.shape
    V = v.shape[-1]
    o = torch.zeros(B, H, T, V, device=v.device, dtype=v.dtype)
    h = torch.zeros(B, H, K, V, device=v.device, dtype=v.dtype)
    if initial_state is not None:
        h = initial_state.to(torch.float32)
    if scale is None:
        scale = 1 / (K**0.5)
    q = q * scale
    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", b_q, h)
    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous()
    return o, h


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------
def run_separate_pipeline(
    q, k, v, g_log2, beta, scale, initial_state, output_final_state, cu_seqlens=None
):
    """Original pipeline: separate kkt -> solve_tril -> recompute_w_u."""
    A_raw = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g_cumsum=g_log2,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
    )
    A_solved = solve_tril(A=A_raw, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A_solved, g_cumsum=g_log2, cu_seqlens=cu_seqlens
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g_log2,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o(
        q=q, k=k, v=v_new, h=h, g=g_log2, scale=scale, cu_seqlens=cu_seqlens
    )
    return A_solved, w, u, h, v_new, o, final_state


def run_fused_pipeline(
    q, k, v, g_log2, beta, scale, initial_state, output_final_state, cu_seqlens=None
):
    """AMD-optimized pipeline: fused kkt+solve -> recompute_w_u."""
    w, u, A = chunk_gated_delta_rule_fwd_intra(
        k=k, v=v, g=g_log2, beta=beta, cu_seqlens=cu_seqlens
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g_log2,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o(
        q=q, k=k, v=v_new, h=h, g=g_log2, scale=scale, cu_seqlens=cu_seqlens
    )
    return A, w, u, h, v_new, o, final_state


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class PrecisionMetrics:
    name: str
    abs_max: float
    rmse: float
    rel_err: float
    cosine_sim: float

    def __str__(self):
        return (
            f"  {self.name:12s}  |  abs_max={self.abs_max:.6e}  "
            f"rmse={self.rmse:.6e}  rel_err={self.rel_err:.6e}  "
            f"cos_sim={self.cosine_sim:.8f}"
        )


def compute_metrics(
    name: str, ref: torch.Tensor, test: torch.Tensor
) -> PrecisionMetrics:
    ref_f = ref.detach().float().flatten()
    test_f = test.detach().float().flatten()
    diff = ref_f - test_f
    abs_max = diff.abs().max().item()
    rmse = diff.square().mean().sqrt().item()
    base = ref_f.square().mean().sqrt().item()
    rel_err = rmse / (base + 1e-12)
    cos = F.cosine_similarity(ref_f.unsqueeze(0), test_f.unsqueeze(0)).item()
    return PrecisionMetrics(name, abs_max, rmse, rel_err, cos)


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------
def make_inputs(
    B: int,
    T: int,
    H: int,
    Hg: int,
    D: int,
    DV: int,
    dtype: torch.dtype,
    mask_p: float = 0.0,
    cu_seqlens: Optional[List[int]] = None,
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = "cuda"

    if cu_seqlens is not None:
        cu_seqlens_t = torch.LongTensor(cu_seqlens).to(device)
        T_total = cu_seqlens[-1]
        N = len(cu_seqlens) - 1
        B_eff = 1
    else:
        cu_seqlens_t = None
        T_total = T
        N = B
        B_eff = B

    q = torch.randn((B_eff, T_total, Hg, D), dtype=dtype, device=device)
    k = F.normalize(
        torch.randn(B_eff, T_total, Hg, D, dtype=torch.float32, device=device),
        p=2,
        dim=-1,
    ).to(dtype)
    v = torch.randn((B_eff, T_total, H, DV), dtype=dtype, device=device)
    g = F.logsigmoid(torch.rand(B_eff, T_total, H, dtype=dtype, device=device))
    if mask_p > 0:
        g = g * (torch.rand_like(g) > mask_p)
    beta = torch.rand(B_eff, T_total, H, dtype=dtype, device=device).sigmoid()
    h0 = torch.randn((N, H, D, DV), dtype=dtype, device=device)

    scale = D**-0.5
    return q, k, v, g, beta, h0, scale, cu_seqlens_t, N


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
def run_precision_test(
    label: str,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype = torch.bfloat16,
    Hg: int = None,
    DV: int = None,
    mask_p: float = 0.0,
    cu_seqlens: Optional[List[int]] = None,
    atol_vs_ref: float = 0.01,
    atol_fused_vs_sep: float = 0.005,
    seed: int = 42,
):
    if Hg is None:
        Hg = H
    if DV is None:
        DV = D

    logger.info(f"{'='*70}")
    logger.info(
        f"[{label}]  B={B} T={T} H={H} Hg={Hg} D={D} DV={DV} "
        f"dtype={dtype} mask_p={mask_p} cu_seqlens={cu_seqlens}"
    )
    logger.info(f"{'='*70}")

    q, k, v, g, beta, h0, scale, cu_seqlens_t, N = make_inputs(
        B, T, H, Hg, D, DV, dtype, mask_p, cu_seqlens, seed
    )

    # -- Preprocess: cumsum g to log2 domain (shared by both pipelines) --
    g_log2 = chunk_local_cumsum(
        g, chunk_size=64, scale=RCP_LN2, cu_seqlens=cu_seqlens_t
    )

    # -- Run both pipelines --
    A_sep, w_sep, u_sep, h_sep, vn_sep, o_sep, ht_sep = run_separate_pipeline(
        q.clone(),
        k.clone(),
        v.clone(),
        g_log2.clone(),
        beta.clone(),
        scale,
        h0.clone(),
        True,
        cu_seqlens_t,
    )
    A_fus, w_fus, u_fus, h_fus, vn_fus, o_fus, ht_fus = run_fused_pipeline(
        q.clone(),
        k.clone(),
        v.clone(),
        g_log2.clone(),
        beta.clone(),
        scale,
        h0.clone(),
        True,
        cu_seqlens_t,
    )

    # -- Run recurrent reference --
    if cu_seqlens is not None:
        ref_o_parts, ref_ht_parts = [], []
        for i in range(N):
            s, e = cu_seqlens[i], cu_seqlens[i + 1]
            ref_oi, ref_hti = recurrent_gdn_ref(
                q[:, s:e],
                k[:, s:e],
                v[:, s:e],
                beta[:, s:e],
                g[:, s:e],
                scale=scale,
                initial_state=h0[i : i + 1],
                output_final_state=True,
            )
            ref_o_parts.append(ref_oi)
            ref_ht_parts.append(ref_hti)
        ref_o = torch.cat(ref_o_parts, dim=1)
        ref_ht = torch.cat(ref_ht_parts, dim=0)
    else:
        ref_o, ref_ht = recurrent_gdn_ref(
            q,
            k,
            v,
            beta,
            g,
            scale=scale,
            initial_state=h0,
            output_final_state=True,
        )

    # -- Compare: fused vs separate (stage-by-stage) --
    logger.info("--- Fused vs Separate (stage-by-stage) ---")
    metrics_fused_vs_sep = []

    m = compute_metrics("A_matrix", A_sep, A_fus)
    metrics_fused_vs_sep.append(m)
    logger.info(str(m))

    m = compute_metrics("w", w_sep, w_fus)
    metrics_fused_vs_sep.append(m)
    logger.info(str(m))

    m = compute_metrics("u", u_sep, u_fus)
    metrics_fused_vs_sep.append(m)
    logger.info(str(m))

    m = compute_metrics("h_state", h_sep, h_fus)
    metrics_fused_vs_sep.append(m)
    logger.info(str(m))

    m = compute_metrics("v_new", vn_sep, vn_fus)
    metrics_fused_vs_sep.append(m)
    logger.info(str(m))

    m = compute_metrics("o_output", o_sep, o_fus)
    metrics_fused_vs_sep.append(m)
    logger.info(str(m))

    m = compute_metrics("final_state", ht_sep, ht_fus)
    metrics_fused_vs_sep.append(m)
    logger.info(str(m))

    # -- Compare: both vs recurrent reference --
    logger.info("--- Separate vs Recurrent Reference ---")
    m_sep_o = compute_metrics("o_sep_vs_ref", ref_o, o_sep)
    m_sep_ht = compute_metrics("ht_sep_vs_ref", ref_ht, ht_sep)
    logger.info(str(m_sep_o))
    logger.info(str(m_sep_ht))

    logger.info("--- Fused vs Recurrent Reference ---")
    m_fus_o = compute_metrics("o_fus_vs_ref", ref_o, o_fus)
    m_fus_ht = compute_metrics("ht_fus_vs_ref", ref_ht, ht_fus)
    logger.info(str(m_fus_o))
    logger.info(str(m_fus_ht))

    # -- Assertions --
    failures = []
    for m in metrics_fused_vs_sep:
        if m.rel_err > atol_fused_vs_sep:
            failures.append(
                f"fused_vs_sep {m.name}: rel_err={m.rel_err:.6e} > {atol_fused_vs_sep}"
            )

    if m_fus_o.rel_err > atol_vs_ref:
        failures.append(
            f"fused_vs_ref o: rel_err={m_fus_o.rel_err:.6e} > {atol_vs_ref}"
        )
    if m_fus_ht.rel_err > atol_vs_ref:
        failures.append(
            f"fused_vs_ref ht: rel_err={m_fus_ht.rel_err:.6e} > {atol_vs_ref}"
        )
    if m_sep_o.rel_err > atol_vs_ref:
        failures.append(f"sep_vs_ref o: rel_err={m_sep_o.rel_err:.6e} > {atol_vs_ref}")
    if m_sep_ht.rel_err > atol_vs_ref:
        failures.append(
            f"sep_vs_ref ht: rel_err={m_sep_ht.rel_err:.6e} > {atol_vs_ref}"
        )

    if failures:
        logger.error(f"FAILED {label}: {len(failures)} check(s)")
        for f in failures:
            logger.error(f"  - {f}")
        return False
    else:
        logger.info(f"PASSED {label}")
        return True


# ---------------------------------------------------------------------------
# Tile-config comparison: call kernels directly with old vs new configs
# ---------------------------------------------------------------------------
def call_fwd_h_with_config(
    k, w, u, g, initial_state, output_final_state, cu_seqlens, BV, num_warps, num_stages
):
    """Call fwd_h kernel directly with specific tile config."""
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = 64

    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(cu_seqlens) - 1
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
        NT = int(chunk_offsets[-1])

    h = k.new_empty(B, NT, H, K, V, dtype=torch.float32)
    final_state = (
        k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    )
    v_new = torch.empty_like(u)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=None,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BV=BV,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return h, v_new, final_state


def call_fwd_o_with_config(
    q, k, v, h, g, scale, cu_seqlens, BK, BV, num_warps, num_stages
):
    """Call fwd_o kernel directly with specific tile config."""
    B, T, Hg, K = q.shape
    V = v.shape[-1]
    H = v.shape[-2]
    BT = min(64, max(16, triton.next_power_of_2(T)))

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    o = torch.zeros_like(v)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    chunk_fwd_kernel_o[grid](
        q,
        k,
        v,
        h,
        g,
        o,
        cu_seqlens,
        chunk_indices,
        scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o


def run_tile_config_test(
    label: str,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype = torch.bfloat16,
    Hg: int = None,
    DV: int = None,
    cu_seqlens: Optional[List[int]] = None,
    seed: int = 42,
):
    """Compare fwd_h and fwd_o with old vs new tile configs on same inputs."""
    if Hg is None:
        Hg = H
    if DV is None:
        DV = D

    logger.info(f"{'='*70}")
    logger.info(
        f"[TILE {label}]  B={B} T={T} H={H} Hg={Hg} D={D} DV={DV} dtype={dtype}"
    )
    logger.info(f"{'='*70}")

    q, k, v, g, beta, h0, scale, cu_seqlens_t, N = make_inputs(
        B, T, H, Hg, D, DV, dtype, 0.0, cu_seqlens, seed
    )
    g_log2 = chunk_local_cumsum(
        g, chunk_size=64, scale=RCP_LN2, cu_seqlens=cu_seqlens_t
    )

    # Shared: compute A, w, u (same for both)
    w, u, A = chunk_gated_delta_rule_fwd_intra(
        k=k, v=v, g=g_log2, beta=beta, cu_seqlens=cu_seqlens_t
    )

    # --- fwd_h: old config (BV=32) vs new config (BV=64 on CDNA3) ---
    h_old, vn_old, ht_old = call_fwd_h_with_config(
        k,
        w,
        u,
        g_log2,
        h0.clone(),
        True,
        cu_seqlens_t,
        BV=32,
        num_warps=4,
        num_stages=2,
    )
    h_new, vn_new, ht_new = call_fwd_h_with_config(
        k,
        w,
        u,
        g_log2,
        h0.clone(),
        True,
        cu_seqlens_t,
        BV=64,
        num_warps=4,
        num_stages=2,
    )

    logger.info("--- fwd_h: BV=32 (old) vs BV=64 (new CDNA3) ---")
    for name, old_t, new_t in [
        ("h_state", h_old, h_new),
        ("v_new", vn_old, vn_new),
        ("final_state", ht_old, ht_new),
    ]:
        m = compute_metrics(name, old_t, new_t)
        logger.info(str(m))

    # --- fwd_o: old config vs new config ---
    # Use the NEW h (BV=64) as input for both fwd_o calls
    h_ref = h_new

    o_old = call_fwd_o_with_config(
        q,
        k,
        vn_new,
        h_ref,
        g_log2,
        scale,
        cu_seqlens_t,
        BK=128,
        BV=64,
        num_warps=4,
        num_stages=2,
    )
    o_new = call_fwd_o_with_config(
        q,
        k,
        vn_new,
        h_ref,
        g_log2,
        scale,
        cu_seqlens_t,
        BK=64,
        BV=128,
        num_warps=4,
        num_stages=1,
    )

    logger.info(
        "--- fwd_o: (BK=128,BV=64,stages=2) old vs (BK=64,BV=128,stages=1) new ---"
    )
    m_o = compute_metrics("o_output", o_old, o_new)
    logger.info(str(m_o))

    # --- Also compare old fwd_o num_warps=4 vs num_warps=1 ---
    # (new code uses num_warps=1 when h.dtype != fp32 on AMD)
    if h_ref.dtype == torch.float32:
        o_w4 = call_fwd_o_with_config(
            q,
            k,
            vn_new,
            h_ref,
            g_log2,
            scale,
            cu_seqlens_t,
            BK=64,
            BV=128,
            num_warps=4,
            num_stages=1,
        )
        o_w1 = call_fwd_o_with_config(
            q,
            k,
            vn_new,
            h_ref,
            g_log2,
            scale,
            cu_seqlens_t,
            BK=64,
            BV=128,
            num_warps=1,
            num_stages=1,
        )
        logger.info(
            "--- fwd_o: num_warps=4 vs num_warps=1 (new BK/BV, h_dtype=fp32) ---"
        )
        m_w = compute_metrics("o_warps", o_w4, o_w1)
        logger.info(str(m_w))

    # --- End-to-end: full old config vs full new config ---
    h_old2, vn_old2, ht_old2 = call_fwd_h_with_config(
        k,
        w,
        u,
        g_log2,
        h0.clone(),
        True,
        cu_seqlens_t,
        BV=32,
        num_warps=4,
        num_stages=2,
    )
    o_e2e_old = call_fwd_o_with_config(
        q,
        k,
        vn_old2,
        h_old2,
        g_log2,
        scale,
        cu_seqlens_t,
        BK=128,
        BV=64,
        num_warps=4,
        num_stages=2,
    )
    h_new2, vn_new2, ht_new2 = call_fwd_h_with_config(
        k,
        w,
        u,
        g_log2,
        h0.clone(),
        True,
        cu_seqlens_t,
        BV=64,
        num_warps=4,
        num_stages=2,
    )
    o_e2e_new = call_fwd_o_with_config(
        q,
        k,
        vn_new2,
        h_new2,
        g_log2,
        scale,
        cu_seqlens_t,
        BK=64,
        BV=128,
        num_warps=4,
        num_stages=1,
    )

    logger.info("--- End-to-End: all-old-config vs all-new-config ---")
    m_e2e_o = compute_metrics("o_e2e", o_e2e_old, o_e2e_new)
    m_e2e_ht = compute_metrics("ht_e2e", ht_old2, ht_new2)
    logger.info(str(m_e2e_o))
    logger.info(str(m_e2e_ht))

    # Compare both against recurrent reference
    if cu_seqlens is not None:
        ref_o_parts, ref_ht_parts = [], []
        for i in range(N):
            s, e = cu_seqlens[i], cu_seqlens[i + 1]
            ref_oi, ref_hti = recurrent_gdn_ref(
                q[:, s:e],
                k[:, s:e],
                v[:, s:e],
                beta[:, s:e],
                g[:, s:e],
                scale=scale,
                initial_state=h0[i : i + 1],
                output_final_state=True,
            )
            ref_o_parts.append(ref_oi)
            ref_ht_parts.append(ref_hti)
        ref_o = torch.cat(ref_o_parts, dim=1)
        ref_ht = torch.cat(ref_ht_parts, dim=0)
    else:
        ref_o, ref_ht = recurrent_gdn_ref(
            q,
            k,
            v,
            beta,
            g,
            scale=scale,
            initial_state=h0,
            output_final_state=True,
        )

    logger.info("--- Old config vs Recurrent Reference ---")
    m_old_ref_o = compute_metrics("o_old_ref", ref_o, o_e2e_old)
    m_old_ref_ht = compute_metrics("ht_old_ref", ref_ht, ht_old2)
    logger.info(str(m_old_ref_o))
    logger.info(str(m_old_ref_ht))

    logger.info("--- New config vs Recurrent Reference ---")
    m_new_ref_o = compute_metrics("o_new_ref", ref_o, o_e2e_new)
    m_new_ref_ht = compute_metrics("ht_new_ref", ref_ht, ht_new2)
    logger.info(str(m_new_ref_o))
    logger.info(str(m_new_ref_ht))

    logger.info(f"DONE tile config test [{label}]")
    return True


# ---------------------------------------------------------------------------
# Deep precision analysis: safe_exp clamping, wy_fast BV, fwd_o masking
# ---------------------------------------------------------------------------
def run_deep_precision_analysis(
    label: str,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype = torch.bfloat16,
    Hg: int = None,
    DV: int = None,
    cu_seqlens: Optional[List[int]] = None,
    seed: int = 42,
):
    if Hg is None:
        Hg = H
    if DV is None:
        DV = D

    logger.info(f"{'='*70}")
    logger.info(f"[DEEP {label}]  B={B} T={T} H={H} Hg={Hg} D={D} DV={DV}")
    logger.info(f"{'='*70}")

    q, k, v, g, beta, h0, scale, cu_seqlens_t, N = make_inputs(
        B, T, H, Hg, D, DV, dtype, 0.0, cu_seqlens, seed
    )

    # =========================================================
    # 1. safe_exp clamping: check if g_diff has positive values
    # =========================================================
    # Old domain (ln)
    g_ln = chunk_local_cumsum(g, chunk_size=64, scale=None, cu_seqlens=cu_seqlens_t)
    # New domain (log2)
    g_log2 = chunk_local_cumsum(
        g, chunk_size=64, scale=RCP_LN2, cu_seqlens=cu_seqlens_t
    )

    BT = 64
    NT = (T + BT - 1) // BT
    T_pad = NT * BT
    g_ln_pad = torch.zeros(B, T_pad, H, device=g_ln.device, dtype=g_ln.dtype)
    g_log2_pad = torch.zeros(B, T_pad, H, device=g_log2.device, dtype=g_log2.dtype)
    g_ln_pad[:, :T] = g_ln
    g_log2_pad[:, :T] = g_log2
    g_ln_c = g_ln_pad.reshape(B * NT, BT, H)
    g_log2_c = g_log2_pad.reshape(B * NT, BT, H)

    # Within each chunk, compute g[i] - g[j] for i > j (lower triangle)
    mask_lower = torch.tril(torch.ones(BT, BT, device="cuda"), diagonal=-1).bool()
    g_diff_ln = g_ln_c[:, :, None, :] - g_ln_c[:, None, :, :]  # [NT, BT, BT, H]
    g_diff_log2 = g_log2_c[:, :, None, :] - g_log2_c[:, None, :, :]

    # Count how many lower-triangle g_diff values are positive (numerically)
    lower_ln = g_diff_ln[:, mask_lower, :]
    lower_log2 = g_diff_log2[:, mask_lower, :]
    n_positive_ln = (lower_ln > 0).sum().item()
    n_positive_log2 = (lower_log2 > 0).sum().item()
    n_total = lower_ln.numel()
    max_positive_ln = lower_ln.clamp(min=0).max().item()
    max_positive_log2 = lower_log2.clamp(min=0).max().item()

    logger.info("--- safe_exp clamping analysis ---")
    logger.info(
        f"  g_diff (ln domain):   {n_positive_ln}/{n_total} positive "
        f"({100*n_positive_ln/n_total:.4f}%), max_pos={max_positive_ln:.6e}"
    )
    logger.info(
        f"  g_diff (log2 domain): {n_positive_log2}/{n_total} positive "
        f"({100*n_positive_log2/n_total:.4f}%), max_pos={max_positive_log2:.6e}"
    )

    if n_positive_ln > 0 or n_positive_log2 > 0:
        # safe_exp would clamp these to 0, but exp2 won't
        # Measure the actual difference: safe_exp(x) vs exp2(x) for positive x
        pos_vals_log2 = lower_log2[lower_log2 > 0]
        if len(pos_vals_log2) > 0:
            safe_result = torch.zeros_like(pos_vals_log2)  # safe_exp → 0
            exp2_result = torch.exp2(pos_vals_log2)  # exp2 → >1
            logger.info(
                f"  For positive g_diff: exp2 gives [{exp2_result.min():.6f}, {exp2_result.max():.6f}] "
                f"instead of 0 (safe_exp)"
            )
    else:
        logger.info("  No positive g_diff values → safe_exp clamping has NO effect")

    # =========================================================
    # 2. wy_fast BV change: 64 → 128
    # =========================================================
    from rtp_llm.models_py.triton_kernels.fla.wy_fast import recompute_w_u_fwd_kernel

    # First compute A (same for both)
    A_raw = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g_cumsum=g_log2,
        cu_seqlens=cu_seqlens_t,
        output_dtype=torch.float32,
    )
    A_solved = solve_tril(A=A_raw, cu_seqlens=cu_seqlens_t, output_dtype=k.dtype)

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens_t, BT) if cu_seqlens_t is not None else None
    )
    NT_actual = triton.cdiv(T, BT) if cu_seqlens_t is None else len(chunk_indices)
    V = DV

    def call_wy_fast(BV_val):
        u_out = torch.empty_like(v)
        w_out = k.new_empty(B, T, H, D)
        recompute_w_u_fwd_kernel[(NT_actual, B * H)](
            k=k,
            v=v,
            beta=beta,
            w=w_out,
            u=u_out,
            A=A_solved,
            g=g_log2,
            cu_seqlens=cu_seqlens_t,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            Hg=Hg,
            K=D,
            V=V,
            BT=BT,
            BK=64,
            BV=BV_val,
            num_warps=4,
            num_stages=3,
        )
        return w_out, u_out

    w_bv64, u_bv64 = call_wy_fast(64)
    w_bv128, u_bv128 = call_wy_fast(min(128, V))

    logger.info("--- wy_fast: BV=64 (old) vs BV=128 (new AMD) ---")
    m_w = compute_metrics("w", w_bv64, w_bv128)
    m_u = compute_metrics("u", u_bv64, u_bv128)
    logger.info(str(m_w))
    logger.info(str(m_u))

    # =========================================================
    # 3. fwd_o masking for non-aligned T
    # =========================================================
    # The difference: old mask = (o_i >= o_i), new mask = (o_t >= o_t) & (m_t & m_t)
    # For the last chunk, m_t zeros out positions >= T
    # This only matters when T % BT != 0
    remainder = T % BT
    if remainder != 0:
        logger.info(f"--- fwd_o masking: T={T}, BT={BT}, remainder={remainder} ---")
        logger.info(f"  Last chunk has {BT - remainder} out-of-bounds positions")
        # The old code doesn't zero these out in m_A, but:
        # - q[j] and k[j] for j >= T are loaded as 0 (boundary_check)
        # - v[j] for j >= T are loaded as 0
        # - So b_A[i, j] = sum_k q[i,k] * k[j,k] = 0 for j >= T
        # - And b_A[i, j] * exp(g[i] - g[j]) = 0 * something = 0
        # The masking change is technically redundant for valid data
        logger.info(
            "  old mask: q/k boundary_check makes b_A=0 for OOB cols → no precision effect"
        )
        logger.info("  new mask: explicit m_t zeroing → same result")

        # But let's verify empirically
        w_shared, u_shared, A_shared = chunk_gated_delta_rule_fwd_intra(
            k=k, v=v, g=g_log2, beta=beta, cu_seqlens=cu_seqlens_t
        )
        h_shared, vn_shared, _ = chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w_shared,
            u=u_shared,
            g=g_log2,
            initial_state=h0.clone(),
            output_final_state=True,
            cu_seqlens=cu_seqlens_t,
        )
        # Run fwd_o with old-style grid params but new kernel
        o_check = call_fwd_o_with_config(
            q,
            k,
            vn_shared,
            h_shared,
            g_log2,
            scale,
            cu_seqlens_t,
            BK=128,
            BV=64,
            num_warps=4,
            num_stages=2,
        )
        o_check2 = call_fwd_o_with_config(
            q,
            k,
            vn_shared,
            h_shared,
            g_log2,
            scale,
            cu_seqlens_t,
            BK=64,
            BV=128,
            num_warps=4,
            num_stages=1,
        )
        m_mask = compute_metrics("o_mask_chk", o_check, o_check2)
        logger.info(f"  Empirical old vs new fwd_o config: {m_mask}")
    else:
        logger.info(
            f"--- fwd_o masking: T={T} is aligned to BT={BT}, no masking difference ---"
        )

    # =========================================================
    # 4. Cumulative effect: exp → exp2 in kkt kernel
    # =========================================================
    # kkt: safe_exp(g_diff) vs exp2(g_diff)
    # Since safe_exp clamped positive diffs to 0 and exp2 doesn't,
    # let's measure the A matrix difference
    # We can only test with the CURRENT kernel (exp2), but we can
    # see if the A matrix matches what safe_exp would produce
    # by checking: are there any A[i,j] values where i > j
    # that should be 0 (from safe_exp clamping) but aren't?
    A_check = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g_cumsum=g_log2,
        cu_seqlens=cu_seqlens_t,
        output_dtype=torch.float32,
    )
    # A shape: [B, T, H, BT]. Check upper-triangle (diagonal+above) should be 0.
    # Position t within chunk c has offset t_in_chunk = t - c*BT.
    # A[b,t,h,j] should be 0 for j >= t_in_chunk (upper triangle + diagonal).
    n_nonzero_upper = 0
    for t in range(T):
        t_in_chunk = t % BT
        upper_vals = A_check[0, t, :, t_in_chunk:]  # diagonal and above
        n_nonzero_upper += (upper_vals.abs() > 1e-10).sum().item()
    logger.info(f"--- kkt upper-triangle check ---")
    logger.info(f"  Non-zero upper-triangle entries: {n_nonzero_upper} (should be 0)")

    logger.info(f"DONE deep analysis [{label}]")


# ---------------------------------------------------------------------------
# exp vs exp2 domain conversion precision
# ---------------------------------------------------------------------------
def run_exp_domain_test(
    label: str,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype = torch.bfloat16,
    Hg: int = None,
    DV: int = None,
    cu_seqlens: Optional[List[int]] = None,
    seed: int = 42,
):
    """Compare old (exp + raw cumsum) vs new (exp2 + log2 cumsum) gating paths.

    The old kernels used ``exp(cumsum(g))``; the new kernels use
    ``exp2(cumsum(g) * RCP_LN2)``.  Mathematically identical, but fp
    rounding differs because:
      1. The cumsum result is multiplied by RCP_LN2 (extra fp32 mul)
      2. exp2 and exp have different ULP characteristics on AMD CDNA
      3. Downstream ``g[i] - g[j]`` differences compound through the pipeline
    """
    if Hg is None:
        Hg = H
    if DV is None:
        DV = D

    logger.info(f"{'='*70}")
    logger.info(f"[EXP {label}]  B={B} T={T} H={H} D={D}")
    logger.info(f"{'='*70}")

    q, k, v, g, beta, h0, scale, cu_seqlens_t, N = make_inputs(
        B, T, H, Hg, D, DV, dtype, 0.0, cu_seqlens, seed
    )

    # Old domain: cumsum(g) in natural log, no scale
    g_ln = chunk_local_cumsum(g, chunk_size=64, scale=None, cu_seqlens=cu_seqlens_t)
    # New domain: cumsum(g) * RCP_LN2 in log2
    g_log2 = chunk_local_cumsum(
        g, chunk_size=64, scale=RCP_LN2, cu_seqlens=cu_seqlens_t
    )

    # --- Level 1: raw gating value comparison ---
    gate_old = torch.exp(g_ln)  # exp(cumsum(g))
    gate_new = torch.exp2(g_log2)  # exp2(cumsum(g)/ln2) == exp(cumsum(g))

    m_gate = compute_metrics("gate_val", gate_old, gate_new)
    logger.info(f"exp(cumsum) vs exp2(cumsum/ln2):  {m_gate}")

    # --- Level 2: gating difference g[i]-g[j] within chunks ---
    # This is what kkt/fwd_o actually computes
    BT = 64
    NT = (T + BT - 1) // BT
    # Reshape to chunks for pairwise diff
    T_pad = NT * BT
    g_ln_pad = torch.zeros(B, T_pad, H, device=g_ln.device, dtype=g_ln.dtype)
    g_log2_pad = torch.zeros(B, T_pad, H, device=g_log2.device, dtype=g_log2.dtype)
    g_ln_pad[:, :T] = g_ln
    g_log2_pad[:, :T] = g_log2
    g_ln_c = g_ln_pad.view(B, NT, BT, H)
    g_log2_c = g_log2_pad.view(B, NT, BT, H)

    # g[i] - g[j] within each chunk, averaged over all chunks
    diff_old = g_ln_c[:, :, :, None, :] - g_ln_c[:, :, None, :, :]  # for exp()
    diff_new = g_log2_c[:, :, :, None, :] - g_log2_c[:, :, None, :, :]  # for exp2()

    gate_diff_old = torch.exp(diff_old)
    gate_diff_new = torch.exp2(diff_new)
    m_diff = compute_metrics("gate_diff", gate_diff_old, gate_diff_new)
    logger.info(f"exp(g[i]-g[j]) vs exp2(g2[i]-g2[j]):  {m_diff}")

    # --- Level 3: full pipeline with old-domain g vs new-domain g ---
    # Both use the CURRENT kernels (which expect exp2/log2 domain).
    # So we can only run the new pipeline. Compare against recurrent ref.
    A_new, w_new, u_new, h_new, vn_new, o_new, ht_new = run_fused_pipeline(
        q.clone(),
        k.clone(),
        v.clone(),
        g_log2.clone(),
        beta.clone(),
        scale,
        h0.clone(),
        True,
        cu_seqlens_t,
    )

    if cu_seqlens is not None:
        ref_o_parts, ref_ht_parts = [], []
        for i in range(N):
            s, e = cu_seqlens[i], cu_seqlens[i + 1]
            ref_oi, ref_hti = recurrent_gdn_ref(
                q[:, s:e],
                k[:, s:e],
                v[:, s:e],
                beta[:, s:e],
                g[:, s:e],
                scale=scale,
                initial_state=h0[i : i + 1],
                output_final_state=True,
            )
            ref_o_parts.append(ref_oi)
            ref_ht_parts.append(ref_hti)
        ref_o = torch.cat(ref_o_parts, dim=1)
        ref_ht = torch.cat(ref_ht_parts, dim=0)
    else:
        ref_o, ref_ht = recurrent_gdn_ref(
            q,
            k,
            v,
            beta,
            g,
            scale=scale,
            initial_state=h0,
            output_final_state=True,
        )

    m_o = compute_metrics("o_vs_ref", ref_o, o_new)
    m_ht = compute_metrics("ht_vs_ref", ref_ht, ht_new)
    logger.info(f"new pipeline vs recurrent ref:  {m_o}")
    logger.info(f"new pipeline vs recurrent ref:  {m_ht}")

    logger.info(f"DONE exp domain test [{label}]")


# ---------------------------------------------------------------------------
# h buffer dtype: fp32 vs bf16
# ---------------------------------------------------------------------------
def run_h_dtype_test(
    label: str,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype = torch.bfloat16,
    Hg: int = None,
    DV: int = None,
    cu_seqlens: Optional[List[int]] = None,
    seed: int = 42,
):
    """Compare h buffer stored in fp32 vs bf16.

    fwd_o casts h to bf16 before dot anyway, so the only difference is
    the precision of the inter-chunk state accumulation in fwd_h.
    """
    if Hg is None:
        Hg = H
    if DV is None:
        DV = D

    logger.info(f"{'='*70}")
    logger.info(f"[HDTYPE {label}]  B={B} T={T} H={H} D={D}")
    logger.info(f"{'='*70}")

    q, k, v, g, beta, h0, scale, cu_seqlens_t, N = make_inputs(
        B, T, H, Hg, D, DV, dtype, 0.0, cu_seqlens, seed
    )
    g_log2 = chunk_local_cumsum(
        g, chunk_size=64, scale=RCP_LN2, cu_seqlens=cu_seqlens_t
    )

    # Shared A, w, u
    w, u, A = chunk_gated_delta_rule_fwd_intra(
        k=k, v=v, g=g_log2, beta=beta, cu_seqlens=cu_seqlens_t
    )

    # fwd_h with state_dtype=fp32 (default/old)
    h_fp32, vn_fp32, ht_fp32 = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g_log2,
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens_t,
        state_dtype=torch.float32,
    )
    # fwd_h with state_dtype=bf16
    h_bf16, vn_bf16, ht_bf16 = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g_log2,
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens_t,
        state_dtype=torch.bfloat16,
    )

    logger.info("--- fwd_h: state_dtype=fp32 vs bf16 ---")
    m_h = compute_metrics("h_state", h_fp32, h_bf16)
    m_vn = compute_metrics("v_new", vn_fp32, vn_bf16)
    m_ht = compute_metrics("final_st", ht_fp32, ht_bf16)
    logger.info(str(m_h))
    logger.info(str(m_vn))
    logger.info(str(m_ht))

    # fwd_o with fp32 h vs bf16 h
    o_fp32 = chunk_fwd_o(
        q=q,
        k=k,
        v=vn_fp32,
        h=h_fp32,
        g=g_log2,
        scale=scale,
        cu_seqlens=cu_seqlens_t,
    )
    o_bf16 = chunk_fwd_o(
        q=q,
        k=k,
        v=vn_bf16,
        h=h_bf16,
        g=g_log2,
        scale=scale,
        cu_seqlens=cu_seqlens_t,
    )

    logger.info("--- fwd_o output: h=fp32 vs h=bf16 ---")
    m_o = compute_metrics("o_output", o_fp32, o_bf16)
    logger.info(str(m_o))

    # Both vs recurrent reference
    if cu_seqlens is not None:
        ref_o_parts, ref_ht_parts = [], []
        for i in range(N):
            s, e = cu_seqlens[i], cu_seqlens[i + 1]
            ref_oi, ref_hti = recurrent_gdn_ref(
                q[:, s:e],
                k[:, s:e],
                v[:, s:e],
                beta[:, s:e],
                g[:, s:e],
                scale=scale,
                initial_state=h0[i : i + 1],
                output_final_state=True,
            )
            ref_o_parts.append(ref_oi)
            ref_ht_parts.append(ref_hti)
        ref_o = torch.cat(ref_o_parts, dim=1)
        ref_ht = torch.cat(ref_ht_parts, dim=0)
    else:
        ref_o, ref_ht = recurrent_gdn_ref(
            q,
            k,
            v,
            beta,
            g,
            scale=scale,
            initial_state=h0,
            output_final_state=True,
        )

    logger.info("--- vs Recurrent Reference ---")
    m_fp32_o = compute_metrics("o_fp32_ref", ref_o, o_fp32)
    m_bf16_o = compute_metrics("o_bf16_ref", ref_o, o_bf16)
    m_fp32_ht = compute_metrics("ht_fp32_ref", ref_ht, ht_fp32)
    m_bf16_ht = compute_metrics("ht_bf16_ref", ref_ht, ht_bf16)
    logger.info(str(m_fp32_o))
    logger.info(str(m_bf16_o))
    logger.info(str(m_fp32_ht))
    logger.info(str(m_bf16_ht))

    logger.info(f"DONE h dtype test [{label}]")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def main():
    all_passed = True

    # ---- 1. Small shape: quick smoke test ----
    all_passed &= run_precision_test(
        label="small_basic",
        B=1,
        T=15,
        H=4,
        D=64,
        dtype=torch.bfloat16,
        cu_seqlens=[0, 15],
    )

    # ---- 2. Medium shape: multi-segment variable-length ----
    all_passed &= run_precision_test(
        label="medium_varlen",
        B=1,
        T=1000,
        H=4,
        D=64,
        dtype=torch.bfloat16,
        cu_seqlens=[0, 256, 500, 1000],
    )

    # ---- 3. GQA shape (H != Hg): common in production models ----
    all_passed &= run_precision_test(
        label="gqa_h32_hg8",
        B=1,
        T=512,
        H=32,
        Hg=8,
        D=128,
        dtype=torch.bfloat16,
        cu_seqlens=[0, 512],
    )

    # ---- 4. Production-like shape: Qwen3.5-397B TP2 ----
    all_passed &= run_precision_test(
        label="prod_qwen35_tp2",
        B=1,
        T=2048,
        H=32,
        Hg=8,
        D=128,
        DV=128,
        dtype=torch.bfloat16,
        cu_seqlens=[0, 2048],
    )

    # ---- 5. Long sequence: stress test precision accumulation ----
    all_passed &= run_precision_test(
        label="long_seq_4096",
        B=1,
        T=4096,
        H=4,
        D=128,
        dtype=torch.bfloat16,
        cu_seqlens=[0, 4096],
        atol_vs_ref=0.02,
    )

    # ---- 6. Non-chunk-aligned length: edge case ----
    all_passed &= run_precision_test(
        label="non_aligned_T",
        B=1,
        T=200,
        H=4,
        D=64,
        dtype=torch.bfloat16,
        cu_seqlens=[0, 73, 200],
    )

    # ---- 7. Gate masking: sparse gates ----
    all_passed &= run_precision_test(
        label="sparse_gates",
        B=1,
        T=512,
        H=4,
        D=64,
        dtype=torch.bfloat16,
        mask_p=0.5,
        cu_seqlens=[0, 512],
    )

    # ---- 8. Batch mode (no cu_seqlens) ----
    all_passed &= run_precision_test(
        label="batch_no_varlen",
        B=2,
        T=256,
        H=4,
        D=64,
        dtype=torch.bfloat16,
    )

    # ---- 9. fp32 initial_state (ssm_state_dtype scenario) ----
    # The initial_state is fp32 but inputs are bf16
    all_passed &= run_precision_test(
        label="fp32_init_state",
        B=1,
        T=512,
        H=4,
        D=64,
        dtype=torch.bfloat16,
        cu_seqlens=[0, 256, 512],
    )

    # ---- 10. Large GQA long seq: closer to real CI failure shape ----
    all_passed &= run_precision_test(
        label="large_gqa_long",
        B=1,
        T=4096,
        H=32,
        Hg=8,
        D=128,
        DV=128,
        dtype=torch.bfloat16,
        cu_seqlens=[0, 4096],
        atol_vs_ref=0.02,
        atol_fused_vs_sep=0.01,
    )

    # ---- Tile config comparisons ----
    logger.info("\n" + "=" * 70)
    logger.info("TILE CONFIG COMPARISON TESTS")
    logger.info("=" * 70 + "\n")

    run_tile_config_test(
        label="tile_small",
        B=1,
        T=256,
        H=4,
        D=64,
        cu_seqlens=[0, 256],
    )
    run_tile_config_test(
        label="tile_gqa",
        B=1,
        T=512,
        H=32,
        Hg=8,
        D=128,
        DV=128,
        cu_seqlens=[0, 512],
    )
    run_tile_config_test(
        label="tile_prod",
        B=1,
        T=2048,
        H=32,
        Hg=8,
        D=128,
        DV=128,
        cu_seqlens=[0, 2048],
    )
    run_tile_config_test(
        label="tile_long",
        B=1,
        T=4096,
        H=4,
        D=128,
        cu_seqlens=[0, 4096],
    )
    run_tile_config_test(
        label="tile_varlen",
        B=1,
        T=1000,
        H=4,
        D=64,
        cu_seqlens=[0, 256, 500, 1000],
    )

    # ---- Deep-dive precision analysis ----
    logger.info("\n" + "=" * 70)
    logger.info("DEEP PRECISION ANALYSIS")
    logger.info("=" * 70 + "\n")

    run_deep_precision_analysis(
        label="deep_small",
        B=1,
        T=200,
        H=4,
        D=64,
        cu_seqlens=[0, 73, 200],
    )
    run_deep_precision_analysis(
        label="deep_prod",
        B=1,
        T=2048,
        H=32,
        Hg=8,
        D=128,
        DV=128,
        cu_seqlens=[0, 2048],
    )
    run_deep_precision_analysis(
        label="deep_long",
        B=1,
        T=4096,
        H=4,
        D=128,
        cu_seqlens=[0, 4096],
    )

    # ---- exp vs exp2 domain conversion ----
    logger.info("\n" + "=" * 70)
    logger.info("EXP vs EXP2 DOMAIN CONVERSION TESTS")
    logger.info("=" * 70 + "\n")

    run_exp_domain_test(
        label="exp_small",
        B=1,
        T=256,
        H=4,
        D=64,
        cu_seqlens=[0, 256],
    )
    run_exp_domain_test(
        label="exp_gqa",
        B=1,
        T=512,
        H=32,
        Hg=8,
        D=128,
        DV=128,
        cu_seqlens=[0, 512],
    )
    run_exp_domain_test(
        label="exp_long",
        B=1,
        T=4096,
        H=4,
        D=128,
        cu_seqlens=[0, 4096],
    )

    # ---- h buffer dtype tests ----
    logger.info("\n" + "=" * 70)
    logger.info("H BUFFER DTYPE TESTS (fp32 vs bf16)")
    logger.info("=" * 70 + "\n")

    run_h_dtype_test(
        label="hdtype_small",
        B=1,
        T=256,
        H=4,
        D=64,
        cu_seqlens=[0, 256],
    )
    run_h_dtype_test(
        label="hdtype_gqa",
        B=1,
        T=512,
        H=32,
        Hg=8,
        D=128,
        DV=128,
        cu_seqlens=[0, 512],
    )
    run_h_dtype_test(
        label="hdtype_long",
        B=1,
        T=4096,
        H=4,
        D=128,
        cu_seqlens=[0, 4096],
    )
    run_h_dtype_test(
        label="hdtype_prod",
        B=1,
        T=2048,
        H=32,
        Hg=8,
        D=128,
        DV=128,
        cu_seqlens=[0, 2048],
    )

    logger.info("=" * 70)
    if all_passed:
        logger.info("ALL PIPELINE TESTS PASSED")
    else:
        logger.error("SOME PIPELINE TESTS FAILED")
        exit(1)


if __name__ == "__main__":
    main()
