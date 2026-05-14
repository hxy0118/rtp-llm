# -*- coding: utf-8 -*-
"""
Standalone exp vs exp2 precision & performance comparison for chunk-GDN.

No rtp-llm kernel imports — pure PyTorch chunk reference + Triton microbench.
Tests the hypothesis: is the exp→exp2 domain conversion the root cause of
token-level output changes in CI?

Usage:
    /opt/conda310/bin/python3 test_exp_vs_exp2.py
"""

import logging
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

logging.basicConfig(
    level="INFO", format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

RCP_LN2 = 1.0 / 0.6931471805599453


# ============================================================================
# 1. Recurrent fp32 reference (ground truth, uses torch.exp)
# ============================================================================
def recurrent_gdn_ref(q, k, v, beta, g, scale, h0=None):
    """Step-by-step recurrent GDN in fp32. Returns (o, h_final)."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    q, k, v, beta, g = [x.float() for x in (q, k, v, beta, g)]
    o = torch.zeros(B, T, H, V, device=q.device)
    h = torch.zeros(B, H, K, V, device=q.device)
    if h0 is not None:
        h = h0.float().clone()
    for i in range(T):
        qi, ki, vi = q[:, i], k[:, i], v[:, i].clone()
        h = h * g[:, i, :, None, None].exp()
        bi = beta[:, i]
        vi = vi - (h * ki[:, :, :, None]).sum(-2)
        vi = vi * bi[:, :, None]
        h = h + ki[:, :, :, None] * vi[:, :, None, :]
        o[:, i] = torch.einsum("bhd,bhdv->bhv", qi * scale, h)
    return o, h


# ============================================================================
# 2. PyTorch chunk reference — parameterized on exp_fn / g_scale
# ============================================================================
def chunk_gdn_pytorch_ref(
    q, k, v, beta, g_raw, scale, h0=None, use_exp2=False, chunk_size=64
):
    """
    Full chunk-GDN pipeline in PyTorch fp32, mirroring the Triton pipeline.

    Args:
        use_exp2: if True, use exp2 domain (g_cumsum *= RCP_LN2, then exp2).
                  if False, use exp domain (raw g_cumsum, then exp).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    q, k, v, beta, g_raw = [x.float() for x in (q, k, v, beta, g_raw)]

    pad_len = (BT - T % BT) % BT
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
        beta = F.pad(beta, (0, 0, 0, pad_len))
        g_raw = F.pad(g_raw, (0, 0, 0, pad_len))

    T_pad = q.shape[1]
    NT = T_pad // BT

    # Chunk-local cumsum of g
    g_cs = g_raw.reshape(B, NT, BT, H).cumsum(dim=2).reshape(B, T_pad, H)

    if use_exp2:
        g_cs = g_cs * RCP_LN2
        exp_fn = torch.exp2
    else:
        exp_fn = torch.exp

    # Reshape to chunks: [B, NT, BT, ...]
    qc = q.reshape(B, NT, BT, H, K)
    kc = k.reshape(B, NT, BT, H, K)
    vc = v.reshape(B, NT, BT, H, V)
    bc = beta.reshape(B, NT, BT, H)
    gc = g_cs.reshape(B, NT, BT, H)

    # ---- Stage 1: KKT + gating + solve ----
    # A[i,j] = beta[i] * (k[i] @ k[j]) * exp_fn(g[i] - g[j])  for i > j
    # Then solve (I + A)^{-1}
    A_all = torch.zeros(B, NT, BT, BT, H, device=q.device)
    eye = torch.eye(BT, device=q.device).unsqueeze(0)
    for n in range(NT):
        kk = kc[:, n]  # [B, BT, H, K]
        for h_idx in range(H):
            kkt = kk[:, :, h_idx, :] @ kk[:, :, h_idx, :].transpose(-1, -2)
            g_diff = gc[:, n, :, h_idx : h_idx + 1] - gc[
                :, n, :, h_idx : h_idx + 1
            ].transpose(-1, -2)
            gated = kkt * exp_fn(g_diff)
            gated = gated.tril(diagonal=-1)
            gated = gated * bc[:, n, :, h_idx : h_idx + 1]
            IpA = eye + gated
            A_all[:, n, :, :, h_idx] = torch.linalg.solve_triangular(
                IpA, eye, upper=False
            )

    # ---- Stage 2: recompute w, u ----
    w_all = torch.zeros_like(kc)
    u_all = torch.zeros_like(vc)
    for n in range(NT):
        Ainv = A_all[:, n]  # [B, BT, BT, H]
        g_exp = exp_fn(gc[:, n])  # [B, BT, H]
        for h_idx in range(H):
            A_h = Ainv[:, :, :, h_idx]  # [B, BT, BT]
            vb = vc[:, n, :, h_idx, :] * bc[:, n, :, h_idx : h_idx + 1]  # [B, BT, V]
            u_all[:, n, :, h_idx, :] = A_h @ vb
            kb = (
                kc[:, n, :, h_idx, :]
                * bc[:, n, :, h_idx : h_idx + 1]
                * g_exp[:, :, h_idx : h_idx + 1]
            )
            w_all[:, n, :, h_idx, :] = A_h @ kb

    # ---- Stage 3: fwd_h (inter-chunk recurrence) ----
    h_state = torch.zeros(B, H, K, V, device=q.device)
    if h0 is not None:
        h_state = h0.float().clone()
    o_out = torch.zeros(B, T_pad, H, V, device=q.device)

    for n in range(NT):
        w_n = w_all[:, n]  # [B, BT, H, K]
        u_n = u_all[:, n]  # [B, BT, H, V]

        # v_new = u - w @ h_state
        v_new = u_n.clone()
        for h_idx in range(H):
            v_new[:, :, h_idx, :] -= w_n[:, :, h_idx, :] @ h_state[:, h_idx]

        # Intra-chunk attention: o_intra = (q @ k^T) * gate * v_new
        for h_idx in range(H):
            qk = qc[:, n, :, h_idx, :] @ kc[:, n, :, h_idx, :].transpose(-1, -2)
            g_diff = gc[:, n, :, h_idx : h_idx + 1] - gc[
                :, n, :, h_idx : h_idx + 1
            ].transpose(-1, -2)
            attn = qk * exp_fn(g_diff)
            attn = attn.tril(diagonal=0)
            o_intra = attn @ v_new[:, :, h_idx, :] * scale

            # Inter-chunk: o_inter = q * exp(g) @ h_state
            q_gated = qc[:, n, :, h_idx, :] * exp_fn(gc[:, n, :, h_idx : h_idx + 1])
            o_inter = (q_gated @ h_state[:, h_idx]) * scale
            o_out[:, n * BT : (n + 1) * BT, h_idx, :] = o_intra + o_inter

        # Update h_state
        for h_idx in range(H):
            g_last = gc[:, n, -1, h_idx]
            h_state[:, h_idx] = h_state[:, h_idx] * exp_fn(g_last)[:, None, None]
            for t in range(BT):
                g_decay = exp_fn(gc[:, n, -1, h_idx] - gc[:, n, t, h_idx])
                kt = kc[:, n, t, h_idx, :]
                vt = v_new[:, t, h_idx, :]
                h_state[:, h_idx] += (kt[:, :, None] * vt[:, None, :]) * g_decay[
                    :, None, None
                ]

    o_out = o_out[:, :T]
    return o_out, h_state


# ============================================================================
# 3. Triton micro-benchmark: exp vs exp2 throughput
# ============================================================================
@triton.jit
def bench_exp_kernel(x_ptr, o_ptr, N: tl.constexpr, USE_EXP2: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * N + tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    if USE_EXP2:
        y = tl.math.exp2(x)
    else:
        y = tl.exp(x)
    tl.store(o_ptr + offs, y)


@triton.jit
def bench_gated_decay_kernel(
    g_ptr,
    h_ptr,
    o_ptr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    """Simulate the fwd_h gating: h = h * exp(g_last) for each chunk."""
    pid = tl.program_id(0)
    h_off = pid * K * V
    b_h = tl.zeros([K, V], dtype=tl.float32)

    for t in range(T):
        g_val = tl.load(g_ptr + pid * T + t).to(tl.float32)
        if USE_EXP2:
            decay = tl.math.exp2(g_val)
        else:
            decay = tl.exp(g_val)
        b_h = b_h * decay

    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    offs = h_off + o_k[:, None] * V + o_v[None, :]
    tl.store(o_ptr + offs, b_h)


def benchmark_exp_throughput():
    """Compare exp vs exp2 raw throughput on typical gating values."""
    logger.info("=" * 60)
    logger.info("Triton micro-benchmark: exp vs exp2 throughput")
    logger.info("=" * 60)

    N = 1024
    n_elements = 1024 * 1024
    x = torch.randn(n_elements, device="cuda") * 0.5 - 1.0  # typical g_cumsum range
    o = torch.empty_like(x)

    grid = (n_elements // N,)

    # Warmup
    for _ in range(10):
        bench_exp_kernel[grid](x, o, N=N, USE_EXP2=False)
        bench_exp_kernel[grid](x, o, N=N, USE_EXP2=True)
    torch.cuda.synchronize()

    n_iters = 200
    # exp
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        bench_exp_kernel[grid](x, o, N=N, USE_EXP2=False)
    torch.cuda.synchronize()
    t_exp = (time.perf_counter() - t0) / n_iters * 1e6

    # exp2
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        bench_exp_kernel[grid](x, o, N=N, USE_EXP2=True)
    torch.cuda.synchronize()
    t_exp2 = (time.perf_counter() - t0) / n_iters * 1e6

    logger.info(f"  exp:  {t_exp:.2f} us  ({n_elements/1e6:.1f}M elements)")
    logger.info(f"  exp2: {t_exp2:.2f} us  ({n_elements/1e6:.1f}M elements)")
    logger.info(f"  speedup: {t_exp/t_exp2:.3f}x")

    # Also check precision
    x_test = torch.randn(10000, device="cuda") * 2.0 - 3.0
    o_exp = torch.empty_like(x_test)
    o_exp2 = torch.empty_like(x_test)
    bench_exp_kernel[(10000 // N + 1,)](x_test, o_exp, N=N, USE_EXP2=False)
    # For exp2, scale input
    x_log2 = x_test * RCP_LN2
    bench_exp_kernel[(10000 // N + 1,)](x_log2, o_exp2, N=N, USE_EXP2=True)
    diff = (o_exp.float() - o_exp2.float()).abs()
    logger.info(
        f"  precision: exp(x) vs exp2(x/ln2) max_diff={diff.max():.6e}, "
        f"mean_diff={diff.mean():.6e}, rel={diff.mean()/o_exp.abs().mean():.6e}"
    )


# ============================================================================
# 4. Precision helpers
# ============================================================================
@dataclass
class Metrics:
    name: str
    abs_max: float
    rel_err: float
    cos_sim: float

    def __str__(self):
        return (
            f"  {self.name:20s}  abs_max={self.abs_max:.6e}  "
            f"rel_err={self.rel_err:.6e}  cos_sim={self.cos_sim:.8f}"
        )


def compare(name, a, b):
    a, b = a.float().flatten(), b.float().flatten()
    d = a - b
    base = a.square().mean().sqrt().item()
    return Metrics(
        name,
        d.abs().max().item(),
        d.square().mean().sqrt().item() / (base + 1e-12),
        F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item(),
    )


# ============================================================================
# 5. Main precision test
# ============================================================================
def run_precision_test(B, T, H, K, V, seed=42):
    logger.info("=" * 60)
    logger.info(f"Precision: B={B} T={T} H={H} K={K} V={V}")
    logger.info("=" * 60)

    torch.manual_seed(seed)
    device = "cuda"

    q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    k = F.normalize(torch.randn(B, T, H, K, device=device), dim=-1).to(torch.bfloat16)
    v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)
    g = F.logsigmoid(torch.rand(B, T, H, device=device, dtype=torch.bfloat16))
    beta = torch.rand(B, T, H, device=device, dtype=torch.bfloat16).sigmoid()
    h0 = torch.randn(B, H, K, V, device=device, dtype=torch.bfloat16)
    scale = K**-0.5

    # 1. Recurrent reference (ground truth, uses exp)
    ref_o, ref_h = recurrent_gdn_ref(q, k, v, beta, g, scale, h0)

    # 2. Chunk reference with exp (old path)
    chunk_exp_o, chunk_exp_h = chunk_gdn_pytorch_ref(
        q, k, v, beta, g, scale, h0, use_exp2=False
    )

    # 3. Chunk reference with exp2 (new path)
    chunk_exp2_o, chunk_exp2_h = chunk_gdn_pytorch_ref(
        q, k, v, beta, g, scale, h0, use_exp2=True
    )

    logger.info("--- chunk_exp (old) vs recurrent reference ---")
    logger.info(str(compare("o_exp_vs_ref", ref_o, chunk_exp_o)))
    logger.info(str(compare("h_exp_vs_ref", ref_h, chunk_exp_h)))

    logger.info("--- chunk_exp2 (new) vs recurrent reference ---")
    logger.info(str(compare("o_exp2_vs_ref", ref_o, chunk_exp2_o)))
    logger.info(str(compare("h_exp2_vs_ref", ref_h, chunk_exp2_h)))

    logger.info("--- chunk_exp vs chunk_exp2 (the delta from domain change) ---")
    m_o = compare("o_exp_vs_exp2", chunk_exp_o, chunk_exp2_o)
    m_h = compare("h_exp_vs_exp2", chunk_exp_h, chunk_exp2_h)
    logger.info(str(m_o))
    logger.info(str(m_h))

    return m_o, m_h


# ============================================================================
# 6. Performance: benchmark full chunk-GDN with rtp-llm kernels (exp2 only,
#    then measure exp overhead via Triton micro-bench)
# ============================================================================
def benchmark_full_pipeline():
    """Benchmark the actual rtp-llm chunk_gated_delta_rule if available."""
    logger.info("=" * 60)
    logger.info("Full pipeline benchmark (rtp-llm kernels)")
    logger.info("=" * 60)

    try:
        from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule

        has_rtp = True
    except ImportError:
        has_rtp = False
        logger.info("  rtp-llm not importable, skipping full pipeline benchmark")
        return

    for label, B, T, H, K in [
        ("small", 1, 512, 4, 64),
        ("prod", 1, 2048, 32, 128),
        ("long", 1, 8192, 4, 128),
    ]:
        torch.manual_seed(42)
        q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
        k = F.normalize(torch.randn(B, T, H, K, device="cuda"), dim=-1).bfloat16()
        v = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
        g = F.logsigmoid(torch.rand(B, T, H, device="cuda", dtype=torch.bfloat16))
        beta = torch.rand(B, T, H, device="cuda", dtype=torch.bfloat16).sigmoid()
        h0 = torch.randn(B, H, K, K, device="cuda", dtype=torch.bfloat16)

        # Warmup
        for _ in range(3):
            chunk_gated_delta_rule(
                q, k, v, g, beta, initial_state=h0, output_final_state=True
            )
        torch.cuda.synchronize()

        n_iters = 20
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            chunk_gated_delta_rule(
                q, k, v, g, beta, initial_state=h0, output_final_state=True
            )
        torch.cuda.synchronize()
        t_ms = (time.perf_counter() - t0) / n_iters * 1e3
        logger.info(f"  [{label}] B={B} T={T} H={H} K={K}: {t_ms:.2f} ms")


def main():
    # ---- Precision tests ----
    run_precision_test(B=1, T=128, H=4, K=64, V=64)
    run_precision_test(B=1, T=256, H=4, K=64, V=64)
    run_precision_test(B=1, T=512, H=4, K=128, V=128)

    # ---- Throughput micro-benchmark ----
    benchmark_exp_throughput()

    # ---- Full pipeline benchmark ----
    benchmark_full_pipeline()


if __name__ == "__main__":
    main()
