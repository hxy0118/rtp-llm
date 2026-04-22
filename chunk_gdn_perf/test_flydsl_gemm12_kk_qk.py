"""Phase 2.1.3: GEMM1 (kk = K@K^T) + GEMM2 (qk = Q@K^T) fused in one FlyDSL kernel.

Validates the multi-GEMM-within-one-kernel pattern (foundation for megakernel).
For MVP simplicity: each GEMM independently loads K from GMEM (no LDS sharing yet).
Outputs both kk [BT,BT] and qk [BT,BT] to verify correctness.

Shape: BT=64, DK=128.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.nn.functional as F

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 128


def _build_gemm_pair():
    """Returns a kernel that computes 2 separate [64x64x128] bf16 GEMMs."""

    @flyc.kernel
    def kernel(
        K: fx.Tensor,  # [BT, DK] bf16
        Q: fx.Tensor,  # [BT, DK] bf16
        kk_out: fx.Tensor,  # [BT, BT] f32 = K @ K^T
        qk_out: fx.Tensor,  # [BT, BT] f32 = Q @ K^T
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x

        tile_in = fx.make_tile(BLOCK_M, BLOCK_K)
        tile_out = fx.make_tile(BLOCK_M, BLOCK_N)

        K_buf = fx.rocdl.make_buffer_tensor(K)
        Q_buf = fx.rocdl.make_buffer_tensor(Q)
        kk_buf = fx.rocdl.make_buffer_tensor(kk_out)
        qk_buf = fx.rocdl.make_buffer_tensor(qk_out)

        bK = fx.slice(fx.zipped_divide(K_buf, tile_in), (None, bid))
        bQ = fx.slice(fx.zipped_divide(Q_buf, tile_in), (None, bid))
        bKK = fx.slice(fx.zipped_divide(kk_buf, tile_out), (None, bid))
        bQK = fx.slice(fx.zipped_divide(qk_buf, tile_out), (None, bid))

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
        thr_mma = tiled_mma.thr_slice(tid)

        copy_in = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        copy_out = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        tcA = fx.make_tiled_copy_A(copy_in, tiled_mma)
        tcB = fx.make_tiled_copy_B(copy_in, tiled_mma)
        tcC = fx.make_tiled_copy_C(copy_out, tiled_mma)
        tcA_th = tcA.get_slice(tid)
        tcB_th = tcB.get_slice(tid)
        tcC_th = tcC.get_slice(tid)

        # ===== GEMM1: kk = K @ K^T =====
        K_for_A = tcA_th.partition_S(bK)
        K_for_B = tcB_th.partition_S(bK)  # same K, will be reinterpreted as B-operand
        kk_dst = tcC_th.partition_S(bKK)

        fA1 = thr_mma.make_fragment_A(bK)
        fB1 = thr_mma.make_fragment_B(bK)
        fC1 = thr_mma.make_fragment_C(bKK)

        fx.copy(copy_in, K_for_A, tcA_th.retile(fA1), pred=None)
        fx.copy(copy_in, K_for_B, tcB_th.retile(fB1), pred=None)
        fx.gemm(mma_atom, fC1, fA1, fB1, fC1)
        fx.copy(copy_out, tcC_th.retile(fC1), kk_dst, pred=None)

        # ===== GEMM2: qk = Q @ K^T =====
        Q_for_A = tcA_th.partition_S(bQ)
        K_for_B2 = tcB_th.partition_S(bK)
        qk_dst = tcC_th.partition_S(bQK)

        fA2 = thr_mma.make_fragment_A(bQ)
        fB2 = thr_mma.make_fragment_B(bK)
        fC2 = thr_mma.make_fragment_C(bQK)

        fx.copy(copy_in, Q_for_A, tcA_th.retile(fA2), pred=None)
        fx.copy(copy_in, K_for_B2, tcB_th.retile(fB2), pred=None)
        fx.gemm(mma_atom, fC2, fA2, fB2, fC2)
        fx.copy(copy_out, tcC_th.retile(fC2), qk_dst, pred=None)

    return kernel


_kernel = _build_gemm_pair()


@flyc.jit
def gemm_kk_qk(K, Q, kk_out, qk_out, stream=fx.Stream(None)):
    _kernel(K, Q, kk_out, qk_out).launch(
        grid=(1, 1, 1), block=(256, 1, 1), stream=stream
    )


def main():
    torch.manual_seed(0)
    M, N, K_dim = BLOCK_M, BLOCK_N, BLOCK_K
    print(f"[setup] M={M} N={N} K={K_dim}")

    K_t = (
        F.normalize(
            torch.randn(M, K_dim, dtype=torch.float32, device="cuda"), p=2, dim=-1
        )
        .to(torch.bfloat16)
        .contiguous()
    )
    Q_t = torch.randn(M, K_dim, dtype=torch.bfloat16, device="cuda").contiguous()
    kk_out = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    qk_out = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    print(f"[gpu]   {torch.cuda.get_device_name(0)}")
    print("[run]   first call (JIT compile)...")
    gemm_kk_qk(K_t, Q_t, kk_out, qk_out, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    exp_kk = K_t.float() @ K_t.float().T
    exp_qk = Q_t.float() @ K_t.float().T
    diff_kk = (kk_out - exp_kk).abs().max().item()
    diff_qk = (qk_out - exp_qk).abs().max().item()
    rel_kk = diff_kk / exp_kk.abs().max().item()
    rel_qk = diff_qk / exp_qk.abs().max().item()
    print(f"[check] kk: max abs diff = {diff_kk:.4e}, rel = {rel_kk:.4e}")
    print(f"[check] qk: max abs diff = {diff_qk:.4e}, rel = {rel_qk:.4e}")
    if rel_kk < 5e-3 and rel_qk < 5e-3:
        print("[PASS] GEMM1+GEMM2 fused kernel works on MI308X.")
    else:
        print("[FAIL]")
        if rel_kk >= 5e-3:
            print(f"  kk[:4,:4] = {kk_out[:4,:4]}")
            print(f"  exp[:4,:4] = {exp_kk[:4,:4]}")


if __name__ == "__main__":
    main()
