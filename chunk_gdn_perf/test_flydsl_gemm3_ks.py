"""Phase 2.1.5: GEMM3 (KS = K @ S_prev) FlyDSL kernel.

Shape: [BT=64, DK=128] @ [DK=128, DV=128] → [64, 128] (state-output GEMM).
The B operand now is the recurrent state S [DK, DV] (not K^T like in GEMM1).
S is row-major [DK, DV] - we need K @ S (no transpose on B).

For chunk_gdn this corresponds to the cutedsl GEMM3 (mma_warp line 2123):
    cute.gemm(tiled_mma_qs, tCtShared[ks], tCrS_A[s], tCrK_B_qs[k], tCtShared[ks])
Note S is the A operand in cutedsl with this orientation; for our PyTorch-style
formulation we keep K @ S = [BT, DV].
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.nn.functional as F

BLOCK_M = 64  # = BT
BLOCK_N = 128  # = DV
BLOCK_K = 128  # = DK


@flyc.kernel
def gemm_ks_kernel(
    K: fx.Tensor,  # [BT, DK] bf16
    S: fx.Tensor,  # [DK, DV] bf16  (state, not transposed)
    KS_out: fx.Tensor,  # [BT, DV] f32
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    tile_K = fx.make_tile(BLOCK_M, BLOCK_K)
    tile_S = fx.make_tile(BLOCK_N, BLOCK_K)  # logical N x K (we will transpose at MMA)
    tile_C = fx.make_tile(BLOCK_M, BLOCK_N)

    K_buf = fx.rocdl.make_buffer_tensor(K)
    # For K @ S we want S in [N, K] orientation for the standard A @ B^T MFMA.
    # S as stored is [DK, DV] row-major; reinterpret as [DV, DK] col-major would be
    # equivalent. For MVP, treat S as [DV, DK] view (transposed access pattern).
    S_buf = fx.rocdl.make_buffer_tensor(S)
    C_buf = fx.rocdl.make_buffer_tensor(KS_out)

    bK = fx.slice(fx.zipped_divide(K_buf, tile_K), (None, bid))
    bS = fx.slice(fx.zipped_divide(S_buf, tile_S), (None, bid))
    bC = fx.slice(fx.zipped_divide(C_buf, tile_C), (None, bid))

    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    # 4 warps tiled (2,2,1) over 64x128 → each warp does 32x64
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
    thr_mma = tiled_mma.thr_slice(tid)

    copy_in = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
    copy_out = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    tcA = fx.make_tiled_copy_A(copy_in, tiled_mma)
    tcB = fx.make_tiled_copy_B(copy_in, tiled_mma)
    tcC = fx.make_tiled_copy_C(copy_out, tiled_mma)

    K_src = tcA.get_slice(tid).partition_S(bK)
    S_src = tcB.get_slice(tid).partition_S(bS)
    C_dst = tcC.get_slice(tid).partition_S(bC)

    fA = thr_mma.make_fragment_A(bK)
    fB = thr_mma.make_fragment_B(bS)
    fC = thr_mma.make_fragment_C(bC)

    fx.copy(copy_in, K_src, tcA.get_slice(tid).retile(fA), pred=None)
    fx.copy(copy_in, S_src, tcB.get_slice(tid).retile(fB), pred=None)
    fx.gemm(mma_atom, fC, fA, fB, fC)
    fx.copy(copy_out, tcC.get_slice(tid).retile(fC), C_dst, pred=None)


@flyc.jit
def gemm_ks(K, S, KS_out, stream=fx.Stream(None)):
    gemm_ks_kernel(K, S, KS_out).launch(
        grid=(1, 1, 1), block=(256, 1, 1), stream=stream
    )


def main():
    torch.manual_seed(0)
    BT, DK, DV = 64, 128, 128
    print(f"[setup] BT={BT} DK={DK} DV={DV} (chunk_gdn GEMM3 shape)")
    print(f"[gpu]   {torch.cuda.get_device_name(0)}")

    K_t = (
        F.normalize(
            torch.randn(BT, DK, dtype=torch.float32, device="cuda"), p=2, dim=-1
        )
        .to(torch.bfloat16)
        .contiguous()
    )
    # S stored as [DV, DK] (transposed view) so MFMA's A @ B^T = K @ S works
    S_t = torch.randn(DV, DK, dtype=torch.bfloat16, device="cuda").contiguous()
    KS_out = torch.zeros(BT, DV, dtype=torch.float32, device="cuda")

    print("[run]   first call (JIT compile)...")
    gemm_ks(K_t, S_t, KS_out, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    # Reference: K @ S_orig where S_orig = S_t.T (so S_t is [DV, DK])
    expected = K_t.float() @ S_t.float().T
    diff = (KS_out - expected).abs().max().item()
    rel = diff / expected.abs().max().item()
    print(f"[check] max abs diff = {diff:.4e}, rel = {rel:.4e}")
    if rel < 5e-3:
        print("[PASS] GEMM3 (KS = K @ S^T) correct on MI308X.")
    else:
        print("[FAIL]")
        print(f"  KS_out[:2,:6] = {KS_out[:2,:6]}")
        print(f"  exp[:2,:6]    = {expected[:2,:6]}")


if __name__ == "__main__":
    main()
