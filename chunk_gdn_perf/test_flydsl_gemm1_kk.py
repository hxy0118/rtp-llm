"""Phase 2.1.2: GEMM1 (kk = K @ K^T) at chunk_gdn shape on MI308X.

Computes C = A @ B^T for [BT=64, DK=128] @ [BT=64, DK=128] → [64, 64], bf16/f32.
First "real" piece of chunk_gdn megakernel: corresponds to GEMM1 in the cutedsl
mma_warp pipeline (see flashinfer/.../gated_delta_net_chunked.py:2086).

K = 128 = 8 × MFMA_K(16), accumulated via tiled_mma over K dim.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

BLOCK_M = 64  # = BT (chunk_size)
BLOCK_N = 64  # = BT
BLOCK_K = 128  # = DK (head_k_dim)


@flyc.kernel
def gemm_kk_kernel(
    A: fx.Tensor,  # [BT, DK] bf16  (= K)
    B: fx.Tensor,  # [BT, DK] bf16  (= K, transposed at op level)
    C: fx.Tensor,  # [BT, BT] f32   (= K @ K^T)
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    tileA = fx.make_tile(BLOCK_M, BLOCK_K)
    tileB = fx.make_tile(BLOCK_N, BLOCK_K)
    tileC = fx.make_tile(BLOCK_M, BLOCK_N)

    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)

    bA = fx.zipped_divide(A, tileA)
    bB = fx.zipped_divide(B, tileB)
    bC = fx.zipped_divide(C, tileC)

    bA = fx.slice(bA, (None, bid))
    bB = fx.slice(bB, (None, bid))
    bC = fx.slice(bC, (None, bid))

    # bf16 MFMA atom 16x16x16; tiled (2,2,1) over 4 warps
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
    thr_mma = tiled_mma.thr_slice(tid)

    copy_atom_in = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
    copy_atom_out = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    tiled_copy_A = fx.make_tiled_copy_A(copy_atom_in, tiled_mma)
    tiled_copy_B = fx.make_tiled_copy_B(copy_atom_in, tiled_mma)
    tiled_copy_C = fx.make_tiled_copy_C(copy_atom_out, tiled_mma)

    thr_copy_A = tiled_copy_A.get_slice(tid)
    thr_copy_B = tiled_copy_B.get_slice(tid)
    thr_copy_C = tiled_copy_C.get_slice(tid)

    copy_src_A = thr_copy_A.partition_S(bA)
    copy_src_B = thr_copy_B.partition_S(bB)
    copy_dst_C = thr_copy_C.partition_S(bC)

    frag_A = thr_mma.make_fragment_A(bA)
    frag_B = thr_mma.make_fragment_B(bB)
    frag_C = thr_mma.make_fragment_C(bC)

    copy_frag_A = thr_copy_A.retile(frag_A)
    copy_frag_B = thr_copy_B.retile(frag_B)
    copy_frag_C = thr_copy_C.retile(frag_C)

    fx.copy(copy_atom_in, copy_src_A, copy_frag_A, pred=None)
    fx.copy(copy_atom_in, copy_src_B, copy_frag_B, pred=None)

    # gemm: frag_C += frag_A @ frag_B^T (tiled_mma handles K-dim accumulation)
    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

    fx.copy(copy_atom_out, copy_frag_C, copy_dst_C, pred=None)


@flyc.jit
def gemm_kk(A, B, C, stream=fx.Stream(None)):
    gemm_kk_kernel(A, B, C).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)


def main():
    torch.manual_seed(0)
    M, N, K = BLOCK_M, BLOCK_N, BLOCK_K
    print(f"[setup] M={M} N={N} K={K} (chunk_gdn GEMM1 shape: BT={M}, DK={K})")

    # Use F.normalize so K is unit-norm (matches test_chunk_prefill.py pattern)
    import torch.nn.functional as F

    K_tensor = F.normalize(
        torch.randn(M, K, dtype=torch.float32, device="cuda"), p=2, dim=-1
    ).to(torch.bfloat16)
    A = K_tensor.contiguous()
    B = K_tensor.contiguous()  # K@K^T → both operands same
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    print(f"[gpu]   {torch.cuda.get_device_name(0)}")
    print("[run]   first call (JIT compile)...")
    gemm_kk(A, B, C, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    expected = A.float() @ B.float().T
    diff = (C - expected).abs().max().item()
    rel = diff / expected.abs().max().item()
    print(f"[check] C max = {C.abs().max():.4f}, exp max = {expected.abs().max():.4f}")
    print(f"[check] max abs diff = {diff:.4e}  rel = {rel:.4e}")
    if rel < 5e-3:
        print("[PASS] GEMM1 (kk) at chunk_gdn shape works on MI308X.")
    else:
        print(f"[FAIL] mismatch")
        print(f"  C[:4,:4]   = {C[:4,:4]}")
        print(f"  exp[:4,:4] = {expected[:4,:4]}")


if __name__ == "__main__":
    main()
