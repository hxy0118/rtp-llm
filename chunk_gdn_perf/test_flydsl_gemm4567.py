"""Phase 2.1.6: GEMM4-7 FlyDSL kernels.

  GEMM4: QS  = Q @ S       [64, 128] @ [128, 128] -> [64, 128]
  GEMM5: NV  = A_inv @ V   [64, 64]  @ [64, 128]  -> [64, 128]
  GEMM6: qkv = W_qkv @ NV  [64, 64]  @ [64, 128]  -> [64, 128]
  GEMM7: dS  = K^T @ delta [128, 64] @ [64, 128]  -> [128, 128]

All bf16 in / fp32 out, single CTA, 4 warps tiled_mma (2,2,1).
S/V/delta are stored in transposed orientation (B as [N, K]) to match the
A @ B^T MFMA convention. Each GEMM is independent and verified vs torch.matmul.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.nn.functional as F


def make_gemm_kernel(BLOCK_M, BLOCK_N, BLOCK_K, name):
    """Build a bf16 [M,K] @ [N,K]^T -> [M,N] f32 kernel for given shape.

    Picks tiled_mma layout + thread count based on output tile size:
      - (2,2,1) 256 threads for output up to ~64x128
      - (4,2,1) 512 threads for taller output (M>=128)
    """
    if BLOCK_M >= 128:
        warp_layout = (4, 2, 1)
        warp_stride = (1, 4, 0)
        block_threads = 512
    else:
        warp_layout = (2, 2, 1)
        warp_stride = (1, 2, 0)
        block_threads = 256

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def gemm_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x

        tA = fx.make_tile(BLOCK_M, BLOCK_K)
        tB = fx.make_tile(BLOCK_N, BLOCK_K)
        tC = fx.make_tile(BLOCK_M, BLOCK_N)

        Abuf = fx.rocdl.make_buffer_tensor(A)
        Bbuf = fx.rocdl.make_buffer_tensor(B)
        Cbuf = fx.rocdl.make_buffer_tensor(C)

        bA = fx.slice(fx.zipped_divide(Abuf, tA), (None, bid))
        bB = fx.slice(fx.zipped_divide(Bbuf, tB), (None, bid))
        bC = fx.slice(fx.zipped_divide(Cbuf, tC), (None, bid))

        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        tiled_mma = fx.make_tiled_mma(mma, fx.make_layout(warp_layout, warp_stride))
        thr_mma = tiled_mma.thr_slice(tid)

        cin = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        cout = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        tcA = fx.make_tiled_copy_A(cin, tiled_mma)
        tcB = fx.make_tiled_copy_B(cin, tiled_mma)
        tcC = fx.make_tiled_copy_C(cout, tiled_mma)

        Asrc = tcA.get_slice(tid).partition_S(bA)
        Bsrc = tcB.get_slice(tid).partition_S(bB)
        Cdst = tcC.get_slice(tid).partition_S(bC)

        fA = thr_mma.make_fragment_A(bA)
        fB = thr_mma.make_fragment_B(bB)
        fC = thr_mma.make_fragment_C(bC)

        fx.copy(cin, Asrc, tcA.get_slice(tid).retile(fA), pred=None)
        fx.copy(cin, Bsrc, tcB.get_slice(tid).retile(fB), pred=None)
        fx.gemm(mma, fC, fA, fB, fC)
        fx.copy(cout, tcC.get_slice(tid).retile(fC), Cdst, pred=None)

    @flyc.jit
    def gemm(A, B, C, stream=fx.Stream(None)):
        gemm_kernel(A, B, C).launch(
            grid=(1, 1, 1), block=(block_threads, 1, 1), stream=stream
        )

    gemm.__name__ = name
    return gemm


def verify(label, my_C, torch_ref, tol=5e-3):
    diff = (my_C - torch_ref).abs().max().item()
    rel = diff / (torch_ref.abs().max().item() + 1e-9)
    ok = rel < tol
    tag = "✓ PASS" if ok else "✗ FAIL"
    print(f"  [{tag}] {label}: max diff={diff:.3e}  rel={rel:.3e}")
    if not ok:
        print(f"    my[:2,:6]  = {my_C[:2,:6]}")
        print(f"    ref[:2,:6] = {torch_ref[:2,:6]}")
    return ok


def main():
    torch.manual_seed(0)
    print(f"[gpu] {torch.cuda.get_device_name(0)}")

    # === GEMM4: QS = Q @ S  [64,128] @ [128,128] -> [64,128] ===
    print("\n[GEMM4] QS = Q @ S  shape [64, 128] x [128, 128] -> [64, 128]")
    gemm4 = make_gemm_kernel(64, 128, 128, "gemm4")
    Q = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda").contiguous()
    S = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda").contiguous()
    QS = torch.zeros(64, 128, dtype=torch.float32, device="cuda")
    gemm4(Q, S, QS)
    torch.cuda.synchronize()
    ref = Q.float() @ S.float().T
    ok4 = verify("QS", QS, ref)

    # === GEMM5: NV = A_inv @ V  [64,64] @ [64,128] -> [64,128] ===
    print("\n[GEMM5] NV = A_inv @ V  shape [64, 64] x [64, 128] -> [64, 128]")
    gemm5 = make_gemm_kernel(64, 128, 64, "gemm5")
    A_inv = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda").contiguous()
    V = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda").contiguous()
    NV = torch.zeros(64, 128, dtype=torch.float32, device="cuda")
    gemm5(A_inv, V, NV)
    torch.cuda.synchronize()
    ref = A_inv.float() @ V.float().T
    ok5 = verify("NV", NV, ref)

    # === GEMM6: qkv = W_qkv @ NV  [64,64] @ [64,128] -> [64,128] ===
    print("\n[GEMM6] qkv = W_qkv @ NV  shape [64, 64] x [64, 128] -> [64, 128]")
    gemm6 = make_gemm_kernel(64, 128, 64, "gemm6")
    W = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda").contiguous()
    NV_in = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda").contiguous()
    qkv = torch.zeros(64, 128, dtype=torch.float32, device="cuda")
    gemm6(W, NV_in, qkv)
    torch.cuda.synchronize()
    ref = W.float() @ NV_in.float().T
    ok6 = verify("qkv", qkv, ref)

    # === GEMM7: dS = K^T @ delta  [128,64] @ [64,128] -> [128,128] ===
    print("\n[GEMM7] dS = K^T @ delta  shape [128, 64] x [64, 128] -> [128, 128]")
    gemm7 = make_gemm_kernel(128, 128, 64, "gemm7")
    Kt = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda").contiguous()
    delta = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda").contiguous()
    dS = torch.zeros(128, 128, dtype=torch.float32, device="cuda")
    gemm7(Kt, delta, dS)
    torch.cuda.synchronize()
    ref = Kt.float() @ delta.float().T
    ok7 = verify("dS", dS, ref)

    print()
    if all([ok4, ok5, ok6, ok7]):
        print("[ALL PASS] GEMM4-7 all correct on MI308X.")
    else:
        print("[FAIL] some GEMMs failed; see above.")


if __name__ == "__main__":
    main()
