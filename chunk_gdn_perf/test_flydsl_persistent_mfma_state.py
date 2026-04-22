"""Phase 2.2.2c: MFMA inside persistent loop with STATE IN REGISTERS (frag_C).

The cutedsl TMEM-state insight applied to AMD: keep the MFMA accumulator
fragment ALIVE across all chunk iterations. No GMEM writes during the loop.
  - frag_C loaded from S_init at kernel start
  - for each chunk: frag_C += K[chunk]^T @ V[chunk]  (MFMA accumulate)
  - frag_C stored to S_final at kernel end

This mirrors the cutedsl TMEM-resident state pattern. No LDS needed for state
since registers are faster. Only GMEM state at kernel boundaries.

Uses compile-time NT (unrolled) to sidestep the runtime-slice issue (TODO for 2.2.3).

Shape: BT=DK=DV=64, single MFMA atom 16x16x16 × 4x4 output tiles × 4 K-steps per chunk.
Per chunk MFMA count: 16 output atoms × 4 K-steps = 64 MFMA / warp, × 4 warps, = 16 MFMAs
                        per atom position (4 warps distribute).
Actually with (2,2,1) tiled_mma: each warp does 2x2 atoms × 4 K-steps = 16 MFMA / warp.
"""

import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.nn.functional as F

BT = 64
DK = 64
DV = 64


def build(NT_const):
    @flyc.kernel(known_block_size=[256, 1, 1])
    def kernel(
        K_all: fx.Tensor,  # [NT*BT, DK] bf16
        V_all: fx.Tensor,  # [NT*BT, DV] bf16
        S_init: fx.Tensor,  # [DK, DV] f32 — used as initial accumulator
        S_final: fx.Tensor,  # [DK, DV] f32 — written with final accumulator
    ):
        tid = fx.thread_idx.x

        K_buf = fx.rocdl.make_buffer_tensor(K_all)
        V_buf = fx.rocdl.make_buffer_tensor(V_all)
        Si_buf = fx.rocdl.make_buffer_tensor(S_init)
        Sf_buf = fx.rocdl.make_buffer_tensor(S_final)

        # MFMA atoms
        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        tm = fx.make_tiled_mma(mma, fx.make_layout((2, 2, 1), (1, 2, 0)))
        thr = tm.thr_slice(tid)

        cin = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        cout = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        tcA = fx.make_tiled_copy_A(cin, tm)
        tcB = fx.make_tiled_copy_B(cin, tm)
        tcC = fx.make_tiled_copy_C(cout, tm)

        # S_init / S_final are [DK, DV] — single tile, no outer divide needed
        # But tiled_copy expects a tile tensor → use zipped_divide with tile = full
        tileS = fx.make_tile(DK, DV)
        bSi = fx.slice(fx.zipped_divide(Si_buf, tileS), (None, fx.Int32(0)))
        bSf = fx.slice(fx.zipped_divide(Sf_buf, tileS), (None, fx.Int32(0)))

        # Allocate state fragment, load S_init into it
        fC = thr.make_fragment_C(bSi)
        fx.copy(
            cout,
            tcC.get_slice(tid).partition_S(bSi),
            tcC.get_slice(tid).retile(fC),
            pred=None,
        )

        # K_all / V_all tiling: NT tiles of [BT, DK or DV]
        tileK = fx.make_tile(BT, DK)
        tileV = fx.make_tile(BT, DV)
        bK_all = fx.zipped_divide(K_buf, tileK)
        bV_all = fx.zipped_divide(V_buf, tileV)

        # Persistent compile-time unrolled loop
        for chunk_idx in range(NT_const):
            cidx = fx.Int32(chunk_idx)
            bK = fx.slice(bK_all, (None, cidx))
            bV = fx.slice(bV_all, (None, cidx))

            # For GEMM7 dS = K^T @ V:
            # A = K^T (shape [DK, BT], but K is stored as [BT, DK])
            # B = V^T (shape [DV, BT], V stored as [BT, DV])
            # A @ B^T pattern: MFMA computes A @ B^T = [DK, DV]. Need A_shape=[DK,BT], B_shape=[DV,BT].
            # Easiest: re-interpret K as A in transposed layout (but this messes up partition).
            #
            # Alternative: compute kk = K @ K^T (GEMM1, not GEMM7). Shape works: A=K[BT,DK], B=K[BT,DK],
            # A @ B^T = [BT, BT]. Since BT=DK here, fits frag_C shape [DK,DV=DK].
            # For MVP we just verify that PER-CHUNK MFMA accumulates into frag_C correctly.
            # So: frag_C += K[chunk] @ K[chunk]^T   (not the real GEMM7 but same pattern).
            fA = thr.make_fragment_A(bK)
            fB = thr.make_fragment_B(bK)
            fx.copy(
                cin,
                tcA.get_slice(tid).partition_S(bK),
                tcA.get_slice(tid).retile(fA),
                pred=None,
            )
            fx.copy(
                cin,
                tcB.get_slice(tid).partition_S(bK),
                tcB.get_slice(tid).retile(fB),
                pred=None,
            )
            # gemm: D = A @ B^T + C; use fC as both input and output (accumulate)
            fx.gemm(mma, fC, fA, fB, fC)

        # Store final frag_C → S_final
        fx.copy(
            cout,
            tcC.get_slice(tid).retile(fC),
            tcC.get_slice(tid).partition_S(bSf),
            pred=None,
        )

    @flyc.jit
    def launcher(K, V, Si, Sf, stream=fx.Stream(None)):
        kernel(K, V, Si, Sf).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)

    return launcher


def main():
    NT = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    torch.manual_seed(0)
    K = (
        F.normalize(
            torch.randn(NT * BT, DK, dtype=torch.float32, device="cuda"), p=2, dim=-1
        )
        .to(torch.bfloat16)
        .contiguous()
    )
    V = torch.randn(NT * BT, DV, dtype=torch.bfloat16, device="cuda").contiguous()
    Si = torch.randn(DK, DV, dtype=torch.float32, device="cuda").contiguous()
    Sf = torch.zeros(DK, DV, dtype=torch.float32, device="cuda")

    print(f"[gpu] {torch.cuda.get_device_name(0)}, NT={NT}")
    launcher = build(NT)
    launcher(K, V, Si, Sf, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    # Reference: S_final = S_init + sum over chunks of K[chunk] @ K[chunk]^T
    K_chunks = K.view(NT, BT, DK).float()
    dS_total = torch.einsum("nbd,ned->be", K_chunks, K_chunks)  # sum of K@K^T per chunk
    # Wait — einsum nbd,ned is sum over n and d: [b, e] = sum over n,d of K[n,b,d]*K[n,e,d]
    # That's sum of (K_n @ K_n^T). Correct.
    # BUT our state shape is [DK, DV] = [64, 64], and K@K^T = [BT, BT] = [64, 64]. Match.
    S_ref = Si + dS_total

    diff = (Sf - S_ref).abs().max().item()
    rel = diff / S_ref.abs().max().item()
    print(f"[check] max diff = {diff:.4e}, rel = {rel:.4e}")
    if rel < 5e-2:
        print(
            f"[PASS] MFMA-accumulate-in-persistent-loop + register state works for NT={NT}"
        )
    else:
        print("[FAIL]")
        print(f"  Sf[:2,:6]     = {Sf[:2,:6]}")
        print(f"  S_ref[:2,:6]  = {S_ref[:2,:6]}")
        return

    if NT >= 4:
        import statistics

        for _ in range(5):
            launcher(K, V, Si, Sf)
        torch.cuda.synchronize()
        ts = []
        for _ in range(30):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            launcher(K, V, Si, Sf)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        med_us = statistics.median(ts) * 1000
        print(f"[bench] NT={NT}: total {med_us:.1f} us, per-chunk {med_us/NT:.2f} us")


if __name__ == "__main__":
    main()
