"""Phase 2.2.5c FINAL: chunk-GDN with REAL O = Q @ S + real state recurrence.

Previous MVP (2.2.5b) used dummy O = Q @ Q^T. This version uses the true
linear-attention formula:

  for chunk t:
      O[t]  = Q[t] @ S_prev      # [BT, DK] @ [DK, DV] → [BT, DV]
      dS    = K[t]^T @ V[t]       # [DK, BT] @ [BT, DV] → [DK, DV]
      S    += dS

Challenge: MFMA needs bf16 B operand, but state S accumulator is f32.
Solution: maintain TWO forms:
  S_f32: accumulator, read as MFMA C input / written as MFMA D output
  S_T_bf16: [DV, DK] bf16 view (transposed layout so A @ B^T = Q @ S)

The S_T_bf16 is refreshed from S_f32 BEFORE each chunk's O computation.
This costs 1 cooperative convert per chunk (~1us for 16KB) = small overhead.

Shape: DK=DV=BT=64, f32 state 16KB, bf16 state 8KB = 24KB scratch per tile.
"""

import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.nn.functional as F
from flydsl.expr import arith, gpu, range_constexpr, vector
from flydsl.expr.typing import T

BT = 64
DK = 64
DV = 64


def build(NT_const):
    @flyc.kernel(known_block_size=[256, 1, 1])
    def kernel(
        Q_all: fx.Tensor,  # [NT*BT, DK] bf16
        K_T_all: fx.Tensor,  # [NT*DK, BT] bf16
        V_T_all: fx.Tensor,  # [NT*DV, BT] bf16
        S_f32: fx.Tensor,  # [DK, DV] f32 accumulator
        S_T_bf16: fx.Tensor,  # [DV, DK] bf16 staging for MFMA B operand
        O_all: fx.Tensor,  # [NT*BT, DV] f32
    ):
        tid = fx.thread_idx.x

        Q_buf = fx.rocdl.make_buffer_tensor(Q_all)
        KT_buf = fx.rocdl.make_buffer_tensor(K_T_all)
        VT_buf = fx.rocdl.make_buffer_tensor(V_T_all)
        Sf_buf = fx.rocdl.make_buffer_tensor(S_f32)
        SbT_buf = fx.rocdl.make_buffer_tensor(S_T_bf16)
        O_buf = fx.rocdl.make_buffer_tensor(O_all)

        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        tm = fx.make_tiled_mma(mma, fx.make_layout((2, 2, 1), (1, 2, 0)))
        thr = tm.thr_slice(tid)
        cin = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        cout = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        tcA = fx.make_tiled_copy_A(cin, tm)
        tcB = fx.make_tiled_copy_B(cin, tm)
        tcC = fx.make_tiled_copy_C(cout, tm)

        tileQ = fx.make_tile(BT, DK)
        tileKT = fx.make_tile(DK, BT)
        tileVT = fx.make_tile(DV, BT)
        tileO = fx.make_tile(BT, DV)
        tileSf = fx.make_tile(DK, DV)
        tileSbT = fx.make_tile(DV, DK)  # S_T_bf16 shape [DV, DK]

        bQ_all = fx.zipped_divide(Q_buf, tileQ)
        bKT_all = fx.zipped_divide(KT_buf, tileKT)
        bVT_all = fx.zipped_divide(VT_buf, tileVT)
        bO_all = fx.zipped_divide(O_buf, tileO)
        bSf = fx.slice(fx.zipped_divide(Sf_buf, tileSf), (None, fx.Int32(0)))
        bSbT = fx.slice(fx.zipped_divide(SbT_buf, tileSbT), (None, fx.Int32(0)))

        # Thread index layout (256 threads; DK=DV=64 → 4096 cells; 16 cells/thread)
        row_t = tid // 4
        col_base_t = (tid % 4) * 16

        for chunk_idx in range(NT_const):
            cidx = fx.Int32(chunk_idx)
            bQ = fx.slice(bQ_all, (None, cidx))
            bKT = fx.slice(bKT_all, (None, cidx))
            bVT = fx.slice(bVT_all, (None, cidx))
            bO = fx.slice(bO_all, (None, cidx))

            # ---- Refresh S_T_bf16 from S_f32 (cooperative convert) ----
            # S_T_bf16[c, r] = S_f32[r, c].to(bf16)
            # Thread (row_t, col_base_t..+16) maps to S_f32 position (row_t, col)
            # Writes S_T_bf16 at (col, row_t)
            for j in range_constexpr(16):
                col = col_base_t + j
                # Read S_f32[row_t, col] f32
                s_val = Sf_buf[(fx.Int32(row_t), fx.Int32(col))]
                # Convert to bf16
                s_bf = s_val.truncf(T.bf16)
                # Write to S_T_bf16[col, row_t]
                SbT_buf[(fx.Int32(col), fx.Int32(row_t))] = s_bf
            gpu.barrier()

            # ---- O[t] = Q[t] @ S = Q @ B^T  where B = S_T_bf16 [DV, DK] ----
            # A @ B^T with A=Q [BT, DK], B=S_T [DV, DK] → [BT, DV] = Q @ S ✓
            fA_o = thr.make_fragment_A(bQ)
            fB_o = thr.make_fragment_B(bSbT)
            fC_o = thr.make_fragment_C(bO)
            fx.copy(
                cin,
                tcA.get_slice(tid).partition_S(bQ),
                tcA.get_slice(tid).retile(fA_o),
                pred=None,
            )
            fx.copy(
                cin,
                tcB.get_slice(tid).partition_S(bSbT),
                tcB.get_slice(tid).retile(fB_o),
                pred=None,
            )
            # fC_o init to 0 via pre-zeroed bO (host-managed)
            fx.copy(
                cout,
                tcC.get_slice(tid).partition_S(bO),
                tcC.get_slice(tid).retile(fC_o),
                pred=None,
            )
            fx.gemm(mma, fC_o, fA_o, fB_o, fC_o)
            fx.copy(
                cout,
                tcC.get_slice(tid).retile(fC_o),
                tcC.get_slice(tid).partition_S(bO),
                pred=None,
            )

            # ---- S += K^T @ V using MFMA with f32 acc ----
            fA_s = thr.make_fragment_A(bKT)
            fB_s = thr.make_fragment_B(bVT)
            fC_s = thr.make_fragment_C(bSf)
            fx.copy(
                cin,
                tcA.get_slice(tid).partition_S(bKT),
                tcA.get_slice(tid).retile(fA_s),
                pred=None,
            )
            fx.copy(
                cin,
                tcB.get_slice(tid).partition_S(bVT),
                tcB.get_slice(tid).retile(fB_s),
                pred=None,
            )
            fx.copy(
                cout,
                tcC.get_slice(tid).partition_S(bSf),
                tcC.get_slice(tid).retile(fC_s),
                pred=None,
            )
            fx.gemm(mma, fC_s, fA_s, fB_s, fC_s)
            fx.copy(
                cout,
                tcC.get_slice(tid).retile(fC_s),
                tcC.get_slice(tid).partition_S(bSf),
                pred=None,
            )
            gpu.barrier()

    @flyc.jit
    def launcher(Q, KT, VT, Sf, SbT, O, stream=fx.Stream(None)):
        kernel(Q, KT, VT, Sf, SbT, O).launch(
            grid=(1, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launcher


def main():
    NT = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    torch.manual_seed(0)

    Q_chunks = torch.randn(NT, BT, DK, dtype=torch.bfloat16, device="cuda")
    K_chunks = F.normalize(
        torch.randn(NT, BT, DK, dtype=torch.float32, device="cuda"), p=2, dim=-1
    ).to(torch.bfloat16)
    V_chunks = torch.randn(NT, BT, DV, dtype=torch.bfloat16, device="cuda")
    Q_flat = Q_chunks.view(NT * BT, DK).contiguous()
    K_T = K_chunks.transpose(1, 2).contiguous().view(NT * DK, BT).contiguous()
    V_T = V_chunks.transpose(1, 2).contiguous().view(NT * DV, BT).contiguous()

    S_init = torch.randn(DK, DV, dtype=torch.float32, device="cuda").contiguous()
    S_f32 = S_init.clone()
    S_T_bf16 = torch.zeros(DV, DK, dtype=torch.bfloat16, device="cuda").contiguous()
    O_flat = torch.zeros(NT * BT, DV, dtype=torch.float32, device="cuda")

    print(f"[gpu] {torch.cuda.get_device_name(0)}, NT={NT}")
    launcher = build(NT)
    launcher(Q_flat, K_T, V_T, S_f32, S_T_bf16, O_flat, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    # Reference (using f32 throughout for stability):
    Q_f = Q_chunks.float()
    K_f = K_chunks.float()
    V_f = V_chunks.float()
    S_ref = S_init.clone()
    O_ref_chunks = []
    for t in range(NT):
        # O[t] uses S BEFORE update of this chunk (so after t=0 we'd use updated S from t=0)
        # But kernel processes O first, then dS — so O at chunk t uses S after chunk t-1's updates
        # Actually kernel does refresh → O → dS → S += dS. So O at chunk t sees S AFTER all prior updates.
        O_ref_chunks.append(Q_f[t] @ S_ref)
        S_ref = S_ref + K_f[t].T @ V_f[t]

    O_ref = torch.stack(O_ref_chunks).view(NT * BT, DV)

    diff_O = (O_flat - O_ref).abs().max().item()
    diff_S = (S_f32 - S_ref).abs().max().item()
    rel_O = diff_O / (O_ref.abs().max().item() + 1e-9)
    rel_S = diff_S / (S_ref.abs().max().item() + 1e-9)
    print(f"[check] O: max={diff_O:.3e}, rel={rel_O:.3e}")
    print(f"[check] S: max={diff_S:.3e}, rel={rel_S:.3e}")
    # Loose tolerance since bf16 staging of S costs precision
    if rel_O < 0.05 and rel_S < 0.01:
        print(f"[PASS] REAL chunk-GDN MVP (O=Q@S + S+=K^T@V) works for NT={NT}")
    else:
        print("[FAIL]")
        print(f"  O[0,:6]  = {O_flat[0,:6]}")
        print(f"  Oref[0,:6] = {O_ref[0,:6]}")
        return

    if NT >= 4:
        import statistics

        for _ in range(5):
            S_f32.copy_(S_init)
            launcher(Q_flat, K_T, V_T, S_f32, S_T_bf16, O_flat)
        torch.cuda.synchronize()
        ts = []
        for _ in range(30):
            S_f32.copy_(S_init)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            launcher(Q_flat, K_T, V_T, S_f32, S_T_bf16, O_flat)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        med_us = statistics.median(ts) * 1000
        print(
            f"[bench] NT={NT}: total {med_us:.1f} us, per-chunk {med_us/NT:.2f} us (REAL 2-GEMM per chunk)"
        )


if __name__ == "__main__":
    main()
