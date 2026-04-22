"""Phase 2.2.3c: runtime NT × multi-CTA × K=64 manual MFMA.

Combines v7/v8 (runtime NT manual MFMA) with multi-CTA grid.
Each CTA (head_id, batch_id) runs NT chunks with K=64 MFMA.
Grid: (H, B, 1). Single warp per CTA.

This is the closest we get to production: single kernel launch handles
H × B × NT chunks in parallel across CUs.
"""

import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T

BT = 64
DK = 16
DV = 16
K_STEPS = BT // 16


@flyc.kernel(known_block_size=[64, 1, 1])
def kernel(
    K_T_all: fx.Tensor,  # [B*H*NT*DK, BT] bf16
    V_T_all: fx.Tensor,  # [B*H*NT*DV, BT] bf16
    S_init: fx.Tensor,  # [B*H*DK, DV] f32
    S_final: fx.Tensor,  # [B*H*DK, DV] f32
    NT: fx.Int32,
    H: fx.Int32,
):
    tid = fx.thread_idx.x
    bid_h = fx.block_idx.x
    bid_b = fx.block_idx.y
    cta_linear = bid_b * H + bid_h
    lane = tid
    lane_mod16 = lane % 16
    lane_div16 = lane // 16

    KT_buf = fx.rocdl.make_buffer_tensor(K_T_all)
    VT_buf = fx.rocdl.make_buffer_tensor(V_T_all)
    Si_buf = fx.rocdl.make_buffer_tensor(S_init)
    Sf_buf = fx.rocdl.make_buffer_tensor(S_final)

    # CTA's S_init row base = cta_linear * DK
    s_row_base = cta_linear * fx.Int32(DK)

    s0 = Si_buf[(s_row_base + lane_div16 * fx.Int32(4) + fx.Int32(0), lane_mod16)]
    s1 = Si_buf[(s_row_base + lane_div16 * fx.Int32(4) + fx.Int32(1), lane_mod16)]
    s2 = Si_buf[(s_row_base + lane_div16 * fx.Int32(4) + fx.Int32(2), lane_mod16)]
    s3 = Si_buf[(s_row_base + lane_div16 * fx.Int32(4) + fx.Int32(3), lane_mod16)]
    fC_init = vector.from_elements(T.vec(4, T.f32), [s0, s1, s2, s3])

    # CTA's K/V row base = cta_linear * NT * DK (runtime NT)
    cta_chunk_base = cta_linear * NT

    dummy0 = arith.constant(0, type=T.i32)
    for chunk_idx, carry in range(0, NT, init=[fC_init, dummy0]):
        fC_prev = carry[0]
        _d = carry[1]
        cidx_i32 = arith.index_cast(T.i32, chunk_idx)
        # Absolute chunk within KT/VT flat tensors
        abs_chunk = cta_chunk_base + cidx_i32

        fC = fC_prev
        for k_step in range_constexpr(K_STEPS):
            a_row = abs_chunk * fx.Int32(DK) + lane_mod16
            a_col_start = fx.Int32(k_step * 16) + lane_div16 * fx.Int32(4)
            a0 = KT_buf[(a_row, a_col_start + fx.Int32(0))]
            a1 = KT_buf[(a_row, a_col_start + fx.Int32(1))]
            a2 = KT_buf[(a_row, a_col_start + fx.Int32(2))]
            a3 = KT_buf[(a_row, a_col_start + fx.Int32(3))]
            fA = vector.from_elements(T.vec(4, T.bf16), [a0, a1, a2, a3])

            b_row = abs_chunk * fx.Int32(DV) + lane_mod16
            b_col_start = fx.Int32(k_step * 16) + lane_div16 * fx.Int32(4)
            b0 = VT_buf[(b_row, b_col_start + fx.Int32(0))]
            b1 = VT_buf[(b_row, b_col_start + fx.Int32(1))]
            b2 = VT_buf[(b_row, b_col_start + fx.Int32(2))]
            b3 = VT_buf[(b_row, b_col_start + fx.Int32(3))]
            fB = vector.from_elements(T.vec(4, T.bf16), [b0, b1, b2, b3])

            fA_i16 = vector.bitcast(T.vec(4, T.i16), fA)
            fB_i16 = vector.bitcast(T.vec(4, T.i16), fB)
            fC = rocdl.mfma_f32_16x16x16bf16_1k(
                T.vec(4, T.f32), [fA_i16, fB_i16, fC, 0, 0, 0]
            )

        results = yield [fC, _d]

    fC_final = results[0]
    for i in range_constexpr(4):
        val = vector.extract(fC_final, static_position=[i], dynamic_position=[])
        Sf_buf[(s_row_base + lane_div16 * fx.Int32(4) + fx.Int32(i), lane_mod16)] = val


@flyc.jit
def launcher(
    KT, VT, Si, Sf, NT: fx.Int32, H: fx.Int32, B: fx.Int32, stream=fx.Stream(None)
):
    kernel(KT, VT, Si, Sf, NT, H).launch(
        grid=(H, B, 1), block=(64, 1, 1), stream=stream
    )


def main():
    NT = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    torch.manual_seed(0)

    K_chunks = torch.randn(B, H, NT, BT, DK, dtype=torch.bfloat16, device="cuda")
    V_chunks = torch.randn(B, H, NT, BT, DV, dtype=torch.bfloat16, device="cuda")
    K_T = K_chunks.transpose(-2, -1).contiguous().view(B * H * NT * DK, BT).contiguous()
    V_T = V_chunks.transpose(-2, -1).contiguous().view(B * H * NT * DV, BT).contiguous()

    Si_bh = torch.randn(B, H, DK, DV, dtype=torch.float32, device="cuda").contiguous()
    Si = Si_bh.view(B * H * DK, DV).contiguous()
    Sf = torch.zeros(B * H * DK, DV, dtype=torch.float32, device="cuda")

    print(f"[gpu] {torch.cuda.get_device_name(0)}, NT={NT} H={H} B={B}, {H*B} CTAs")
    launcher(K_T, V_T, Si, Sf, NT, H, B, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    # Reference
    S_ref = Si_bh.clone()
    for t in range(NT):
        S_ref = S_ref + torch.einsum(
            "bhtd,bhte->bhde", K_chunks[:, :, t].float(), V_chunks[:, :, t].float()
        )
    S_ref_flat = S_ref.view(B * H * DK, DV)

    diff = (Sf - S_ref_flat).abs().max().item()
    rel = diff / (S_ref_flat.abs().max().item() + 1e-9)
    print(f"[check] max diff = {diff:.4e}, rel = {rel:.4e}")
    if rel < 0.05:
        print(f"[PASS] runtime NT × multi-CTA: {H*B} CTAs × NT={NT} × K=64 works")
        import statistics

        for _ in range(5):
            Sf.zero_()
            launcher(K_T, V_T, Si, Sf, NT, H, B)
        torch.cuda.synchronize()
        ts = []
        for _ in range(10):
            Sf.zero_()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            launcher(K_T, V_T, Si, Sf, NT, H, B)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        med_us = statistics.median(ts) * 1000
        total_chunks = NT * H * B
        print(f"[bench] total {med_us:.1f} us, {total_chunks} chunks parallel")
        print(f"        per-chunk effective: {med_us / total_chunks:.4f} us")
        print(f"        FLA prod ref (MI355X TP2 layer): 6.5 us/chunk")
        print(f"        speedup vs FLA: {6.5 * total_chunks / med_us:.1f}×")
    else:
        print("[FAIL]")


if __name__ == "__main__":
    main()
