"""Phase 2.2.3e FINAL: Production-shape runtime MFMA with multi-CTA grid.

Full 64x64 output per CTA × multi-CTA (H, B, 1).
DK=DV=64, BT=64 (K), 16 atoms/warp, 4 K-steps = 64 MFMA/chunk/CTA.

Single kernel launch handles: H × B CTAs × NT chunks = full-layer state update.
"""

import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T

BT = 64
DK = 64
DV = 64
K_STEPS = BT // 16
M_ATOMS = DK // 16
N_ATOMS = DV // 16


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

    s_row_base = cta_linear * fx.Int32(DK)
    cta_nt_base = cta_linear * NT

    # 16 frag_C from S_init (CTA-specific offset)
    frags_init = []
    for m_atom in range_constexpr(M_ATOMS):
        for n_atom in range_constexpr(N_ATOMS):
            v0 = Si_buf[
                (
                    s_row_base
                    + fx.Int32(16 * m_atom)
                    + lane_div16 * fx.Int32(4)
                    + fx.Int32(0),
                    fx.Int32(16 * n_atom) + lane_mod16,
                )
            ]
            v1 = Si_buf[
                (
                    s_row_base
                    + fx.Int32(16 * m_atom)
                    + lane_div16 * fx.Int32(4)
                    + fx.Int32(1),
                    fx.Int32(16 * n_atom) + lane_mod16,
                )
            ]
            v2 = Si_buf[
                (
                    s_row_base
                    + fx.Int32(16 * m_atom)
                    + lane_div16 * fx.Int32(4)
                    + fx.Int32(2),
                    fx.Int32(16 * n_atom) + lane_mod16,
                )
            ]
            v3 = Si_buf[
                (
                    s_row_base
                    + fx.Int32(16 * m_atom)
                    + lane_div16 * fx.Int32(4)
                    + fx.Int32(3),
                    fx.Int32(16 * n_atom) + lane_mod16,
                )
            ]
            frags_init.append(vector.from_elements(T.vec(4, T.f32), [v0, v1, v2, v3]))

    for chunk_idx, state in range(0, NT, init=frags_init):
        frags = [state[i] for i in range_constexpr(16)]
        cidx_i32 = arith.index_cast(T.i32, chunk_idx)
        abs_chunk = cta_nt_base + cidx_i32

        for k_step in range_constexpr(K_STEPS):
            A_frags = []
            for m_atom in range_constexpr(M_ATOMS):
                a_row = abs_chunk * fx.Int32(DK) + fx.Int32(16 * m_atom) + lane_mod16
                a_col = fx.Int32(k_step * 16) + lane_div16 * fx.Int32(4)
                a0 = KT_buf[(a_row, a_col + fx.Int32(0))]
                a1 = KT_buf[(a_row, a_col + fx.Int32(1))]
                a2 = KT_buf[(a_row, a_col + fx.Int32(2))]
                a3 = KT_buf[(a_row, a_col + fx.Int32(3))]
                A_frags.append(
                    vector.bitcast(
                        T.vec(4, T.i16),
                        vector.from_elements(T.vec(4, T.bf16), [a0, a1, a2, a3]),
                    )
                )

            B_frags = []
            for n_atom in range_constexpr(N_ATOMS):
                b_row = abs_chunk * fx.Int32(DV) + fx.Int32(16 * n_atom) + lane_mod16
                b_col = fx.Int32(k_step * 16) + lane_div16 * fx.Int32(4)
                b0 = VT_buf[(b_row, b_col + fx.Int32(0))]
                b1 = VT_buf[(b_row, b_col + fx.Int32(1))]
                b2 = VT_buf[(b_row, b_col + fx.Int32(2))]
                b3 = VT_buf[(b_row, b_col + fx.Int32(3))]
                B_frags.append(
                    vector.bitcast(
                        T.vec(4, T.i16),
                        vector.from_elements(T.vec(4, T.bf16), [b0, b1, b2, b3]),
                    )
                )

            for m_atom in range_constexpr(M_ATOMS):
                for n_atom in range_constexpr(N_ATOMS):
                    idx = m_atom * N_ATOMS + n_atom
                    frags[idx] = rocdl.mfma_f32_16x16x16bf16_1k(
                        T.vec(4, T.f32),
                        [A_frags[m_atom], B_frags[n_atom], frags[idx], 0, 0, 0],
                    )

        results = yield frags

    fC_final = [results[i] for i in range_constexpr(16)]

    for m_atom in range_constexpr(M_ATOMS):
        for n_atom in range_constexpr(N_ATOMS):
            idx = m_atom * N_ATOMS + n_atom
            for i in range_constexpr(4):
                val = vector.extract(
                    fC_final[idx], static_position=[i], dynamic_position=[]
                )
                row = (
                    s_row_base
                    + fx.Int32(16 * m_atom)
                    + lane_div16 * fx.Int32(4)
                    + fx.Int32(i)
                )
                col = fx.Int32(16 * n_atom) + lane_mod16
                Sf_buf[(row, col)] = val


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

    print(
        f"[gpu] {torch.cuda.get_device_name(0)}, NT={NT} H={H} B={B}, {H*B} CTAs × 64 MFMA/chunk"
    )
    launcher(K_T, V_T, Si, Sf, NT, H, B, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

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
        print(f"[PASS] full production: NT={NT} × H={H} × 64 MFMA/chunk × multi-CTA")
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
        med = statistics.median(ts) * 1000
        total = NT * H * B
        print(f"[bench] total {med:.1f} us, {total} chunks parallel")
        print(f"        per-chunk effective: {med/total:.3f} us")
        print(
            f"        FLA chunk_h on MI308X: 11430 us for 32768 chunks = 0.348 us/chunk"
        )
        print(f"        speedup vs FLA chunk_h (MI308X): {11430/med:.2f}×")
    else:
        print("[FAIL]")


if __name__ == "__main__":
    main()
