"""Phase 2.1.4: ALU epilogue (mask + beta scale) FlyDSL kernel.

Pipeline for chunk_gdn pre-inverse stage:
  1. (already validated 2.1.2) kk = K @ K^T  [BT, BT] fp32
  2. (this step) M = -kk * beta_row * strict_lower_tri_mask  [BT, BT] fp32
     where beta_row[i] is per-row scalar broadcast across columns

Reference: rtp_llm/.../fla/test/test_chunk_prefill.py:107
    attn = -((k_beta @ k^T) * L_mask).masked_fill(mask, 0)
For MVP we drop L_mask (decay) so beta_row substitutes for k_beta scaling.
Strict lower triangle = mask[i,j]=1 if j<i else 0; diagonal cleared too (the +I
comes later as part of the inverse step).

Single CTA, BT=64.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.nn.functional as F

BT = 64


@flyc.kernel
def alu_mask_beta_kernel(
    kk_in: fx.Tensor,  # [BT, BT] fp32
    beta: fx.Tensor,  # [BT]      fp32 (per-row scalar)
    M_out: fx.Tensor,  # [BT, BT] fp32
):
    tid = fx.thread_idx.x  # [0, 256)

    # 256 threads, 64 rows × 64 cols = 4096 elem -> 16 elem/thread
    # Layout: row = tid // 4, col_base = (tid % 4) * 16
    row = tid // 4
    col_base = (tid % 4) * 16

    kk_buf = fx.rocdl.make_buffer_tensor(kk_in)
    beta_buf = fx.rocdl.make_buffer_tensor(beta)
    M_buf = fx.rocdl.make_buffer_tensor(M_out)

    beta_row = beta_buf[(row,)]
    zero_f32 = fx.Float32(0.0)

    for j in fx.range_constexpr(16):
        col = col_base + j
        kk_val = kk_buf[(row, col)]
        # M = -kk * beta if col < row else 0
        prod = fx.arith.negf(kk_val * beta_row)
        masked = fx.arith.select(col < row, prod, zero_f32)
        M_buf[(row, col)] = masked


@flyc.jit
def alu_mask_beta(kk_in, beta, M_out, stream=fx.Stream(None)):
    alu_mask_beta_kernel(kk_in, beta, M_out).launch(
        grid=(1, 1, 1), block=(256, 1, 1), stream=stream
    )


def main():
    torch.manual_seed(0)

    # Synthetic kk + beta (matching what GEMM1 would produce on real K)
    K_t = F.normalize(
        torch.randn(BT, 128, dtype=torch.float32, device="cuda"), p=2, dim=-1
    ).to(torch.bfloat16)
    kk = (K_t.float() @ K_t.float().T).contiguous()
    beta = torch.rand(BT, dtype=torch.float32, device="cuda").sigmoid()
    M_out = torch.zeros(BT, BT, dtype=torch.float32, device="cuda")

    print(f"[setup] BT={BT} dtype=fp32")
    print(f"[gpu]   {torch.cuda.get_device_name(0)}")
    print("[run]   first call (JIT compile)...")
    alu_mask_beta(kk, beta, M_out, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    # torch reference
    mask = torch.tril(torch.ones(BT, BT, device="cuda"), diagonal=-1)  # strict lower
    M_ref = (-kk * beta[:, None]) * mask

    diff = (M_out - M_ref).abs().max().item()
    rel = diff / (M_ref.abs().max().item() + 1e-9)
    print(f"[check] max abs diff = {diff:.4e}, rel = {rel:.4e}")
    if rel < 1e-5:
        print("[PASS] ALU mask+beta kernel correct on MI308X.")
    else:
        print(f"[FAIL]")
        print(f"  M_out[:4,:6] = {M_out[:4,:6]}")
        print(f"  M_ref[:4,:6] = {M_ref[:4,:6]}")


if __name__ == "__main__":
    main()
