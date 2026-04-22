"""Phase 3 MVP: hierarchical inverse stage 1 (Gauss-Jordan 8x8 via warp shuffle).

Single-stage inverse of an 8x8 unit lower-triangular block using warp shuffle.
64-lane wave handles 8 independent 8x8 blocks in parallel (block_id = lane // 8).
Uses ds_bpermute for pivot broadcast (replaces NV shuffle_sync).

This is Phase 1.3 Stage 1 from the checkpoint, implemented as a standalone FlyDSL
kernel to verify the inverse algorithm + AMD warp shuffle pattern.

Input:  M [8, 8] fp16 (strict lower triangular, diagonal = 0)
Output: (I + M)^{-1} [8, 8] fp16 (lower triangular)
"""

import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T

N = 8  # block size


@flyc.kernel(known_block_size=[64, 1, 1])
def inverse_stage1_kernel(M: fx.Tensor, out: fx.Tensor):
    """Gauss-Jordan invert N x N unit lower-triangular block.
    Each lane owns one row of one of 8 blocks (64 lanes / 8 lanes per block = 8 blocks).
    For standalone single-block test: only block_id 0 does real work.
    """
    tid = fx.thread_idx.x
    # For single 8x8 block, use only first 8 lanes
    row_in_blk = tid % 8  # 0..7

    M_buf = fx.rocdl.make_buffer_tensor(M)
    out_buf = fx.rocdl.make_buffer_tensor(out)

    # Load row[row_in_blk] into registers: 8 fp32 values
    # Initialize as identity + M: diagonal = 1.0, off-diag = M
    row = [arith.constant(0.0, type=T.f32) for _ in range(N)]
    # Only lanes 0..7 read (rest ignore for single-block test)
    for i in range_constexpr(N):
        m_val = M_buf[(fx.Int32(row_in_blk), fx.Int32(i))]
        # f16 → f32
        m_f32 = m_val.extf(T.f32)
        row[i] = m_f32
    # Diagonal: if i == row_in_blk, set to 1.0
    for i in range_constexpr(N):
        one_f32 = arith.constant(1.0, type=T.f32)
        is_diag = arith.cmpi(arith.CmpIPredicate.eq, fx.Int32(row_in_blk), fx.Int32(i))
        row[i] = arith.select(is_diag, one_f32, row[i])

    # Gauss-Jordan: N-1 pivot steps
    # For src_row in 0..N-1:
    #   row_scale = -row[src_row]
    #   for i in 0..src_row-1: row[i] += row_scale * row_src[i]
    #   if row > src_row: row[src_row] = row_scale
    # Broadcast row_src[i] using ds_bpermute from lane `src_row` within the block
    # For single block at lanes 0..7: src_lane = src_row (0..6)

    for src_row in range_constexpr(N - 1):
        row_scale = arith.negf(row[src_row])
        for i in range_constexpr(N):
            # Only update if i < src_row
            if_update = i < src_row  # Python bool at unroll time
            if if_update:
                # Broadcast row[i] value from lane `src_row`
                # ds_bpermute: dst_lane reads data FROM lane stored at idx[dst_lane]
                # We want lane X to get row[i] value from lane src_row
                # Index = src_row * 4 (ds_bpermute uses byte-granularity)
                idx_i32 = fx.Int32(src_row * 4)
                # Cast row[i] (f32) to i32 for ds_bpermute, then back
                row_i_i32 = vector.bitcast(T.i32, row[i])
                shfl_val_i32 = rocdl.ds_bpermute(T.i32, idx_i32, row_i_i32)
                shfl_val = vector.bitcast(T.f32, shfl_val_i32)
                # row[i] += row_scale * shfl_val if row_in_blk > src_row
                updated = row[i] + row_scale * shfl_val
                is_below = arith.cmpi(
                    arith.CmpIPredicate.ugt, fx.Int32(row_in_blk), fx.Int32(src_row)
                )
                row[i] = arith.select(is_below, updated, row[i])
        # row[src_row] = row_scale if row_in_blk > src_row
        is_below = arith.cmpi(
            arith.CmpIPredicate.ugt, fx.Int32(row_in_blk), fx.Int32(src_row)
        )
        row[src_row] = arith.select(is_below, row_scale, row[src_row])

    # Write row back (as fp16)
    for i in range_constexpr(N):
        val_f16 = row[i].truncf(T.f16)
        out_buf[(fx.Int32(row_in_blk), fx.Int32(i))] = val_f16


@flyc.jit
def launcher(M, out, stream=fx.Stream(None)):
    inverse_stage1_kernel(M, out).launch(
        grid=(1, 1, 1), block=(64, 1, 1), stream=stream
    )


def main():
    torch.manual_seed(0)
    # Generate random strict lower triangular M (small values for numerical stability)
    M_fp32 = torch.tril(torch.randn(N, N) * 0.1, diagonal=-1)
    M_f16 = M_fp32.to(torch.float16).cuda()
    out = torch.zeros(N, N, dtype=torch.float16, device="cuda")

    print(f"[gpu] {torch.cuda.get_device_name(0)}")
    print(f"[setup] N={N}, Gauss-Jordan inverse of (I + strict_lower_tri M)")

    launcher(M_f16, out, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    # Reference: torch.linalg.inv(I + M)
    I = torch.eye(N, dtype=torch.float32)
    ref = torch.linalg.inv(I + M_fp32).to(torch.float16).cuda()

    diff = (out.float() - ref.float()).abs().max().item()
    rel = diff / (ref.float().abs().max().item() + 1e-9)
    print(f"[check] max diff = {diff:.4e}, rel = {rel:.4e}")
    print(f"[debug] M sample:")
    print(M_fp32[:4, :4])
    print(f"[debug] out sample:")
    print(out[:4, :4].float().cpu())
    print(f"[debug] ref sample:")
    print(ref[:4, :4].float().cpu())
    if rel < 0.05:
        print(
            f"[PASS] Phase 3 stage-1 hierarchical inverse (Gauss-Jordan 8x8) works on CDNA3"
        )
    else:
        print("[FAIL] inverse algorithm buggy")


if __name__ == "__main__":
    main()
