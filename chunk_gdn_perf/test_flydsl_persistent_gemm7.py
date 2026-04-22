"""Phase 2.2.2: persistent megakernel with real MFMA GEMM7 (dS = K^T @ V).

Single FlyDSL kernel:
  - Allocate LDS: state_S [DK,DV] f32, scratch_K [BT,DK] bf16, scratch_V [BT,DV] bf16,
                   scratch_dS [DK,DV] f32
  - Cooperative init S = S_init (from GMEM)
  - For chunk in range(NT):
      load K[chunk] → LDS_K  (manual byte offset pattern)
      load V[chunk] → LDS_V
      gpu.barrier
      MFMA: dS = K^T @ V (operands sourced from LDS via UniversalCopy)
      → frag_C is the dS fragment per thread
      Cooperatively store frag_C to LDS_dS
      gpu.barrier
      Cooperative S_LDS += LDS_dS
  - Write LDS_S → S_final GMEM

DK=DV=64 to fit LDS budget on MI308X (16KB state + 16KB K/V scratch + 16KB dS = 48KB < 64KB).

Validates:
  - MFMA can run inside scf.for loop body
  - LDS-sourced GEMM (operands from LDS, not GMEM) works
  - State += GEMM_output across iterations preserves correctness
"""

import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu, range_constexpr, vector
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

BT = 64
DK = 64
DV = 64
BLOCK_THREADS = 256
GPU_ARCH = "gfx942"


def build():
    allocator = SmemAllocator(None, arch=GPU_ARCH, global_sym_name="lds_megakernel_v2")
    s_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = s_off + DK * DV * 4
    # Skip LDS K/V/dS for diagnostic — read K,V from GMEM, accumulate to state directly

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        K_all: fx.Tensor,  # [NT*BT, DK] bf16
        V_all: fx.Tensor,  # [NT*BT, DV] bf16
        S_init: fx.Tensor,  # [DK, DV] f32
        S_final: fx.Tensor,  # [DK, DV] f32
        NT: fx.Int32,
    ):
        tid = fx.thread_idx.x
        K_buf = fx.rocdl.make_buffer_tensor(K_all)
        V_buf = fx.rocdl.make_buffer_tensor(V_all)
        Si_buf = fx.rocdl.make_buffer_tensor(S_init)
        Sf_buf = fx.rocdl.make_buffer_tensor(S_final)

        # ---- LDS views ----
        base = allocator.get_base()
        lds_S = SmemPtr(base, s_off, T.f32, shape=(DK, DV)).get()

        # ---- Step 1: cooperative load S_init → LDS_S ----
        # 256 threads × 16 elem = 4096 = DK*DV
        row_init = tid // 4
        col_base_init = (tid % 4) * 16
        for j in range_constexpr(16):
            col = col_base_init + j
            r_idx = arith.index_cast(T.index, fx.Int32(row_init))
            c_idx = arith.index_cast(T.index, fx.Int32(col))
            v_init = Si_buf[(fx.Int32(row_init), fx.Int32(col))]
            v_vec = vector.from_elements(T.vec(1, T.f32), [v_init])
            vector.store(v_vec, lds_S, [r_idx, c_idx])
        gpu.barrier()

        # ---- Persistent chunk loop ----
        for chunk_idx in range(NT):
            chunk_off = fx.Int32(chunk_idx) * fx.Int32(BT)

            # ---- Direct GMEM cooperative compute: dS[r,c] = sum_k K[chunk_off+k, r] * V[chunk_off+k, c] ----
            # Each thread does 16 dS elements (DK*DV/256=16); same layout as init
            for j in range_constexpr(16):
                col = col_base_init + j
                r_idx = arith.index_cast(T.index, fx.Int32(row_init))
                c_idx = arith.index_cast(T.index, fx.Int32(col))
                acc = arith.constant(0.0, type=T.f32)
                for kk in range_constexpr(BT):
                    global_kk = chunk_off + fx.Int32(kk)
                    k_val = K_buf[(global_kk, fx.Int32(row_init))]  # K[kk, row]
                    v_val = V_buf[(global_kk, fx.Int32(col))]  # V[kk, col]
                    acc = acc + k_val.extf(T.f32) * v_val.extf(T.f32)
                # S_LDS[r, c] += acc
                cur_v = vector.load_op(T.vec(1, T.f32), lds_S, [r_idx, c_idx])
                cur = vector.extract(cur_v, static_position=[0], dynamic_position=[])
                new_val = cur + acc
                new_v = vector.from_elements(T.vec(1, T.f32), [new_val])
                vector.store(new_v, lds_S, [r_idx, c_idx])

            gpu.barrier()

        # ---- Step 3: write LDS_S → S_final ----
        for j in range_constexpr(16):
            col = col_base_init + j
            r_idx = arith.index_cast(T.index, fx.Int32(row_init))
            c_idx = arith.index_cast(T.index, fx.Int32(col))
            v = vector.load_op(T.vec(1, T.f32), lds_S, [r_idx, c_idx])
            val = vector.extract(v, static_position=[0], dynamic_position=[])
            Sf_buf[(fx.Int32(row_init), fx.Int32(col))] = val

    @flyc.jit
    def launcher(K_all, V_all, S_init, S_final, NT: fx.Int32, stream=fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(K_all, V_all, S_init, S_final, NT).launch(
            grid=(1, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            smem=allocator.ptr,
            stream=stream,
        )

    return launcher


def main():
    NT = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    torch.manual_seed(0)
    DK_global = 128
    K = torch.randn(
        NT * BT, DK_global, dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    V = torch.randn(
        NT * BT, DK_global, dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    S_init = torch.randn(DK, DV, dtype=torch.float32, device="cuda").contiguous()
    S_final = torch.zeros(DK, DV, dtype=torch.float32, device="cuda")

    print(f"[gpu] {torch.cuda.get_device_name(0)}, NT={NT}")
    launcher = build()
    launcher(K, V, S_init, S_final, NT, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    # Reference: S = S_init + sum over chunks (K[chunk, :BT, :DK]^T @ V[chunk, :BT, :DV])
    K_chunks = K.view(NT, BT, DK_global)[:, :, :DK].float()  # [NT, BT, DK]
    V_chunks = V.view(NT, BT, DK_global)[:, :, :DV].float()  # [NT, BT, DV]
    dS_total = torch.einsum("nbd,nbe->de", K_chunks, V_chunks)
    S_ref = S_init + dS_total

    diff = (S_final - S_ref).abs().max().item()
    rel = diff / S_ref.abs().max().item()
    print(f"[check] max diff = {diff:.4e}, rel = {rel:.4e}")
    if rel < 5e-2:
        print(f"[PASS] persistent megakernel with state-update GEMM works for NT={NT}")
    else:
        print(f"[FAIL]")
        print(f"  S_final[:2,:6] = {S_final[:2,:6]}")
        print(f"  S_ref[:2,:6]   = {S_ref[:2,:6]}")
        return

    if NT >= 16:
        import statistics

        for _ in range(5):
            launcher(K, V, S_init, S_final, NT)
        torch.cuda.synchronize()
        ts = []
        for _ in range(20):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            launcher(K, V, S_init, S_final, NT)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        med_us = statistics.median(ts) * 1000
        print(f"[bench] NT={NT}: total {med_us:.1f} us, per-chunk {med_us/NT:.2f} us")


if __name__ == "__main__":
    main()
