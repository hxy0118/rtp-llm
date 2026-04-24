"""Comprehensive benchmark: TP1/TP2 × T=4096..131072."""
import torch, torch.nn.functional as F, triton, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_standalone import (
    chunk_local_cumsum, fused_kkt_solve, recompute_w_u_fwd,
    fwd_h as fwd_h_tuned, fwd_o as fwd_o_tuned,
    new_pipeline, old_pipeline,
    prepare_chunk_indices, prepare_chunk_offsets, RCP_LN2,
)
from bench_gluon import gluon_fwd_h, best_pipeline

def bench(fn, n=20, w=5):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return int(sum(ts) / len(ts))

B = 1
DK, DV = 128, 128
BT = 64

configs = [
    # (Hg, H, label)
    (16, 64, "TP1"),
    (8, 32, "TP2"),
]

seq_lens_tp2 = [4096, 8192, 16384, 32768, 65536, 131072]
seq_lens_tp1 = [4096, 8192, 16384, 32768]

print("=" * 80)
print("MI355X chunk-GDN Comprehensive Benchmark")
print("B=%d DK=%d DV=%d BT=%d" % (B, DK, DV, BT))
print("Production mode: varlen=True, output_final_state=True")
print("=" * 80)

for Hg, H, tp_label in configs:
    print("\n" + "=" * 80)
    print("[%s] Hg=%d H=%d" % (tp_label, Hg, H))
    print("=" * 80)
    SCALE = DK ** -0.5

    # Header
    print("\n%-8s | %10s %10s %10s | %10s %10s %8s | %10s %10s %8s" % (
        "T", "orig(us)", "new(us)", "best(us)",
        "fwd_h_tri", "fwd_h_gl", "speedup",
        "fwd_o_tri", "fwd_o_gl", "speedup"))
    print("-" * 120)

    cur_seq_lens = seq_lens_tp1 if tp_label == "TP1" else seq_lens_tp2
    for T in cur_seq_lens:
        torch.manual_seed(42)
        q = torch.randn(B, T, Hg, DK, device="cuda", dtype=torch.bfloat16)
        k = F.normalize(torch.randn(B, T, Hg, DK, device="cuda", dtype=torch.bfloat16), p=2, dim=-1)
        v = torch.randn(B, T, H, DV, device="cuda", dtype=torch.bfloat16)
        g = F.logsigmoid(torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16))
        beta = torch.rand(B, T, H, device="cuda", dtype=torch.bfloat16).sigmoid()
        cu = torch.tensor([0, T], device="cuda", dtype=torch.long)
        ci = prepare_chunk_indices(cu, BT)
        co = prepare_chunk_offsets(cu, BT)
        pipe_kw = dict(output_final_state=True, cu_seqlens=cu, chunk_indices=ci, chunk_offsets=co)

        # Prepare exp2 data for isolated kernel tests
        gc2 = chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2, cu_seqlens=cu, chunk_indices=ci)
        A2 = fused_kkt_solve(k, gc2, beta, use_exp2=True)
        w2, u2 = recompute_w_u_fwd(k, v, beta, A2, gc2, use_exp2=True)
        h2, vn2, _ = fwd_h_tuned(k, w2, u2, gc2, use_exp2=True,
                                   output_final_state=True, cu_seqlens=cu, chunk_offsets=co)
        torch.cuda.synchronize()

        try:
            # Pipeline benchmarks
            a_old = bench(lambda: old_pipeline(q, k, v, g, beta, SCALE, **pipe_kw))
            torch.cuda.empty_cache()
            a_new = bench(lambda: new_pipeline(q, k, v, g, beta, SCALE, **pipe_kw))
            torch.cuda.empty_cache()
            a_best = bench(lambda: best_pipeline(q, k, v, g, beta, SCALE, **pipe_kw))
            torch.cuda.empty_cache()

            # Isolated fwd_h
            a_h_tri = bench(lambda: fwd_h_tuned(k, w2, u2, gc2, use_exp2=True,
                                                 output_final_state=True, cu_seqlens=cu, chunk_offsets=co))
            a_h_gl = bench(lambda: gluon_fwd_h(k, w2, u2, gc2, use_exp2=True,
                                                output_final_state=True, cu_seqlens=cu, chunk_offsets=co, BV=16))

            # Isolated fwd_o
            a_o_tri = bench(lambda: fwd_o_tuned(q, k, vn2, h2, gc2, SCALE, use_exp2=True))

            h_speedup = a_h_tri / a_h_gl
            print("%-8d | %10d %10d %10d | %10d %10d %7.3fx | %10d %10s %8s" % (
                T, a_old, a_new, a_best,
                a_h_tri, a_h_gl, h_speedup,
                a_o_tri, "—", "—"))

        except Exception as e:
            print("%-8d | ERROR: %s" % (T, str(e)[:80]))

        # Cleanup
        del q, k, v, g, beta, cu, ci, co, gc2, A2, w2, u2, h2, vn2
        torch.cuda.empty_cache()

print("\nDone.")
