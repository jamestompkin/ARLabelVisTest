"""Where does Python's rdg_admm spend its wall-clock?"""
import sys, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arlabelvis.rgd.mesh_ops import build_mesh_ops
from arlabelvis.rgd.admm import rdg_admm
from arlabelvis.off import read_off

V, F = read_off("external/matlab_rgd/e2e_cielab_neural.off")
print(f"mesh: {len(V)} verts, {len(F)} faces")

# Warm up + single-source time with high-res history
ops = build_mesh_ops(V, F)
# Warm-up (JIT-ish caches in scipy)
_ = rdg_admm(ops, x0=0, alpha_hat=0.25)

# Instrumented copy of the inner loop for stage-level timing.
import cProfile, pstats, io
pr = cProfile.Profile()
pr.enable()
for src in range(20):
    rdg_admm(ops, x0=src, alpha_hat=0.25)
pr.disable()
st = pstats.Stats(pr).sort_stats("cumulative")
stream = io.StringIO()
st.stream = stream
st.print_stats(25)
print(stream.getvalue())

# Fine-grained breakdown of a single rdg_admm call
from arlabelvis.rgd.mesh_ops import MeshOps
from scipy.sparse import diags
from scipy.sparse.linalg import factorized

def time_stage(n=5):
    ms = []
    for src in range(n):
        phases = {}
        t0 = time.perf_counter()
        # -- eliminate x0 --
        keep = np.ones(ops.nv, dtype=bool); keep[src] = False
        Ww_p = ops.Ww[keep][:, keep].tocsc()
        G_p = ops.G[:, keep].tocsr()
        G_pt = G_p.T.tocsr()
        ta_tiled = np.tile(ops.ta, 3)
        div_p = (G_pt @ diags(ta_tiled)).tocsr()
        t1 = time.perf_counter(); phases["eliminate_x0"] = t1-t0
        # -- cholesky factor --
        solve = factorized(Ww_p)
        t2 = time.perf_counter(); phases["factorize"] = t2-t1
        # -- ADMM iters --
        u, hist = rdg_admm(ops, x0=src, alpha_hat=0.25)
        t3 = time.perf_counter(); phases["full_rdg_admm"] = t3-t0
        phases["admm_iters_approx"] = phases["full_rdg_admm"] - phases["eliminate_x0"] - phases["factorize"]
        phases["iters"] = hist.iters
        ms.append(phases)
    for k in ["eliminate_x0", "factorize", "admm_iters_approx", "full_rdg_admm"]:
        mean = np.mean([m[k] for m in ms]) * 1000
        print(f"  {k:25s}  {mean:7.1f} ms")
    print(f"  mean iters: {np.mean([m['iters'] for m in ms]):.1f}")

print("\n--- fine-grained stage timing (5 sources) ---")
time_stage(5)
