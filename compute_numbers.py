"""Every number quoted in the skew Brownian motion note. Run: python3 compute_numbers.py"""
import json, time
import numpy as np
import skew as S

A, T = S.HOUSE["alpha"], S.HOUSE["t"]
out, t0 = {"house": dict(S.HOUSE)}, time.time()
out["exact"] = {"p_positive": S.p_positive(A), "mean": S.mean(T, A),
                "second_moment": S.second_moment(T), "occupation": S.mean_occupation(T, A)}

def se(x): return float(np.std(x, ddof=1) / np.sqrt(len(x)))

print("main run ...", flush=True)
r = S.walk(S.HOUSE["n_paths"], h=S.HOUSE["h"], seed=20260923)
x, occ, zt = r["x"], r["occupation"], r["zero_time"]
pos = (x > 0).astype(float)
p_zero = float(np.mean(x == 0))
# The lattice has an atom at the origin that the continuum has not: it is O(h) and it
# belongs to NEITHER side. Conditioning on X != 0 removes it; and the time spent AT the
# origin is apportioned by alpha, because an excursion leaves upward with that chance.
pos_corr = pos / (1 - p_zero)
occ_corr = occ + A * zt
out["main"] = dict(n_paths=len(x), h=r["h"], steps=r["steps"],
    p_positive=float(pos.mean()), p_positive_se=se(pos),
    p_zero=p_zero, p_positive_corrected=float(pos_corr.mean()), p_corr_se=se(pos_corr),
    zero_time=float(zt.mean()),
    occupation_corrected=float(occ_corr.mean()), occ_corr_se=se(occ_corr),
    mean=float(x.mean()), mean_se=se(x),
    second=float((x**2).mean()), second_se=se(x**2),
    occupation=float(occ.mean()), occupation_se=se(occ))
for k, ex in (("p_positive", A), ("mean", S.mean(T, A)),
              ("second", 1.0), ("occupation", A * T)):
    out["main"][k + "_dev_se"] = (out["main"][k] - ex) / out["main"][k + "_se"]
out["main"]["p_corr_dev_se"] = (out["main"]["p_positive_corrected"] - A) / out["main"]["p_corr_se"]
out["main"]["occ_corr_dev_se"] = (out["main"]["occupation_corrected"] - A * T) / out["main"]["occ_corr_se"]
print(f"  P(X>0) {out['main']['p_positive']:.4f}  E[X] {out['main']['mean']:.4f}", flush=True)

print("convergence in h ...", flush=True)
conv = []
for h in (0.04, 0.02, 0.01, 0.005):
    rr = S.walk(40000, h=h, seed=11)
    p = (rr["x"] > 0).astype(float)
    pz = float(np.mean(rr["x"] == 0))
    pc = p / (1 - pz)
    oc = rr["occupation"] + A * rr["zero_time"]
    conv.append(dict(h=h, steps=rr["steps"], parity="even" if rr["steps"] % 2 == 0 else "odd",
                     p=float(p.mean()), se=se(p), p_zero=pz,
                     p_corr=float(pc.mean()), err=float(pc.mean() - A),
                     occ=float(rr["occupation"].mean()), occ_corr=float(oc.mean()),
                     occ_err=float(oc.mean() - A * T)))
    print(f"  h={h}: P={conv[-1]['p']:.4f} err {conv[-1]['err']:+.4f}", flush=True)
out["convergence"] = conv

print("alpha sweep ...", flush=True)
sweep = []
for a in (0.1, 0.25, 0.5, 0.75, 0.9):
    rr = S.walk(40000, alpha=a, h=0.01, seed=7)
    p = (rr["x"] > 0).astype(float)
    pz = float(np.mean(rr["x"] == 0))
    sweep.append(dict(alpha=a, p=float(p.mean()), se=se(p), p_zero=pz,
                      p_corr=float((p / (1 - pz)).mean()),
                      occ_corr=float((rr["occupation"] + a * rr["zero_time"]).mean()),
                      mean=float(rr["x"].mean()), mean_exact=S.mean(T, a),
                      occ=float(rr["occupation"].mean()), occ_exact=a * T))
    print(f"  a={a}: P={sweep[-1]['p']:.4f}", flush=True)
out["alpha_sweep"] = sweep

print("density fit ...", flush=True)
edges = np.linspace(-3.5, 3.5, 57)
hist, _ = np.histogram(x, bins=edges, density=True)
mid = 0.5 * (edges[1:] + edges[:-1])
th = S.density(mid, T, A)
out["density"] = dict(max_abs_err=float(np.max(np.abs(hist - th))),
                      rel_err_at_peak=float(abs(hist[np.argmax(th)] - th.max()) / th.max()),
                      ratio_pos_neg=float(np.mean(hist[mid > 0] / th[mid > 0])),
                      ratio_neg=float(np.mean(hist[mid < 0] / th[mid < 0])))
out["runtime_s"] = time.time() - t0
json.dump(out, open("numbers.json", "w"), indent=1)
print(f"\nwrote numbers.json in {out['runtime_s']:.0f}s")
