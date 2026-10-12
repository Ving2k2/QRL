"""The output head and the factorial at 120 episodes (Sections 4 and 5.1).

Cell means and intervals, the three-way ANOVA with Holm over its seven terms, the robustness checks of the two
interactions (Freedman-Lane permutation, aligned rank transform, Welch contrast, seed as a block, score windows),
simple effects, minimum detectable effects, five-seed subsets, the initial networks (output range, input
sensitivity, the Heisenberg images of the readouts), Acrobot-v1, the capacity-matched MLPs and the MLP frontier
at 120 episodes, and the seed-0 reproduction of the April 2026 archive.

Reads results/cartpole_120, mlp_matched_120, mlp_frontier_120, acrobot_head_ablation, initialization and
archive_2026_04. -> analysis/factorial_120.txt
"""
import glob
import io
import itertools
import json
import os

import numpy as np
from scipy import stats

from qrl import data
from qrl.blindness import image as _img, qubits as _qb

_out = io.StringIO()


def print(*args, sep=" ", end="\n", **_):          # every line goes to the output file
    _out.write(sep.join(str(a) for a in args) + end)


RESULTS = data.RESULTS


def load(sub):
    out = {}
    for f in glob.glob(os.path.join(RESULTS, sub, "*.json")):
        j = json.load(open(f))
        if "label" in j:
            out.setdefault(j["label"], []).append((j["seed"], np.mean(j["history"][-20:]), j["params"]))
        else:
            for k, v in j.items():
                for r in v:
                    out.setdefault(k, []).append((r["seed"], np.mean(r["history"][-20:]), r["params"]))
    return {k: (np.array([x[1] for x in sorted(v)]), sorted(v)[0][2]) for k, v in out.items()}


Q = load("cartpole_120")
C = load("mlp_matched_120")
AC = {}
_ac = {}
for _p in glob.glob(os.path.join(RESULTS, "acrobot_head_ablation", "*.json")):
    _d = json.load(open(_p))
    _ac.setdefault(_d["label"], []).append(_d)
for k, v in _ac.items():
    AC[k] = (np.array([np.mean(r["history"][-20:]) for r in sorted(v, key=lambda r: r["seed"])]),
             v[0]["params"])


def welch(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return p, (a.mean() - b.mean()) / sp


def fmt(a):
    return "%.1f +- %.1f" % (a.mean(), a.std(ddof=1))


print("=" * 72)
print("TABLE: CartPole arms  (cartpole_120)")
print("=" * 72)
print("%-26s %6s %4s %18s" % ("variant", "params", "n", "mean +- sample sd"))
for k in ["Naive QDQN", "QDQN-R no-scale", "QDQN-R no-reup", "QDQN-R (full)",
          "QDQN-R full hi-scale", "QDQN-R global lo-scale", "QDQN-R global-obs"]:
    a, p = Q[k]
    print("%-26s %6d %4d %18s" % (k, p, len(a), fmt(a)))

print()
print("=" * 72)
print("TABLE: capacity-matched MLPs  (mlp_matched_120)")
print("=" * 72)
for k in sorted(C, key=lambda k: (C[k][1], k)):
    a, p = C[k]
    print("  %-34s %5d params  n=%d  %s" % (k, p, len(a), fmt(a)))

print()
print("=" * 72)
print("2x2  (all four cells, n=15)")
print("=" * 72)
cell = {("Zi", "LO"): Q["QDQN-R (full)"][0][:15], ("Zi", "HI"): Q["QDQN-R full hi-scale"][0][:15],
        ("ZZ", "LO"): Q["QDQN-R global lo-scale"][0][:15], ("ZZ", "HI"): Q["QDQN-R global-obs"][0][:15]}
for o, nm in [("Zi", "per-qubit <Zi>"), ("ZZ", "Z0Z2,Z1Z3")]:
    print("  %-16s init3.2: %-18s init30: %-18s" % (nm, fmt(cell[(o, "LO")]), fmt(cell[(o, "HI")])))
y = np.concatenate([cell[k] for k in [("Zi", "LO"), ("Zi", "HI"), ("ZZ", "LO"), ("ZZ", "HI")]])
obs = np.repeat([0, 0, 1, 1], 15); ini = np.repeat([0, 1, 0, 1], 15); gm = y.mean()
SSo = 30 * sum((y[obs == g].mean() - gm) ** 2 for g in (0, 1))
SSi = 30 * sum((y[ini == g].mean() - gm) ** 2 for g in (0, 1))
SSc = sum(15 * (cell[k].mean() - gm) ** 2 for k in cell)
SSw = sum(((cell[k] - cell[k].mean()) ** 2).sum() for k in cell)
MSw = SSw / 56
print("  ANOVA (df=1,56):")
for lab, SS in [("observable", SSo), ("init scale", SSi), ("interaction", SSc - SSo - SSi)]:
    F = SS / MSw
    print("    %-12s F=%5.2f  p=%.4f  eta2p=%.3f" % (lab, F, 1 - stats.f.cdf(F, 1, 56), SS / (SS + SSw)))

print()
print("=" * 72)
print("HEADLINE CONTRASTS")
print("=" * 72)
G, F, NS, NR, NA = (Q["QDQN-R global-obs"][0], Q["QDQN-R (full)"][0], Q["QDQN-R no-scale"][0],
                    Q["QDQN-R no-reup"][0], Q["Naive QDQN"][0])
c64 = C["Classical MLP 2x64 (lr=0.001)"][0]
m12 = C["Classical MLP 1x12 (lr=0.005)"][0]
best_cls = max(C.values(), key=lambda t: t[0].mean())
for lab, a, b in [("F1  full vs no-scale", F, NS),
                  ("F1  global-obs vs no-scale", G, NS),
                  ("    no-scale vs naive", NS, NA),
                  ("F2  full vs no-reup", F, NR),
                  ("    global-obs vs classical 2x64", G, c64),
                  ("    global-obs vs matched 1x12", G, m12),
                  ("    full vs classical 2x64", F, c64)]:
    p, d = welch(a, b)
    print("  %-34s dmean=%+8.1f  p=%.5f  d=%+6.2f" % (lab, a.mean() - b.mean(), p, d))
print()
print("  ratio global-obs / classical2x64 = %.2fx  at %.1fx fewer params"
      % (G.mean() / c64.mean(), 4610 / 84))
print("  ratio global-obs / matched 1x12  = %.2fx  (84 vs 86 params)" % (G.mean() / m12.mean()))
print("  best classical at ANY width      = %.1f (%s)" % (best_cls[0].mean(), best_cls[1]))
print("  seed sensitivity: global-obs n=5 %.1f -> n=15 %.1f ; full n=5 %.1f -> n=15 %.1f"
      % (G[:5].mean(), G.mean(), F[:5].mean(), F.mean()))

print()
print("=" * 72)
print("ACROBOT  (acrobot_head_ablation)")
print("=" * 72)
for k in sorted(AC):
    a, p = AC[k]
    print("  %-30s %5d params  n=%d  %s" % (k, p, len(a), fmt(a)))
for lab, a, b in [("QDQN-R vs classical", AC["QDQN-R (Acrobot)"][0], AC["Classical DQN (Acrobot)"][0]),
                  ("QDQN-R vs no-scale", AC["QDQN-R (Acrobot)"][0], AC["QDQN-R no-scale (Acrobot)"][0]),
                  ("QDQN-R vs no-reup", AC["QDQN-R (Acrobot)"][0], AC["QDQN-R no-reup (Acrobot)"][0])]:
    p, d = welch(a, b)
    print("  %-24s dmean=%+8.1f  p=%.5f  d=%+7.2f" % (lab, a.mean() - b.mean(), p, d))


# ---------------------------------------------------------------------------
# Everything below is deterministic: the resampling seed is fixed so the intervals and permutation p-values
# regenerate exactly.
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(20260924)


def tci(a):
    """t-based 95% CI of the mean. Used instead of the percentile bootstrap,
    which under-covers at n = 5."""
    if a.std(ddof=1) == 0:
        return a.mean(), a.mean()
    h = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return a.mean() - h, a.mean() + h


def welch_full(a, b):
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    diff = a.mean() - b.mean()
    h = stats.t.ppf(0.975, df) * np.sqrt(va + vb)
    p, d = welch(a, b)
    return diff, p, d, diff - h, diff + h


print()
print("=" * 72)
print("t-BASED 95% CIs (n-1 sd)")
print("=" * 72)
for k in ["Naive QDQN", "QDQN-R no-scale", "QDQN-R no-reup", "QDQN-R (full)",
          "QDQN-R full hi-scale", "QDQN-R global lo-scale", "QDQN-R global-obs"]:
    a = Q[k][0]
    print("  %-26s n=%2d  %s   CI [%.1f, %.1f]" % (k, len(a), fmt(a), *tci(a)))
for k in ["Classical MLP 1x12 (lr=0.005)", "Classical MLP 2x64 (lr=0.001)"]:
    a = C[k][0]
    print("  %-34s n=%d  %s   CI [%.1f, %.1f]" % (k, len(a), fmt(a), *tci(a)))
for k in sorted(AC):
    a = AC[k][0]
    print("  %-30s n=%d  %s   CI [%.1f, %.1f]" % (k, len(a), fmt(a), *tci(a)))

print()
print("=" * 72)
print("PAIRWISE FAMILY, Welch, Holm over the family")
print("=" * 72)
GL, FH = Q["QDQN-R global lo-scale"][0], Q["QDQN-R full hi-scale"][0]
fam = [("global-obs vs no-scale", G, NS), ("full vs no-scale", F, NS),
       ("no-reup vs classical 2x64", NR, c64), ("no-reup vs matched 1x12", NR, m12),
       ("global-obs vs matched 1x12", G, m12), ("global-obs vs classical 2x64", G, c64),
       ("global-obs vs no-reup", G, NR), ("lo-scale vs no-reup", GL, NR),
       ("lo-scale vs classical 2x64", GL, c64), ("hi-scale vs classical 2x64", FH, c64),
       ("full vs classical 2x64", F, c64), ("full vs no-reup", F, NR),
       ("no-scale vs naive", NS, NA)]
res = [(lab,) + welch_full(a, b) for lab, a, b in fam]
order = np.argsort([r[2] for r in res]); m = len(res); holm = [0.0] * m; run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (m - rank) * res[i][2])); holm[i] = run
for (lab, diff, p, d, lo, hi), ph in zip(res, holm):
    print("  %-30s diff=%+7.1f  CI [%+7.1f, %+7.1f]  p=%.5f  holm=%.4f  d=%+.2f" % (lab, diff, lo, hi, p, ph, d))

print()
print("=" * 72)
print("2x2 ROBUSTNESS")
print("=" * 72)
yy = np.concatenate([F, FH, GL, G]); nn = len(F)
ob = np.repeat([0, 0, 1, 1], nn); ii = np.repeat([0, 1, 0, 1], nn)
eo = yy[ob == 1].mean() - yy[ob == 0].mean(); ei = yy[ii == 1].mean() - yy[ii == 0].mean()
B = 20000; po = pi_ = 0
for _ in range(B):
    yp = RNG.permutation(yy)
    po += abs(yp[ob == 1].mean() - yp[ob == 0].mean()) >= abs(eo)
    pi_ += abs(yp[ii == 1].mean() - yp[ii == 0].mean()) >= abs(ei)
print("  row effect (observable) = %+.1f   permutation p = %.4f" % (eo, po / B))
print("  column effect (init)    = %+.1f   permutation p = %.4f" % (ei, pi_ / B))
ic = (G.mean() - GL.mean()) - (FH.mean() - F.mean())
bs = [(c[3] - c[2]) - (c[1] - c[0]) for c in
      ([x[RNG.integers(0, len(x), len(x))].mean() for x in (F, FH, GL, G)] for _ in range(10000))]
print("  interaction contrast = %+.1f   bootstrap 95%% CI [%.1f, %.1f]" % (ic, *np.percentile(bs, [2.5, 97.5])))
pa = sorted([(1 - stats.f.cdf(SSo / MSw, 1, 56), "observable"), (1 - stats.f.cdf(SSi / MSw, 1, 56), "init"),
             (1 - stats.f.cdf((SSc - SSo - SSi) / MSw, 1, 56), "interaction")])
print("  Holm over the 3 ANOVA terms: " + ", ".join(
    "%s %s" % (k, "sig" if p <= 0.05 / (3 - r) else "ns") for r, (p, k) in enumerate(pa)))
print("  sd by cell: " + ", ".join("%s %.1f" % (k, v.std(ddof=1)) for k, v in
      sorted({"Zi/3.2": F, "Zi/30": FH, "ZZ/3.2": GL, "ZZ/30": G}.items(), key=lambda kv: -kv[1].std(ddof=1))))
print("  Brown-Forsythe, Zi row vs ZZ row: p = %.3f" % stats.levene(np.r_[F, FH], np.r_[GL, G], center="median").pvalue)
print("  global-obs sd on seeds 0-4 = %.1f" % G[:5].std(ddof=1))

print()
print("=" * 72)
print("2x2 AT FIVE SEEDS (seeds 0-4 of the same runs)")
print("=" * 72)
c5 = [x[:5] for x in (F, FH, GL, G)]; y5 = np.concatenate(c5); g5 = y5.mean()
o5 = np.repeat([0, 0, 1, 1], 5); i5 = np.repeat([0, 1, 0, 1], 5)
S_o = 10 * sum((y5[o5 == g].mean() - g5) ** 2 for g in (0, 1))
S_i = 10 * sum((y5[i5 == g].mean() - g5) ** 2 for g in (0, 1))
S_c = 5 * sum((x.mean() - g5) ** 2 for x in c5); S_w = sum(((x - x.mean()) ** 2).sum() for x in c5)
for lab, SS in [("observable", S_o), ("init", S_i), ("interaction", S_c - S_o - S_i)]:
    Fv = SS / (S_w / 16)
    print("  %-12s F=%5.2f  p=%.4f  eta2p=%.3f" % (lab, Fv, 1 - stats.f.cdf(Fv, 1, 16), SS / (SS + S_w)))

print()
print("=" * 72)
print("EVERY FIVE-SEED SUBSET OF THE FIFTEEN (order-free seed sensitivity)")
print("=" * 72)
for lab, a in [("global-obs", G), ("no-reup", NR), ("full", F)]:
    ms = np.array([a[list(c)].mean() for c in itertools.combinations(range(len(a)), 5)])
    rel = np.abs(ms / a.mean() - 1)
    pr = 100 * (ms < a[:5].mean()).mean()
    print("  %-10s subsets=%d  5-95%% [%.1f, %.1f]  off by >20%%: %.0f%%   seeds 0-4 mean %.1f sits at percentile %.0f"
          % (lab, len(ms), *np.percentile(ms, [5, 95]), 100 * (rel > 0.2).mean(), a[:5].mean(), pr))

print()
print("=" * 72)
print("SEED-0 REPRODUCTION OF THE APRIL 2026 ARCHIVE")
print("=" * 72)
_apr = json.load(open(os.path.join(RESULTS, "archive_2026_04", "results.json")))
_now = {"Naive QDQN": Q["Naive QDQN"][0], "QDQN-R no-scale": NS, "QDQN-R (full)": F,
        "QDQN-R global-obs": G, "Classical DQN": c64}
for k in ["Naive QDQN", "QDQN-R no-scale", "Classical DQN", "QDQN-R (full)", "QDQN-R global-obs"]:
    s0 = [np.mean(r["history"][-20:]) for r in _apr[k] if r["seed"] == 0][0]
    print("  %-18s stored %6.1f  regenerated %6.1f  delta %+6.1f" % (k, s0, _now[k][0], _now[k][0] - s0))

# ---------------------------------------------------------------------------
# The re-uploading-off half of the factorial, making it a
# 2x2x2 of observable x initial output range x re-uploading, n = 15 per cell.
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("2x2x2  observable x initial range x re-uploading (n=15 per cell)")
print("=" * 72)
A1 = {("Zi", "LO", 1): F, ("Zi", "HI", 1): FH, ("ZZ", "LO", 1): GL, ("ZZ", "HI", 1): G,
      ("Zi", "LO", 0): NR, ("Zi", "HI", 0): Q["QDQN-R no-reup hi-scale"][0],
      ("ZZ", "LO", 0): Q["QDQN-R global no-reup lo-scale"][0], ("ZZ", "HI", 0): Q["QDQN-R global no-reup"][0]}
for k in ["QDQN-R no-reup hi-scale", "QDQN-R global no-reup lo-scale", "QDQN-R global no-reup"]:
    a, p = Q[k]
    print("  %-32s params %d  n=%d  %s  CI [%.1f, %.1f]" % (k, p, len(a), fmt(a), *tci(a)))
k8 = list(A1); y8 = np.concatenate([A1[k] for k in k8]); n8 = 15
cO = np.repeat([1 if k[0] == "ZZ" else -1 for k in k8], n8)
cI = np.repeat([1 if k[1] == "HI" else -1 for k in k8], n8)
cR = np.repeat([1 if k[2] == 1 else -1 for k in k8], n8)
X8 = {"O": cO, "I": cI, "R": cR, "OxI": cO * cI, "OxR": cO * cR, "IxR": cI * cR, "OxIxR": cO * cI * cR}
cellid = np.repeat(np.arange(8), n8)


def anova8(yv):
    ssw = sum(((yv[cellid == c] - yv[cellid == c].mean()) ** 2).sum() for c in range(8))
    out = {}
    for t, x in X8.items():
        est = (yv * x).mean(); ss = len(yv) * est ** 2; f = ss / (ssw / (len(yv) - 8))
        out[t] = (2 * est, f, 1 - stats.f.cdf(f, 1, len(yv) - 8), ss / (ss + ssw))
    return out


r8 = anova8(y8)
ordr = sorted(r8, key=lambda t: r8[t][2]); holm8 = {}; run = 0.0
for rank, t in enumerate(ordr):
    run = max(run, min(1.0, (7 - rank) * r8[t][2])); holm8[t] = run
cnt8 = {t: 0 for t in r8}
RNG8 = np.random.default_rng(20260924)
for _ in range(20000):
    rp = anova8(RNG8.permutation(y8))
    for t in r8:
        cnt8[t] += rp[t][1] >= r8[t][1]
rk8 = anova8(stats.rankdata(y8))
print("  three-way ANOVA (df=1,112); effect = mean(+) - mean(-)")
for t in X8:
    e, f, p, e2 = r8[t]
    print("    %-6s effect %+7.1f  F=%6.2f  p=%.4f  holm=%.4f  eta2p=%.3f  perm p=%.4f  rank p=%.4f"
          % (t, e, f, p, holm8[t], e2, cnt8[t] / 20000, rk8[t][2]))
print("  Brown-Forsythe across the 8 cells: p = %.3f" % stats.levene(*[A1[k] for k in k8], center="median").pvalue)
for rr, lab in [(1, "with re-uploading"), (0, "without re-uploading")]:
    zi = np.r_[A1[("Zi", "LO", rr)], A1[("Zi", "HI", rr)]]; zz = np.r_[A1[("ZZ", "LO", rr)], A1[("ZZ", "HI", rr)]]
    lo_ = np.r_[A1[("Zi", "LO", rr)], A1[("ZZ", "LO", rr)]]; hi_ = np.r_[A1[("Zi", "HI", rr)], A1[("ZZ", "HI", rr)]]
    cc = [A1[(o, i, rr)] for o in ("Zi", "ZZ") for i in ("LO", "HI")]; gm2 = np.concatenate(cc).mean()
    ssw2 = sum(((c - c.mean()) ** 2).sum() for c in cc)
    so = 30 * ((zi.mean() - gm2) ** 2 + (zz.mean() - gm2) ** 2); si = 30 * ((lo_.mean() - gm2) ** 2 + (hi_.mean() - gm2) ** 2)
    print("  2x2 %-21s observable %+6.1f F=%.2f p=%.4f | init %+6.1f F=%.2f p=%.4f"
          % (lab, zz.mean() - zi.mean(), so / (ssw2 / 56), 1 - stats.f.cdf(so / (ssw2 / 56), 1, 56),
             hi_.mean() - lo_.mean(), si / (ssw2 / 56), 1 - stats.f.cdf(si / (ssw2 / 56), 1, 56)))
# observable x re-uploading contrast, bootstrap CI and every five-seed subset
oxr = lambda m: (np.mean([m[("ZZ", i, 1)] - m[("Zi", i, 1)] for i in ("LO", "HI")])
                 - np.mean([m[("ZZ", i, 0)] - m[("Zi", i, 0)] for i in ("LO", "HI")]))
bs8 = [oxr({k: A1[k][RNG8.integers(0, 15, 15)].mean() for k in k8}) for _ in range(10000)]
print("  OxR contrast = %+.1f  bootstrap 95%% CI [%.1f, %.1f]"
      % (oxr({k: A1[k].mean() for k in k8}), *np.percentile(bs8, [2.5, 97.5])))
sub = np.array([oxr({k: A1[k][list(c)].mean() for k in k8}) for c in itertools.combinations(range(15), 5)])
print("  OxR over all %d five-seed subsets: 5-95%% [%.1f, %.1f], share <= 0: %.1f%%"
      % (len(sub), *np.percentile(sub, [5, 95]), 100 * (sub <= 0).mean()))
print("  SIMPLE EFFECTS, Welch, Holm over the eight rows")
simple = [("ZZ vs Zi | reup, 3.2", ("ZZ", "LO", 1), ("Zi", "LO", 1)), ("ZZ vs Zi | reup, 30", ("ZZ", "HI", 1), ("Zi", "HI", 1)),
          ("ZZ vs Zi | no reup, 3.2", ("ZZ", "LO", 0), ("Zi", "LO", 0)), ("ZZ vs Zi | no reup, 30", ("ZZ", "HI", 0), ("Zi", "HI", 0)),
          ("reup vs none | Zi, 3.2", ("Zi", "LO", 1), ("Zi", "LO", 0)), ("reup vs none | Zi, 30", ("Zi", "HI", 1), ("Zi", "HI", 0)),
          ("reup vs none | ZZ, 3.2", ("ZZ", "LO", 1), ("ZZ", "LO", 0)), ("reup vs none | ZZ, 30", ("ZZ", "HI", 1), ("ZZ", "HI", 0))]
sr = [(lab,) + welch_full(A1[a], A1[b]) for lab, a, b in simple]
so_ = np.argsort([r[2] for r in sr]); sh = [0.0] * 8; run = 0.0
for rank, i in enumerate(so_):
    run = max(run, min(1.0, (8 - rank) * sr[i][2])); sh[i] = run
for (lab, diff, p, d, lo, hi), ph in zip(sr, sh):
    print("    %-26s diff=%+7.1f  CI [%+7.1f, %+7.1f]  p=%.5f  holm=%.4f  d=%+.2f" % (lab, diff, lo, hi, p, ph, d))
print("  plateau: OLS slope over episodes 81-120, t-CI across seeds")
_h8 = {}
for _f in glob.glob(os.path.join(RESULTS, "cartpole_120", "*.json")):
    _j = json.load(open(_f))
    for _r in ([_j] if "label" in _j else [r for v in _j.values() for r in v]):
        _h8.setdefault(_r.get("label", ""), {})[_r["seed"]] = _r["history"]
_lab8 = {("Zi", "LO", 1): "QDQN-R (full)", ("Zi", "HI", 1): "QDQN-R full hi-scale",
         ("ZZ", "LO", 1): "QDQN-R global lo-scale", ("ZZ", "HI", 1): "QDQN-R global-obs",
         ("Zi", "LO", 0): "QDQN-R no-reup", ("Zi", "HI", 0): "QDQN-R no-reup hi-scale",
         ("ZZ", "LO", 0): "QDQN-R global no-reup lo-scale", ("ZZ", "HI", 0): "QDQN-R global no-reup"}
for k in k8:
    hs = _h8[_lab8[k]]
    sl = np.array([np.polyfit(np.arange(80, 120), hs[s][80:], 1)[0] for s in range(15)])
    lo, hi = stats.t.interval(0.95, 14, sl.mean(), sl.std(ddof=1) / np.sqrt(15))
    print("    %-32s slope %+.2f  CI [%+.2f, %+.2f]" % (_lab8[k], sl.mean(), lo, hi))
ep0 = np.array([[_h8[_lab8[k]][s][0] for s in range(15)] for k in k8])
print("  episode-0 return identical across all eight cells at every seed: %s" % bool((ep0 == ep0[0]).all()))

# ---------------------------------------------------------------------------
# Initial Q-value spread (results/initialization/output_range.json).
# No training: each cell's initial network, seed by seed, on states from
# random-policy episodes. Ratio = sd over states / |mean over states|.
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("INITIAL Q-VALUE SPREAD (no training; results/initialization/output_range.json)")
print("=" * 72)
_iq = json.load(open(os.path.join(RESULTS, "initialization", "output_range.json")))
_rows = _iq["rows"]; _cells = []
for _r in _rows:
    if _r["cell"] not in _cells:
        _cells.append(_r["cell"])
print("  states: %d" % _iq["n_states"])
for _c in _cells:
    _rr = [r for r in _rows if r["cell"] == _c]
    _m = lambda k: np.mean([r[k] for r in _rr])
    print("  %-12s qmax %6.2f  const %6.2f  sd %6.2f  sd/const %.2f  greedy-0 share %.2f  final %.1f"
          % (_c, _m("qmax"), _m("q_const"), _m("q_sd"), _m("q_sd") / _m("q_const"), _m("greedy0"), _m("final20")))
_z1, _z2 = [], []
for _c in _cells:
    _rr = [r for r in _rows if r["cell"] == _c]
    _a = stats.rankdata([r["q_sd"] / r["q_const"] for r in _rr]); _b = stats.rankdata([r["final20"] for r in _rr])
    _z1 += list(_a - _a.mean()); _z2 += list(_b - _b.mean())
print("  pooled within-cell rank correlation, sd/const vs final-20: r = %+.2f, p = %.2f" % stats.pearsonr(_z1, _z2))
_z1, _z2 = [], []
for _c in _cells:
    _rr = [r for r in _rows if r["cell"] == _c]
    _a = stats.rankdata([abs(r["greedy0"] - .5) for r in _rr]); _b = stats.rankdata([r["final20"] for r in _rr])
    _z1 += list(_a - _a.mean()); _z2 += list(_b - _b.mean())
print("  pooled within-cell rank correlation, greedy imbalance vs final-20: r = %+.2f, p = %.2f" % stats.pearsonr(_z1, _z2))

# ---------------------------------------------------------------------------
# Robustness of the two interactions and further reporting details.
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("INTERACTION ROBUSTNESS: blocking, windows, MDE, truncation")
print("=" * 72)
_H8 = {k: np.array([_h8[_lab8[k]][s] for s in range(15)]) for k in k8}   # (15, 120) per cell


def _terms_from(yv):
    return {t: (yv * x).mean() for t, x in X8.items()}


def _F8(yv, t):
    return anova8(yv)[t][1]


RNGR = np.random.default_rng(20260925)
for t in ("OxR", "IxR"):
    # Freedman-Lane: permute residuals of the model without term t
    cm = np.array([y8[cellid == c].mean() for c in range(8)])[cellid]
    b = (y8 * X8[t]).mean()
    fit_r = cm - b * X8[t]; res_r = y8 - fit_r
    f0 = _F8(y8, t); cnt = 0
    for _ in range(20000):
        cnt += _F8(fit_r + RNGR.permutation(res_r), t) >= f0
    # aligned rank transform: align on term t, rank, full-factorial ANOVA, read term t
    art = anova8(stats.rankdata(res_r))[t]
    # Welch-type contrast: interaction as a contrast of cell means, unequal variances
    w = np.array([X8[t][cellid == c][0] for c in range(8)]) / 2.0
    ms = np.array([y8[cellid == c].mean() for c in range(8)])
    vs = np.array([y8[cellid == c].var(ddof=1) for c in range(8)]) / 15
    est = (w * ms).sum(); se = np.sqrt((w ** 2 * vs).sum())
    dfw = (w ** 2 * vs).sum() ** 2 / ((w ** 2 * vs) ** 2 / 14).sum()
    pw = 2 * stats.t.sf(abs(est / se), dfw)
    print("  %-4s Freedman-Lane p=%.4f  ART F=%.2f p=%.4f  Welch contrast %+.1f (df %.0f) p=%.4f"
          % (t, cnt / 20000, art[1], art[2], est, dfw, pw))
# seed as a block (randomized complete block: 8 cells x 15 seeds, no replication)
Y = np.array([A1[k] for k in k8])                      # (8, 15)
ss_seed = 8 * ((Y.mean(0) - Y.mean()) ** 2).sum()
ss_cell = 15 * ((Y.mean(1) - Y.mean()) ** 2).sum()
ss_tot = ((Y - Y.mean()) ** 2).sum(); ss_err = ss_tot - ss_seed - ss_cell; ms_err = ss_err / 98
print("  seed block: F(14,98) = %.2f, p = %.4f" % (ss_seed / 14 / ms_err, stats.f.sf(ss_seed / 14 / ms_err, 14, 98)))
for t in ("O", "I", "R", "OxR", "IxR", "OxI", "OxIxR"):
    ss_t = 120 * (y8 * X8[t]).mean() ** 2
    print("    blocked %-6s F(1,98) = %6.2f  p = %.4f" % (t, ss_t / ms_err, stats.f.sf(ss_t / ms_err, 1, 98)))
# alternative score windows
for lab, sl in [("final-10", slice(110, 120)), ("final-40", slice(80, 120)), ("all 120", slice(0, 120))]:
    yw = np.concatenate([_H8[k][:, sl].mean(1) for k in k8]); aw = anova8(yw)
    print("  window %-9s OxR F=%.2f p=%.4f | IxR F=%.2f p=%.4f" % (lab, aw["OxR"][1], aw["OxR"][2], aw["IxR"][1], aw["IxR"][2]))
# minimum detectable simple effect at 80% power (two-sided 0.05, Welch)
mde = []
for lab, a, b in simple:
    va, vb = A1[a].var(ddof=1) / 15, A1[b].var(ddof=1) / 15
    df_ = (va + vb) ** 2 / (va ** 2 / 14 + vb ** 2 / 14)
    mde.append((stats.t.ppf(0.975, df_) + stats.t.ppf(0.80, df_)) * np.sqrt(va + vb))
    print("    MDE %-26s %.0f" % (lab, mde[-1]))
print("  MDE range over the eight simple effects: %.0f to %.0f" % (min(mde), max(mde)))
# truncation and environment steps
for k in k8:
    h = _H8[k]
    print("  %-32s final-20 episodes at 500: %4.1f%%   env steps per run: %6.0f"
          % (_lab8[k], 100 * (h[:, 100:] >= 500).mean(), h.sum(1).mean()))
for lab in ("Classical MLP 1x12 (lr=0.005)", "Classical MLP 2x64 (lr=0.001)"):
    runs = [json.load(open(f)) for f in glob.glob(os.path.join(RESULTS, "mlp_matched_120", "*.json"))]
    hh = np.array([r["history"] for r in sorted((r for r in runs if r["label"] == lab), key=lambda r: r["seed"])])
    print("  %-32s final-20 episodes at 500: %4.1f%%   env steps per run: %6.0f"
          % (lab, 100 * (hh[:, 100:] >= 500).mean(), hh.sum(1).mean()))
print("  epsilon over episodes 100-119: %.4f to %.4f, mean %.4f"
      % (0.97 ** 100, 0.97 ** 119, np.mean([0.97 ** e for e in range(100, 120)])))

# ---------------------------------------------------------------------------
# The initial networks and the pole-angle blind spot (results/initialization, qrl/blindness.py)
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("INITIAL NETWORKS: realized range and pole-angle sensitivity")
print("=" * 72)
_isn = json.load(open(os.path.join(RESULTS, "initialization", "input_sensitivity.json")))
for _c in _cells:
    _q = np.mean([r["qmax"] for r in _rows if r["cell"] == _c])
    _ps = [np.mean([r["q_range_%d" % j] for r in _isn["rows"] if r["cell"] == _c]) for j in range(4)]
    print("  %-12s realized max|Q| %6.2f   Q range per input sweep (cart pos, cart vel, pole angle, pole vel): %s"
          % (_c, _q, ", ".join("%.2f" % v for v in _ps)))
print("  zero variational angles, two-body no-reup, pole-angle range: %.1e" % _isn["zero_angle_pole_range_global_noreup"])
for _L in (3, 4, 5):
    print("  L=%d  Z_i images: %s   two-body images: %s" % (
        _L, [_qb(_img(1 << q, _L)) for q in range(4)],
        {"Z%dZ%d" % (a, b): _qb(_img((1 << a) | (1 << b), _L)) for a, b in itertools.combinations(range(4), 2)}))

# ---------------------------------------------------------------------------
# The MLP frontier at 120 episodes
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("CLASSICAL FRONTIER: raw and tanh input, five learning rates, selection on seeds 15-19")
print("=" * 72)
_FD = {}
for _f in glob.glob(os.path.join(RESULTS, "mlp_frontier_120", "*.json")):
    _j = json.load(open(_f)); _FD[(_j["width"], _j["input"], "%g" % _j["lr"], _j["seed"])] = float(np.mean(_j["history"][-20:]))
print("  runs: %d" % len(_FD))
_FW = {"1x12": 86, "1x16": 114, "2x8": 130, "2x16": 386, "2x64": 4610}
_FLR = ["0.001", "0.002", "0.005", "0.01", "0.02"]
_sel = {}
for _w in _FW:
    _row = []
    for _i in ("raw", "tanh"):
        _lr = max(_FLR, key=lambda l: np.mean([_FD[(_w, _i, "%g" % float(l), s)] for s in range(15, 20)]))
        _a = np.array([_FD[(_w, _i, "%g" % float(_lr), s)] for s in range(15)]); _sel[(_w, _i)] = (_lr, _a)
        _row.append("%s lr %-5s %6.1f +- %5.1f" % (_i, _lr, _a.mean(), _a.std(ddof=1)))
    _b10 = max(((i, l) for i in ("raw", "tanh") for l in _FLR), key=lambda k: np.mean([_FD[(_w, k[0], "%g" % float(k[1]), s)] for s in range(15)]))
    _bv = np.mean([_FD[(_w, _b10[0], "%g" % float(_b10[1]), s)] for s in range(15)])
    print("  %-5s %5d | %s | best of ten %.1f (%s, %s)" % (_w, _FW[_w], " | ".join(_row), _bv, *_b10))
_m12t = np.array([_FD[("1x12", "tanh", "0.01", s)] for s in range(15)])
for _lab, _v in [("global-obs", G), ("no-reup", NR)]:
    _p, _d = welch(_v, _m12t)
    print("  %-10s vs 1x12 best of ten (120.1): diff %+.1f p=%.6f d=%.2f ratio %.2f" % (_lab, _v.mean() - _m12t.mean(), _p, _d, _v.mean() / _m12t.mean()))
print("  global-obs / selected 1x12 tanh: %.2f" % (G.mean() / _sel[("1x12", "tanh")][1].mean()))
print("  max selected classical at any width: %.1f" % max(_sel[k][1].mean() for k in _sel))

open(data.output("factorial_120.txt"), "w", encoding="utf-8").write(_out.getvalue())
