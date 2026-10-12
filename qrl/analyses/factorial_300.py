"""The factorial, the rewired cells and the head ablation at 300 episodes (cartpole_300).

  A  rewiring tests (pole angle moved to qubit 1, sham rewiring), paired by seed, at 120 episodes (prefix of the
     300-episode runs) and at 300;
  B  three-way ANOVA at episode 300 and on the area under the learning curve, with the robustness checks used at
     120, and collapse rates per cell;
  C  greedy evaluation of the final networks (results/greedy/);
  D  the MLP frontier at 300 episodes (mlp_frontier_300).
The reproducibility checks compare cartpole_300 with earlier copies of its runs (cartpole_300_first_pass) and with
the separate 120-episode runs (cartpole_120, cartpole_120_rewired).
-> analysis/factorial_300.txt
"""
import glob
import itertools
import os

import numpy as np
from scipy import stats

from .. import data
from ..io import read_json

OUT = data.output("factorial_300.txt")
_lines = []


def say(s=""):
    print(s); _lines.append(s)


def head(s):
    say(); say("=" * 78); say(s); say("=" * 78)


def safe(s):
    return s.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("=", "")


def load(label, n=15):
    H = []
    for s in range(n):
        j = read_json(data.path("cartpole_300", "%s_seed%d" % (safe(label), s)))
        assert j["seed"] == s and len(j["history"]) == 300
        H.append(j["history"])
    return np.array(H, dtype=float)                       # (n, 300)


F8 = {("Zi", "LO", 1): "QDQN-R (full)", ("Zi", "HI", 1): "QDQN-R full hi-scale",
      ("ZZ", "LO", 1): "QDQN-R global lo-scale", ("ZZ", "HI", 1): "QDQN-R global-obs",
      ("Zi", "LO", 0): "QDQN-R no-reup", ("Zi", "HI", 0): "QDQN-R no-reup hi-scale",
      ("ZZ", "LO", 0): "QDQN-R global no-reup lo-scale", ("ZZ", "HI", 0): "QDQN-R global no-reup"}
REWIRED = ["QDQN-R global no-reup perm", "QDQN-R no-reup hi-scale perm", "QDQN-R global-obs perm",
           "QDQN-R full hi-scale perm", "QDQN-R global no-reup sham", "QDQN-R global no-reup lo-scale perm"]
COLLAPSED = ["Naive QDQN", "QDQN-R no-scale"]
H = {lab: load(lab) for lab in list(F8.values()) + REWIRED}
H.update({lab: load(lab, 5) for lab in COLLAPSED})

SCORES = {"120": lambda h: h[:, 100:120].mean(1),          # final 20 of the first 120 episodes
          "300": lambda h: h[:, 280:300].mean(1),          # final 20 of 300
          "AUC": lambda h: h.mean(1)}                      # mean return over all 300 episodes
S = {w: {lab: f(h) for lab, h in H.items()} for w, f in SCORES.items()}


def tci(a):
    h = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return a.mean() - h, a.mean() + h


def holm(ps):
    order = np.argsort(ps); out = [0.0] * len(ps); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i])); out[i] = run
    return out


# ---------------------------------------------------------------------------
head("CELL SUMMARY (mean +- s.d. over seeds)")
say("%-38s %2s  %-16s %-16s %-16s %s" % ("cell", "n", "final-20 @120", "final-20 @300", "AUC 0-300", "collapsed"))
for lab in list(F8.values()) + REWIRED + COLLAPSED:
    a, b, c = S["120"][lab], S["300"][lab], S["AUC"][lab]
    col = int((b < 0.5 * a).sum())
    say("%-38s %2d  %6.1f +- %5.1f   %6.1f +- %5.1f   %6.1f +- %5.1f   %d/%d"
        % (lab, len(a), a.mean(), a.std(ddof=1), b.mean(), b.std(ddof=1), c.mean(), c.std(ddof=1), col, len(a)))
say("collapsed = final-20 at 300 below half of final-20 at 120")

# sanity: the 120-episode prefix must reproduce the paper's cell means
PAPER120 = {"QDQN-R (full)": 213.0, "QDQN-R full hi-scale": 263.3, "QDQN-R global lo-scale": 290.6,
            "QDQN-R global-obs": 366.8, "QDQN-R no-reup": 295.4, "QDQN-R no-reup hi-scale": 231.8,
            "QDQN-R global no-reup lo-scale": 246.7, "QDQN-R global no-reup": 164.1}
ok = all(abs(round(S["120"][k].mean(), 1) - v) < 1e-9 for k, v in PAPER120.items())
say("120-episode prefix reproduces the paper's eight cell means: %s" % ok)

# ---------------------------------------------------------------------------
# Factorial machinery (same definitions as src/verify_numbers.py)
k8 = list(F8); n8 = 15
cO = np.repeat([1 if k[0] == "ZZ" else -1 for k in k8], n8)
cI = np.repeat([1 if k[1] == "HI" else -1 for k in k8], n8)
cR = np.repeat([1 if k[2] == 1 else -1 for k in k8], n8)
X8 = {"O": cO, "I": cI, "R": cR, "OxI": cO * cI, "OxR": cO * cR, "IxR": cI * cR, "OxIxR": cO * cI * cR}
cellid = np.repeat(np.arange(8), n8)
TERMS = list(X8)


def anova8(yv):
    ssw = sum(((yv[cellid == c] - yv[cellid == c].mean()) ** 2).sum() for c in range(8))
    out = {}
    for t, x in X8.items():
        est = (yv * x).mean(); ss = len(yv) * est ** 2; f = ss / (ssw / (len(yv) - 8))
        out[t] = (2 * est, f, stats.f.sf(f, 1, len(yv) - 8), ss / (ss + ssw))
    return out


def factorial_report(y, lab, rng_seed, nperm=20000):
    r = anova8(y); hp = dict(zip(TERMS, holm([r[t][2] for t in TERMS])))
    say("  three-way ANOVA, %s (df 1,112); effect = mean(+) - mean(-); Holm over 7 terms" % lab)
    for t in TERMS:
        e, f, p, e2 = r[t]
        say("    %-6s effect %+7.1f  F=%6.2f  p=%.4f  Holm=%.4f  eta2p=%.3f" % (t, e, f, p, hp[t], e2))
    say("  Brown-Forsythe across the 8 cells: p = %.4f"
        % stats.levene(*[y[cellid == c] for c in range(8)], center="median").pvalue)
    rng = np.random.default_rng(rng_seed)
    for t in ("OxR", "IxR"):
        cm = np.array([y[cellid == c].mean() for c in range(8)])[cellid]
        b = (y * X8[t]).mean(); fit_r = cm - b * X8[t]; res_r = y - fit_r
        f0 = anova8(y)[t][1]; cnt = 0
        for _ in range(nperm):
            cnt += anova8(fit_r + rng.permutation(res_r))[t][1] >= f0
        art = anova8(stats.rankdata(res_r))[t]
        w = np.array([X8[t][cellid == c][0] for c in range(8)]) / 2.0
        ms = np.array([y[cellid == c].mean() for c in range(8)])
        vs = np.array([y[cellid == c].var(ddof=1) for c in range(8)]) / 15
        est = (w * ms).sum(); se = np.sqrt((w ** 2 * vs).sum())
        dfw = (w ** 2 * vs).sum() ** 2 / ((w ** 2 * vs) ** 2 / 14).sum()
        say("    %-4s Welch contrast %+7.1f (df %3.0f) p=%.4f | Freedman-Lane p=%.4f | ART F=%.2f p=%.4f"
            % (t, est, dfw, 2 * stats.t.sf(abs(est / se), dfw), cnt / nperm, art[1], art[2]))
    Y = y.reshape(8, 15)
    ss_seed = 8 * ((Y.mean(0) - Y.mean()) ** 2).sum(); ss_cell = 15 * ((Y.mean(1) - Y.mean()) ** 2).sum()
    ms_err = (((Y - Y.mean()) ** 2).sum() - ss_seed - ss_cell) / 98
    say("  seed as block: seed F(14,98)=%.2f p=%.4f" % (ss_seed / 14 / ms_err, stats.f.sf(ss_seed / 14 / ms_err, 14, 98)))
    for t in ("OxR", "IxR"):
        ss_t = 120 * (y * X8[t]).mean() ** 2
        say("    blocked %-4s F(1,98)=%6.2f p=%.4f" % (t, ss_t / ms_err, stats.f.sf(ss_t / ms_err, 1, 98)))
    return r, hp


def cellvec(w):
    return np.concatenate([S[w][F8[k]] for k in k8])


def oxr_contrast(m):
    return (np.mean([m[("ZZ", i, 1)] - m[("Zi", i, 1)] for i in ("LO", "HI")])
            - np.mean([m[("ZZ", i, 0)] - m[("Zi", i, 0)] for i in ("LO", "HI")]))


# ---------------------------------------------------------------------------
head("B. FACTORIAL AT 300 EPISODES")
say("Cell means (readout / range / re-uploading):")
say("  %-22s %8s %8s %8s" % ("cell", "@120", "@300", "AUC"))
for k in k8:
    say("  %-22s %8.1f %8.1f %8.1f" % ("%s/%s/%s" % (k[0], "3.2" if k[1] == "LO" else "30", "reup" if k[2] else "none"),
                                      S["120"][F8[k]].mean(), S["300"][F8[k]].mean(), S["AUC"][F8[k]].mean()))
RES = {}
for w, lab, seed in [("120", "final-20 at episode 120 (check against the paper)", 20260924),
                     ("300", "final-20 at episode 300 (B.1, primary)", 20260926),
                     ("AUC", "mean return over episodes 0-300 (B.2, secondary)", 20260927)]:
    say(); say("-- %s" % lab)
    RES[w] = factorial_report(cellvec(w), lab, seed)
    m = {k: S[w][F8[k]].mean() for k in k8}
    rngb = np.random.default_rng(7 + len(w))
    bs = [oxr_contrast({k: S[w][F8[k]][rngb.integers(0, 15, 15)].mean() for k in k8}) for _ in range(10000)]
    say("  OxR contrast (two-body minus per-qubit, reup minus none) = %+.1f, bootstrap 95%% CI [%.1f, %.1f]"
        % (oxr_contrast(m), *np.percentile(bs, [2.5, 97.5])))

say(); say("-- B.3 side by side (p, Holm p)")
for t in TERMS:
    say("  %-6s  @120 p=%.4f (Holm %.4f)   @300 p=%.4f (Holm %.4f)   AUC p=%.4f (Holm %.4f)"
        % (t, RES["120"][0][t][2], RES["120"][1][t], RES["300"][0][t][2], RES["300"][1][t],
           RES["AUC"][0][t][2], RES["AUC"][1][t]))
say(); say("-- B.4 collapse rate per cell (final-20 at 300 < 0.5 x final-20 at 120)")
for k in k8:
    a, b = S["120"][F8[k]], S["300"][F8[k]]
    say("  %-32s %2d/15" % (F8[k], int((b < 0.5 * a).sum())))

# ---------------------------------------------------------------------------
head("A. REWIRING TESTS (paired by seed, n = 15)")
TESTS = [("A.1 ZZ/30/none: pole on q1 vs original (primary)", "QDQN-R global no-reup perm", "QDQN-R global no-reup"),
         ("A.2 Zi/30/none: pole on q1 vs original (control)", "QDQN-R no-reup hi-scale perm", "QDQN-R no-reup hi-scale"),
         ("A.3 ZZ/30/reup: pole on q1 vs original", "QDQN-R global-obs perm", "QDQN-R global-obs"),
         ("A.5 Zi/30/reup: pole on q1 vs original", "QDQN-R full hi-scale perm", "QDQN-R full hi-scale"),
         ("A.6 ZZ/30/none: pole on q1 vs sham rewiring", "QDQN-R global no-reup perm", "QDQN-R global no-reup sham"),
         ("A.7 ZZ/3.2/none: pole on q1 vs original", "QDQN-R global no-reup lo-scale perm", "QDQN-R global no-reup lo-scale")]


def paired(a, b):
    d = a - b
    lo, hi = tci(d)
    pt = stats.ttest_rel(a, b).pvalue
    pw = stats.wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
    return d.mean(), lo, hi, pt, pw, int((d > 0).sum()), int((d < 0).sum())


for w in ("120", "300"):
    say(); say("-- at %s episodes%s" % (w, " (primary for the mechanism)" if w == "120" else ""))
    rows = [(name,) + paired(S[w][a], S[w][b]) for name, a, b in TESTS]
    hp = holm([r[4] for r in rows])
    for (name, m, lo, hi, pt, pw, up, dn), h in zip(rows, hp):
        say("  %-48s diff %+7.1f  CI [%+7.1f, %+7.1f]  t p=%.4f  Holm %.4f  Wilcoxon p=%.4f  up/down %d/%d"
            % (name, m, lo, hi, pt, h, pw, up, dn))
    for lab in ("QDQN-R global no-reup sham",):
        m, lo, hi, pt, pw, up, dn = paired(S[w][lab], S[w]["QDQN-R global no-reup"])
        say("  (descriptive) sham vs original ZZ/30/none: diff %+7.1f  CI [%+7.1f, %+7.1f]  t p=%.4f  up/down %d/%d"
            % (m, lo, hi, pt, up, dn))
    # A.5: readout x re-uploading contrast at range 30, rewired vs original
    P4 = {("ZZ", 1): "QDQN-R global-obs perm", ("Zi", 1): "QDQN-R full hi-scale perm",
          ("ZZ", 0): "QDQN-R global no-reup perm", ("Zi", 0): "QDQN-R no-reup hi-scale perm"}
    O4 = {("ZZ", 1): "QDQN-R global-obs", ("Zi", 1): "QDQN-R full hi-scale",
          ("ZZ", 0): "QDQN-R global no-reup", ("Zi", 0): "QDQN-R no-reup hi-scale"}
    wts = {("ZZ", 1): 1, ("Zi", 1): -1, ("ZZ", 0): -1, ("Zi", 0): 1}

    def contrast(cells, idx=slice(None)):
        return sum(wts[k] * S[w][cells[k]][idx].mean() for k in wts)

    for tag, cells in (("rewired", P4), ("original", O4)):
        vs = np.array([S[w][cells[k]].var(ddof=1) / 15 for k in wts])
        est = contrast(cells); se = np.sqrt(vs.sum()); dfw = vs.sum() ** 2 / (vs ** 2 / 14).sum()
        say("  A.5 %-8s readout x re-uploading contrast at range 30: %+7.1f  (Welch df %3.0f) p=%.4f"
            % (tag, est, dfw, 2 * stats.t.sf(abs(est / se), dfw)))
    rng = np.random.default_rng(20260926 + int(w))
    bd = []
    for _ in range(10000):
        idx = rng.integers(0, 15, 15)
        bd.append(contrast(P4, idx) - contrast(O4, idx))
    bd = np.array(bd)
    say("  A.5 rewired minus original contrast: %+7.1f, seed-paired bootstrap 95%% CI [%+.1f, %+.1f], share >= 0: %.3f"
        % (contrast(P4) - contrast(O4), *np.percentile(bd, [2.5, 97.5]), (bd >= 0).mean()))

# ---------------------------------------------------------------------------
head("C. GREEDY EVALUATION (10 greedy episodes per network, env seeds 10000+k)")
for which, fn, w in (("120", "cartpole_120", "120"), ("300", "cartpole_300", "300")):
    G = data.greedy(fn)["results"]
    have = all(F8[k] in G and len(G[F8[k]]) == 15 for k in k8)
    say(); say("-- networks at %s episodes (%s)" % (which, "primary" if which == "120" else "secondary"))
    for lab in list(F8.values()) + REWIRED + COLLAPSED:
        if lab in G:
            g = np.array([np.mean(G[lab][str(s)]) for s in sorted(map(int, G[lab]))])
            say("  %-38s n=%2d  greedy %6.1f +- %5.1f" % (lab, len(g), g.mean(), g.std(ddof=1)))
    if have:
        yg = np.concatenate([[np.mean(G[F8[k]][str(s)]) for s in range(15)] for k in k8])
        r = anova8(yg); hp = dict(zip(TERMS, holm([r[t][2] for t in TERMS])))
        for t in TERMS:
            say("    %-6s effect %+7.1f  F=%6.2f  p=%.4f  Holm=%.4f" % (t, r[t][0], r[t][1], r[t][2], hp[t]))
        tr = cellvec(w)
        say("  per-run correlation, greedy vs final-20 at %s: Pearson r=%.2f, Spearman rho=%.2f"
            % (which, stats.pearsonr(yg, tr)[0], stats.spearmanr(yg, tr)[0]))

# ---------------------------------------------------------------------------
head("D. FAIR CLASSICAL FRONTIER AT 300 EPISODES")
PARAMS = {"1x12": 86, "1x16": 114, "2x8": 130, "2x16": 386, "2x64": 4610}
LRS = ["0.001", "0.002", "0.005", "0.01", "0.02"]
D3, D1 = {}, {}
for f in glob.glob(os.path.join(data.RESULTS, "mlp_frontier_300", "*.json")):
    j = read_json(f); assert len(j["history"]) == 300
    key = (j["width"], j["input"], "%g" % j["lr"], j["seed"])
    D3[key] = float(np.mean(j["history"][-20:]))
    g = os.path.join(data.RESULTS, "mlp_frontier_120", os.path.basename(f))
    if os.path.exists(g):
        D1[key] = read_json(g)["history"] == j["history"][:120]
say("runs: %d; 120-episode prefix identical to the 120-episode frontier runs: %d/%d"
    % (len(D3), sum(D1.values()), len(D1)))
front = {}
for wd in PARAMS:
    for inp in ("raw", "tanh"):
        tune = {lr: np.mean([D3[(wd, inp, "%g" % float(lr), s)] for s in range(15, 20)]) for lr in LRS}
        best = max(tune, key=tune.get)
        a = np.array([D3[(wd, inp, "%g" % float(best), s)] for s in range(15)])
        front[(wd, inp)] = (best, a, tune[best])
opt = {}
for wd in PARAMS:
    best = max(((i, lr) for i in ("raw", "tanh") for lr in LRS),
               key=lambda k: np.mean([D3[(wd, k[0], "%g" % float(k[1]), s)] for s in range(15)]))
    opt[wd] = (best, np.array([D3[(wd, best[0], "%g" % float(best[1]), s)] for s in range(15)]))
say("%-5s %5s | %-24s | %-24s | best of ten on 0-14" % ("MLP", "params", "raw (lr on 15-19)", "tanh (lr on 15-19)"))
for wd in PARAMS:
    r_, t_ = front[(wd, "raw")], front[(wd, "tanh")]
    say("%-5s %5d | %6.1f +- %5.1f (%-5s)  | %6.1f +- %5.1f (%-5s)  | %6.1f (%s, %s)"
        % (wd, PARAMS[wd], r_[1].mean(), r_[1].std(ddof=1), r_[0], t_[1].mean(), t_[1].std(ddof=1), t_[0],
           opt[wd][1].mean(), *opt[wd][0]))
sel_best = {wd: max(("raw", "tanh"), key=lambda i: front[(wd, i)][2]) for wd in PARAMS}
# as in the paper's table of the MLP frontier: the largest reported mean over every
# (width, input) whose learning rate was selected on seeds 15-19
maxsel = max(front[k][1].mean() for k in front)
maxopt = max(opt[wd][1].mean() for wd in PARAMS)
say("max over widths and inputs: lr selected on seeds 15-19 %.1f; best of ten on seeds 0-14 %.1f" % (maxsel, maxopt))
say("comparisons below use, per width, the input whose selected lr scored higher on seeds 15-19")
for lab in ("QDQN-R global-obs", "QDQN-R no-reup", "QDQN-R (full)", "QDQN-R global lo-scale"):
    v = S["300"][lab]
    for wd in ("1x12", "2x64"):
        for tag, c in (("selected", front[(wd, sel_best[wd])][1]), ("best-of-ten", opt[wd][1])):
            p = stats.ttest_ind(v, c, equal_var=False).pvalue
            say("  %-24s @300 %6.1f vs %-5s %-11s %6.1f: diff %+7.1f  Welch p=%.5f"
                % (lab, v.mean(), wd, tag, c.mean(), v.mean() - c.mean(), p))

# ---------------------------------------------------------------------------
head("REPRODUCIBILITY CHECKS")
same = diff = 0
for f in sorted(glob.glob(os.path.join(data.RESULTS, "cartpole_300_first_pass", "*.json"))):
    a = read_json(f)["history"]; b = read_json(os.path.join(data.RESULTS, "cartpole_300", os.path.basename(f)))["history"]
    same += a == b; diff += a != b
say("(ii) first-pass 300-episode runs vs rerun: %d identical, %d differ" % (same, diff))
same = diff = 0
LATER = ("QDQN-R_full_hi-scale_seed", "QDQN-R_global_lo-scale_seed")   # 120-episode runs of a later batch, not checked
for g in sorted(glob.glob(os.path.join(data.RESULTS, "cartpole_300", "*.json"))):
    name = os.path.basename(g)
    r = os.path.join(data.RESULTS, "cartpole_120_rewired", name)
    if not os.path.exists(r) and not name.startswith(LATER):
        r = os.path.join(data.RESULTS, "cartpole_120", name)
    if os.path.exists(r):
        a = read_json(r)["history"]; b = read_json(g)["history"]
        same += a == b[:len(a)]; diff += a != b[:len(a)]
say("(i) 120-episode runs vs first 120 episodes of the 300-episode runs: %d identical, %d differ" % (same, diff))
ep0 = np.array([H[lab][:, 0] for lab in list(F8.values()) + REWIRED])
say("episode-0 return identical across all 14 cells at every seed: %s" % bool((ep0 == ep0[0]).all()))

# ---------------------------------------------------------------------------
head("EXPLORATORY")
for w in ("120", "300"):
    say("-- scale sensitivity of the two interactions at %s (OxR, IxR p; Holm over 7)" % w)
    for tag, fn in (("raw", lambda y: y), ("sqrt", np.sqrt), ("log", np.log)):
        y = fn(cellvec(w)); r = anova8(y); hp = dict(zip(TERMS, holm([r[t][2] for t in TERMS])))
        say("  %-5s OxR p=%.4f (Holm %.4f)   IxR p=%.4f (Holm %.4f)" % (tag, r["OxR"][2], hp["OxR"], r["IxR"][2], hp["IxR"]))
say("-- simple effects at 300 (Welch; Holm over the eight rows)")
simple = [("ZZ vs Zi | reup, 3.2", ("ZZ", "LO", 1), ("Zi", "LO", 1)), ("ZZ vs Zi | reup, 30", ("ZZ", "HI", 1), ("Zi", "HI", 1)),
          ("ZZ vs Zi | none, 3.2", ("ZZ", "LO", 0), ("Zi", "LO", 0)), ("ZZ vs Zi | none, 30", ("ZZ", "HI", 0), ("Zi", "HI", 0)),
          ("reup vs none | Zi, 3.2", ("Zi", "LO", 1), ("Zi", "LO", 0)), ("reup vs none | Zi, 30", ("Zi", "HI", 1), ("Zi", "HI", 0)),
          ("reup vs none | ZZ, 3.2", ("ZZ", "LO", 1), ("ZZ", "LO", 0)), ("reup vs none | ZZ, 30", ("ZZ", "HI", 1), ("ZZ", "HI", 0))]
sr = []
for lab, a, b in simple:
    x, y = S["300"][F8[a]], S["300"][F8[b]]
    sr.append((lab, x.mean() - y.mean(), stats.ttest_ind(x, y, equal_var=False).pvalue))
for (lab, dm, p), h in zip(sr, holm([r[2] for r in sr])):
    say("  %-24s diff %+7.1f  p=%.4f  Holm %.4f" % (lab, dm, p, h))
say("-- collapse (final-20 at 300 < half of final-20 at 120) by re-uploading, Fisher exact")
col = {r: sum(int((S["300"][F8[k]] < 0.5 * S["120"][F8[k]]).sum()) for k in k8 if k[2] == r) for r in (0, 1)}
say("  re-uploading %d/60, none %d/60, Fisher p=%.5f"
    % (col[1], col[0], stats.fisher_exact([[col[1], 60 - col[1]], [col[0], 60 - col[0]]])[1]))
y3 = cellvec("300")
sdw = np.sqrt(sum(((y3[cellid == c] - y3[cellid == c].mean()) ** 2).sum() for c in range(8)) / 112)
say("-- pooled within-cell s.d. at 300: %.1f; MDE of a main effect (60 vs 60, 80%% power, two-sided 0.05): %.0f"
    % (sdw, (stats.t.ppf(0.975, 112) + stats.t.ppf(0.8, 112)) * sdw * np.sqrt(2 / 60)))
say("-- each cell vs the scaling-only cell (Zi/3.2/none) at 300, Welch")
base = S["300"][F8[("Zi", "LO", 0)]]
for k in k8:
    if k != ("Zi", "LO", 0):
        a = S["300"][F8[k]]
        say("  %-32s %+7.1f  p=%.3f" % (F8[k], a.mean() - base.mean(), stats.ttest_ind(a, base, equal_var=False).pvalue))
say("-- CartPole-v0-style criterion on the v1 runs (returns capped at 200; 100-episode mean >= 195)")
tot_end = tot_any = 0
for k in k8:
    c200 = np.minimum(H[F8[k]], 200)
    roll = np.array([[c200[s, i:i + 100].mean() for i in range(201)] for s in range(15)])
    tot_end += int((c200[:, 200:].mean(1) >= 195).sum()); tot_any += int((roll >= 195).any(1).sum())
say("  factorial runs meeting it over episodes 200-300: %d/120; in some 100-episode window: %d/120" % (tot_end, tot_any))
say("-- share of final-20 episodes at the 500-step limit, and env steps per run (300 episodes)")
for k in k8:
    h = H[F8[k]]
    say("  %-32s at 500: %4.1f%%   steps per run: %7.0f" % (F8[k], 100 * (h[:, 280:] >= 500).mean(), h.sum(1).mean()))

open(OUT, "w", encoding="utf-8").write("\n".join(_lines) + "\n")
print("\nwrote", OUT)
