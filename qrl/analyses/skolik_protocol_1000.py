"""The factorial and the Skolik agent under the Skolik protocol, 1000 episodes (skolik_protocol_1000).

The eight factorial cells and the Skolik et al. agent, CartPole-v1, 15 seeds, 1000 episodes.
  0  reproducibility checks;
  1  confirmatory: three-way ANOVA on the greedy score of the best checkpoint (selected on environment seeds
     20000+k, reported on 10000+k; results/greedy/), Holm over 7 terms, with Welch contrasts, Freedman-Lane
     permutation and seed-as-block checks;
  2  time to solve: Cox model with the three factors and their interactions, runs that never solve censored at
     1000; solve rate per cell;
  3  stability: logistic regression of collapse on the three factors;
  4  final level (greedy at episode 1000, final 20 training episodes) and AUC: the same three-way ANOVA;
  5  the Skolik agent: Welch on the best-checkpoint score against two cells, Holm over 2;
  6  plateau: slope of the last 200 episodes.
-> analysis/skolik_protocol_1000.txt
"""
import warnings

import numpy as np
from scipy import stats

from .. import data
from ..io import read_json

OUT = data.output("skolik_protocol_1000.txt")
SOLVED, EPISODES, NSEED = 475.0, 1000, 15
_lines = []


def say(s=""):
    print(s); _lines.append(s)


def head(s):
    say(); say("=" * 78); say(s); say("=" * 78)


F8 = {("Zi", "LO", 1): "QDQN-R_full", ("Zi", "HI", 1): "QDQN-R_full_hi-scale",
      ("ZZ", "LO", 1): "QDQN-R_global_lo-scale", ("ZZ", "HI", 1): "QDQN-R_global-obs",
      ("Zi", "LO", 0): "QDQN-R_no-reup", ("Zi", "HI", 0): "QDQN-R_no-reup_hi-scale",
      ("ZZ", "LO", 0): "QDQN-R_global_no-reup_lo-scale", ("ZZ", "HI", 0): "QDQN-R_global_no-reup"}
k8 = list(F8)
ARMS = list(F8.values()) + ["Skolik"]


def tag(k):
    return "%s/%s/%s" % (k[0], "3.2" if k[1] == "LO" else "30", "reup" if k[2] else "none")


H = {}
for a in ARMS:
    rows = []
    for s in range(NSEED):
        j = read_json(data.path("skolik_protocol_1000", "%s_seed%d" % (a, s)))
        assert j["label"] == a and j["seed"] == s and len(j["history"]) == EPISODES
        rows.append(j["history"])
    H[a] = np.array(rows, dtype=float)                     # (15, 1000)
G = data.greedy("skolik_protocol_1000")["results"]


def roll(h, w):
    c = np.cumsum(np.insert(h, 0, 0.0))
    return (c[w:] - c[:-w]) / w


def per_run(h):
    m100 = roll(h, 100)
    hit = np.where(m100 >= SOLVED)[0]
    return {"t": hit[0] + 100 if len(hit) else EPISODES, "event": int(len(hit) > 0),
            "final20": h[-20:].mean(), "auc": h.mean(),
            "collapse": int(h[-20:].mean() < 0.5 * roll(h, 20).max()),
            "hold": int(h[-100:].mean() >= SOLVED),
            "slope": np.polyfit(np.arange(200), h[-200:], 1)[0]}


M = {a: [per_run(h) for h in H[a]] for a in ARMS}


def col(a, key):
    if key in ("best", "final_greedy", "best_ep"):
        f = {"best": lambda r: np.mean(r["best_report"]), "final_greedy": lambda r: np.mean(r["final_report"]),
             "best_ep": lambda r: r["best_ep"]}[key]
        return np.array([f(G[a][str(s)]) for s in range(NSEED)], dtype=float)
    return np.array([r[key] for r in M[a]], dtype=float)


def tci(x):
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def holm(ps):
    order = np.argsort(ps); out = [0.0] * len(ps); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i])); out[i] = run
    return out


# ---------------------------------------------------------------------------
head("0. REPRODUCIBILITY CHECKS")
ok0 = all(len({H[a][s, 0] for a in ARMS}) == 1 for s in range(NSEED))
say("episode-0 return identical across the 9 arms at every seed: %s" % ok0)
for a, ref in [("Skolik", "SK-SKP-v1"), ("QDQN-R_global-obs", "GO-SKP-v1")]:
    bad = [s for s in range(NSEED) if list(H[a][s, :300]) != data.history("skolik_comparison", "%s_seed%d" % (ref, s))[:300]]
    say("first 300 episodes of %-18s bit-identical to skolik_comparison/%s at all 15 seeds: %s%s"
        % (a, ref, not bad, "" if not bad else "  (seeds %s differ)" % bad))

# ---------------------------------------------------------------------------
head("CELL SUMMARY (mean +- s.d. over 15 seeds)")
say("%-28s %-15s %-15s %-13s %-13s %6s %8s %6s" % ("cell", "best greedy", "greedy @1000", "final-20",
                                                   "AUC", "solved", "collapse", "hold"))
for a, t in [(F8[k], tag(k)) for k in k8] + [("Skolik", "Skolik agent")]:
    b, f, l, u = col(a, "best"), col(a, "final_greedy"), col(a, "final20"), col(a, "auc")
    say("%-28s %6.1f +- %5.1f %6.1f +- %5.1f %5.1f +- %5.1f %5.1f +- %5.1f %3d/15 %5d/15 %3d/15"
        % (t, b.mean(), b.std(ddof=1), f.mean(), f.std(ddof=1), l.mean(), l.std(ddof=1), u.mean(), u.std(ddof=1),
           col(a, "event").sum(), col(a, "collapse").sum(), col(a, "hold").sum()))
say("best greedy: selected snapshot, 10 greedy episodes on env seeds 10000+k (selection on 20000+k)")
say("solved: some 100-episode training mean >= 475; collapse: final-20 < half the best 20-episode mean;")
say("hold: mean of the last 100 training episodes >= 475")

# ---------------------------------------------------------------------------
# Factorial machinery (same definitions as src/analyze_all300.py), +-1 coding
n8 = NSEED
cO = np.repeat([1 if k[0] == "ZZ" else -1 for k in k8], n8)
cI = np.repeat([1 if k[1] == "HI" else -1 for k in k8], n8)
cR = np.repeat([1 if k[2] == 1 else -1 for k in k8], n8)
X8 = {"O": cO, "I": cI, "R": cR, "OxI": cO * cI, "OxR": cO * cR, "IxR": cI * cR, "OxIxR": cO * cI * cR}
cellid = np.repeat(np.arange(8), n8)
TERMS = list(X8)
say(); say("Terms: O = readout (ZZ +, Zi -), I = initial range (30 +, 3.2 -), R = re-uploading (on +, off -).")
say("effect = mean(+) - mean(-) of the +-1 contrast; for interactions, half the difference of simple effects.")


def anova8(yv):
    ssw = sum(((yv[cellid == c] - yv[cellid == c].mean()) ** 2).sum() for c in range(8))
    out = {}
    for t, x in X8.items():
        est = (yv * x).mean(); ss = len(yv) * est ** 2; f = ss / (ssw / (len(yv) - 8))
        out[t] = (2 * est, f, stats.f.sf(f, 1, len(yv) - 8), ss / (ss + ssw))
    return out


def anova_report(y, lab):
    r = anova8(y); hp = dict(zip(TERMS, holm([r[t][2] for t in TERMS])))
    say("  three-way ANOVA, %s (df 1,112); Holm over 7 terms" % lab)
    for t in TERMS:
        e, f, p, e2 = r[t]
        say("    %-6s effect %+7.1f  F=%7.2f  p=%.4g  Holm=%.4g  eta2p=%.3f" % (t, e, f, p, hp[t], e2))
    return r, hp


def robustness(y, rng_seed, nperm=20000):
    say("  Brown-Forsythe across the 8 cells: p = %.4g"
        % stats.levene(*[y[cellid == c] for c in range(8)], center="median").pvalue)
    rng = np.random.default_rng(rng_seed)
    cm = np.array([y[cellid == c].mean() for c in range(8)])
    vs = np.array([y[cellid == c].var(ddof=1) for c in range(8)]) / n8
    for t in TERMS:
        b = (y * X8[t]).mean(); fit_r = cm[cellid] - b * X8[t]; res_r = y - fit_r
        f0 = anova8(y)[t][1]
        cnt = sum(anova8(fit_r + rng.permutation(res_r))[t][1] >= f0 for _ in range(nperm))
        w = np.array([X8[t][cellid == c][0] for c in range(8)]) / 4.0
        est = (w * cm).sum(); se = np.sqrt((w ** 2 * vs).sum())
        dfw = (w ** 2 * vs).sum() ** 2 / ((w ** 2 * vs) ** 2 / (n8 - 1)).sum() if se > 0 else np.nan
        pw = 2 * stats.t.sf(abs(est / se), dfw) if se > 0 else np.nan
        say("    %-6s Welch contrast %+7.1f (df %5.1f) p=%.4g | Freedman-Lane p=%.4g"
            % (t, est, dfw, pw, cnt / nperm))
    Y = y.reshape(8, n8)
    ss_seed = 8 * ((Y.mean(0) - Y.mean()) ** 2).sum(); ss_cell = n8 * ((Y.mean(1) - Y.mean()) ** 2).sum()
    ms_err = (((Y - Y.mean()) ** 2).sum() - ss_seed - ss_cell) / 98
    say("  seed as block: seed F(14,98)=%.2f p=%.4g" % (ss_seed / 14 / ms_err, stats.f.sf(ss_seed / 14 / ms_err, 14, 98)))
    for t in TERMS:
        ss_t = len(y) * (y * X8[t]).mean() ** 2
        say("    blocked %-6s F(1,98)=%7.2f p=%.4g" % (t, ss_t / ms_err, stats.f.sf(ss_t / ms_err, 1, 98)))


def cellvec(key):
    return np.concatenate([col(F8[k], key) for k in k8])


# ---------------------------------------------------------------------------
head("1. PRIMARY: BEST-CHECKPOINT GREEDY SCORE")
say("Cell means: " + "  ".join("%s %.1f" % (tag(k), col(F8[k], "best").mean()) for k in k8))
PRIM = anova_report(cellvec("best"), "best-checkpoint greedy score")
robustness(cellvec("best"), 20261003)

# ---------------------------------------------------------------------------
head("2. SECONDARY: TIME TO SOLVE (Cox model, censored at 1000)")
from statsmodels.duration.hazard_regression import PHReg
for k in k8:
    t, e = col(F8[k], "t"), col(F8[k], "event")
    say("  %-14s solved %2d/15   median episode of those solved: %s"
        % (tag(k), e.sum(), "%.0f" % np.median(t[e == 1]) if e.sum() else "-"))
Xc = np.column_stack([X8[t] for t in TERMS]).astype(float)
n_ev = int(cellvec("event").sum())
with warnings.catch_warnings(record=True) as wlist:
    warnings.simplefilter("always")
    cox = PHReg(cellvec("t"), Xc, status=cellvec("event"), ties="efron").fit()
converged = cox.mle_retvals.get("converged", False) if hasattr(cox, "mle_retvals") else False
say("  Cox proportional hazards, +-1 coding, Efron ties; hazard ratio = exp(2*beta) for + vs -")
if converged and n_ev >= 10 * len(TERMS):                 # at least 10 events per parameter
    hp = holm(list(cox.pvalues))
    for i, t in enumerate(TERMS):
        say("    %-6s beta %+7.3f  se %6.3f  HR %8.3g  p=%.4g  Holm=%.4g"
            % (t, cox.params[i], cox.bse[i], np.exp(2 * cox.params[i]), cox.pvalues[i], hp[i]))
else:
    say("  NOT ESTIMABLE: %d of 120 runs reached the solved criterion (fit converged: %s; 10 events per"
        % (n_ev, converged))
    say("  parameter would need %d). No coefficients are reported. Deviation from" % (10 * len(TERMS)))
    say("  the time-to-solve comparison is reported as the solve counts per cell above.")
if wlist:
    say("  fit warnings: " + "; ".join(sorted({str(w.message)[:90] for w in wlist})))

# ---------------------------------------------------------------------------
head("3. SECONDARY: STABILITY (logistic regression of collapse on the three factors)")
import statsmodels.api as sm
for k in k8:
    say("  %-14s collapse %2d/15   hold %2d/15" % (tag(k), col(F8[k], "collapse").sum(), col(F8[k], "hold").sum()))
Xl = sm.add_constant(np.column_stack([cO, cI, cR]).astype(float))
logit = sm.Logit(cellvec("collapse"), Xl).fit(disp=0)
hp = holm(list(logit.pvalues[1:]))
say("  logit(collapse) ~ O + I + R, +-1 coding; odds ratio = exp(2*beta) for + vs -")
for i, t in enumerate(["O", "I", "R"]):
    b = logit.params[i + 1]
    say("    %-2s beta %+6.3f  se %5.3f  OR %6.3f  p=%.4g  Holm=%.4g" % (t, b, logit.bse[i + 1], np.exp(2 * b),
                                                                     logit.pvalues[i + 1], hp[i]))

# ---------------------------------------------------------------------------
head("4. SECONDARY: FINAL LEVEL AND AUC (three-way ANOVA)")
for key, lab in [("final_greedy", "greedy score of the network at episode 1000"),
                 ("final20", "mean of the final 20 training episodes"),
                 ("auc", "mean training return over all 1000 episodes (AUC)")]:
    say(); say("-- %s" % lab)
    say("  Cell means: " + "  ".join("%s %.1f" % (tag(k), col(F8[k], key).mean()) for k in k8))
    anova_report(cellvec(key), lab)

# ---------------------------------------------------------------------------
head("5. SKOLIK AGENT VS TWO CELLS (Welch, Holm over 2)")
sk = col("Skolik", "best")
cmp = [("(a) ZZ/30/reup", F8[("ZZ", "HI", 1)]), ("(b) head only Zi/3.2/none", F8[("Zi", "LO", 0)])]
ps = [stats.ttest_ind(sk, col(a, "best"), equal_var=False).pvalue for _, a in cmp]
for (lab, a), p, hpv in zip(cmp, ps, holm(ps)):
    x = col(a, "best")
    say("  Skolik %.1f vs %-26s %.1f  diff %+6.1f  Welch p=%.4g  Holm=%.4g" % (sk.mean(), lab, x.mean(),
                                                                           sk.mean() - x.mean(), p, hpv))
say("  described: solved %d/15, median solve episode %s, collapse %d/15, hold %d/15, greedy @1000 %.1f"
    % (col("Skolik", "event").sum(),
       "%.0f" % np.median(col("Skolik", "t")[col("Skolik", "event") == 1]) if col("Skolik", "event").sum() else "-",
       col("Skolik", "collapse").sum(), col("Skolik", "hold").sum(), col("Skolik", "final_greedy").mean()))

# ---------------------------------------------------------------------------
head("6. PLATEAU: SLOPE OF THE LAST 200 EPISODES (return per 100 episodes)")
rising = 0
for a in ARMS:
    s = col(a, "slope") * 100; lo, hi = tci(s); rising += lo > 0
    say("  %-32s %+6.1f  95%% CI [%+6.1f, %+6.1f]%s" % (a, s.mean(), lo, hi, "  still rising" if lo > 0 else ""))
say("  extension rule: extend to 1500 episodes if more than half of the 9 arms are still rising")
say("  arms still rising: %d/9 -> %s" % (rising, "EXTEND" if rising > 4.5 else "do not extend"))

# ---------------------------------------------------------------------------
head("EXPLORATORY")
yb = cellvec("best")
say("ceiling: %d of 120 factorial runs score exactly 500 on the best-checkpoint greedy score (Skolik %d/15)"
    % (int((yb == 500).sum()), int((sk == 500).sum())))
say("rank-based three-way ANOVA on the best-checkpoint score (ranks of y):")
rr = anova8(stats.rankdata(yb)); hr = dict(zip(TERMS, holm([rr[t][2] for t in TERMS])))
for t in TERMS:
    say("    %-6s F=%6.2f  p=%.4g  Holm=%.4g" % (t, rr[t][1], rr[t][2], hr[t]))
say("pooled by factor (60 runs each side), Mann-Whitney:")
for t, lab in [("O", "readout ZZ vs Zi"), ("I", "range 30 vs 3.2"), ("R", "re-uploading on vs off")]:
    for key in ("best", "final_greedy"):
        y = cellvec(key); a, b = y[X8[t] > 0], y[X8[t] < 0]
        say("  %-24s %-13s %6.1f vs %6.1f  p=%.4g" % (lab, key, a.mean(), b.mean(), stats.mannwhitneyu(a, b).pvalue))
    c = cellvec("collapse")
    say("  %-24s %-13s %2d/60 vs %2d/60  Fisher p=%.4g" % (lab, "collapse", c[X8[t] > 0].sum(), c[X8[t] < 0].sum(),
        stats.fisher_exact([[c[X8[t] > 0].sum(), 60 - c[X8[t] > 0].sum()], [c[X8[t] < 0].sum(), 60 - c[X8[t] < 0].sum()]])[1]))
say("episode of the selected checkpoint (median per cell): " + "  ".join(
    "%s %.0f" % (tag(k), np.median(col(F8[k], "best_ep"))) for k in k8) + "  Skolik %.0f" % np.median(col("Skolik", "best_ep")))
say("300-episode collapse (final-20 at 300 < half final-20 at 120), first 300 episodes:")
for t, lab in [("O", "readout ZZ / Zi"), ("R", "re-uploading on / off")]:
    v = np.concatenate([(H[F8[k]][:, 280:300].mean(1) < 0.5 * H[F8[k]][:, 100:120].mean(1)) for k in k8]).astype(int)
    say("  %-24s %2d/60 vs %2d/60" % (lab, v[X8[t] > 0].sum(), v[X8[t] < 0].sum()))

open(OUT, "w", encoding="utf-8").write("\n".join(_lines) + "\n")
print("\nwrote", OUT)
