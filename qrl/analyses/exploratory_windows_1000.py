"""Exploratory: robustness of the 1000-episode contrasts of head_lr_time_limit_1000.

  1. variance model: the confirmatory contrasts use the pooled within-cell variance. Here the same contrasts with
     unequal variances (Welch-Satterthwaite df) and a percentile bootstrap that resamples runs within each
     (arm, cell) group.
  2. score window: final-20 (confirmatory), final-50, final-100 and the mean over episodes 501-1000.
-> analysis/exploratory_windows_1000.txt
"""
import numpy as np
from scipy import stats

from .. import data
from . import head_lr_time_limit_1000 as Q
from . import bootstrap_arms_1000 as X

OUT = data.output("exploratory_windows_1000.txt")
lines = []


def say(s=""):
    print(s); lines.append(s)


def welch(y, w):
    groups = {g: np.array([v for (a, s), v in y[g[0]].items() if a == g[1]]) for g in w}
    est = sum(c * groups[g].mean() for g, c in w.items())
    terms = [c * c * groups[g].var(ddof=1) / len(groups[g]) for g, c in w.items()]
    se = np.sqrt(sum(terms))
    df = sum(terms) ** 2 / sum(t * t / (len(groups[g]) - 1) for t, g in zip(terms, w))
    return est, se, df


def boot(y, w, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    groups = {g: np.array([v for (a, s), v in y[g[0]].items() if a == g[1]]) for g in w}
    draws = np.empty(n)
    for i in range(n):
        draws[i] = sum(c * rng.choice(groups[g], len(groups[g])).mean() for g, c in w.items())
    return np.percentile(draws, [2.5, 97.5]), np.mean(draws <= 0) if np.mean(draws) > 0 else np.mean(draws >= 0)


H = Q.histories()
H["E"] = {(a, s): X.load("E", a, s) for a in Q.CELLS for s in X.SEEDS}
windows = {"final-20 (primary)": lambda h: h[980:1000].mean(), "final-50": lambda h: h[950:1000].mean(),
           "final-100": lambda h: h[900:1000].mean(), "mean 501-1000": lambda h: h[500:1000].mean()}
tests = [("D(A)-D(C)", {**Q.D("A"), **Q.D("C", -1.0)}), ("D(B)", Q.D("B")), ("D(B)-D(A)", {**Q.D("B"), **Q.D("A", -1.0)}),
         ("U(B)", Q.U("B")), ("R = level(B)-level(A)", {**X.level("B"), **X.level("A", -1.0)})]

say("EXPLORATORY robustness of the 1000-episode contrasts")
for wname, f in windows.items():
    y = {x: {k: f(h) for k, h in H[x].items()} for x in H}
    say()
    say("== score: %s" % wname)
    for name, w in tests:
        est, se, df = Q.contrast(y, w)
        pp = Q.p_two(est, se, df)
        we, wse, wdf = welch(y, w)
        wp = 2 * stats.t.sf(abs(we / wse), wdf)
        (lo, hi), _ = boot(y, w)
        say("  %-22s %+7.1f  pooled CI [%+7.1f, %+7.1f] p=%.3g | Welch CI [%+7.1f, %+7.1f] df=%5.1f p=%.3g | bootstrap 95%% [%+7.1f, %+7.1f]"
            % (name, est, *Q.ci(est, se, df), pp, we - stats.t.ppf(.975, wdf) * wse, we + stats.t.ppf(.975, wdf) * wse, wdf, wp, lo, hi))
open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\nwrote", OUT)
