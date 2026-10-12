"""The factorial under the bootstrap rule at 300 episodes, and a classical DQN tuned for 300 episodes.

  DQN  a classical DQN tuned for 300 episodes (classical_dqn_300), seeds 0-14: mean final-20 at episode 300 with a
       t-interval, read against the range of the eight QDQN-R cells at 300 episodes (251.3 to 331.2, cartpole_300).
  BOOT the eight cells under the QDQN-R protocol with the bootstrap rule (factorial_300_bootstrap), seeds 0-14,
       300 episodes, against the terminal-rule runs of the same cells and seeds (cartpole_300). Confirmatory, Holm
       over three:
         H1 readout x re-uploading contrast at 120 episodes under the bootstrap rule (pooled within-cell variance);
         H2 the same contrast at 300 episodes under the bootstrap rule;
         H3 bootstrap minus terminal difference of that contrast at 300 episodes, paired by seed (df 14).
       The contrast is the difference of differences averaged over the initial range.
-> analysis/bootstrap_and_dqn_300.txt
"""
import os
import sys

import numpy as np
from scipy import stats

from .. import data
from ..io import read_json

CELLS = ["QDQN-R (full)", "QDQN-R full hi-scale", "QDQN-R global lo-scale", "QDQN-R global-obs",
            "QDQN-R no-reup", "QDQN-R no-reup hi-scale", "QDQN-R global no-reup lo-scale", "QDQN-R global no-reup"]
SEEDS = list(range(15))
DQN300_KEY, DQN300_SEEDS = "2x256|0.0003|100|0.97", list(range(15))

OUT = []
K8 = [("Zi", "LO", 1), ("Zi", "HI", 1), ("ZZ", "LO", 1), ("ZZ", "HI", 1), ("Zi", "LO", 0), ("Zi", "HI", 0), ("ZZ", "LO", 0), ("ZZ", "HI", 0)]
F8 = dict(zip(K8, CELLS))
W = {"OxR": {k: 0.5 * (1 if k[0] == "ZZ" else -1) * (1 if k[2] else -1) for k in K8},
     "IxR": {k: 0.5 * (1 if k[1] == "HI" else -1) * (1 if k[2] else -1) for k in K8}}
QRP_RANGE = (251.3, 331.2)


def say(s=""):
    OUT.append(s); print(s)


def head(s):
    say(); say("=" * 78); say(s); say("=" * 78)


def holm(ps):
    order = np.argsort(ps); out = [0.0] * len(ps); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i])); out[i] = run
    return out


def hist(path, n):
    h = np.array(read_json(path)["history"], float)
    assert len(h) >= n, path
    return h


def need(paths):
    miss = [p for p in paths if not os.path.exists(p)]
    if miss:
        sys.exit("missing %d result files, e.g. %s" % (len(miss), miss[0]))


def safe_long(s):
    return s.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("=", "")


def contrast(y, term):
    """pooled within-cell variance, df = N - 8"""
    w = W[term]; n = {k: len(y[k]) for k in K8}
    est = sum(w[k] * y[k].mean() for k in K8)
    df = sum(n.values()) - 8
    s2 = sum(((y[k] - y[k].mean()) ** 2).sum() for k in K8) / df
    se = np.sqrt(s2 * sum(w[k] ** 2 / n[k] for k in K8)); t = stats.t.ppf(0.975, df)
    return est, est - t * se, est + t * se, 2 * stats.t.sf(abs(est / se), df), df


def paired(d):
    d = np.asarray(d, float); se = d.std(ddof=1) / np.sqrt(len(d)); t = stats.t.ppf(0.975, len(d) - 1)
    return d.mean(), d.mean() - t * se, d.mean() + t * se, 2 * stats.t.sf(abs(d.mean() / se), len(d) - 1)


def analyse_dqn_300():
    head("CLASSICAL DQN TUNED FOR 300 EPISODES (setting %s, seeds 0-14, 300 episodes)" % DQN300_KEY)
    paths = [data.path("classical_dqn_300", "2x256_lr0.0003_target100_eps0.97_seed%d" % s) for s in DQN300_SEEDS]
    need(paths)
    y = np.array([hist(p, 300)[280:300].mean() for p in paths])
    t = stats.t.ppf(0.975, len(y) - 1); se = y.std(ddof=1) / np.sqrt(len(y))
    lo, hi = y.mean() - t * se, y.mean() + t * se
    say("final-20 @300: %.1f +- %.1f (s.d.), 95%% CI [%.1f, %.1f]; QDQN-R cells at 300 (seeds 0-14): %.1f to %.1f"
        % (y.mean(), y.std(ddof=1), lo, hi, *QRP_RANGE))
    if y.mean() >= QRP_RANGE[0]:
        reading = "mean within or above the range of the QDQN-R cells: a classical DQN tuned for 300 episodes matches the circuits"
    elif hi < QRP_RANGE[0]:
        reading = "interval below the lowest QDQN-R cell: the circuits exceed a classical DQN tuned for 300 episodes"
    else:
        reading = "mean below the lowest QDQN-R cell, interval reaching it: not established either way"
    say("        reading: " + reading)
    say("        described: runs with final-20 >= 475 at 300: %d/15; selection-seed value (seeds 15-19) 283.0" % (y >= 475).sum())


def analyse_bootstrap():
    head("EIGHT CELLS, QDQN-R PROTOCOL, BOOTSTRAP RULE (seeds 0-14, 300 episodes) AGAINST THE TERMINAL RULE")
    bp = {k: [data.path("factorial_300_bootstrap", "%s_seed%d" % (safe_long(F8[k]), s)) for s in SEEDS] for k in K8}
    tp = {k: [data.path("cartpole_300", "%s_seed%d" % (safe_long(F8[k]), s)) for s in SEEDS] for k in K8}
    need([p for v in bp.values() for p in v] + [p for v in tp.values() for p in v])
    B = {k: [hist(p, 300) for p in bp[k]] for k in K8}
    T = {k: [hist(p, 300) for p in tp[k]] for k in K8}
    NS = len(SEEDS)
    same0 = sum(B[k][i][0] == T[k][i][0] for k in K8 for i in range(NS))
    say("integrity: episode-0 return equal to the terminal-rule run in %d of %d pairs" % (same0, 8 * NS))
    f = lambda H, a, b: {k: np.array([h[a:b].mean() for h in H[k]]) for k in K8}
    b120, b300, t300 = f(B, 100, 120), f(B, 280, 300), f(T, 280, 300)
    say("cell means, final-20:      bootstrap @120   bootstrap @300   terminal @300 (same seeds)")
    for k in K8:
        say("  %-32s %8.1f        %8.1f        %8.1f" % (F8[k], b120[k].mean(), b300[k].mean(), t300[k].mean()))
    h1, h2 = contrast(b120, "OxR"), contrast(b300, "OxR")
    cs = lambda y: np.array([sum(W["OxR"][k] * y[k][i] for k in K8) for i in range(NS)])
    h3 = paired(cs(b300) - cs(t300))
    hp = holm([h1[3], h2[3], h3[3]])
    say()
    say("PRIMARY: readout x re-uploading contrast; Holm over the three tests")
    say("  H1 bootstrap @120          %+7.1f  95%% CI [%+7.1f, %+7.1f]  p=%.4f  Holm=%.4f" % (h1[0], h1[1], h1[2], h1[3], hp[0]))
    say("  H2 bootstrap @300          %+7.1f  95%% CI [%+7.1f, %+7.1f]  p=%.4f  Holm=%.4f" % (h2[0], h2[1], h2[2], h2[3], hp[1]))
    say("  H3 bootstrap - terminal @300, paired by seed %+7.1f  95%% CI [%+7.1f, %+7.1f]  p=%.4f  Holm=%.4f"
        % (h3[0], h3[1], h3[2], h3[3], hp[2]))
    say("  terminal @300 on the same seeds (reference): %+7.1f" % contrast(t300, "OxR")[0])
    r1 = "detected (positive, as at 120 under the terminal rule)" if hp[0] < 0.05 and h1[0] > 0 else "not detected"
    r2_ = "detected: under the bootstrap rule the interaction persists at 300 episodes" if hp[1] < 0.05 else "not detected at 300 episodes"
    r3 = "the rule changes the 300-episode interaction" if hp[2] < 0.05 else "no detectable effect of the rule on the 300-episode interaction"
    say("        reading: H1 %s; H2 %s; H3 %s" % (r1, r2_, r3))
    say()
    say("SECONDARY (described, not used for the reading)")
    i1, i2 = contrast(b120, "IxR"), contrast(b300, "IxR")
    say("  range x re-uploading, bootstrap: @120 %+7.1f (p=%.3f), @300 %+7.1f (p=%.3f)" % (i1[0], i1[3], i2[0], i2[3]))
    ch = paired(cs(b300) - cs(b120))
    say("  change of the readout x re-uploading contrast 120 -> 300 under the bootstrap rule, paired by seed: %+7.1f [%+.1f, %+.1f] p=%.4f" % ch)
    lev = paired([np.mean([b300[k][i] for k in K8]) - np.mean([t300[k][i] for k in K8]) for i in range(NS)])
    say("  rule effect on the mean level at 300 (bootstrap - terminal), paired by seed: %+7.1f [%+.1f, %+.1f] p=%.4f" % lev)
    for k in K8:
        say("    %-32s capped share @101-120 bootstrap %.3f terminal %.3f"
            % (F8[k], np.mean([np.mean(h[100:120] >= 500) for h in B[k]]), np.mean([np.mean(h[100:120] >= 500) for h in T[k]])))


def main():
    analyse_dqn_300()
    analyse_bootstrap()
    open(data.output("bootstrap_and_dqn_300.txt"), "w", encoding="utf-8").write("\n".join(OUT) + "\n")
    print("wrote analysis/bootstrap_and_dqn_300.txt")

