"""Exploratory: the factorial at 300 episodes under the two protocols, on the same cells and seeds 0-14.

The Skolik-protocol runs are the first 300 episodes of skolik_protocol_1000; the QDQN-R-protocol runs are cartpole_300.
Per protocol, the factorial terms on the final-20 score at episode 300 (pooled within-cell variance, df 112), and the
difference of the readout gap between the protocols, paired by seed.
-> analysis/exploratory_protocol_300.txt
"""
import numpy as np
from scipy import stats

from .. import data

CELLS = {("Zi", "LO", 1): "QDQN-R_full", ("Zi", "HI", 1): "QDQN-R_full_hi-scale", ("ZZ", "LO", 1): "QDQN-R_global_lo-scale",
         ("ZZ", "HI", 1): "QDQN-R_global-obs", ("Zi", "LO", 0): "QDQN-R_no-reup", ("Zi", "HI", 0): "QDQN-R_no-reup_hi-scale",
         ("ZZ", "LO", 0): "QDQN-R_global_no-reup_lo-scale", ("ZZ", "HI", 0): "QDQN-R_global_no-reup"}
K = list(CELLS)
EXPERIMENT = {"QDQN-R": "cartpole_300", "Skolik": "skolik_protocol_1000"}
TERMS = {"readout": lambda k: 1 if k[0] == "ZZ" else -1, "range": lambda k: 1 if k[1] == "HI" else -1,
         "re-up": lambda k: 1 if k[2] else -1}
TERMS["readout x re-up"] = lambda k: TERMS["readout"](k) * TERMS["re-up"](k)


def final20(protocol, cell, seed, ep=300):
    h = data.history(EXPERIMENT[protocol], "%s_seed%d" % (cell, seed))
    assert len(h) >= ep
    return float(np.mean(h[ep - 20:ep]))


def term(y, f):
    """effect = mean of the + cells minus mean of the - cells, pooled within-cell variance (df 112)"""
    est = 2 * np.mean([f(k) * y[k].mean() for k in K])
    ssw = sum(((y[k] - y[k].mean()) ** 2).sum() for k in K)
    se = np.sqrt(ssw / 112 * sum((f(k) / 4) ** 2 / 15 for k in K))
    h = stats.t.ppf(0.975, 112) * se
    return est, est - h, est + h, (est / se) ** 2, 2 * stats.t.sf(abs(est / se), 112)


def main():
    y = {p: {k: np.array([final20(p, c, s) for s in range(15)]) for k, c in CELLS.items()} for p in EXPERIMENT}
    lines = ["EXPLORATORY: the factorial at 300 episodes under the two protocols (same cells, seeds 0-14)"]
    for p in EXPERIMENT:
        lines.append("== %s protocol, final-20 at episode 300" % p)
        for name, f in TERMS.items():
            lines.append("  %-16s %+7.1f  95%% CI [%+7.1f, %+7.1f]  F=%6.2f  p=%.2g" % ((name,) + term(y[p], f)))
        lines.append("  level (mean over the eight cells) %.1f" % np.mean([y[p][k].mean() for k in K]))
    gap = {p: np.array([np.mean([y[p][k][s] for k in K if k[0] == "ZZ"]) - np.mean([y[p][k][s] for k in K if k[0] == "Zi"])
                        for s in range(15)]) for p in EXPERIMENT}
    d = gap["Skolik"] - gap["QDQN-R"]
    h = stats.t.ppf(0.975, 14) * d.std(ddof=1) / np.sqrt(15)
    lines.append("== readout gap, Skolik minus QDQN-R protocol, paired by seed: %+.1f  95%% CI [%+.1f, %+.1f]  p=%.2g"
                 % (d.mean(), d.mean() - h, d.mean() + h, 2 * stats.t.sf(abs(d.mean() / (h / stats.t.ppf(0.975, 14))), 14)))
    open(data.output("exploratory_protocol_300.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
