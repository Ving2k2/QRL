"""The head type and the observable, both stages (observable_vs_head and skolik_protocol_1000).

Per-qubit observables with the dense head (Zi), two-body observables with the scalar head (ZZs) and two-body
observables with the dense head (ZZd), initial ranges 3.2 and 30, no re-uploading, head lr 0.1, 300 episodes.
Stage 1: seeds 0-14; stage 2: seeds 15-37. Per seed, H = ZZs - ZZd (head type) and O = ZZd - Zi (observable),
averaged over the two ranges. Confirmatory: an inverse-normal combination test per part,
    Z = sqrt(15/38) z1 + sqrt(23/38) z2,   z_k = Phi^-1(F_t(t_k; n_k - 1)),
with t_k the one-sample t statistic of stage k; two-sided p from Z; Holm over the two parts. Stage 2 was decided after
stage 1 had been seen; the combination test keeps its level in that case.
-> analysis/observable_vs_head.txt
"""
import os
import sys

import numpy as np
from scipy import stats

from .. import data
from ..io import read_json

STAGE2 = list(range(15, 38))

OUT = []
STAGE1 = list(range(15))
CELLS = {("Zi", "LO"): "QDQN-R no-reup", ("Zi", "HI"): "QDQN-R no-reup hi-scale",
         ("ZZs", "LO"): "QDQN-R global no-reup lo-scale", ("ZZs", "HI"): "QDQN-R global no-reup",
         ("ZZd", "LO"): "ZZ-dense no-reup lo-scale", ("ZZd", "HI"): "ZZ-dense no-reup"}


def say(s=""):
    OUT.append(s); print(s)


def safe_conv(c):
    return c.replace(" ", "_").replace("(", "").replace(")", "")


def path(kind, rng, s):
    c = CELLS[(kind, rng)]
    if s in STAGE1 and kind != "ZZd":
        return data.path("skolik_protocol_1000", "%s_seed%d" % (safe_conv(c), s))
    return data.path("observable_vs_head", "%s_lr0.1_seed%d" % (safe_conv(c), s))


def f20(kind, rng, s):
    h = np.array(read_json(path(kind, rng, s))["history"], float)
    assert len(h) >= 300
    return h[280:300].mean()


def holm(ps):
    order = np.argsort(ps); out = [0.0] * len(ps); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i])); out[i] = run
    return out


def tstat(d):
    d = np.asarray(d, float); se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean() / se, len(d) - 1


def interval(d):
    d = np.asarray(d, float); se = d.std(ddof=1) / np.sqrt(len(d)); t = stats.t.ppf(0.975, len(d) - 1)
    return d.mean(), d.mean() - t * se, d.mean() + t * se


def main():
    seeds = STAGE1 + STAGE2
    miss = [path(k, r, s) for s in seeds for (k, r) in CELLS if not os.path.exists(path(k, r, s))]
    if miss:
        sys.exit("missing %d result files, e.g. %s" % (len(miss), miss[0]))
    for s in STAGE2:
        for (k, r) in CELLS:
            j = read_json(path(k, r, s))
            assert j["protocol"] == "SKP" and j["bootstrap_truncation"] is False and j["lr_classical"] == 0.1
    y = {(k, r, s): f20(k, r, s) for s in seeds for (k, r) in CELLS}
    part = lambda a, b, ss: np.array([0.5 * sum(y[(a, r, s)] - y[(b, r, s)] for r in ("LO", "HI")) for s in ss])
    say("READOUT GAP AT HEAD LR 0.1: HEAD TYPE (H = ZZs - ZZd) AND OBSERVABLE (O = ZZd - Zi), seeds 0-37")
    say("cell means, final-20 @300:          stage 1 (seeds 0-14)   stage 2 (seeds 15-37)")
    for (k, r) in CELLS:
        say("  %-3s range %-3s  %-32s %8.1f   %8.1f" % (k, "3.2" if r == "LO" else "30", CELLS[(k, r)],
                                                   np.mean([y[(k, r, s)] for s in STAGE1]), np.mean([y[(k, r, s)] for s in STAGE2])))
    w1, w2 = np.sqrt(15 / 38), np.sqrt(23 / 38)
    res = {}
    for lab, (a, b) in (("H (head type: ZZs - ZZd)", ("ZZs", "ZZd")), ("O (observable: ZZd - Zi)", ("ZZd", "Zi"))):
        t1, df1 = tstat(part(a, b, STAGE1)); t2, df2 = tstat(part(a, b, STAGE2))
        z1, z2 = stats.norm.ppf(stats.t.cdf(t1, df1)), stats.norm.ppf(stats.t.cdf(t2, df2))
        Z = w1 * z1 + w2 * z2
        res[lab] = (Z, 2 * stats.norm.sf(abs(Z)), interval(part(a, b, seeds)), interval(part(a, b, STAGE2)), z1, z2)
    hp = dict(zip(res, holm([v[1] for v in res.values()])))
    say()
    say("PRIMARY: inverse-normal combination of stage 1 (n=15) and stage 2 (n=23), weights sqrt(15/38), sqrt(23/38); Holm over 2")
    for lab, (Z, p, (m, lo, hi), (m2, lo2, hi2), z1, z2) in res.items():
        say("  %-26s Z=%+.2f  p=%.4f  Holm=%.4f | all 38 seeds %+7.1f [%+.1f, %+.1f] | stage 2 %+7.1f [%+.1f, %+.1f] | z1 %+.2f z2 %+.2f"
            % (lab, Z, p, hp[lab], m, lo, hi, m2, lo2, hi2, z1, z2))
    D = part("ZZs", "Zi", seeds)
    say("  readout gap D = H + O over 38 seeds: %+.1f" % D.mean())
    hk, ok_ = list(res)
    sigH, sigO = hp[hk] < 0.05, hp[ok_] < 0.05
    reading = ("mainly the head type" if sigH and not sigO else "mainly the observable" if sigO and not sigH else
               "both contribute" if sigH and sigO else "not separated (report the intervals; no claim of no effect)")
    say("        reading: %s" % reading)
    say()
    say("SECONDARY (described)")
    for (k, label) in (("Zi", "Zi"), ("ZZs", "ZZs"), ("ZZd", "ZZd")):
        col = sum(y[(k, r, s)] < 0.5 * np.array(read_json(path(k, r, s))["history"], float)[100:120].mean()
                  for s in STAGE2 for r in ("LO", "HI"))
        say("  300-episode collapses in stage 2, %-3s: %d/46" % (label, col))
    open(data.output("observable_vs_head.txt"), "w", encoding="utf-8").write("\n".join(OUT) + "\n")
    print("wrote analysis/observable_vs_head.txt")

