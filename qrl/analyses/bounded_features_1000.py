"""Classical control for the head effects (bounded_features_1000): a classical network whose four features are bounded
in (-1, 1), read through the three heads of the circuits (dense on the four features, Zi; dense on the products
f0*f2 and f1*f3, ZZd; one scalar pair per action on the same products, ZZs), initial ranges 3.2 and 30, head learning
rates 0.1 and 0.01, Skolik protocol, terminal rule, 1000 episodes, seeds 0-14.

Score: final-20 at episode 1000; per seed and readout, the mean over the two ranges. Confirmatory (one-sample t-tests
of per-seed differences, df 14, Holm over four): H = ZZs - ZZd at 0.1; O = ZZd - Zi at 0.1; G = Zi(0.01) - Zi(0.1);
dD = D(0.01) - D(0.1) with D = ZZs - Zi. Described: cell means, 1000-episode collapses, D, H and O at both rates.
Exploratory: H, G and dD of the circuit without re-uploading minus those of the classical network, paired by seed.
-> analysis/bounded_features_1000.txt
"""
import os
import sys

import numpy as np
from scipy import stats

from .. import data
from ..io import read_json

SEEDS = list(range(15))
READOUTS, RANGES, LRS = ("Zi", "ZZd", "ZZs"), ("LO", "HI"), ("0.1", "0.01")
CELLS = ["%s_%s_%s" % (r, g, lr) for lr in LRS for r in READOUTS for g in RANGES]
OUT = []


def say(s=""):
    OUT.append(s); print(s)


def t1(d):
    d = np.asarray(d, float); se = d.std(ddof=1) / np.sqrt(len(d)); hw = stats.t.ppf(0.975, len(d) - 1) * se
    return d.mean(), d.mean() - hw, d.mean() + hw, stats.ttest_1samp(d, 0.0).pvalue


def mde(d, a):
    n, sd = len(d), np.std(d, ddof=1); tc = stats.t.ppf(1 - a / 2, n - 1); g = np.arange(1, 400, 0.5)
    return g[np.argmax([stats.nct.sf(tc, n - 1, x * np.sqrt(n) / sd) >= 0.8 for x in g])]


def holm(ps):
    o = np.argsort(ps); out = [0.0] * len(ps); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); out[i] = run
    return out


def main():
    p = lambda c, s: data.path("bounded_features_1000", "%s_seed%d" % (c, s))
    miss = [p(c, s) for c in CELLS for s in SEEDS if not os.path.exists(p(c, s))]
    if miss:
        sys.exit("missing %d result files, e.g. %s" % (len(miss), miss[0]))
    H = {(c, s): np.array(read_json(p(c, s))["history"], float) for c in CELLS for s in SEEDS}
    f20 = lambda h: h[980:1000].mean()
    collapsed = lambda h: f20(h) < 0.5 * np.convolve(h, np.ones(20) / 20, "valid").max()
    say("CLASSICAL NETWORK WITH BOUNDED FEATURES, Skolik protocol, terminal rule, 1000 episodes, seeds 0-14")
    say("cell means (s.d.), 1000-episode collapses")
    for c in CELLS:
        v = np.array([f20(H[(c, s)]) for s in SEEDS])
        say("  %-12s %6.1f (%5.1f)  collapsed %2d/15" % (c, v.mean(), v.std(ddof=1), sum(collapsed(H[(c, s)]) for s in SEEDS)))
    R = {(r, lr): np.array([0.5 * sum(f20(H[("%s_%s_%s" % (r, g, lr), s)]) for g in RANGES) for s in SEEDS])
         for r in READOUTS for lr in LRS}
    tests = [("H  = ZZs - ZZd at head lr 0.1", R[("ZZs", "0.1")] - R[("ZZd", "0.1")]),
             ("O  = ZZd - Zi at head lr 0.1", R[("ZZd", "0.1")] - R[("Zi", "0.1")]),
             ("G  = Zi(0.01) - Zi(0.1)", R[("Zi", "0.01")] - R[("Zi", "0.1")]),
             ("dD = D(0.01) - D(0.1), D = ZZs - Zi", (R[("ZZs", "0.01")] - R[("Zi", "0.01")]) - (R[("ZZs", "0.1")] - R[("Zi", "0.1")]))]
    res = [t1(d) for _, d in tests]
    hp = holm([r[3] for r in res])
    say("")
    say("CONFIRMATORY: one-sample t-tests of per-seed differences, df 14, Holm over 4")
    for (lab, _), r, h in zip(tests, res, hp):
        say("  %-40s %+7.1f  95%% CI [%+7.1f, %+7.1f]  p=%.3g  Holm=%.3g" % (lab, r[0], r[1], r[2], r[3], h))
    say("")
    say("DESCRIBED")
    for lr in LRS:
        D, hh, oo = t1(R[("ZZs", lr)] - R[("Zi", lr)]), t1(R[("ZZs", lr)] - R[("ZZd", lr)]), t1(R[("ZZd", lr)] - R[("Zi", lr)])
        say("  head lr %-4s readout gap D %+7.1f [%+7.1f, %+7.1f]; head type %+7.1f [%+7.1f, %+7.1f]; feature part %+7.1f [%+7.1f, %+7.1f]"
            % (lr, D[0], D[1], D[2], hh[0], hh[1], hh[2], oo[0], oo[1], oo[2]))
    say("  smallest effect detectable with 80% power (df 14), at 0.05 and at the strictest Holm threshold 0.0125:")
    for (lab, d) in tests:
        say("    %-40s %5.0f  %5.0f" % (lab, mde(d, 0.05), mde(d, 0.0125)))

    # circuit minus classical on the same seeds (head-type control and arms A and C, without re-uploading)
    from . import head_lr_1000_paired as hlp
    from . import observable_vs_head_1000 as ovh
    q = lambda k, s: 0.5 * sum(f20(np.array(read_json(ovh.path(k, g, s))["history"], float)) for g in RANGES)
    arm = lambda a, cells, s: np.mean([f20(np.array(read_json(hlp.path(a, c, s))["history"][:1000], float)) for c in cells])
    pq, tb = hlp.PER_QUBIT[:2], hlp.TWO_BODY[:2]
    qH = np.array([q("ZZs", s) - q("ZZd", s) for s in SEEDS])
    qG = np.array([arm("A", pq, s) - arm("C", pq, s) for s in SEEDS])
    qdD = np.array([(arm("A", tb, s) - arm("A", pq, s)) - (arm("C", tb, s) - arm("C", pq, s)) for s in SEEDS])
    say("")
    say("EXPLORATORY: circuit minus classical, paired by seed (circuit without re-uploading)")
    for lab, qd, (_, cd) in (("head type H", qH, tests[0]), ("head lr G", qG, tests[2]), ("gap change dD", qdD, tests[3])):
        a, b = t1(qd), t1(qd - cd)
        say("  %-13s circuit %+7.1f  classical %+7.1f  difference %+7.1f  95%% CI [%+7.1f, %+7.1f]  p=%.3g"
            % (lab, a[0], cd.mean(), b[0], b[1], b[2], b[3]))
    open(data.output("bounded_features_1000.txt"), "w", encoding="utf-8").write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
