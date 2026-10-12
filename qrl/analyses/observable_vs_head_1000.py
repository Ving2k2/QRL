"""The head type and the observable at 1000 episodes (observable_vs_head_1000 and skolik_protocol_1000).

Per-qubit observables with the dense head (Zi), two-body observables with the scalar head (ZZs) and two-body
observables with the dense head (ZZd), initial ranges 3.2 and 30, no re-uploading, head lr 0.1, seeds 0-14.
Confirmatory: final-20 score at episode 1000 (episodes 981-1000); per seed, H = ZZs - ZZd (head type) and
O = ZZd - Zi (observable), averaged over the two ranges; one-sample t-test of the per-seed differences (df 14),
two-sided; Holm over the two parts. Check: the first 300 episodes of every ZZd run equal the 300-episode run of the
same cell and seed. Described: the parts over the last 50 and 100 episodes and episodes 501-1000; cell means;
1000-episode collapses (final-20 below half of the best 20-episode mean) by readout.
-> analysis/observable_vs_head_1000.txt
"""
import os
import sys

import numpy as np
from scipy import stats

from .. import data
from ..io import read_json

SEEDS = list(range(15))
CELLS = {("Zi", "LO"): "QDQN-R no-reup", ("Zi", "HI"): "QDQN-R no-reup hi-scale",
         ("ZZs", "LO"): "QDQN-R global no-reup lo-scale", ("ZZs", "HI"): "QDQN-R global no-reup",
         ("ZZd", "LO"): "ZZ-dense no-reup lo-scale", ("ZZd", "HI"): "ZZ-dense no-reup"}
WINDOWS = [("last 50", 950, 1000), ("last 100", 900, 1000), ("episodes 501-1000", 500, 1000)]
OUT = []


def say(s=""):
    OUT.append(s); print(s)


def safe(c):
    return c.replace(" ", "_")


def path(kind, rng, s):
    c = CELLS[(kind, rng)]
    if kind == "ZZd":
        return data.path("observable_vs_head_1000", "%s_seed%d" % (safe(c), s))
    return data.path("skolik_protocol_1000", "%s_seed%d" % (safe(c), s))


def holm(ps):
    order = np.argsort(ps); out = [0.0] * len(ps); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i])); out[i] = run
    return out


def ttest(d):
    d = np.asarray(d, float); se = d.std(ddof=1) / np.sqrt(len(d)); tq = stats.t.ppf(0.975, len(d) - 1)
    return d.mean(), d.mean() - tq * se, d.mean() + tq * se, 2 * stats.t.sf(abs(d.mean() / se), len(d) - 1)


def main():
    miss = [path(k, r, s) for s in SEEDS for (k, r) in CELLS if not os.path.exists(path(k, r, s))]
    if miss:
        sys.exit("missing %d result files, e.g. %s" % (len(miss), miss[0]))
    H = {(k, r, s): np.array(read_json(path(k, r, s))["history"], float) for s in SEEDS for (k, r) in CELLS}
    assert all(len(h) >= 1000 for h in H.values())
    same = 0
    for s in SEEDS:
        for r in ("LO", "HI"):
            j = read_json(path("ZZd", r, s))
            assert j["protocol"] == "SKP" and j["bootstrap_truncation"] is False and j["lr_classical"] == 0.1
            short = read_json(data.path("observable_vs_head", "%s_lr0.1_seed%d" % (safe(CELLS[("ZZd", r)]), s)))
            same += list(H[("ZZd", r, s)][:300]) == short["history"][:300]
    say("READOUT GAP AT HEAD LR 0.1 AND 1000 EPISODES, NO RE-UPLOADING: HEAD TYPE (H = ZZs - ZZd) AND OBSERVABLE (O = ZZd - Zi)")
    say("check: first 300 episodes of the ZZd runs identical to the 300-episode runs: %d/30" % same)
    say()
    say("cell means, final-20 at 1000 (seeds 0-14):")
    for (k, r) in CELLS:
        say("  %-3s range %-3s  %-32s %7.1f" % (k, "3.2" if r == "LO" else "30", CELLS[(k, r)],
                                               np.mean([H[(k, r, s)][980:1000].mean() for s in SEEDS])))

    def part(a, b, lo, hi):
        return np.array([0.5 * sum(H[(a, r, s)][lo:hi].mean() - H[(b, r, s)][lo:hi].mean() for r in ("LO", "HI"))
                         for s in SEEDS])

    res = {lab: ttest(part(a, b, 980, 1000)) for lab, (a, b) in
           (("H (head type: ZZs - ZZd)", ("ZZs", "ZZd")), ("O (observable: ZZd - Zi)", ("ZZd", "Zi")))}
    hp = dict(zip(res, holm([v[3] for v in res.values()])))
    say()
    say("PRIMARY: final-20 at 1000, one-sample t over seeds 0-14 (df 14), Holm over 2")
    for lab, (m, lo, hi, p) in res.items():
        say("  %-26s %+7.1f  95%% CI [%+.1f, %+.1f]  p=%.2g  Holm=%.2g" % (lab, m, lo, hi, p, hp[lab]))
    say("  readout gap D = H + O: %+.1f" % part("ZZs", "Zi", 980, 1000).mean())
    hk, ok_ = list(res)
    sigH, sigO = hp[hk] < 0.05, hp[ok_] < 0.05
    reading = ("mainly the head type" if sigH and not sigO else "mainly the observable" if sigO and not sigH else
               "both contribute" if sigH and sigO else "not separated (report the intervals; no claim of no effect)")
    say("        reading: %s" % reading)
    say()
    say("SECONDARY (described)")
    for lab, lo, hi in WINDOWS:
        h_, o_ = ttest(part("ZZs", "ZZd", lo, hi)), ttest(part("ZZd", "Zi", lo, hi))
        say("  %-18s H %+7.1f [%+.1f, %+.1f]   O %+7.1f [%+.1f, %+.1f]" % (lab, h_[0], h_[1], h_[2], o_[0], o_[1], o_[2]))
    for k in ("Zi", "ZZs", "ZZd"):
        col = 0
        for s in SEEDS:
            for r in ("LO", "HI"):
                h = H[(k, r, s)][:1000]
                best = max(h[i:i + 20].mean() for i in range(0, 981))
                col += h[980:1000].mean() < 0.5 * best
        say("  1000-episode collapses, %-3s: %d/30" % (k, col))
    open(data.output("observable_vs_head_1000.txt"), "w", encoding="utf-8").write("\n".join(OUT) + "\n")
    print("wrote analysis/observable_vs_head_1000.txt")
