"""The readout gap at 1000 episodes at head learning rate 0.01 against 0.1, paired by seed (head_lr_time_limit_1000,
arm A, and skolik_protocol_1000, seeds 0-14).

D(seed) = mean final-20 score at episode 1000 (episodes 981-1000) of the four two-body cells minus that of the four
per-qubit cells (averaged over initial range and re-uploading). Confirmatory: the change D(A) - D(C), paired by seed
over seeds 0-14 (df 14), two-sided t-test, Holm-corrected together with D(A) alone. Check: the first 300 episodes of
the seed 5-9 runs of arm A equal the 300-episode runs they continue. Described: the share of D(C) removed, an unpaired
Welch version, longer score windows, seeds 0-4 against 5-14, the change on seeds 5-14
alone, 1000-episode collapses per readout.
-> analysis/head_lr_1000_paired.txt
"""
import os
import sys

import numpy as np
from scipy import stats

from .. import data
from ..io import read_json

SEEDS = list(range(15))
PER_QUBIT = ["QDQN-R_no-reup", "QDQN-R_no-reup_hi-scale", "QDQN-R_full", "QDQN-R_full_hi-scale"]
TWO_BODY = ["QDQN-R_global_no-reup_lo-scale", "QDQN-R_global_no-reup", "QDQN-R_global_lo-scale", "QDQN-R_global-obs"]
CELLS = PER_QUBIT + TWO_BODY
OUT = []


def say(s=""):
    OUT.append(s); print(s)


def path(arm, cell, s):
    if arm == "A":
        return data.path("head_lr_time_limit_1000", "A_%s_seed%d" % (cell, s))
    return data.path("skolik_protocol_1000", "%s_seed%d" % (cell, s))


def ci_t(d):
    d = np.asarray(d, float); se = d.std(ddof=1) / np.sqrt(len(d)); q = stats.t.ppf(0.975, len(d) - 1)
    return d.mean(), d.mean() - q * se, d.mean() + q * se, 2 * stats.t.sf(abs(d.mean() / se), len(d) - 1)


def holm(ps):
    o = np.argsort(ps); out = [0.0] * len(ps); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); out[i] = run
    return out


def main():
    miss = [path(a, c, s) for a in "AC" for c in CELLS for s in SEEDS if not os.path.exists(path(a, c, s))]
    if miss:
        sys.exit("missing %d result files, e.g. %s" % (len(miss), miss[0]))
    H = {(a, c, s): np.array(read_json(path(a, c, s))["history"][:1000], float) for a in "AC" for c in CELLS for s in SEEDS}
    same = sum(list(H[("A", c, s)][:300]) == read_json(data.path("head_lr_300", "%s_seed%d" % (c, s)))["history"][:300]
               for c in CELLS for s in range(5, 10))
    say("READOUT GAP AT 1000 EPISODES, HEAD LR 0.01 (ARM A) vs 0.1 (ARM C), SEEDS 0-14, SKOLIK PROTOCOL, TERMINAL RULE")
    say("check: first 300 episodes of the seed 5-9 runs equal to the runs they continue: %d/40" % same)

    def D(a, sl=slice(980, 1000), seeds=SEEDS):
        return np.array([np.mean([H[(a, c, s)][sl].mean() for c in TWO_BODY]) -
                         np.mean([H[(a, c, s)][sl].mean() for c in PER_QUBIT]) for s in seeds])

    dA, dC = D("A"), D("C")
    pri, sec = ci_t(dA - dC), ci_t(dA)
    hp = dict(zip(("pri", "sec"), holm([pri[3], sec[3]])))
    say()
    say("level, final-20 at 1000: arm A %.1f, arm C %.1f (mean over 8 cells and 15 seeds)" % (
        np.mean([H[("A", c, s)][980:].mean() for c in CELLS for s in SEEDS]),
        np.mean([H[("C", c, s)][980:].mean() for c in CELLS for s in SEEDS])))
    say("D(C) %+.1f   D(A) %+.1f" % (dC.mean(), dA.mean()))
    say()
    say("CONFIRMATORY: D(A) - D(C), paired by seed over seeds 0-14 (df 14), Holm over 2")
    say("  D(A)-D(C)  %+7.1f  95%% CI [%+.1f, %+.1f]  p=%.4g  Holm=%.4g" % (pri[0], pri[1], pri[2], pri[3], hp["pri"]))
    say("  D(A)       %+7.1f  95%% CI [%+.1f, %+.1f]  p=%.4g  Holm=%.4g" % (sec[0], sec[1], sec[2], sec[3], hp["sec"]))
    say()
    say("DESCRIBED")
    say("  share of D(C) removed: %.0f%% (from the CI of the change: %.0f%% to %.0f%%)"
        % (-100 * pri[0] / dC.mean(), -100 * pri[2] / dC.mean(), -100 * pri[1] / dC.mean()))
    w = stats.ttest_ind(dA, dC, equal_var=False)
    se = np.sqrt(dA.var(ddof=1) / 15 + dC.var(ddof=1) / 15); q = stats.t.ppf(0.975, w.df)
    say("  unpaired Welch: %+.1f  95%% CI [%+.1f, %+.1f]  p=%.4g" % (dA.mean() - dC.mean(), dA.mean() - dC.mean() - q * se,
                                                                  dA.mean() - dC.mean() + q * se, w.pvalue))
    for lab, sl in (("last 50", slice(950, 1000)), ("last 100", slice(900, 1000)), ("episodes 501-1000", slice(500, 1000))):
        a_, c_ = D("A", sl), D("C", sl)
        m, l_, h_, _ = ci_t(a_ - c_)
        say("  %-18s D(C) %+6.1f  D(A) %+6.1f  change %+6.1f [%+.1f, %+.1f]  share removed %.0f%%"
            % (lab, c_.mean(), a_.mean(), m, l_, h_, -100 * m / c_.mean()))
    for lab, seeds in (("seeds 0-4", range(5)), ("seeds 5-14", range(5, 15))):
        say("  %-10s D(A) %+6.1f   D(C) %+6.1f" % (lab, D("A", seeds=seeds).mean(), D("C", seeds=seeds).mean()))
    new = list(range(5, 15))
    c5, a5 = ci_t(D("A", seeds=new) - D("C", seeds=new)), ci_t(D("A", seeds=new))
    say("  seeds 5-14 only (not analysed before the extension): D(A)-D(C) %+.1f [%+.1f, %+.1f] p=%.3g;  D(A) %+.1f [%+.1f, %+.1f]"
        % (c5[0], c5[1], c5[2], c5[3], a5[0], a5[1], a5[2]))
    for a in "AC":
        for lab, cells in (("per-qubit", PER_QUBIT), ("two-body", TWO_BODY)):
            n = sum(H[(a, c, s)][980:].mean() < 0.5 * max(H[(a, c, s)][i:i + 20].mean() for i in range(981))
                    for c in cells for s in SEEDS)
            say("  1000-episode collapses, arm %s, %-9s: %d/60" % (a, lab, n))
    open(data.output("head_lr_1000_paired.txt"), "w", encoding="utf-8").write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
