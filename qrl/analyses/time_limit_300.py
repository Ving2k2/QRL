"""The time-limit rule at 300 episodes (time_limit_300 against the terminal-rule runs).

The eight factorial cells x head learning rate 0.1 / 0.01 x seeds 0-4, 300 episodes, under two rules for the
500-step time limit:
  new  bootstrap at truncation   time_limit_300/lr<lr>_<cell>_seed<k>
  old  truncation as terminal    lr 0.1: first 300 episodes of skolik_protocol_1000; lr 0.01: head_lr_300
Outcome: final-20 at episode 300. Within each re-uploading level, contrasts of cell means, each averaged equally over
the two initial ranges, with the pooled within-cell variance of that level's 8 cells (5 runs each, 32 df):
  E_Zi = Zi(0.01) - Zi(0.1)      E_ZZ = ZZ(0.01) - ZZ(0.1)      I = E_ZZ - E_Zi
  D_low = ZZ(0.01) - Zi(0.01)    D_high = ZZ(0.1) - Zi(0.1)
Confirmatory: E_Zi and I at each re-uploading level (4 tests), Holm over the four.
-> analysis/time_limit_300.txt
"""
import numpy as np
from scipy import stats

from .. import data
from ..io import read_json

OUT = data.output("time_limit_300.txt")
CELLS = {"QDQN-R_no-reup": ("Zi", "3.2", "off"), "QDQN-R_no-reup_hi-scale": ("Zi", "30", "off"),
         "QDQN-R_global_no-reup_lo-scale": ("ZZ", "3.2", "off"), "QDQN-R_global_no-reup": ("ZZ", "30", "off"),
         "QDQN-R_full": ("Zi", "3.2", "on"), "QDQN-R_full_hi-scale": ("Zi", "30", "on"),
         "QDQN-R_global_lo-scale": ("ZZ", "3.2", "on"), "QDQN-R_global-obs": ("ZZ", "30", "on")}
LRS, SEEDS, EP, REUP = (0.1, 0.01), range(5), 300, ("off", "on")
PRIMARY = [(r, k) for r in REUP for k in ("E_Zi", "I")]
_lines = []


def say(s=""):
    print(s); _lines.append(s)


def head(s):
    say(); say("=" * 78); say(s); say("=" * 78)


def load(path, n=EP):
    h = read_json(path)["history"]
    assert len(h) >= n
    return np.array(h[:n], float)


def histories():
    H = {"new": {}, "old": {}}
    for arm, (ro, _, re) in CELLS.items():
        for lr in LRS:
            for s in SEEDS:
                j = read_json(data.path("time_limit_300", "lr%g_%s_seed%d" % (lr, arm, s)))
                assert j["label"] == arm and j["seed"] == s and j["lr_classical"] == lr and j["bootstrap_truncation"] is True
                assert len(j["history"]) == EP
                H["new"][arm, lr, s] = np.array(j["history"], float)
                old = "skolik_protocol_1000" if lr == 0.1 else "head_lr_300"
                H["old"][arm, lr, s] = load(data.path(old, "%s_seed%d" % (arm, s)))
    return H


def contrasts(y, reup):
    """y[(arm, lr, seed)] -> {name: (estimate, se, df)} from the 8 cell means of one re-uploading level."""
    arms = [a for a in CELLS if CELLS[a][2] == reup]
    mu = {(arm, lr): np.mean([y[arm, lr, s] for s in SEEDS]) for arm in arms for lr in LRS}
    resid = [y[arm, lr, s] - mu[arm, lr] for arm in arms for lr in LRS for s in SEEDS]
    df = len(resid) - len(mu)
    s2 = np.sum(np.square(resid)) / df
    w = {}
    for arm in arms:
        ro = CELLS[arm][0]
        for lr in LRS:
            sign_lr = 1 if lr == 0.01 else -1
            sign_ro = 1 if ro == "ZZ" else -1
            w.setdefault("E_Zi", {})[arm, lr] = 0.5 * sign_lr if ro == "Zi" else 0.0
            w.setdefault("E_ZZ", {})[arm, lr] = 0.5 * sign_lr if ro == "ZZ" else 0.0
            w.setdefault("I", {})[arm, lr] = 0.5 * sign_lr * sign_ro
            w.setdefault("D_low", {})[arm, lr] = 0.5 * sign_ro if lr == 0.01 else 0.0
            w.setdefault("D_high", {})[arm, lr] = 0.5 * sign_ro if lr == 0.1 else 0.0
    n = len(SEEDS)
    return {k: (sum(c * mu[key] for key, c in wk.items()), np.sqrt(s2 * sum(c * c for c in wk.values()) / n), df)
            for k, wk in w.items()}


def p_two(est, se, df):
    return 2 * stats.t.sf(abs(est / se), df)


def holm(ps):
    order = np.argsort(ps); m = len(ps); out = np.empty(m); run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[j])); out[j] = run
    return out


def reading(new, old, p_new, p_old):
    if p_old >= 0.05:
        return "not established under the old rule: described only"
    same = np.sign(new) == np.sign(old)
    if same and p_new < 0.05:
        return "robust: same direction, established under both rules"
    if same:
        return "direction kept, not established under the new rule"
    if p_new < 0.05:
        return "reversed: the conclusion depends on how truncation is handled"
    return "not preserved: opposite direction, not established"


def main():
    H = histories()
    f20 = {r: {k: h[280:300].mean() for k, h in H[r].items()} for r in H}
    C = {(rule, re): contrasts(f20[rule], re) for rule in H for re in REUP}

    head("0. CHECKS")
    same0 = sum(H["new"][k][0] == H["old"][k][0] for k in H["new"])
    say("episode-0 return equal under both rules (same cell, rate, seed): %d/%d" % (same0, len(H["new"])))

    head("1. PRIMARY: E_Zi and I at each re-uploading level (final-20 at episode 300), Holm over 4")
    P = {rule: dict(zip(PRIMARY, holm([p_two(*C[rule, re][k]) for re, k in PRIMARY]))) for rule in H}
    for re, k in PRIMARY:
        for rule in ("old", "new"):
            est, se, df = C[rule, re][k]
            t = stats.t.ppf(0.975, df)
            say("  re-up %-3s %-5s %-3s %+7.1f  95%% CI [%+7.1f, %+7.1f]  p=%.3g  Holm=%.3g"
                % (re, k, rule, est, est - t * se, est + t * se, p_two(est, se, df), P[rule][re, k]))
        say("        reading: " + reading(C["new", re][k][0], C["old", re][k][0],
                                                          P["new"][re, k], P["old"][re, k]))

    head("2. SECONDARY (described, not used for the reading)")
    for re in REUP:
        for k in ("E_ZZ", "D_low", "D_high", "E_Zi", "I"):
            (en, sn, _), (eo, so, _) = C["new", re][k], C["old", re][k]
            say("  re-up %-3s %-6s old %+7.1f   new %+7.1f   new - old %+7.1f (se %5.1f)"
                % (re, k, eo, en, en - eo, np.hypot(sn, so)))
    say("  per cell: final-20 @300 | share of final-20 episodes at 500 | collapse | AUC 0-300")
    for arm, (ro, init, re) in CELLS.items():
        for lr in LRS:
            row = []
            for rule in ("old", "new"):
                hs = [H[rule][arm, lr, s] for s in SEEDS]
                row.append("%s %6.1f %4.0f%% %d/5 %6.1f" % (rule, np.mean([h[280:].mean() for h in hs]),
                           100 * np.mean([np.mean(h[280:] == 500) for h in hs]),
                           sum(h[280:].mean() < 0.5 * h[100:120].mean() for h in hs), np.mean([h.mean() for h in hs])))
            say("    %s/%-3s/%-3s lr %-4g  %s   |   %s" % (ro, init, re, lr, row[0], row[1]))

    open(OUT, "w", encoding="utf-8").write("\n".join(_lines) + "\n")
    print("\nwrote", OUT)
