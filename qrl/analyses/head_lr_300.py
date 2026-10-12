"""The head learning rate at 300 episodes, per-qubit cells without re-uploading.

The two per-qubit cells without re-uploading, seeds 0-9, 300 episodes, three arms:
  new  Skolik protocol, head lr 0.01   head_lr_300
  skp  Skolik protocol, head lr 0.1    first 300 episodes of skolik_protocol_1000
  qrp  QDQN-R protocol                 cartpole_300
Confirmatory: final-20 at episode 300, OLS y ~ cell + arm on new and skp (40 runs); the effect new - skp is read
against the gap G = qrp - skp. Described: collapse, AUC, greedy scores at episode 300 (results/greedy/), and the
largest |Q| the output head can produce and its movement over the snapshots at episodes 50..300 (results/heads/).
-> analysis/head_lr_300.txt
"""
import numpy as np
from scipy import stats

from .. import data
from ..io import read_json

OUT = data.output("head_lr_300.txt")
CELLS = {"QDQN-R_no-reup": "QDQN-R no-reup", "QDQN-R_no-reup_hi-scale": "QDQN-R no-reup hi-scale"}  # -> label in greedy/cartpole_300
SEEDS, EP, N, HEAD_LR = range(10), 300, 10, 0.01
_lines = []


def say(s=""):
    print(s); _lines.append(s)


def head(s):
    say(); say("=" * 78); say(s); say("=" * 78)


def histories():
    H = {"new": {}, "skp": {}, "qrp": {}}
    for c in CELLS:
        for s in SEEDS:
            j = read_json(data.path("head_lr_300", "%s_seed%d" % (c, s)))
            assert j["label"] == c and j["seed"] == s and j["lr_classical"] == HEAD_LR and len(j["history"]) == EP
            H["new"][c, s] = np.array(j["history"], float)
            H["skp"][c, s] = np.array(data.history("skolik_protocol_1000", "%s_seed%d" % (c, s))[:EP], float)
            h = data.history("cartpole_300", "%s_seed%d" % (c, s))
            assert len(h) == EP
            H["qrp"][c, s] = np.array(h, float)
    return H


def arm_effect(y):
    """OLS y ~ cell + arm on the new and skp arms; returns (effect new - skp, ci low, ci high, p)."""
    import pandas as pd, statsmodels.formula.api as smf
    df = pd.DataFrame([(c, a, y[a][c, s]) for a in ("new", "skp") for c in CELLS for s in SEEDS], columns=["cell", "arm", "y"])
    m = smf.ols("y ~ C(cell) + C(arm, Treatment('skp'))", df).fit()
    k = "C(arm, Treatment('skp'))[T.new]"
    lo, hi = m.conf_int().loc[k]
    return m.params[k], lo, hi, m.pvalues[k]


def reading(est, lo, hi, G):
    if lo > 0 and est >= 0.5 * G:
        return "row 1: the head learning rate of 0.1 is the main cause of the per-qubit drop under SKP"
    if lo > 0:
        return "row 2: the head learning rate is part of the cause; other SKP differences contribute"
    if hi < 0:
        return "row 3: the lower head learning rate makes the per-qubit cells worse"
    if hi < 0.5 * G:
        return "row 4: the head learning rate is not the main cause (recovery above half the gap excluded)"
    return "row 5: inconclusive with 10 seeds"


def greedy_scores():
    cache = data.greedy("head_lr_300_per_qubit")        # "arm|cell|seed" -> greedy returns at episode 300
    q = data.greedy("cartpole_300")
    assert q["env_seeds"] == "10000+k"
    G = {a: {(c, s): np.mean(cache["%s|%s|%d" % (a, c, s)]) for c in CELLS for s in SEEDS} for a in ("new", "skp")}
    G["qrp"] = {(c, s): np.mean(q["results"][lab][str(s)]) for c, lab in CELLS.items() for s in SEEDS}
    return G


def head_stats(arm, c, s):
    """Largest |Q| the head can output and its relative movement, snapshots ep50..300."""
    import torch
    sds = [data.head("head_lr_300" if arm == "new" else "skolik_protocol_1000", "%s_seed%d" % (c, s), e)
           for e in range(50, EP + 1, 50)]
    qmax = [(d["output_head.weight"].abs().sum(1) + d["output_head.bias"].abs()).max().item() for d in sds]
    flat = [torch.cat([d["output_head.weight"].flatten(), d["output_head.bias"]]) for d in sds]
    move = [((b - a).norm() / a.norm()).item() for a, b in zip(flat, flat[1:])]
    return qmax, move


def main(snapshots=True):
    H = histories()
    f20 = {a: {k: h[280:300].mean() for k, h in H[a].items()} for a in H}
    tag = {"QDQN-R_no-reup": "Zi/3.2/none", "QDQN-R_no-reup_hi-scale": "Zi/30/none"}

    head("0. REPRODUCIBILITY")
    same = sum(H["new"][k][0] == H["skp"][k][0] for k in H["new"])
    say("episode-0 return equal to the skp run (same cell and seed): %d/%d" % (same, len(H["new"])))

    head("1. PRIMARY: final-20 at episode 300, OLS y ~ cell + arm (new vs skp, 40 runs)")
    for c in CELLS:
        say("  %-12s " % tag[c] + "   ".join("%s %6.1f (sd %5.1f)" % (a, np.mean([f20[a][c, s] for s in SEEDS]),
                                                                  np.std([f20[a][c, s] for s in SEEDS], ddof=1)) for a in ("new", "skp", "qrp")))
    G = np.mean([np.mean([f20["qrp"][c, s] for s in SEEDS]) - np.mean([f20["skp"][c, s] for s in SEEDS]) for c in CELLS])
    est, lo, hi, p = arm_effect(f20)
    say("  gap G = qrp - skp (mean of the two cells): %.1f" % G)
    say("  arm effect new - skp: %+.1f, 95%% CI [%+.1f, %+.1f], p = %.4g" % (est, lo, hi, p))
    say("  as a fraction of G:   %.2f, 95%% CI [%.2f, %.2f]" % (est / G, lo / G, hi / G))
    say("  reading: " + reading(est, lo, hi, G))
    yn, ys = [f20["new"][k] for k in sorted(H["new"])], [f20["skp"][k] for k in sorted(H["new"])]
    say("  checks: Welch new vs skp p = %.4g; paired t over the 20 (cell, seed) pairs p = %.4g"
        % (stats.ttest_ind(yn, ys, equal_var=False).pvalue, stats.ttest_rel(yn, ys).pvalue))

    head("2. SECONDARY (described, not used for the reading)")
    for a in ("new", "skp", "qrp"):
        col = [H[a][k][280:300].mean() < 0.5 * H[a][k][100:120].mean() for k in sorted(H[a])]
        say("  %-4s collapse (final-20 @300 < half final-20 @120) %2d/20   AUC 0-300 %6.1f"
            % (a, sum(col), np.mean([h.mean() for h in H[a].values()])))
    if snapshots:
        g = greedy_scores()
        for c in CELLS:
            say("  greedy @300, %-12s " % tag[c] + "   ".join("%s %6.1f" % (a, np.mean([g[a][c, s] for s in SEEDS])) for a in ("new", "skp", "qrp")))
        est, lo, hi, p = arm_effect(g)
        say("  greedy @300, arm effect new - skp: %+.1f, 95%% CI [%+.1f, %+.1f], p = %.4g" % (est, lo, hi, p))
        say("  output head, median over runs, snapshots ep50..300:")
        for a in ("new", "skp"):
            st = [head_stats(a, c, s) for c in CELLS for s in SEEDS]
            say("    %-4s largest |Q| the head can output: %s" % (a, " ".join("%7.1f" % x for x in np.median([q for q, _ in st], 0))))
            say("    %-4s relative head movement per 50 ep:  %s" % (a, " ".join("%7.2f" % x for x in np.median([m for _, m in st], 0))))

    open(OUT, "w", encoding="utf-8").write("\n".join(_lines) + "\n")
    print("\nwrote", OUT)
