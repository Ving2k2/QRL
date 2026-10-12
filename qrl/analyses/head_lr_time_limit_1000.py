"""Head learning rate x time-limit rule at 1000 episodes, quantum cells.

The eight factorial cells, Skolik protocol, 1000 episodes, three arms:
  A  head lr 0.01, truncation stored as terminal    head_lr_time_limit_1000/A_*   seeds 0-4
  B  head lr 0.01, bootstrap at the 500-step limit  head_lr_time_limit_1000/B_*   seeds 0-4
  C  head lr 0.1,  truncation stored as terminal    skolik_protocol_1000          seeds 0-14
Outcome: final-20 at episode 1000 (episodes 981-1000). Contrasts of cell means, each averaged equally over the other
two factors:
  D(X) readout gap, two-body minus per-qubit
  U(X) re-uploading effect, on minus off
and R = mean(B) - mean(A) over the eight cells. Standard errors use the pooled within-cell variance of the groups each
contrast involves.
  0  checks: A and B continue their 300-episode sources bit for bit; episode-0 returns agree across A, B and C;
  1  confirmatory, Holm over seven: D(B), D(A) - D(C), D(B) - D(A), U(B), U(A) - U(C), U(B) - U(A), R;
  2  described: D(C), U(C), D(A), U(A); per cell final-20, collapse (final-20 < half the best 20-episode mean),
     AUC 0-1000, share of episodes 300-1000 at 500; the greedy score of the episode-1000 network (10 episodes,
     environment seeds 10000+k; results/greedy/).
-> analysis/head_lr_time_limit_1000.txt
"""
import numpy as np
from scipy import stats

from .. import data
from ..io import read_json

OUT = data.output("head_lr_time_limit_1000.txt")
CELLS = {"QDQN-R_no-reup": ("Zi", "3.2", "off"), "QDQN-R_no-reup_hi-scale": ("Zi", "30", "off"),
         "QDQN-R_global_no-reup_lo-scale": ("ZZ", "3.2", "off"), "QDQN-R_global_no-reup": ("ZZ", "30", "off"),
         "QDQN-R_full": ("Zi", "3.2", "on"), "QDQN-R_full_hi-scale": ("Zi", "30", "on"),
         "QDQN-R_global_lo-scale": ("ZZ", "3.2", "on"), "QDQN-R_global-obs": ("ZZ", "30", "on")}
SEEDS = {"A": range(5), "B": range(5), "C": range(15)}
EP = 1000
_lines = []


def say(s=""):
    print(s); _lines.append(s)


def head(s):
    say(); say("=" * 78); say(s); say("=" * 78)


def source_json(x, arm, s):
    if x == "B":
        return data.path("time_limit_300", "lr0.01_%s_seed%d" % (arm, s))
    return data.path("head_lr_300", "%s_seed%d" % (arm, s))


def histories():
    H = {"A": {}, "B": {}, "C": {}}
    for arm in CELLS:
        for x in ("A", "B"):
            for s in SEEDS[x]:
                j = read_json(data.path("head_lr_time_limit_1000", "%s_%s_seed%d" % (x, arm, s)))
                assert j["label"] == arm and j["seed"] == s and j["armset"] == x and j["lr_classical"] == 0.01
                assert j["bootstrap_truncation"] is (x == "B") and len(j["history"]) == EP
                H[x][arm, s] = np.array(j["history"], float)
        for s in SEEDS["C"]:
            h = data.history("skolik_protocol_1000", "%s_seed%d" % (arm, s))
            assert len(h) == EP
            H["C"][arm, s] = np.array(h, float)
    return H


def contrast(y, w):
    """y[x][(arm, seed)] -> value; w[(x, arm)] -> weight. Returns (estimate, se, df)."""
    groups = {g: np.array([v for (a, s), v in y[g[0]].items() if a == g[1]]) for g in w}
    df = sum(len(v) for v in groups.values()) - len(groups)
    s2 = sum(((v - v.mean()) ** 2).sum() for v in groups.values()) / df
    est = sum(c * groups[g].mean() for g, c in w.items())
    return est, np.sqrt(s2 * sum(c * c / len(groups[g]) for g, c in w.items())), df


def D(x, sign=1.0):
    return {(x, a): sign * (0.25 if ro == "ZZ" else -0.25) for a, (ro, _, _) in CELLS.items()}


def U(x, sign=1.0):
    return {(x, a): sign * (0.25 if re == "on" else -0.25) for a, (_, _, re) in CELLS.items()}


def ci(est, se, df):
    t = stats.t.ppf(0.975, df)
    return est - t * se, est + t * se


def p_two(est, se, df):
    return 2 * stats.t.sf(abs(est / se), df)


def holm(ps):
    order = np.argsort(ps); m = len(ps); out = np.empty(m); run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[j])); out[j] = run
    return out


def reading(xB, xC, hB, hAC, hBA):
    if hB < 0.05 and np.sign(xB) == np.sign(xC):
        return "row 1: the effect persists under the corrected settings"
    if hB < 0.05:
        return "row 2: the effect reverses under the corrected settings"
    if hAC < 0.05 or hBA < 0.05:
        which = " and ".join(n for n, h in (("the head learning rate", hAC), ("the truncation rule", hBA)) if h < 0.05)
        return "row 3: not detected under the corrected settings; it changes with " + which
    return "row 4: inconclusive with 5 seeds per cell"


def greedy_scores():
    cache = data.greedy("head_lr_time_limit_1000")       # "arm|cell|seed" -> greedy returns at episode 1000
    g = {x: {(a, s): np.mean(cache["%s|%s|%d" % (x, a, s)]) for a in CELLS for s in SEEDS[x]} for x in ("A", "B")}
    conv = data.greedy("skolik_protocol_1000")["results"]
    g["C"] = {(a, s): np.mean(conv[a][str(s)]["final_report"]) for a in CELLS for s in SEEDS["C"]}
    return g


def main(greedy=True):
    H = histories()
    y = {x: {k: h[980:1000].mean() for k, h in H[x].items()} for x in H}

    head("0. CHECKS")
    for x in ("A", "B"):
        same = sum(list(H[x][a, s][:300]) == read_json(source_json(x, a, s))["history"] for a in CELLS for s in SEEDS[x])
        say("arm %s: first 300 episodes equal the 300-episode source runs: %d/40" % (x, same))
    e0 = sum(H["A"][k][0] == H["B"][k][0] == H["C"][k][0] for k in H["A"])
    say("episode-0 return equal across A, B and C: %d/40" % e0)

    head("1. PRIMARY: final-20 at episode 1000, Holm over seven")
    tests = [("D(B)", D("B")), ("D(A)-D(C)", {**D("A"), **D("C", -1.0)}), ("D(B)-D(A)", {**D("B"), **D("A", -1.0)}),
             ("U(B)", U("B")), ("U(A)-U(C)", {**U("A"), **U("C", -1.0)}), ("U(B)-U(A)", {**U("B"), **U("A", -1.0)}),
             ("R", {**{("B", a): 1 / 8 for a in CELLS}, **{("A", a): -1 / 8 for a in CELLS}})]
    res = {k: contrast(y, w) for k, w in tests}
    hp = dict(zip(res, holm([p_two(*res[k]) for k in res])))
    for k, (est, se, df) in res.items():
        say("  %-10s %+7.1f  95%% CI [%+7.1f, %+7.1f]  p=%.3g  Holm=%.3g" % (k, est, *ci(est, se, df), p_two(est, se, df), hp[k]))
    DC, UC = contrast(y, D("C")), contrast(y, U("C"))
    say("  reference (convergence, head lr 0.1): D(C) %+7.1f [%+.1f, %+.1f]   U(C) %+7.1f [%+.1f, %+.1f]"
        % (DC[0], *ci(*DC), UC[0], *ci(*UC)))
    say("  reading, readout gap: " + reading(res["D(B)"][0], DC[0], hp["D(B)"], hp["D(A)-D(C)"], hp["D(B)-D(A)"]))
    say("  reading, re-uploading: " + reading(res["U(B)"][0], UC[0], hp["U(B)"], hp["U(A)-U(C)"], hp["U(B)-U(A)"]))
    say("  reading, truncation rule: " + (
        "row A: the rule changes the level reached at 1000 episodes" if hp["R"] < 0.05 else
        "row B: no detectable effect of the rule on the level at 1000 episodes"))

    head("2. SECONDARY (described, not used for the reading)")
    for k, w in (("D(A)", D("A")), ("U(A)", U("A"))):
        e = contrast(y, w); say("  %-5s %+7.1f  95%% CI [%+7.1f, %+7.1f]" % (k, e[0], *ci(*e)))
    say("  per cell: final-20 @1000 | collapse | AUC 0-1000 | share of episodes 300-1000 at 500")
    for arm, (ro, init, re) in CELLS.items():
        row = []
        for x in ("C", "A", "B"):
            hs = [H[x][arm, s] for s in SEEDS[x]]
            col = sum(h[980:].mean() < 0.5 * np.convolve(h, np.ones(20) / 20, "valid").max() for h in hs)
            row.append("%s %6.1f %2d/%-2d %6.1f %4.0f%%" % (x, np.mean([h[980:].mean() for h in hs]), col, len(hs),
                                                       np.mean([h.mean() for h in hs]), 100 * np.mean([np.mean(h[300:] == 500) for h in hs])))
        say("    %s/%-3s/%-3s  %s" % (ro, init, re, "  |  ".join(row)))
    if greedy:
        g = greedy_scores()
        for x in ("C", "A", "B"):
            dg, ug = contrast(g, D(x)), contrast(g, U(x))
            say("  greedy @1000, arm %s: D %+7.1f [%+.1f, %+.1f]   U %+7.1f [%+.1f, %+.1f]" % (x, dg[0], *ci(*dg), ug[0], *ci(*ug)))

    open(OUT, "w", encoding="utf-8").write("\n".join(_lines) + "\n")
    print("\nwrote", OUT)
