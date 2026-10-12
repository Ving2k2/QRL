"""The Skolik agent and QDQN-R global-obs under both protocols, 300 episodes, seeds 0-14.

Cells on CartPole-v1: model in {QDQN-R global-obs (GO), Skolik et al. (SK)} x protocol in {QDQN-R (OURP),
Skolik et al. (SKP)}. GO-OURP-v1 and NR-OURP-v1 are runs of cartpole_300; the others are skolik_comparison.
SK-SKP-v0 (their environment) is described, not tested. -> analysis/skolik_comparison.txt
"""
import numpy as np
from scipy import stats

from ..data import greedy as stored_greedy, history, output

OUT = output("skolik_comparison.txt")
_lines = []


def say(s=""):
    print(s); _lines.append(s)


def head(s):
    say(); say("=" * 78); say(s); say("=" * 78)


def hist(cell):
    if cell in ("GO-OURP-v1", "NR-OURP-v1"):
        lab = "QDQN-R_global-obs" if cell == "GO-OURP-v1" else "QDQN-R_no-reup"
        return np.array([history("cartpole_300", "%s_seed%d" % (lab, s)) for s in range(15)], dtype=float)
    return np.array([history("skolik_comparison", "%s_seed%d" % (cell, s)) for s in range(15)], dtype=float)


G300 = stored_greedy("cartpole_300")["results"]
GSK = stored_greedy("skolik_comparison")["results"]


def greedy(cell):
    src, lab = {"GO-OURP-v1": (G300, "QDQN-R global-obs"), "NR-OURP-v1": (G300, "QDQN-R no-reup")}.get(cell, (GSK, cell))
    return np.array([np.mean(src[lab][str(s)]) for s in range(15)])


CELLS = ["GO-OURP-v1", "GO-SKP-v1", "SK-OURP-v1", "SK-SKP-v1", "SK-SKP-v0", "NR-OURP-v1"]
H = {c: hist(c) for c in CELLS}
SC = {c: {"greedy": greedy(c), "final20": H[c][:, 280:].mean(1), "AUC": H[c].mean(1)} for c in CELLS}


def tci(a):
    h = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return a.mean() - h, a.mean() + h


def holm(ps):
    order = np.argsort(ps); out = [0.0] * len(ps); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i])); out[i] = run
    return out


head("CELLS (300 episodes, 15 seeds; mean +- s.d.)")
say("%-11s %-22s %-22s %-22s" % ("cell", "greedy (10 ep, eps=0)", "final-20 training", "AUC 0-300"))
for c in CELLS:
    g, f, a = SC[c]["greedy"], SC[c]["final20"], SC[c]["AUC"]
    say("%-11s %6.1f +- %5.1f          %6.1f +- %5.1f          %6.1f +- %5.1f"
        % (c, g.mean(), g.std(ddof=1), f.mean(), f.std(ddof=1), a.mean(), a.std(ddof=1)))
say("GO = QDQN-R global-obs (84 params), SK = Skolik et al. (62 params), NR = QDQN-R no-reup (90; 74 receive gradient)")
say("OURP = QDQN-R protocol, SKP = Skolik et al. Table 2 'Cart Pole v0, optimal'")

# ---------------------------------------------------------------------------
head("PRIMARY: 2x2 model x protocol on CartPole-v1 (Holm over 3 terms)")
K = [("GO", "OURP"), ("GO", "SKP"), ("SK", "OURP"), ("SK", "SKP")]
cid = np.repeat(np.arange(4), 15)
xM = np.repeat([-1, -1, 1, 1], 15)        # +1 = Skolik model
xP = np.repeat([-1, 1, -1, 1], 15)        # +1 = Skolik protocol
X = {"model (SK - GO)": xM, "protocol (SKP - OURP)": xP, "model x protocol": xM * xP}
for score in ("greedy", "final20", "AUC"):
    y = np.concatenate([SC["%s-%s-v1" % k][score] for k in K])
    ssw = sum(((y[cid == c] - y[cid == c].mean()) ** 2).sum() for c in range(4))
    rows = []
    for t, x in X.items():
        est = (y * x).mean(); f = len(y) * est ** 2 / (ssw / (len(y) - 4))
        rows.append((t, 2 * est, f, stats.f.sf(f, 1, len(y) - 4)))
    hp = holm([r[3] for r in rows])
    say(); say("-- %s%s" % (score, " (primary)" if score == "greedy" else " (secondary)"))
    for (t, e, f, p), h in zip(rows, hp):
        say("  %-22s effect %+7.1f  F(1,56)=%6.2f  p=%.4f  Holm=%.4f" % (t, e, f, p, h))
    w = np.array([1, -1, -1, 1]) / 2.0
    ms = np.array([y[cid == c].mean() for c in range(4)]); vs = np.array([y[cid == c].var(ddof=1) for c in range(4)]) / 15
    est = (w * ms).sum(); se = np.sqrt((w ** 2 * vs).sum()); dfw = (w ** 2 * vs).sum() ** 2 / ((w ** 2 * vs) ** 2 / 14).sum()
    say("  interaction, Welch-type contrast %+7.1f (df %.0f) p=%.4f" % (est, dfw, 2 * stats.t.sf(abs(est / se), dfw)))

# ---------------------------------------------------------------------------
head("DIRECT COMPARISON: Skolik agent vs QDQN-R, each under its own protocol (v1)")
for score in ("greedy", "final20"):
    comps = [("SK-SKP-v1 vs GO-OURP-v1", "SK-SKP-v1", "GO-OURP-v1"), ("SK-SKP-v1 vs NR-OURP-v1", "SK-SKP-v1", "NR-OURP-v1")]
    rows = []
    for name, a, b in comps:
        x, y = SC[a][score], SC[b][score]
        va, vb = x.var(ddof=1) / 15, y.var(ddof=1) / 15; dfw = (va + vb) ** 2 / (va ** 2 / 14 + vb ** 2 / 14)
        hw = stats.t.ppf(0.975, dfw) * np.sqrt(va + vb)
        rows.append((name, x.mean() - y.mean(), x.mean() - y.mean() - hw, x.mean() - y.mean() + hw,
                     stats.ttest_ind(x, y, equal_var=False).pvalue))
    hp = holm([r[4] for r in rows])
    say("-- %s%s" % (score, " (primary)" if score == "greedy" else " (secondary)"))
    for (name, d, lo, hi, p), h in zip(rows, hp):
        say("  %-26s diff %+7.1f  95%% CI [%+.1f, %+.1f]  Welch p=%.4f  Holm=%.4f" % (name, d, lo, hi, p, h))

# ---------------------------------------------------------------------------
head("CARTPOLE-v0 REPLICATION (described, not tested)")
h0 = H["SK-SKP-v0"]
reach, first, keep = 0, [], 0
for s in range(15):
    roll = np.array([h0[s, i:i + 100].mean() for i in range(0, 201)])
    ok = np.where(roll >= 195)[0]
    if len(ok):
        reach += 1; first.append(int(ok[0]) + 100)
    keep += h0[s, 200:].mean() >= 195
    say("  seed %2d  best 100-episode mean %6.1f  %s  final-100 mean %6.1f"
        % (s, roll.max(), ("first reaches 195 at episode %d" % (ok[0] + 100)) if len(ok) else "never reaches 195", h0[s, 200:].mean()))
say("  seeds reaching the v0 criterion within 300 episodes: %d/15; still >= 195 over the final 100: %d/15" % (reach, keep))
say("  share of episodes at the 200-step limit, final 20: %.1f%%" % (100 * (h0[:, 280:] >= 200).mean()))

# ---------------------------------------------------------------------------
head("EXPLORATORY: drops from the best level within 300 episodes")
say("drop = final-20 score below half of the run's best 20-episode moving average")
k20 = np.ones(20) / 20
for c in CELLS:
    best = np.array([np.convolve(h, k20, "valid").max() for h in H[c]])
    fin = H[c][:, -20:].mean(1)
    say("  %-11s mean best-20 %6.1f  final-20 %6.1f  dropped %2d/15" % (c, best.mean(), fin.mean(), int((fin < 0.5 * best).sum())))

# ---------------------------------------------------------------------------
head("CHECKS")
ep0 = np.array([H[c][:, 0] for c in CELLS])
say("episode-0 return identical across all six cells at every seed: %s" % bool((ep0 == ep0[0]).all()))
same = [int(np.array_equal(H["SK-SKP-v0"][s, :k], H["SK-SKP-v1"][s, :k])) for s in range(15)
        for k in [next((i for i, v in enumerate(H["SK-SKP-v1"][s]) if v >= 200), 300)]]
say("SK-SKP v0 and v1 identical up to the first episode of 200 or more steps: %d/15 seeds" % sum(same))
say("epsilon at episode 300: SKP %.3f, OURP %.3f" % (max(0.01, 0.99 ** 300), max(0.01, 0.97 ** 300)))

open(OUT, "w", encoding="utf-8").write("\n".join(_lines) + "\n")
print("\nwrote", OUT)
