"""Exploratory checks of the factorial, on the stored runs only.

  1. The two 120-episode interactions (readout x re-uploading, range x re-uploading) on seeds 0-4 (the first
     five-seed pass) and on seeds 5-14 (added afterwards), QDQN-R protocol (cartpole_300, first 120 episodes).
  2. The same two interaction contrasts with 95% CIs at 120 and 300 episodes, on the final-20 and the greedy score
     (15 seeds).
  3. Shots needed to read the trained heads: with N shots per observable the standard error of Q(s,a) is at most
     ||w_a||_2 / sqrt(N), so N = ||w_a||^2 gives a standard error of at most one reward point. Per run the largest
     ||w_a||_2 over actions (results/heads/); medians over runs.
-> analysis/exploratory_factorial.txt
"""
import numpy as np
from scipy import stats

from .. import data

OUT = data.output("exploratory_factorial.txt")
lines = []


def say(s=""):
    print(s); lines.append(s)


def safe(s):
    return s.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("=", "")


F8 = {("Zi", "LO", 1): "QDQN-R (full)", ("Zi", "HI", 1): "QDQN-R full hi-scale",
      ("ZZ", "LO", 1): "QDQN-R global lo-scale", ("ZZ", "HI", 1): "QDQN-R global-obs",
      ("Zi", "LO", 0): "QDQN-R no-reup", ("Zi", "HI", 0): "QDQN-R no-reup hi-scale",
      ("ZZ", "LO", 0): "QDQN-R global no-reup lo-scale", ("ZZ", "HI", 0): "QDQN-R global no-reup"}
H = {k: np.array([data.history("cartpole_300", "%s_seed%d" % (safe(v), s))
                  for s in range(15)], float) for k, v in F8.items()}
G = {w: data.greedy(f)["results"] for w, f in (("120", "cartpole_120"), ("300", "cartpole_300"))}

# contrast weights: difference of differences, averaged over the third factor (paper's +148.8 convention)
W = {"readout x re-up": {k: 0.5 * (1 if k[0] == "ZZ" else -1) * (1 if k[2] else -1) for k in F8},
     "range x re-up": {k: 0.5 * (1 if k[1] == "HI" else -1) * (1 if k[2] else -1) for k in F8}}


def contrast(y, w, seeds):
    g = {k: y[k][seeds] for k in F8}
    df = sum(len(v) for v in g.values()) - len(g)
    s2 = sum(((v - v.mean()) ** 2).sum() for v in g.values()) / df
    est = sum(c * g[k].mean() for k, c in w.items())
    se = np.sqrt(s2 * sum(c * c / len(g[k]) for k, c in w.items()))
    t = stats.t.ppf(0.975, df)
    return est, est - t * se, est + t * se, 2 * stats.t.sf(abs(est / se), df), df


final20 = {w: {k: h[:, int(w) - 20:int(w)].mean(1) for k, h in H.items()} for w in ("120", "300")}
greedy = {w: {k: np.array([np.mean(G[w][F8[k]][str(s)]) for s in range(15)]) for k in F8} for w in G}

say("EXPLORATORY checks of the factorial")
say()
say("1. 120-episode interactions by seed block (final-20 at 120, QDQN-R protocol, pooled within-cell variance)")
for name, w in W.items():
    for lab, seeds in (("seeds 0-4 ", np.arange(5)), ("seeds 5-14", np.arange(5, 15)), ("seeds 0-14", np.arange(15))):
        est, lo, hi, p, df = contrast(final20["120"], w, seeds)
        say("  %-16s %s  %+7.1f  95%% CI [%+7.1f, %+7.1f]  p=%.4f (df %d)" % (name, lab, est, lo, hi, p, df))
say()
say("2. Interaction contrasts with 95% CIs, 15 seeds")
for name, w in W.items():
    for lab, y in (("final-20 @120", final20["120"]), ("final-20 @300", final20["300"]),
                   ("greedy   @120", greedy["120"]), ("greedy   @300", greedy["300"])):
        est, lo, hi, p, df = contrast(y, w, np.arange(15))
        say("  %-16s %s  %+7.1f  95%% CI [%+7.1f, %+7.1f]  p=%.4f" % (name, lab, est, lo, hi, p))

say()
say("2b. Change of each interaction between 120 and 300 episodes, paired by run (score = final-20 @300 minus final-20 @120)")
diff = {k: final20["300"][k] - final20["120"][k] for k in F8}
for name, w in W.items():
    est, lo, hi, p, df = contrast(diff, w, np.arange(15))
    say("  %-16s change %+7.1f  95%% CI [%+7.1f, %+7.1f]  p=%.4f" % (name, est, lo, hi, p))
s2 = sum(((final20["300"][k] - final20["300"][k].mean()) ** 2).sum() for k in F8) / 112
se = np.sqrt(s2 * 8 * 0.25 / 15)
say("  minimum detectable interaction contrast at 300 (80%% power, two-sided 0.05): %.1f points (pooled SD %.1f)"
    % ((stats.t.ppf(0.975, 112) + stats.t.ppf(0.8, 112)) * se, np.sqrt(s2)))

say()
say("3. Shots per observable needed for a standard error of at most 1 point on Q (N = max_a ||w_a||_2^2), median over runs")


def head_norm(snapshot):
    d = data.head(*snapshot)
    if "output_head.weight" in d:                        # per-qubit readout: dense head, rows w_a
        return float(d["output_head.weight"].norm(dim=1).max())
    w = [v for k, v in d.items() if k.endswith("output_w")]   # two-body readout: one scalar w_a per action
    return float(w[0].abs().max()) if w else np.nan


def report(label, snapshots):
    v = np.array([head_norm(p) for p in snapshots])
    if len(v):
        say("  %-58s n=%2d  max_a||w_a||_2 median %7.1f  -> N for SE<=1: median %9.0f (range %.0f-%.0f)"
            % (label, len(v), np.median(v), np.median(v ** 2), (v ** 2).min(), (v ** 2).max()))
    else:
        say("  %-58s no snapshots found" % label)


NOREUP_ZI = ("QDQN-R_no-reup", "QDQN-R_no-reup_hi-scale")
report("per-qubit, no re-up, 300 ep, head lr 0.01 (Skolik protocol)",
       [("head_lr_300", "%s_seed%d" % (c, s), 300) for c in NOREUP_ZI for s in range(10)])
report("per-qubit, no re-up, 300 ep, head lr 0.1 (Skolik protocol)",
       [("skolik_protocol_1000", "%s_seed%d" % (c, s), 300) for c in NOREUP_ZI for s in range(10)])
CELLS = ["QDQN-R_no-reup", "QDQN-R_no-reup_hi-scale", "QDQN-R_full", "QDQN-R_full_hi-scale",
         "QDQN-R_global_no-reup_lo-scale", "QDQN-R_global_no-reup", "QDQN-R_global_lo-scale", "QDQN-R_global-obs"]
for ro, cells in (("per-qubit", CELLS[:4]), ("two-body", CELLS[4:])):
    report("%s, 1000 ep, arm C (head lr 0.1, terminal)" % ro,
           [("skolik_protocol_1000", "%s_seed%d" % (c, s), 1000) for c in cells for s in range(15)])
    for x, lab in (("A", "head lr 0.01, terminal"), ("B", "head lr 0.01, bootstrap"), ("E", "head lr 0.1, bootstrap")):
        report("%s, 1000 ep, arm %s (%s)" % (ro, x, lab),
               [("head_lr_time_limit_1000", "%s_%s_seed%d" % (x, c, s), 1000) for c in cells for s in range(5)])
open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\nwrote", OUT)
