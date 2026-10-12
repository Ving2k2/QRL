"""Exploratory cross-checks on the stored runs.

  A. Replication seeds 15-44 at 120 episodes: main effects and simple effects of re-uploading.
  B. Interactions at 300 episodes (seeds 0-14) against their replicated 120-episode values (seeds 15-44).
  C. Readout gap D and its change with the head learning rate, per scoring window, arms C, E, A, B.
  D. Time-limit exposure: does the share of capped episodes at episodes 101-120 predict the change from 120 to 300
     episodes within cells? Interaction contrasts at matched numbers of environment steps.
  E. Classical DQN grid (classical_dqn_own selection runs, seeds 15-19) scored at 300 episodes instead of 1000.
  F. Paper-level Holm over the four confirmatory claims of the abstract (reads the outputs of head_lr_300,
     head_lr_time_limit_1000_robustness and skolik_protocol_1000).
  G. Arms on the same seeds 0-4.
-> analysis/exploratory_cross_checks.txt
"""
import os
import re

import numpy as np
from scipy import stats

from .. import data
from ..io import read_json
from .replication_and_controls import (DQN_GRID, DQN_SELECTION_SEEDS, F8, K8, REPLICATION_SEEDS, contrast, holm,
                                       run_path, safe)

OUT = []


def say(s=""):
    OUT.append(s); print(s)


def h(path):
    return np.array(read_json(path)["history"], float)


def orig(k, s):
    return h(data.path("cartpole_300", "%s_seed%d" % (safe(F8[k]), s)))


MAIN = {"readout": lambda k: 1 if k[0] == "ZZ" else -1, "range": lambda k: 1 if k[1] == "HI" else -1,
        "re-up": lambda k: 1 if k[2] else -1}


def main_effect(y, f):
    """mean of the +1 cells minus mean of the -1 cells, pooled within-cell variance"""
    w = {k: f(k) / 4.0 for k in K8}
    n = {k: len(y[k]) for k in K8}
    est = sum(w[k] * y[k].mean() for k in K8)
    df = sum(n.values()) - 8
    s2 = sum(((y[k] - y[k].mean()) ** 2).sum() for k in K8) / df
    se = np.sqrt(s2 * sum(w[k] ** 2 / n[k] for k in K8))
    t = stats.t.ppf(0.975, df)
    return est, est - t * se, est + t * se, 2 * stats.t.sf(abs(est / se), df), se


def contrast_se(y, term):
    est, lo, hi, p, df = contrast(y, term)
    return est, (hi - est) / stats.t.ppf(0.975, df)


say("EXPLORATORY cross-checks")
say()

# ------------------------------------------------------------------ A
say("== A. Replication seeds 15-44, final-20 at 120 episodes")
new = {k: np.array([h(run_path("replication", F8[k], s))[100:120].mean() for s in REPLICATION_SEEDS]) for k in K8}
o120 = {k: np.array([orig(k, s)[100:120].mean() for s in range(15)]) for k in K8}
o300 = {k: np.array([orig(k, s)[280:300].mean() for s in range(15)]) for k in K8}
res = {name: main_effect(new, f) for name, f in MAIN.items()}
ps = holm([r[3] for r in res.values()])
for (name, (est, lo, hi, p, se)), hp in zip(res.items(), ps):
    eo = main_effect(o120, MAIN[name])
    say("  main effect %-8s new %+7.1f  95%% CI [%+7.1f, %+7.1f]  p=%.2g  Holm(3)=%.2g   seeds 0-14 %+7.1f (p=%.2g)"
        % (name, est, lo, hi, p, hp, eo[0], eo[3]))
say("  simple effect of re-uploading (re-up minus none), Welch:")
for ro in ("Zi", "ZZ"):
    for ra in ("LO", "HI"):
        a, b = new[(ro, ra, 1)], new[(ro, ra, 0)]
        t = stats.ttest_ind(a, b, equal_var=False)
        a0, b0 = o120[(ro, ra, 1)], o120[(ro, ra, 0)]
        say("    %s/%s  new %+7.1f (p=%.3f)   seeds 0-14 %+7.1f" % (ro, {"LO": "3.2", "HI": "30"}[ra], a.mean() - b.mean(), t.pvalue,
                                                               a0.mean() - b0.mean()))
say()

# ------------------------------------------------------------------ B
say("== B. Interaction at 300 episodes (seeds 0-14) against its value at 120 episodes on the new seeds 15-44")
say("   independent samples, normal approximation; avoids regression from the selected seeds-0-14 values at 120")
for term in ("OxR", "IxR"):
    e3, s3 = contrast_se(o300, term)
    e1, s1 = contrast_se(new, term)
    d, se = e3 - e1, np.hypot(s3, s1)
    say("  %s  @300 seeds 0-14 %+7.1f (se %.1f)   @120 seeds 15-44 %+7.1f (se %.1f)   difference %+7.1f  95%% CI [%+7.1f, %+7.1f]  z=%.2f  p=%.4f"
        % (term, e3, s3, e1, s1, d, d - 1.96 * se, d + 1.96 * se, d / se, 2 * stats.norm.sf(abs(d / se))))
say()

# ------------------------------------------------------------------ C
say("== C. Readout gap D (two-body minus per-qubit, averaged over range and re-uploading) per scoring window")
C8 = [("Zi", "LO", 0, "QDQN-R_no-reup"), ("Zi", "HI", 0, "QDQN-R_no-reup_hi-scale"),
      ("ZZ", "LO", 0, "QDQN-R_global_no-reup_lo-scale"), ("ZZ", "HI", 0, "QDQN-R_global_no-reup"),
      ("Zi", "LO", 1, "QDQN-R_full"), ("Zi", "HI", 1, "QDQN-R_full_hi-scale"),
      ("ZZ", "LO", 1, "QDQN-R_global_lo-scale"), ("ZZ", "HI", 1, "QDQN-R_global-obs")]


def h1000(arm, c, s):
    p = (data.path("skolik_protocol_1000", "%s_seed%d" % (c, s)) if arm == "C" else
         data.path("head_lr_time_limit_1000", "%s_%s_seed%d" % (arm, c, s)))
    return h(p)[:1000]


ARMS = (("C", 15), ("E", 5), ("A", 5), ("B", 5))
HH = {(a, c[3]): [h1000(a, c[3], s) for s in range(n)] for a, n in ARMS for c in C8}
WINS = (("final-20", slice(980, 1000)), ("final-50", slice(950, 1000)), ("final-100", slice(900, 1000)),
        ("episodes 501-1000", slice(500, 1000)))


def D(arm, sl):
    m = {c[3]: np.mean([x[sl].mean() for x in HH[arm, c[3]]]) for c in C8}
    return np.mean([m[C8[i + 2][3]] - m[C8[i][3]] for i in (0, 1, 4, 5)])


say("  %-18s %8s %8s %8s %8s %12s %10s" % ("window", "D(C)", "D(E)", "D(A)", "D(B)", "D(A)-D(C)", "D(A)/D(C)"))
for lab, sl in WINS:
    d = {a: D(a, sl) for a, _ in ARMS}
    say("  %-18s %+8.1f %+8.1f %+8.1f %+8.1f %+12.1f %10.2f" % (lab, d["C"], d["E"], d["A"], d["B"], d["A"] - d["C"], d["A"] / d["C"]))
say("  (intervals for D(A)-D(C) per window: analysis/exploratory_windows_1000.txt)")
say()

# ------------------------------------------------------------------ D
say("== D. Time-limit exposure under the QDQN-R protocol, seeds 0-14")
x, y, y0, cell = [], [], [], []
for k in K8:
    for s in range(15):
        r = orig(k, s)
        x.append(np.mean(r[100:120] >= 500)); y.append(r[280:300].mean() - r[100:120].mean()); y0.append(r[100:120].mean())
        cell.append(k)
x, y, y0 = np.array(x), np.array(y), np.array(y0)


def within(v):
    v = v.copy()
    for k in K8:
        m = np.array([c == k for c in cell]); v[m] -= v[m].mean()
    return v


xc, yc, y0c = within(x), within(y), within(y0)
r, p = stats.pearsonr(xc, yc)
say("  within-cell correlation of capped share (episodes 101-120) with the change 120->300: r=%.2f, p=%.3g" % (r, p))
say("  (confounded with regression to the mean: the capped share rises with the score at 120)")
X = np.column_stack([y0c, xc]); beta, *_ = np.linalg.lstsq(X, yc, rcond=None)
res_ = yc - X @ beta; dfr = len(yc) - 8 - 2
cov = res_ @ res_ / dfr * np.linalg.inv(X.T @ X)
t_ = beta[1] / np.sqrt(cov[1, 1])
say("  within cells, change ~ score at 120 + capped share: capped-share coefficient %+.0f points per unit share (se %.0f), t=%.2f, p=%.3g;"
    " score-at-120 coefficient %+.2f" % (beta[1], np.sqrt(cov[1, 1]), t_, 2 * stats.t.sf(abs(t_), dfr), beta[0]))
for k in K8:
    m = np.array([c == k for c in cell])
    say("    %-32s capped share %.3f   change %+7.1f" % (F8[k], x[m].mean(), y[m].mean()))
st = {k: np.mean([orig(k, s)[:300].sum() for s in range(15)]) for k in K8}
say("  environment steps in 300 episodes, cell means: %d to %d (single runs %d to %d)"
    % (min(st.values()), max(st.values()), min(orig(k, s)[:300].sum() for k in K8 for s in range(15)),
       max(orig(k, s)[:300].sum() for k in K8 for s in range(15))))
for S in (15000, 20000, 30000, 40000, 50000):
    lvl = {}
    for k in K8:
        v = []
        for s in range(15):
            r = orig(k, s); cs = np.cumsum(r)
            i = int(np.searchsorted(cs, S))
            v.append(r[max(0, i - 19):i + 1].mean() if i < len(r) else np.nan)
        lvl[k] = np.array(v)
    if any(np.isnan(v).any() for v in lvl.values()):
        say("  at %d steps: not reached by every run" % S); continue
    e1, l1, u1, p1, _ = contrast(lvl, "OxR"); e2, l2, u2, p2, _ = contrast(lvl, "IxR")
    say("  at %6d environment steps (mean of the 20 episodes ending there): OxR %+7.1f [%+.1f, %+.1f] p=%.3g   IxR %+7.1f [%+.1f, %+.1f] p=%.3g"
        % (S, e1, l1, u1, p1, e2, l2, u2, p2))
say()

# ------------------------------------------------------------------ E
say("== E. Classical DQN grid (selection runs, seeds 15-19) scored by final-20 at episode 300")
grid = ["|".join((g["width"], g["lr"], g["target"], g["eps"])) for g in DQN_GRID]
sc = {key: np.mean([h(run_path("dqn_selection", key, s))[280:300].mean() for s in DQN_SELECTION_SEEDS]) for key in grid}
best = max(sc, key=sc.get)
say("  best setting at 300 episodes: %s, %.1f (chosen and scored on the same five seeds, so biased upward)" % (best, sc[best]))
for key in sorted(sc, key=sc.get, reverse=True)[:5]:
    say("    %-24s %.1f" % (key, sc[key]))
sel = "2x256|0.0003|1000|0.99"
say("  setting selected by final-20 at 1000: %.1f at 300 on seeds 15-19" % sc[sel])
say()

# ------------------------------------------------------------------ F
say("== F. Paper-level Holm over the four confirmatory claims of the abstract")


def grab(fname, pattern):
    return float(re.search(pattern, open(os.path.join(data.ANALYSIS, fname), encoding="utf-8").read(), re.S).group(1))


R2 = contrast(new, "OxR")[3]
claims = [("readout x re-uploading, replication", R2),
          ("head lr, per-qubit cells without re-uploading, 300 ep", grab("head_lr_300.txt", r"arm effect new - skp: \+247.8.*?p = ([0-9.eE-]+)")),
          ("D(A)-D(C), 1000 ep", grab("exploratory_windows_1000.txt", r"D\(A\)-D\(C\)\s+-125.2\s+pooled CI \[[^\]]+\] p=([0-9.eE-]+)")),
          ("readout gap, arm C, 1000 ep", grab("skolik_protocol_1000.txt", r"O\s+effect\s+\+159.3\s+F=\s*[0-9.]+\s+p=([0-9.eE-]+)"))]
for (lab, p), hp in zip(claims, holm([c[1] for c in claims])):
    say("  %-55s p=%.2g  Holm(4)=%.2g" % (lab, p, hp))

open(data.output("exploratory_cross_checks.txt"), "w", encoding="utf-8").write("\n".join(OUT) + "\n")

# ------------------------------------------------------------------ G
say()
say("== G. Arms on the same seeds 0-4: D and level of arm C on seeds 0-4, and seed-paired arm contrasts (df 4)")


def Ds(arm, s, sl=slice(980, 1000)):
    m = {c[3]: HH[arm, c[3]][s][sl].mean() for c in C8}
    return np.mean([m[C8[i + 2][3]] - m[C8[i][3]] for i in (0, 1, 4, 5)])


def lev(arm, s, sl=slice(980, 1000)):
    return np.mean([HH[arm, c[3]][s][sl].mean() for c in C8])


for lab, f in (("D", Ds), ("level", lev)):
    say("  %s, final-20: C seeds 0-14 %.1f   C seeds 0-4 %.1f   E %.1f   A %.1f   B %.1f"
        % (lab, np.mean([f("C", s) for s in range(15)]), np.mean([f("C", s) for s in range(5)]),
           np.mean([f("E", s) for s in range(5)]), np.mean([f("A", s) for s in range(5)]), np.mean([f("B", s) for s in range(5)])))
for lab, a, b in (("D(A)-D(C)", "A", "C"), ("D(B)-D(A)", "B", "A"), ("D(E)-D(C)", "E", "C")):
    d = np.array([Ds(a, s) - Ds(b, s) for s in range(5)])
    se = d.std(ddof=1) / np.sqrt(5); t = stats.t.ppf(0.975, 4)
    say("  %-10s seed-paired %+7.1f  95%% CI [%+7.1f, %+7.1f]  p=%.3f" % (lab, d.mean(), d.mean() - t * se, d.mean() + t * se,
                                                                     2 * stats.t.sf(abs(d.mean() / se), 4)))

open(data.output("exploratory_cross_checks.txt"), "w", encoding="utf-8").write("\n".join(OUT) + "\n")
