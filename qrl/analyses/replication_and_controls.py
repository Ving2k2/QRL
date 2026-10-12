"""The replication on new seeds, the head learning rate on Acrobot, the classical DQNs and the head-type control.

  replication  the factorial at 120 episodes on new seeds 15-44 (replication_120): the OxR and IxR interactions,
               Holm over 2
  acrobot      head lr 0.01 vs 0.1 on Acrobot-v1, seeds 0-9 (acrobot_head_lr)
  classical    the classical DQN of Skolik et al. on CartPole-v0 (classical_dqn_skolik), and a classical DQN with
               its own protocol (classical_dqn_own): a 24-point grid on seeds 15-19, the selected setting on 0-14
  observable   two-body observables through the dense head (observable_vs_head, seeds 0-14 at head lr 0.1 and 0-9
               at 0.01): H (head type) and O (observable), Holm over 2; observable_vs_head adds the second stage
-> analysis/replication_and_controls.txt
"""
import os
import sys

import numpy as np
from scipy import stats

from .. import data
from ..classical_dqn import GRID, selected
from ..io import read_json

REPLICATION_SEEDS = list(range(15, 45))
ACROBOT_CELLS = ["QDQN-R (Acrobot)", "QDQN-R no-reup (Acrobot)"]
ACROBOT_HEAD_LR = ["0.01", "0.1"]
ACROBOT_SEEDS = list(range(10))
SKOLIK_DQN_WIDTHS = ["2x20", "2x64"]
SKOLIK_DQN_SEEDS = list(range(15))
DQN_GRID = GRID
DQN_SELECTION_SEEDS, DQN_FINAL_SEEDS = list(range(15, 20)), list(range(15))
OBSERVABLE_SEEDS = {"0.1": list(range(15)), "0.01": list(range(10))}

OUT = data.output("replication_and_controls.txt")
lines = []


def say(s=""):
    print(s); lines.append(s)


def head(s):
    say(); say("=" * 78); say(s); say("=" * 78)


def holm(ps):
    order = np.argsort(ps); out = [0.0] * len(ps); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i])); out[i] = run
    return out


def load(path, n_min):
    h = read_json(path)["history"]
    assert len(h) >= n_min, (path, len(h))
    return np.array(h[:n_min], float)


def need(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print("MISSING %d run(s), e.g. %s" % (len(missing), missing[:3]))
        sys.exit(1)


def run_path(group, key, seed):
    """Result file of one run; `key` is the cell, "cell|head lr", the width or the classical DQN setting."""
    if group == "replication":
        return data.path("replication_120", "%s_seed%d" % (safe(key), seed))
    if group in ("acrobot", "observable"):
        cell, lr = key.split("|")
        return data.path("acrobot_head_lr" if group == "acrobot" else "observable_vs_head",
                         "%s_lr%s_seed%d" % (safe(cell), lr, seed))
    if group == "skolik_dqn":
        return data.path("classical_dqn_skolik", "%s_seed%d" % (key, seed))
    width, lr, target, eps = key.split("|")                  # dqn_selection: the grid; dqn_final: the selected setting
    return data.path("classical_dqn_own", "%s%s_lr%s_target%s_eps%s_seed%d"
                     % ("selection_" if group == "dqn_selection" else "", width, lr, target, eps, seed))


def paired(d):
    """one-sample t on per-seed contrasts: estimate, 95% CI, p"""
    d = np.asarray(d, float); se = d.std(ddof=1) / np.sqrt(len(d)); t = stats.t.ppf(0.975, len(d) - 1)
    return d.mean(), d.mean() - t * se, d.mean() + t * se, 2 * stats.t.sf(abs(d.mean() / se), len(d) - 1)


def safe(s):
    return s.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("=", "")


# ---------------------------------------------------------------------------
# replication of the 120-episode interactions
F8 = {("Zi", "LO", 1): "QDQN-R (full)", ("Zi", "HI", 1): "QDQN-R full hi-scale",
      ("ZZ", "LO", 1): "QDQN-R global lo-scale", ("ZZ", "HI", 1): "QDQN-R global-obs",
      ("Zi", "LO", 0): "QDQN-R no-reup", ("Zi", "HI", 0): "QDQN-R no-reup hi-scale",
      ("ZZ", "LO", 0): "QDQN-R global no-reup lo-scale", ("ZZ", "HI", 0): "QDQN-R global no-reup"}
K8 = list(F8)
SIGN = {"OxR": {k: (1 if k[0] == "ZZ" else -1) * (1 if k[2] else -1) for k in K8},
        "IxR": {k: (1 if k[1] == "HI" else -1) * (1 if k[2] else -1) for k in K8}}


def contrast(y, term, welch=False):
    """difference of differences averaged over the third factor (the paper's +148.8 convention);
    pooled within-cell variance as in the three-way ANOVA (F = t^2), or Welch."""
    w = {k: 0.5 * SIGN[term][k] for k in K8}
    n = {k: len(y[k]) for k in K8}
    est = sum(w[k] * y[k].mean() for k in K8)
    if welch:
        v = {k: w[k] ** 2 * y[k].var(ddof=1) / n[k] for k in K8}
        se = np.sqrt(sum(v.values())); df = sum(v.values()) ** 2 / sum(v[k] ** 2 / (n[k] - 1) for k in K8)
    else:
        df = sum(n.values()) - 8
        s2 = sum(((y[k] - y[k].mean()) ** 2).sum() for k in K8) / df
        se = np.sqrt(s2 * sum(w[k] ** 2 / n[k] for k in K8))
    t = stats.t.ppf(0.975, df)
    return est, est - t * se, est + t * se, 2 * stats.t.sf(abs(est / se), df), df


def analyse_replication():
    head("REPLICATION OF THE 120-EPISODE INTERACTIONS (seeds 15-44, QDQN-R protocol, terminal rule)")
    paths = {k: [run_path("replication", F8[k], s) for s in REPLICATION_SEEDS] for k in K8}
    need([p for v in paths.values() for p in v])
    for k in K8:
        for p in paths[k]:
            j = read_json(p)
            assert j["protocol"] == "QDQN-R" and j["bootstrap_truncation"] is False and len(j["history"]) == 120
    new = {k: np.array([load(p, 120)[100:120].mean() for p in paths[k]]) for k in K8}
    orig = {k: np.array([load(data.path("cartpole_300", "%s_seed%d" % (safe(F8[k]), s)), 120)[100:120].mean()
                         for s in range(15)]) for k in K8}
    say("cell means, final-20 @120:   seeds 0-14 (original)   seeds 15-44 (new)")
    for k in K8:
        say("  %-32s %8.1f   %8.1f" % (F8[k], orig[k].mean(), new[k].mean()))
    res = {t: contrast(new, t) for t in ("OxR", "IxR")}
    hp = dict(zip(res, holm([res[t][3] for t in res])))
    say()
    say("PRIMARY: difference of differences, pooled within-cell variance (df %d), Holm over 2" % res["OxR"][4])
    for t, (est, lo, hi, p, df) in res.items():
        o = contrast(orig, t)[0]
        ok = hp[t] < 0.05 and np.sign(est) == np.sign(o)
        say("  %-4s new %+7.1f  95%% CI [%+7.1f, %+7.1f]  p=%.4f  Holm=%.4f   original (seeds 0-14) %+7.1f"
            % (t, est, lo, hi, p, hp[t], o))
        say("        reading: %s" % ("replicated" if ok else "not replicated"))
    say()
    say("SECONDARY (described, not used for the reading)")
    for t in ("OxR", "IxR"):
        est, lo, hi, p, df = contrast({k: np.log(v) for k, v in new.items()}, t)
        say("  %-4s log scale        %+7.3f  95%% CI [%+7.3f, %+7.3f]  p=%.4f" % (t, est, lo, hi, p))
        est, lo, hi, p, df = contrast(new, t, welch=True)
        say("  %-4s Welch            %+7.1f  95%% CI [%+7.1f, %+7.1f]  p=%.4f (df %.0f)" % (t, est, lo, hi, p, df))
        est, lo, hi, p, df = contrast({k: np.concatenate([orig[k], new[k]]) for k in K8}, t)
        say("  %-4s seeds 0-44 pooled %+7.1f  95%% CI [%+7.1f, %+7.1f]  p=%.4f" % (t, est, lo, hi, p))
    a, b = new[("ZZ", "HI", 1)], new[("ZZ", "HI", 0)]
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    say("  simple effect of re-uploading at ZZ/30: %+7.1f (se %.1f; original +202.8)" % (a.mean() - b.mean(), se))


# ---------------------------------------------------------------------------
# head learning rate on Acrobot
def analyse_acrobot():
    head("HEAD LEARNING RATE ON ACROBOT-v1 (seeds 0-9, 500 episodes, QDQN-R protocol, bootstrap)")
    keys = [(c, lr) for c in ACROBOT_CELLS for lr in ACROBOT_HEAD_LR]
    paths = {k: [run_path("acrobot", "|".join(k), s) for s in ACROBOT_SEEDS] for k in keys}
    need([p for v in paths.values() for p in v])
    for v in paths.values():
        for p in v:
            j = read_json(p)
            assert j["bootstrap_truncation"] is True and j["env"] == "Acrobot-v1" and len(j["history"]) == 500
    H = {k: np.array([load(p, 500) for p in paths[k]]) for k in keys}
    f20 = {k: h[:, 480:500].mean(1) for k, h in H.items()}
    auc = {k: h.mean(1) for k, h in H.items()}
    say("cell means (seeds 0-9):   final-20 @500   AUC 0-500   runs with final-20 = -500")
    for k in keys:
        say("  %-26s lr %-5s %8.1f   %8.1f   %d/10" % (k[0], k[1], f20[k].mean(), auc[k].mean(), int((f20[k] <= -500).sum())))
    c0, c1 = ACROBOT_CELLS
    gain = {c: f20[(c, "0.01")] - f20[(c, "0.1")] for c in ACROBOT_CELLS}
    est, lo, hi, p = paired(0.5 * (gain[c0] + gain[c1]))
    say()
    say("PRIMARY: head lr 0.01 minus 0.1, mean over re-uploading, paired by seed (df 9)")
    say("  effect %+7.1f  95%% CI [%+7.1f, %+7.1f]  p=%.4f" % (est, lo, hi, p))
    reading = ("lowering the head lr also helps on Acrobot" if p < 0.05 and est > 0 else
               "the head lr effect reverses on Acrobot" if p < 0.05 else "not established on Acrobot")
    say("        reading: %s" % reading)
    say()
    say("SECONDARY (described)")
    for c in ACROBOT_CELLS:
        e, l, h_, pp = paired(gain[c])
        say("  head lr effect in %-26s %+7.1f  95%% CI [%+7.1f, %+7.1f]  p=%.4f" % (c, e, l, h_, pp))
    e, l, h_, pp = paired(gain[c0] - gain[c1])
    say("  interaction (effect with re-up minus without) %+7.1f  95%% CI [%+7.1f, %+7.1f]  p=%.4f" % (e, l, h_, pp))
    e, l, h_, pp = paired(0.5 * ((auc[(c0, "0.01")] - auc[(c0, "0.1")]) + (auc[(c1, "0.01")] - auc[(c1, "0.1")])))
    say("  head lr effect on AUC 0-500 %+7.1f  95%% CI [%+7.1f, %+7.1f]  p=%.4f" % (e, l, h_, pp))
    for c, f in ((c0, "QDQN-R_Acrobot"), (c1, "QDQN-R_no-reup_Acrobot")):
        ablation = [data.path("acrobot_head_ablation", "%s_seed%d" % (f, s)) for s in range(5)]
        if all(os.path.exists(p) for p in ablation):
            say("  head ablation run (terminal rule, lr 0.01, seeds 0-4) %-26s final-20 @500 %8.1f"
                % (c, np.mean([load(p, 500)[480:].mean() for p in ablation])))


# ---------------------------------------------------------------------------
# classical DQNs
def solve_episode(h, thr=195.0, win=100):
    c = np.convolve(h, np.ones(win) / win, mode="valid")          # c[i] = mean of h[i:i+win]
    idx = np.nonzero(c >= thr)[0]
    return int(idx[0] + win) if len(idx) else None                 # 1-based episode that closes the window


def analyse_classical():
    head("CLASSICAL DQN OF SKOLIK ET AL. ON CARTPOLE-v0 (seeds 0-14, 1000 episodes)")
    paths = {w: [run_path("skolik_dqn", w, s) for s in SKOLIK_DQN_SEEDS] for w in SKOLIK_DQN_WIDTHS}
    need([p for v in paths.values() for p in v])
    for w in SKOLIK_DQN_WIDTHS:
        hs = [load(p, 1000) for p in paths[w]]
        assert max(h.max() for h in hs) <= 200
        se = [solve_episode(h) for h in hs]
        solved = [e for e in se if e is not None]
        med = np.median([e if e is not None else np.inf for e in se])
        say("  %s: solved %2d/15 within 1000 episodes; median episode to solve %s; solved runs: mean %s, range %s"
            % (w, len(solved), "%.0f" % med if np.isfinite(med) else ">1000",
               "%.0f" % np.mean(solved) if solved else "-", "%d-%d" % (min(solved), max(solved)) if solved else "-"))
        say("        final-20 @1000 mean %.1f" % np.mean([h[-20:].mean() for h in hs]))
        if w == "2x20":
            ok = len(solved) >= 8 and med <= 500
            say("        reference: Skolik et al. report about 250 episodes on average for (20, 20)")
            say("        reading: %s" % ("reproduced" if ok else "not reproduced"))

    head("CLASSICAL DQN WITH ITS OWN PROTOCOL ON CARTPOLE-v1 (bootstrap, 1000 episodes)")
    grid = ["|".join((g["width"], g["lr"], g["target"], g["eps"])) for g in DQN_GRID]
    need([run_path("dqn_selection", k, s) for k in grid for s in DQN_SELECTION_SEEDS])
    say("stage 1 (seeds 15-19): mean final-20 @1000, mean AUC")
    for k in grid:
        hs = [load(run_path("dqn_selection", k, s), 1000) for s in DQN_SELECTION_SEEDS]
        say("  %-26s %7.1f  %7.1f" % (k, np.mean([h[-20:].mean() for h in hs]), np.mean([h.mean() for h in hs])))
    sel = selected(os.path.join(data.RESULTS, "classical_dqn_own"))
    say("selected setting: %s" % sel)
    paths = [run_path("dqn_final", sel, s) for s in DQN_FINAL_SEEDS]
    need(paths)
    hs = np.array([load(p, 1000) for p in paths])
    f300, f1000 = hs[:, 280:300].mean(1), hs[:, 980:1000].mean(1)
    say("stage 2 (seeds 0-14): final-20 @300 %.1f +- %.1f   final-20 @1000 %.1f +- %.1f   AUC 0-1000 %.1f   final-20 >= 475 @1000: %d/15"
        % (f300.mean(), f300.std(ddof=1), f1000.mean(), f1000.std(ddof=1), hs.mean(), int((f1000 >= 475).sum())))
    q = {c: np.mean([load(data.path("cartpole_300", "%s_seed%d" % (safe(c), s)), 300)[280:300].mean()
                     for s in range(15)]) for c in F8.values()}
    lo_q, hi_q = min(q.values()), max(q.values())
    assert abs(lo_q - 251.3) < 0.05 and abs(hi_q - 331.2) < 0.05, (lo_q, hi_q)   # the range of the eight cells at 300 episodes
    m = f300.mean()
    reading = ("above every QDQN-R cell at 300" if m > hi_q else
               "within the range of the QDQN-R cells at 300" if m >= lo_q else "below every QDQN-R cell at 300")
    say("        QDQN-R cells at 300 (QDQN-R protocol, terminal rule, seeds 0-14): %.1f to %.1f" % (lo_q, hi_q))
    say("        reading: %s" % reading)
    b = [data.path("head_lr_time_limit_1000", "B_%s_seed%d" % (safe(c), s)) for c in F8.values() for s in range(5)]
    if all(os.path.exists(p) for p in b):
        say("        described: arm B (head lr 0.01, bootstrap), final-20 @1000 over 8 cells x 5 seeds: %.1f"
            % np.mean([load(p, 1000)[980:].mean() for p in b]))


# ---------------------------------------------------------------------------
# observable vs head type
TRIPLES = {"LO": {"Zi": "QDQN-R_no-reup", "ZZs": "QDQN-R_global_no-reup_lo-scale", "ZZd": "ZZ-dense no-reup lo-scale"},
              "HI": {"Zi": "QDQN-R_no-reup_hi-scale", "ZZs": "QDQN-R_global_no-reup", "ZZd": "ZZ-dense no-reup"}}


def triple_path(kind, init, lr, s):
    c = TRIPLES[init][kind]
    if kind == "ZZd":
        return run_path("observable", c + "|" + lr, s)
    return data.path("skolik_protocol_1000" if lr == "0.1" else "head_lr_300", "%s_seed%d" % (c, s))


def head_reach(snapshot):
    d = data.head(*snapshot)
    if "output_head.weight" in d:
        return (d["output_head.weight"].abs().sum(1) + d["output_head.bias"].abs()).max().item()
    return (d["output_w"].abs() + d["output_b"].abs()).max().item()


def triple_snapshot(kind, init, lr, s):
    c = TRIPLES[init][kind]
    if kind == "ZZd":
        return "observable_vs_head", "%s_lr%s_seed%d" % (safe(c), lr, s), 300
    return ("skolik_protocol_1000" if lr == "0.1" else "head_lr_300"), "%s_seed%d" % (c, s), 300


def analyse_observable():
    head("OBSERVABLE VS HEAD TYPE, FIRST STAGE (no re-uploading, Skolik protocol, terminal rule, 300 episodes)")
    need([triple_path(k, i, lr, s) for lr in ("0.1", "0.01") for s in OBSERVABLE_SEEDS[lr] for i in ("LO", "HI") for k in ("Zi", "ZZs", "ZZd")])
    for lr in ("0.1", "0.01"):
        for s in OBSERVABLE_SEEDS[lr]:
            for i in ("LO", "HI"):
                j = read_json(triple_path("ZZd", i, lr, s))
                assert j["protocol"] == "SKP" and j["bootstrap_truncation"] is False and j["lr_classical"] == float(lr)
    H = {(k, i, lr): np.array([load(triple_path(k, i, lr, s), 300) for s in OBSERVABLE_SEEDS[lr]])
         for lr in ("0.1", "0.01") for i in ("LO", "HI") for k in ("Zi", "ZZs", "ZZd")}
    f20 = {key: h[:, 280:300].mean(1) for key, h in H.items()}
    say("cell means, final-20 @300 (Zi = per-qubit + dense; ZZs = two-body + (w, b); ZZd = two-body + dense, new)")
    for lr in ("0.1", "0.01"):
        for i in ("LO", "HI"):
            say("  head lr %-4s range %s   Zi %6.1f   ZZs %6.1f   ZZd %6.1f   (n=%d)" % (
                lr, "3.2" if i == "LO" else "30 ", f20[("Zi", i, lr)].mean(), f20[("ZZs", i, lr)].mean(),
                f20[("ZZd", i, lr)].mean(), len(f20[("Zi", i, lr)])))

    def part(lr, a, b):                          # per seed, mean over the two ranges of (a - b)
        return 0.5 * sum(f20[(a, i, lr)] - f20[(b, i, lr)] for i in ("LO", "HI"))

    Hc, Oc = part("0.1", "ZZs", "ZZd"), part("0.1", "ZZd", "Zi")
    res = {"H (head type: ZZs - ZZd)": paired(Hc), "O (observable: ZZd - Zi)": paired(Oc)}
    hp = dict(zip(res, holm([r[3] for r in res.values()])))
    say()
    say("PRIMARY: head lr 0.1, seeds 0-14, mean over the two ranges, paired by seed (df 14), Holm over 2")
    for k, (e, l, h_, p) in res.items():
        say("  %-26s %+7.1f  95%% CI [%+7.1f, %+7.1f]  p=%.4f  Holm=%.4f" % (k, e, l, h_, p, hp[k]))
    D = part("0.1", "ZZs", "Zi")
    say("  readout gap D = H + O    %+7.1f (check: %+.1f)" % (D.mean(), Hc.mean() + Oc.mean()))
    hk, ok_ = list(res)
    sigH, sigO = hp[hk] < 0.05, hp[ok_] < 0.05
    reading = ("mainly the head type" if sigH and not sigO else "mainly the observable" if sigO and not sigH else
               "both contribute" if sigH and sigO else "not separated (report the intervals; no claim of no effect)")
    say("        reading: %s" % reading)
    if sigH and sigO:
        say("        shares of D: H %.0f%%, O %.0f%%" % (100 * Hc.mean() / D.mean(), 100 * Oc.mean() / D.mean()))
    say()
    say("SECONDARY (described)")
    for k, d in (("H at head lr 0.01 (seeds 0-9)", part("0.01", "ZZs", "ZZd")), ("O at head lr 0.01 (seeds 0-9)", part("0.01", "ZZd", "Zi"))):
        e, l, h_, p = paired(d)
        say("  %-36s %+7.1f  95%% CI [%+7.1f, %+7.1f]  p=%.4f" % (k, e, l, h_, p))
    n10 = len(OBSERVABLE_SEEDS["0.01"])
    gains = {k: 0.5 * sum(f20[(k, i, "0.01")] - f20[(k, i, "0.1")][:n10] for i in ("LO", "HI")) for k in ("Zi", "ZZs", "ZZd")}
    for k in ("Zi", "ZZs", "ZZd"):
        e, l, h_, p = paired(gains[k])
        say("  lowering the head lr (0.01 - 0.1), %-3s, seeds 0-9 %+7.1f  95%% CI [%+7.1f, %+7.1f]  p=%.4f" % (k, e, l, h_, p))
    e, l, h_, p = paired(gains["ZZd"] - gains["ZZs"])
    say("  head lr x head type, two-body observable (gain ZZd - gain ZZs) %+7.1f  95%% CI [%+7.1f, %+7.1f]  p=%.4f" % (e, l, h_, p))
    say("  collapse (final-20 @300 < half final-20 @120):")
    for lr in ("0.1", "0.01"):
        say("    head lr %-4s " % lr + "   ".join("%s %d/%d" % (k, sum(int((H[(k, i, lr)][:, 280:300].mean(1) < 0.5 * H[(k, i, lr)][:, 100:120].mean(1)).sum())
                                                        for i in ("LO", "HI")), 2 * len(OBSERVABLE_SEEDS[lr])) for k in ("Zi", "ZZs", "ZZd")))
    say("  head reach at ep300, max_a (sum |w_a| + |b_a|), median over runs:")
    for lr in ("0.1", "0.01"):
        row = []
        for k in ("Zi", "ZZs", "ZZd"):
            ps = [triple_snapshot(k, i, lr, s) for i in ("LO", "HI") for s in OBSERVABLE_SEEDS[lr]]
            v = [head_reach(p) for p in ps]
            row.append("%s %s (n=%d)" % (k, "%.1f" % np.median(v) if v else "-", len(v)))
        say("    head lr %-4s " % lr + "   ".join(row))


def main():
    for analyse in (analyse_replication, analyse_acrobot, analyse_classical, analyse_observable):
        analyse()
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\nwrote", OUT)

