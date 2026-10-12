"""Arms E and S added to head_lr_time_limit_1000.

  E  head lr 0.1, bootstrap at the 500-step limit   head_lr_time_limit_1000/E_*  seeds 0-4
  S  the Skolik agent, own protocol, bootstrap      head_lr_time_limit_1000/S_*  seeds 0-4
so that the eight cells form a 2 x 2 of head lr (0.1, 0.01) x rule (terminal, bootstrap) at 1000 episodes:
C, A (terminal) and E, B (bootstrap). Outcome: final-20 at episode 1000. With the contrasts of
head_lr_time_limit_1000 (level = mean over the eight cells; D = readout gap; U = re-uploading effect):
  confirmatory, Holm over six:
    rule at head lr 0.1:            level(E) - level(C), D(E) - D(C), U(E) - U(C)
    rule x head lr interaction:     [X(B) - X(A)] - [X(E) - X(C)] for X = level, D, U
  described: the Skolik agent under the bootstrap rule (S) against its terminal-rule runs (15 seeds) and against
  two-body/30/re-up and per-qubit/3.2/none in arm E, at 300 and 1000 episodes (Welch t).
-> analysis/bootstrap_arms_1000.txt
"""
import numpy as np
from scipy import stats

from .. import data
from ..io import read_json
from . import head_lr_time_limit_1000 as Q

OUT = data.output("bootstrap_arms_1000.txt")
SEEDS = range(5)
_lines = []


def say(s=""):
    print(s); _lines.append(s)


def load(x, arm, s):
    j = read_json(data.path("head_lr_time_limit_1000", "%s_%s_seed%d" % (x, arm, s)))
    assert j["label"] == arm and j["seed"] == s and j["armset"] == x and j["lr_classical"] == 0.1
    assert j["bootstrap_truncation"] is True and len(j["history"]) == Q.EP
    return np.array(j["history"], float)


def level(x, sign=1.0):
    return {(x, a): sign / 8 for a in Q.CELLS}


def main():
    H = Q.histories()
    H["E"] = {(a, s): load("E", a, s) for a in Q.CELLS for s in SEEDS}
    S = [load("S", "Skolik", s) for s in SEEDS]
    y = {x: {k: h[980:1000].mean() for k, h in H[x].items()} for x in H}

    say("=" * 78); say("0. CHECKS"); say("=" * 78)
    same = sum(list(H["E"][a, s][:300]) == data.history("time_limit_300", "lr0.1_%s_seed%d" % (a, s))
               for a in Q.CELLS for s in SEEDS)
    say("arm E: first 300 episodes equal the truncation-check runs (head lr 0.1): %d/40" % same)
    say("episode-0 return equal across A, B, C and E: %d/40" % sum(H["A"][k][0] == H["B"][k][0] == H["C"][k][0] == H["E"][k][0] for k in H["A"]))

    say(); say("=" * 78); say("1. PRIMARY: 2 x 2 of head lr x rule at 1000 episodes, Holm over six"); say("=" * 78)
    tests = [
        ("level(E)-level(C)", {**level("E"), **level("C", -1)}),
        ("D(E)-D(C)", {**Q.D("E"), **Q.D("C", -1.0)}),
        ("U(E)-U(C)", {**Q.U("E"), **Q.U("C", -1.0)}),
        ("level x", {**level("B"), **level("A", -1), **level("E", -1), **level("C")}),
        ("D x", {**Q.D("B"), **Q.D("A", -1.0), **Q.D("E", -1.0), **Q.D("C")}),
        ("U x", {**Q.U("B"), **Q.U("A", -1.0), **Q.U("E", -1.0), **Q.U("C")}),
    ]
    res = {k: Q.contrast(y, w) for k, w in tests}
    hp = dict(zip(res, Q.holm([Q.p_two(*res[k]) for k in res])))
    for k, (est, se, df) in res.items():
        verdict = (("the rule changes %s at head lr 0.1" if k.endswith("(C)") else "the rule's effect on %s depends on the head lr")
                   % k.split("(")[0].split(" ")[0]) if hp[k] < 0.05 else "not detected"
        say("  %-18s %+7.1f  95%% CI [%+7.1f, %+7.1f]  p=%.3g  Holm=%.3g  -> %s"
            % (k, est, *Q.ci(est, se, df), Q.p_two(est, se, df), hp[k], verdict))

    say(); say("=" * 78); say("2. SECONDARY (described): the Skolik agent under the bootstrap rule"); say("=" * 78)
    conv = [np.array(data.history("skolik_protocol_1000", "Skolik_seed%d" % s), float)
            for s in range(15)]
    for b in (300, 1000):
        f = lambda hs: np.array([h[b - 20:b].mean() for h in hs])
        sk, sc = f(S), f(conv)
        say("  ep%-4d Skolik bootstrap %6.1f (5 seeds) vs terminal %6.1f (15 seeds): Welch p=%.3g"
            % (b, sk.mean(), sc.mean(), stats.ttest_ind(sk, sc, equal_var=False).pvalue))
        for arm in ("QDQN-R_global-obs", "QDQN-R_no-reup"):
            cell = f([H["E"][arm, s] for s in SEEDS])
            say("         vs %s/%s/%s in arm E %6.1f: difference %+6.1f, Welch p=%.3g"
                % (*Q.CELLS[arm], cell.mean(), sk.mean() - cell.mean(), stats.ttest_ind(sk, cell, equal_var=False).pvalue))

    open(OUT, "w", encoding="utf-8").write("\n".join(_lines) + "\n")
    print("\nwrote", OUT)
