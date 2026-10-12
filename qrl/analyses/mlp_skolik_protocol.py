"""The MLP frontier under the Skolik protocol against the QDQN-R cells, at 300 and 1000 episodes.

The MLP frontier (mlp_skolik_protocol) is scored for each rule for the 500-step limit (old: terminal; boot:
bootstrap) and each budget (final-20 at that budget):
  selected      per width and input, the learning rate with the best mean on seeds 15-19, reported as its mean on
                seeds 0-14; the larger of the raw- and tanh-input values
  best of ten   per width, the best of the ten (input x lr) settings chosen on seeds 0-14 (favours the classical side)
  bound         the largest best-of-ten value over the five widths
It is compared with the eight QDQN-R cells under the Skolik protocol with head lr 0.01 and the same rule:
  300  old   head_lr_300                     seeds 0-9
  300  boot  time_limit_300 (lr0.01_*)       seeds 0-4
  1000 old   head_lr_time_limit_1000 (A_*)   seeds 0-4
  1000 boot  head_lr_time_limit_1000 (B_*)   seeds 0-4
and, for reference, with head lr 0.1 (skolik_protocol_1000, old rule).
Reading for each rule and budget: "holds" if every cell exceeds the bound; "partly" if at least one cell exceeds the
largest selected value but not every cell exceeds the bound; "fails" otherwise.
-> analysis/mlp_skolik_protocol.txt
"""
import numpy as np

from .. import data

OUT = data.output("mlp_skolik_protocol.txt")
WIDTHS = ["1x12", "1x16", "2x8", "2x16", "2x64"]
INPUTS = ["raw", "tanh"]
LRS = ["0.001", "0.002", "0.005", "0.01", "0.02"]
CELLS = {"QDQN-R_no-reup": "Zi/3.2/off", "QDQN-R_no-reup_hi-scale": "Zi/30/off",
         "QDQN-R_global_no-reup_lo-scale": "ZZ/3.2/off", "QDQN-R_global_no-reup": "ZZ/30/off",
         "QDQN-R_full": "Zi/3.2/on", "QDQN-R_full_hi-scale": "Zi/30/on",
         "QDQN-R_global_lo-scale": "ZZ/3.2/on", "QDQN-R_global-obs": "ZZ/30/on"}
_lines = []


def say(s=""):
    print(s); _lines.append(s)


def f20(h, budget):
    return float(np.mean(h[budget - 20:budget]))


def mlp_frontier(rule, budget):
    score = {}
    for w in WIDTHS:
        for i in INPUTS:
            for lr in LRS:
                for s in range(20):
                    h = data.history("mlp_skolik_protocol", "%s_%s_%s_lr%s_seed%d" % (
                        "terminal" if rule == "old" else "bootstrap", w, i, lr, s))
                    score[w, i, lr, s] = f20(h, budget)
    mean = lambda w, i, lr, seeds: np.mean([score[w, i, lr, s] for s in seeds])
    rows = {}
    for w in WIDTHS:
        sel = max(mean(w, i, max(LRS, key=lambda lr: mean(w, i, lr, range(15, 20))), range(15)) for i in INPUTS)
        best = max(mean(w, i, lr, range(15)) for i in INPUTS for lr in LRS)
        rows[w] = (sel, best)
    return rows


def quantum(rule, budget, head_lr=0.01):
    out = {}
    for arm in CELLS:
        if head_lr == 0.1:
            runs = [("skolik_protocol_1000", "%s_seed%d" % (arm, s)) for s in range(15)]
        elif budget == 1000:
            runs = [("head_lr_time_limit_1000", "%s_%s_seed%d" % ("A" if rule == "old" else "B", arm, s))
                    for s in range(5)]
        elif rule == "boot":
            runs = [("time_limit_300", "lr0.01_%s_seed%d" % (arm, s)) for s in range(5)]
        else:
            runs = [("head_lr_300", "%s_seed%d" % (arm, s)) for s in range(10)]
        out[arm] = np.mean([f20(data.history(e, r), budget) for e, r in runs])
    return out


def main():
    for budget in (300, 1000):
        for rule in ("old", "boot"):
            rows = mlp_frontier(rule, budget)
            bound, top_sel = max(b for _, b in rows.values()), max(s for s, _ in rows.values())
            q = quantum(rule, budget)
            say("=" * 78)
            say("budget %d, rule %s" % (budget, rule))
            say("  MLP (SKP)  " + "  ".join("%s sel %.1f best %.1f" % (w, *rows[w]) for w in WIDTHS))
            say("  bound (largest best of ten) %.1f   largest selected %.1f" % (bound, top_sel))
            say("  QDQN-R head lr 0.01: " + "  ".join("%s %.1f" % (CELLS[a], v) for a, v in q.items()))
            if rule == "old":
                ref = quantum(rule, budget, head_lr=0.1)
                say("  QDQN-R head lr 0.1 (reference): " + "  ".join("%s %.1f" % (CELLS[a], v) for a, v in ref.items()))
            above = sum(v > bound for v in q.values())
            verdict = ("holds" if above == len(q) else
                       "partly" if any(v > top_sel for v in q.values()) else "fails")
            say("  cells above the bound: %d/8 -> reading: %s" % (above, verdict))
    open(OUT, "w", encoding="utf-8").write("\n".join(_lines) + "\n")
    print("\nwrote", OUT)
