"""The head learning rate x readout at 300 episodes, no re-uploading.

Per-qubit (Zi) and two-body (ZZ) readout x initial range 3.2 / 30 x head lr 0.1 (skolik_protocol_1000, first 300
episodes) / 0.01 (head_lr_300), seeds 0-9. Final-20 at episode 300; per seed, the readout gap ZZ - Zi averaged over
the two ranges. Confirmatory, Holm over two: the interaction gap(0.01) - gap(0.1) (one-sample t) and equivalence of
gap(0.01) within +-50 (TOST). The greedy scores at episode 300 (results/greedy/) are described.
-> analysis/head_lr_300_readout.txt (JSON)
"""
import json
import os

from ..data import greedy as stored_greedy, output, path as result_path
from ..io import durable_write

MATCHES = {"low": {"Zi": "QDQN-R_no-reup", "ZZ": "QDQN-R_global_no-reup_lo-scale"},
           "high": {"Zi": "QDQN-R_no-reup_hi-scale", "ZZ": "QDQN-R_global_no-reup"}}
SEEDS = list(range(10))
BOUND = 50.0


def load_histories():
    histories = {}
    missing = []
    for seed in SEEDS:
        for initial, cells in MATCHES.items():
            for readout, cell in cells.items():
                for lr in (0.1, 0.01):
                    experiment = "skolik_protocol_1000" if lr == 0.1 else "head_lr_300"
                    path = result_path(experiment, "%s_seed%d" % (cell, seed))
                    if not os.path.exists(path):
                        missing.append(str(path))
                        continue
                    with open(path, encoding="utf-8") as stream:
                        data = json.load(stream)
                    assert data["label"] == cell and data["seed"] == seed
                    assert data["env"] == "CartPole-v1" and data["protocol"] == "SKP"
                    assert len(data["history"]) >= 300
                    if lr == 0.01:
                        assert data["lr_classical"] == lr and len(data["history"]) == 300
                    histories[seed, initial, readout, lr] = data["history"][:300]
    if missing:
        raise RuntimeError("Analysis waits for all registered runs; missing %d completed files:\n%s" %
                           (len(missing), "\n".join(missing)))
    assert len(histories) == 80
    for seed in SEEDS:
        for initial in MATCHES:
            for readout in ("Zi", "ZZ"):
                assert histories[seed, initial, readout, 0.01][0] == histories[seed, initial, readout, 0.1][0]
    return histories


def holm_two(pvalues):
    import numpy as np
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    result = np.empty(2)
    result[order[0]] = min(1.0, 2*pvalues[order[0]])
    result[order[1]] = max(result[order[0]], min(1.0, pvalues[order[1]]))
    return result.tolist()


def contrast_summary(values, confidence=0.95):
    import numpy as np
    from scipy import stats
    values = np.asarray(values, dtype=float)
    average, sd = values.mean(), values.std(ddof=1)
    se = sd / np.sqrt(len(values))
    radius = stats.t.ppf((1+confidence)/2, len(values)-1) * se
    return {"n_seed_blocks": len(values), "mean": float(average), "sd": float(sd),
            "ci": [float(average-radius), float(average+radius)], "confidence": confidence}


def primary_statistics(scores):
    import numpy as np
    from scipy import stats
    gaps = {lr: np.array([np.mean([scores[seed, initial, "ZZ", lr] - scores[seed, initial, "Zi", lr]
                                  for initial in MATCHES]) for seed in SEEDS]) for lr in (0.1, 0.01)}
    interaction = gaps[0.01] - gaps[0.1]
    interaction_p = (float(stats.ttest_1samp(interaction, 0.0).pvalue)
                     if interaction.std(ddof=1) > 0 else (1.0 if interaction.mean() == 0 else 0.0))
    low = gaps[0.01]
    se = low.std(ddof=1)/np.sqrt(len(low))
    if se == 0:
        lower_p = 0.0 if low.mean() > -BOUND else 1.0
        upper_p = 0.0 if low.mean() < BOUND else 1.0
    else:
        lower_p = float(stats.t.sf((low.mean()+BOUND)/se, len(low)-1))
        upper_p = float(stats.t.cdf((low.mean()-BOUND)/se, len(low)-1))
    equivalence_p = max(lower_p, upper_p)
    adjusted = holm_two([interaction_p, equivalence_p])
    return {"gap_high_lr": contrast_summary(gaps[0.1]),
            "gap_low_lr": contrast_summary(low), "interaction": contrast_summary(interaction),
            "interaction_p": interaction_p, "interaction_p_holm": adjusted[0],
            "low_lr_equivalence_bounds": [-BOUND, BOUND],
            "low_lr_ci90": contrast_summary(low, confidence=0.9)["ci"],
            "tost_p_lower": lower_p, "tost_p_upper": upper_p,
            "tost_p": equivalence_p, "tost_p_holm": adjusted[1],
            "equivalent_within_50": adjusted[1] < 0.05,
            "interaction_shapiro_p": float(stats.shapiro(interaction).pvalue) if interaction.std() > 0 else None,
            "low_gap_shapiro_p": float(stats.shapiro(low).pvalue) if low.std() > 0 else None,
            "interaction_wilcoxon_p_descriptive": float(stats.wilcoxon(interaction).pvalue) if np.any(interaction) else 1.0,
            "seed_contrasts": [{"seed": seed, "gap_high_lr": float(gaps[0.1][seed]),
                                "gap_low_lr": float(low[seed]), "interaction": float(interaction[seed])}
                               for seed in SEEDS]}


def greedy_scores(keys):
    import numpy as np
    cache = stored_greedy("head_lr_300_two_body")       # "seed|range|readout|lr" -> greedy returns at episode 300
    return {key: float(np.mean(cache["|".join(map(str, key))])) for key in keys}


def main(evaluate_greedy=True):
    import numpy as np
    histories = load_histories()
    final20 = {key: float(np.mean(history[280:300])) for key, history in histories.items()}
    table = []
    for initial in MATCHES:
        for readout in ("Zi", "ZZ"):
            for lr in (0.1, 0.01):
                selected = [histories[seed, initial, readout, lr] for seed in SEEDS]
                values = [np.mean(history[280:300]) for history in selected]
                table.append({"initial_range": initial, "readout": readout, "head_lr": lr,
                              "n": 10, "final20_mean": float(np.mean(values)),
                              "final20_sd": float(np.std(values, ddof=1)),
                              "collapse": sum(float(np.mean(history[280:300])) <
                                              0.5*float(np.mean(history[100:120])) for history in selected),
                              "auc_mean": float(np.mean([np.mean(history) for history in selected]))})
    result = {"scope": "SKP, no re-uploading, 300 episodes; extension after seeing Zi interim data",
              "primary_endpoint": "mean return episodes 281..300",
              "reproducibility_episode0": "40/40 same-cell/seed comparisons",
              "descriptives": table, "primary": primary_statistics(final20),
              "greedy_evaluated": evaluate_greedy}
    if evaluate_greedy:
        result["greedy_secondary_descriptive"] = primary_statistics(greedy_scores(list(histories)))
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    durable_write(output("head_lr_300_readout.txt"),
                  lambda stream: stream.write(text), "w")
    print(text, flush=True)

