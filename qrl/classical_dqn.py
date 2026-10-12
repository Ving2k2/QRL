"""The classical DQN with its own protocol: its 24-point grid, run names and selection rule.

Kept free of torch so that the analyses can import it (see analyze.py).
"""
import itertools
import os

import numpy as np

from .io import read_json

# width x learning rate x target copy interval (steps) x epsilon decay per episode
GRID = [dict(width=w, lr=lr, target=t, eps=e) for w, lr, t, e in
        itertools.product(["2x64", "2x256"], ["0.001", "0.0003"], ["1", "100", "1000"], ["0.99", "0.97"])]


def key(setting):
    """'width|lr|target|eps' of a grid point."""
    return "|".join((setting["width"], setting["lr"], setting["target"], setting["eps"]))


def run_name(setting_key, seed, prefix=""):
    width, lr, target, eps = setting_key.split("|")
    return "%s%s_lr%s_target%s_eps%s_seed%d" % (prefix, width, lr, target, eps, seed)


def selected(results_dir):
    """The setting of the grid with the highest mean final-20 score at episode 1000 over seeds 15-19 (ties: the
    higher mean return over all episodes). None until every selection run exists."""
    best = None
    for g in GRID:
        k = key(g)
        paths = [os.path.join(results_dir, run_name(k, s, "selection_") + ".json") for s in range(15, 20)]
        if not all(os.path.exists(p) for p in paths):
            return None
        hs = [read_json(p)["history"] for p in paths]
        score = (np.mean([np.mean(h[-20:]) for h in hs]), np.mean([np.mean(h) for h in hs]))
        if best is None or score > best[0]:
            best = (score, k)
    return best[1]
