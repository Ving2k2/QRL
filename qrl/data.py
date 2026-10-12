"""Where the stored results and the analysis outputs live."""
import functools
import os

from .io import read_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
ANALYSIS = os.path.join(ROOT, "analysis")


def path(experiment, run):
    """Result file of one run: results/<experiment>/<run>.json."""
    return os.path.join(RESULTS, experiment, run + ".json")


def history(experiment, run):
    """Training returns of one run, one value per episode."""
    return read_json(path(experiment, run))["history"]


def greedy(name):
    """Stored greedy evaluations (results/greedy/<name>.json)."""
    return read_json(os.path.join(RESULTS, "greedy", name + ".json"))


def output(name):
    """Output file of an analysis: analysis/<name>."""
    os.makedirs(ANALYSIS, exist_ok=True)
    return os.path.join(ANALYSIS, name)


@functools.lru_cache(maxsize=None)
def _heads(experiment):
    return read_json(os.path.join(RESULTS, "heads", experiment + ".json"))


def head(experiment, run, episode):
    """Output-head parameters of the network snapshot of `run` after `episode` episodes, as float32 tensors
    (results/heads/<experiment>.json: the only part of the snapshots that the analyses read)."""
    import torch
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in _heads(experiment)[run][str(episode)].items()}
