"""Every experiment of the paper, as a list of runs.

A run is one agent trained from one seed. Its result is results/<experiment>/<run name>.json. "terminal" and
"bootstrap" are the two rules for the 500-step time limit (qrl.training.train).
"""
import os
from dataclasses import dataclass, field

from . import classical_dqn, protocols
from .cells import CELLS, FACTORIAL, slug
from .models import TanhInput, mlp

MLP_WIDTHS = {"1x12": [12], "1x16": [16], "2x8": [8, 8], "2x16": [16, 16], "2x64": [64, 64]}
MLP_INPUTS = ["raw", "tanh"]
MLP_LRS = ["0.001", "0.002", "0.005", "0.01", "0.02"]
CLASSICAL_WIDTHS = {"2x64": [64, 64], "2x256": [256, 256]}
REWIRED = ["QDQN-R global-obs perm", "QDQN-R full hi-scale perm", "QDQN-R global no-reup perm",
           "QDQN-R no-reup hi-scale perm", "QDQN-R global no-reup lo-scale perm", "QDQN-R global no-reup sham"]
HEAD_ABLATION = ["Naive QDQN", "QDQN-R no-scale"]
OBSERVABLE_CELLS = ["QDQN-R no-reup", "QDQN-R no-reup hi-scale", "QDQN-R global no-reup lo-scale",
                    "QDQN-R global no-reup", "ZZ-dense no-reup lo-scale", "ZZ-dense no-reup"]


@dataclass
class Run:
    name: str                  # file name of the result, without .json
    model: object              # constructor of the Q-network, or the name of a cell in qrl.cells.CELLS
    seed: int
    episodes: int
    env: str = "CartPole-v1"
    train: dict = field(default_factory=dict)    # keyword arguments of qrl.training.train
    meta: dict = field(default_factory=dict)     # extra fields of the result file
    snapshot_every: int = None                   # save the network every that many episodes (greedy evaluation)

    def constructor(self):
        return CELLS[self.model][0] if isinstance(self.model, str) else self.model


def _qdqnr_protocol(cell, **changes):
    _, lr, lr_head = CELLS[cell]
    return {**protocols.QDQNR, "lr": lr, "lr_head": lr_head, **changes}


def _cell_runs(cells, seeds, episodes, train_of, env="CartPole-v1", prefix="", snapshot_every=None, **meta):
    return [Run(prefix + "%s_seed%d" % (slug(c), s), c, s, episodes, env, train_of(c), dict(meta, cell=c),
                snapshot_every) for s in seeds for c in cells]


def _mlp(width, inp):
    base = mlp(MLP_WIDTHS[width])
    return base if inp == "raw" else (lambda: TanhInput(base()))


def _mlp_runs(episodes, train_of, rules=("",), seeds=range(20)):
    return [Run("%s%s_%s_lr%s_seed%d" % (rule, w, i, lr, s), _mlp(w, i), s, episodes, "CartPole-v1", train_of(lr, rule),
                dict(width=w, input=i, lr=float(lr), **({"rule": rule.rstrip("_")} if rule else {})))
            for s in seeds for rule in rules for w in MLP_WIDTHS for i in MLP_INPUTS for lr in MLP_LRS]


def _classical_dqn(key, seed, episodes, prefix=""):
    g = dict(zip(("width", "lr", "target", "eps"), key.split("|")))
    name = classical_dqn.run_name(key, seed, prefix)
    train = dict(protocols.CLASSICAL_OWN, lr=float(g["lr"]), target_update_steps=int(g["target"]),
                 eps_decay=float(g["eps"]))
    return Run(name, mlp(CLASSICAL_WIDTHS[g["width"]]), seed, episodes, "CartPole-v1", train, g)


RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def experiments(results_root=RESULTS):
    """name -> (description, list of runs). classical_dqn_own adds its final runs once its selection runs exist."""
    qp = _qdqnr_protocol
    sk = protocols.skolik
    E = {}
    E["cartpole_120"] = ("The factorial and the head ablation at 120 episodes, QDQN-R protocol",
                         _cell_runs(FACTORIAL, range(15), 120, qp) + _cell_runs(HEAD_ABLATION, range(5), 120, qp))
    E["cartpole_300"] = ("The factorial, the rewired cells and the head ablation at 300 episodes",
                         _cell_runs(FACTORIAL + REWIRED, range(15), 300, qp) + _cell_runs(HEAD_ABLATION, range(5), 300, qp))
    E["replication_120"] = ("The factorial at 120 episodes on new seeds 15-44",
                            _cell_runs(FACTORIAL, range(15, 45), 120, qp))
    E["factorial_300_bootstrap"] = ("The factorial under the bootstrap rule, paired with cartpole_300",
                                    _cell_runs(FACTORIAL, range(15), 300, lambda c: qp(c, bootstrap_truncation=True)))
    E["acrobot_head_ablation"] = ("Acrobot-v1 with and without the output head, 500 episodes",
                                  _cell_runs(["Classical DQN (Acrobot)", "Naive QDQN (Acrobot)", "QDQN-R (Acrobot)",
                                              "QDQN-R no-reup (Acrobot)", "QDQN-R no-scale (Acrobot)"],
                                             range(5), 500, qp, env="Acrobot-v1"))
    E["skolik_protocol_1000"] = ("The factorial and the Skolik agent, Skolik protocol, 1000 episodes "
                                 "(arm C)", _cell_runs(FACTORIAL + ["Skolik"], range(15), 1000, lambda c: sk(),
                                                       snapshot_every=50))
    E["head_lr_300"] = ("The factorial under the Skolik protocol with head learning rate 0.01",
                        _cell_runs(FACTORIAL, range(10), 300, lambda c: sk(0.01), snapshot_every=50))
    E["acrobot_head_lr"] = ("Acrobot-v1, head learning rate 0.01 and 0.1, bootstrap rule",
                            [r for lr in ("0.01", "0.1") for r in _cell_runs(
                                ["QDQN-R (Acrobot)", "QDQN-R no-reup (Acrobot)"], range(10), 500,
                                lambda c, lr=lr: qp(c, lr_head=float(lr), bootstrap_truncation=True),
                                env="Acrobot-v1", snapshot_every=50, head_lr=float(lr))])
    for r in E["acrobot_head_lr"][1]:
        r.name = r.name.replace("_seed", "_lr%g_seed" % r.meta["head_lr"])
    E["observable_vs_head"] = (
        "Two-body observables through the dense head; seeds 15-37 add the second stage",
        [r for lr, seeds in (("0.1", range(15)), ("0.01", range(10))) for r in _cell_runs(
            ["ZZ-dense no-reup lo-scale", "ZZ-dense no-reup"], seeds, 300, lambda c, lr=lr: sk(float(lr)),
            snapshot_every=50, head_lr=float(lr))]
        + _cell_runs(OBSERVABLE_CELLS, range(15, 38), 300, lambda c: sk(0.1), snapshot_every=50, head_lr=0.1))
    for r in E["observable_vs_head"][1]:
        r.name = r.name.replace("_seed", "_lr%g_seed" % r.meta["head_lr"])
    E["observable_vs_head_1000"] = (
        "Two-body observables through the dense head at head learning rate 0.1, seeds 0-14, 1000 episodes",
        _cell_runs(["ZZ-dense no-reup lo-scale", "ZZ-dense no-reup"], range(15), 1000, lambda c: sk(0.1),
                   snapshot_every=50, head_lr=0.1))
    E["head_lr_time_limit_1000"] = (
        "Head learning rate x time-limit rule, 1000 episodes (arms A, B, E; S: the Skolik agent)",
        [r for arm, lr, boot in (("A", 0.01, False), ("B", 0.01, True), ("E", 0.1, True)) for r in _cell_runs(
            FACTORIAL, range(5), 1000, lambda c, lr=lr, boot=boot: sk(lr, bootstrap_truncation=boot), prefix=arm + "_",
            snapshot_every=50, arm=arm, head_lr=lr, bootstrap_truncation=boot)]
        + _cell_runs(["Skolik"], range(5), 1000, lambda c: sk(bootstrap_truncation=True), prefix="S_",
                     snapshot_every=50, arm="S", head_lr=0.1, bootstrap_truncation=True))
    E["time_limit_300"] = ("The factorial under the bootstrap rule, Skolik protocol, 300 episodes",
                           [r for lr in (0.1, 0.01) for r in _cell_runs(
                               FACTORIAL, range(5), 300, lambda c, lr=lr: sk(lr, bootstrap_truncation=True),
                               prefix="lr%g_" % lr, snapshot_every=50, head_lr=lr, bootstrap_truncation=True)])
    E["skolik_comparison"] = (
        "The Skolik agent and QDQN-R global-obs under both protocols, 300 episodes",
        [Run("%s_seed%d" % (arm, s), model, s, 300, env, train, dict(arm=arm))
         for s in range(15) for arm, model, env, train in (
             ("SK-SKP-v1", "Skolik", "CartPole-v1", sk()), ("SK-SKP-v0", "Skolik", "CartPole-v0", sk()),
             ("GO-SKP-v1", "QDQN-R global-obs", "CartPole-v1", sk()),
             ("SK-OURP-v1", "Skolik", "CartPole-v1", dict(protocols.QDQNR, lr=5e-3, lr_head=1e-2)))])
    E["mlp_matched_120"] = ("MLPs of 86 to 4,610 parameters at two learning rates, 120 episodes",
                            [Run("%s_lr%s_seed%d" % (w, lr, s), mlp(MLP_WIDTHS[w]), s, 120, "CartPole-v1",
                                 dict(protocols.QDQNR, lr=float(lr), lr_head=None), dict(width=w, lr=float(lr)))
                             for w in MLP_WIDTHS for lr in ("0.001", "0.005") for s in range(15)])
    E["mlp_frontier_120"] = ("The MLP frontier (raw and tanh input, five learning rates), 120 episodes",
                             _mlp_runs(120, lambda lr, rule: dict(protocols.QDQNR, lr=float(lr), lr_head=None)))
    E["mlp_frontier_300"] = ("The MLP frontier, 300 episodes",
                             _mlp_runs(300, lambda lr, rule: dict(protocols.QDQNR, lr=float(lr), lr_head=None)))
    E["classical_dqn_skolik"] = (
        "The classical DQN of Skolik et al. on CartPole-v0, 1000 episodes",
        [Run("%s_seed%d" % (w, s), mlp(hidden), s, 1000, "CartPole-v0", dict(protocols.SKOLIK_CLASSICAL,
             batch_size=batch), dict(width=w, batch_size=batch))
         for s in range(15) for w, (hidden, batch) in (("2x20", ([20, 20], 64)), ("2x64", ([64, 64], 16)))])
    selection = [_classical_dqn(classical_dqn.key(g), s, 1000, "selection_") for s in range(15, 20)
                 for g in classical_dqn.GRID]
    chosen = classical_dqn.selected(os.path.join(results_root, "classical_dqn_own"))
    E["classical_dqn_own"] = ("A classical DQN with its own protocol: 24 settings on seeds 15-19, "
                              "the selected one on seeds 0-14, 1000 episodes",
                              selection + ([_classical_dqn(chosen, s, 1000) for s in range(15)] if chosen else []))
    E["classical_dqn_300"] = ("The classical DQN setting best at 300 episodes, on seeds 0-14",
                              [_classical_dqn("2x256|0.0003|100|0.97", s, 300) for s in range(15)])
    E["mlp_skolik_protocol"] = (
        "The MLP frontier under the Skolik protocol, both time-limit rules, 1000 episodes",
        _mlp_runs(1000, lambda lr, rule: protocols.skolik(None, lr=float(lr), bootstrap_truncation=rule == "bootstrap_"),
                  rules=("terminal_", "bootstrap_")))
    return E
