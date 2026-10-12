"""Recompute the statistics of the paper from the stored results.

    python analyze.py --list          # the analyses
    python analyze.py factorial_120   # one analysis -> analysis/factorial_120.txt
    python analyze.py --all           # every analysis, in order

Each analysis reads results/ and writes one text file to analysis/. Under --all each runs in its own process: on
Windows, numpy's least squares can crash once torch has been loaded into the same process.
"""
import argparse
import importlib
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore", message=".*out of date.*")

# module in qrl/analyses -> what it tests; exploratory_cross_checks reads the outputs of three earlier analyses
ANALYSES = {
    "factorial_120": "The output head and the factorial at 120 episodes",
    "factorial_300": "The factorial, the rewired cells and the head ablation at 300 episodes",
    "exploratory_factorial": "Exploratory: the interactions by seed block and at 120 vs 300 episodes",
    "bootstrap_and_dqn_300": "The factorial under the bootstrap rule; a classical DQN tuned for 300 episodes",
    "skolik_protocol_1000": "The factorial and the Skolik agent at 1000 episodes",
    "exploratory_protocol_300": "Exploratory: the factorial at 300 episodes under the two protocols",
    "head_lr_300": "The head learning rate at 300 episodes",
    "head_lr_300_readout": "Head learning rate x readout at 300 episodes",
    "head_lr_300_reuploading": "Head learning rate x readout x re-uploading at 300 episodes",
    "observable_vs_head": "Head type and observable, both stages",
    "observable_vs_head_1000": "Head type and observable at 1000 episodes",
    "head_lr_time_limit_1000": "Head learning rate x time-limit rule at 1000 episodes",
    "bootstrap_arms_1000": "Arm E and the Skolik agent under the bootstrap rule at 1000 episodes",
    "exploratory_windows_1000": "Exploratory: variance model and score window at 1000 episodes",
    "time_limit_300": "The time-limit rule at 300 episodes",
    "skolik_comparison": "The Skolik agent under both protocols",
    "mlp_skolik_protocol": "The MLP frontier under the Skolik protocol",
    "replication_and_controls": "Replication on new seeds, Acrobot, classical DQNs, head type vs observable",
    "exploratory_cross_checks": "Exploratory cross-checks",
}


def run(name):
    module = importlib.import_module("qrl.analyses." + name)    # the analyses run on import
    if hasattr(module, "main"):
        module.main()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("analysis", nargs="?", choices=list(ANALYSES), metavar="analysis")
    ap.add_argument("--all", action="store_true", help="run every analysis")
    ap.add_argument("--list", action="store_true", help="list the analyses")
    a = ap.parse_args()
    if a.all:
        for name in ANALYSES:
            print("== %s" % name, flush=True)
            subprocess.run([sys.executable, __file__, name], check=True, stdout=subprocess.DEVNULL)
    elif a.analysis:
        run(a.analysis)
    else:
        for name, what in ANALYSES.items():
            print("%-28s %s" % (name, what))


if __name__ == "__main__":
    main()
