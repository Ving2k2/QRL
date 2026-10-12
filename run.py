"""Train the agents of one experiment of the paper.

    python run.py --list                         # experiments and their number of runs
    python run.py cartpole_300 --workers 8       # every unfinished run of an experiment
    python run.py cartpole_300 --only QDQN-R_global-obs_seed0

Results go to results/<experiment>/<run>.json; a run that exists is skipped and an interrupted run resumes from its
checkpoint. Every run is single-threaded and fixed by its seed, so it reproduces the stored result bit for bit on
the software versions of requirements.txt.

    python run.py cartpole_300 --only QDQN-R_global-obs_seed0 --episodes 10 --out scratch
writes the first 10 episodes of a run to scratch/; they equal the first 10 episodes of the stored result.
"""
import argparse
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import warnings
warnings.filterwarnings("ignore", message=".*CartPole-v0 is out of date.*")

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    from qrl.experiments import experiments
    from qrl.runner import run_experiment
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", nargs="?")
    ap.add_argument("--list", action="store_true", help="list the experiments")
    ap.add_argument("--workers", type=int, default=1, help="parallel runs (one CPU core each)")
    ap.add_argument("--only", nargs="+", help="run only these runs (file names without .json)")
    ap.add_argument("--episodes", type=int, help="shorten the runs (requires --out)")
    ap.add_argument("--out", help="results directory (default: results/)")
    a = ap.parse_args()
    root = os.path.abspath(a.out) if a.out else os.path.join(ROOT, "results")
    if a.episodes is not None and root == os.path.join(ROOT, "results"):
        ap.error("--episodes writes shortened runs; give a scratch directory with --out")
    if a.list or not a.experiment:
        for name, (description, runs) in experiments().items():
            print("%-26s %5d runs  %s" % (name, len(runs), description))
        return
    run_experiment(root, a.experiment, a.workers, a.only, a.episodes)


if __name__ == "__main__":
    main()
