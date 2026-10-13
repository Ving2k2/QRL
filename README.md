# Quantum Components or Classical Settings?

Code for *Quantum Components or Classical Settings? A Controlled Factorial Study of Variational Quantum Deep
Q-Learning*.

The study tests whether the credit given to data re-uploading, a two-body readout and the initial range of the
output head in variational deep Q-networks holds across configurations, training budgets and the classical settings
of the training protocol, on CartPole-v1 and Acrobot-v1.

## Installation

Python 3.13 on CPU:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run.py --list                      # the experiments of the paper
python run.py cartpole_300 --workers 8    # train every run of one experiment
python analyze.py --all                   # recompute the statistics of the paper
```

Each run is single-threaded and fixed by its seed, checkpoints every five episodes and resumes after an
interruption. Results go to `results/<experiment>/`, one JSON file per run; analyses write to `analysis/`.

## Structure

```
run.py              train the runs of an experiment
analyze.py          run the analyses
qrl/                Q-networks, training loop, protocols and the list of experiments
qrl/analyses/       one module per family of tests
```

## Data

The result files of the paper are in `results/`, one JSON file per run with its training returns, together with
the greedy evaluations (`results/greedy/`) and the output heads of the saved networks (`results/heads/`).
`analyze.py` recomputes every statistic from them without retraining.

## License

MIT, see `LICENSE`.
