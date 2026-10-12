"""Run the runs of an experiment, one process per run, resuming unfinished runs from their checkpoints."""
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from .experiments import experiments
from .io import read_json, write_json


def paths(root, experiment, name):
    d = os.path.join(root, experiment)
    return {"result": os.path.join(d, name + ".json"), "partial": os.path.join(d, "partial", name + ".json"),
            "checkpoint": os.path.join(d, "checkpoints", name + ".pt"),
            "snapshot": os.path.join(d, "snapshots", name + "_ep%d.pt"), "weights": os.path.join(d, "weights", name + ".pt")}


def find(root, experiment, name):
    for run in experiments()[experiment][1]:
        if run.name == name:
            return run
    raise KeyError("%s has no run %s" % (experiment, name))


def done(root, experiment, run):
    p = paths(root, experiment, run.name)["result"]
    return os.path.exists(p) and len(read_json(p)["history"]) >= run.episodes


def run_one(root, experiment, name, episodes=None):
    """Train one run and write its result file; `episodes` shortens it (only into a scratch directory)."""
    import torch
    torch.set_num_threads(1)
    from .training import train
    run = find(root, experiment, name)
    n = run.episodes if episodes is None else episodes
    p = paths(root, experiment, name)
    for key in ("partial", "checkpoint", "snapshot", "weights"):
        os.makedirs(os.path.dirname(p[key]), exist_ok=True)

    def progress(history, n_params):
        write_json(p["partial"], {"history": list(history), "params": int(n_params), "complete": False}, indent=None)

    r = train(run.constructor(), run.seed, n, env_name=run.env, label=name, checkpoint=p["checkpoint"],
              snapshot_every=run.snapshot_every, snapshot_fmt=p["snapshot"], on_progress=progress, **run.train)
    torch.save(r["agent"].state_dict(), p["weights"])
    settings = {k: v for k, v in run.train.items()}
    write_json(p["result"], dict(run.meta, experiment=experiment, seed=run.seed, env=run.env, episodes=n,
                                 settings=settings, history=r["history"], params=r["params"], time_s=r["time_s"]))
    for leftover in (p["partial"], p["checkpoint"]):
        if os.path.exists(leftover):
            os.remove(leftover)
    return name


def run_experiment(root, experiment, workers=1, only=None, episodes=None):
    """Run every unfinished run of `experiment` (or those named in `only`) with `workers` processes."""
    description, runs = experiments()[experiment]
    if only:
        runs = [r for r in runs if r.name in only]
    todo = [r for r in runs if episodes is not None or not done(root, experiment, r)]
    print("%s: %s\n%d of %d runs to do, %d workers" % (experiment, description, len(todo), len(runs), workers),
          flush=True)
    t0 = time.time()
    if workers <= 1:
        for r in todo:
            run_one(root, experiment, r.name, episodes)
            print("done %s  (%.0f min)" % (r.name, (time.time() - t0) / 60), flush=True)
    else:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), max_tasks_per_child=1) as pool:
            futures = [pool.submit(run_one, root, experiment, r.name, episodes) for r in todo]
            for k, f in enumerate(as_completed(futures), 1):
                print("done %s  [%d/%d, %.0f min]" % (f.result(), k, len(todo), (time.time() - t0) / 60), flush=True)
    if experiment == "classical_dqn_own" and only is None and episodes is None:
        final = [r for r in experiments()[experiment][1] if not done(root, experiment, r)]
        if final:                       # the selection runs exist now, so the final runs can be added
            run_experiment(root, experiment, workers)
