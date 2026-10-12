"""Greedy evaluation of a stored network: epsilon = 0, fixed environment seeds.

The scores used by the analyses are stored in results/greedy/; they were computed with `play` from the network
snapshots that run.py writes to results/<experiment>/snapshots/ (not part of this repository, about 2 GB).
"""
import gymnasium as gym
import torch


def play(agent, base, n, env_name="CartPole-v1"):
    """Returns of n greedy episodes on environment seeds base, base + 1, ..., base + n - 1."""
    env = gym.make(env_name)
    out = []
    for k in range(n):
        s, _ = env.reset(seed=base + k)
        done, ret = False, 0.0
        while not done:
            with torch.no_grad():
                a = agent(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).argmax(1).item()
            s, r, term, trunc, _ = env.step(a)
            ret += r
            done = term or trunc
        out.append(ret)
    env.close()
    return out


def evaluate(run, snapshot, base=10000, n=10):
    """Greedy returns of the network saved in `snapshot` for a qrl.experiments.Run."""
    agent = run.constructor()()
    agent.load_state_dict(torch.load(snapshot, weights_only=False))
    agent.eval()
    return play(agent, base, n, run.env)
