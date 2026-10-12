"""Deep Q-learning loop with checkpoint and resume.

One loop trains every agent of the study. A run is fixed by its seed: the five random streams (Python, NumPy,
PyTorch, the environment and its action space) are seeded at the start and stored in every checkpoint, together with
both networks, the Adam moments, the replay buffer, epsilon and the step counter, so a resumed run continues the
uninterrupted one bit for bit. Nothing depends on the total number of episodes, so a run of N episodes is the prefix
of a longer run with the same seed.
"""
import os
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .io import durable_write
from .models import count_parameters


def _rng_state(env):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "env": env.unwrapped.np_random.bit_generator.state,
            "action_space": env.action_space.np_random.bit_generator.state}


def _set_rng_state(env, st):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    env.unwrapped.np_random.bit_generator.state = st["env"]
    env.action_space.np_random.bit_generator.state = st["action_space"]


def train(make_model, seed, episodes, env_name="CartPole-v1", lr=1e-3, lr_head=None, gamma=0.99, batch_size=32,
          replay_size=10000, eps_start=1.0, eps_end=0.01, eps_decay=0.97, target_update=10, target_update_steps=None,
          train_every=1, grad_clip=1.0, use_huber=True, bootstrap_truncation=False, label="agent",
          checkpoint=None, checkpoint_every=5, snapshot_every=None, snapshot_fmt=None, on_progress=None):
    """Train one agent and return its per-episode returns.

    lr, lr_head      learning rates of the circuit (with the input weights) and of the classical head; with lr_head
                     None, or for a model without a circuit, every parameter uses lr.
    target_update    copy the target network every that many episodes, unless target_update_steps is set, in which
                     case it is copied every that many environment steps.
    bootstrap_truncation
                     False: a transition that ends at the time limit is stored as terminal (the terminal rule);
                     True: only a failure is terminal and the target bootstraps at the limit (the bootstrap rule).
    checkpoint       path of the checkpoint; if it exists, training resumes from it.
    snapshot_every   save the online network every that many episodes to snapshot_fmt % episodes_done.
    on_progress      called as on_progress(history, n_params) every five episodes.
    """
    # The construction order below fixes the point at which every random stream is consumed. Do not reorder.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = gym.make(env_name)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    agent, target_agent = make_model(), make_model()
    target_agent.load_state_dict(agent.state_dict())
    for p in target_agent.parameters():
        p.requires_grad_(False)

    if lr_head is not None and hasattr(agent, "vqc"):
        circuit_params = list(agent.vqc.parameters())
        circuit_ids = {id(p) for p in circuit_params}
        opt = optim.Adam([{"params": circuit_params, "lr": lr},
                          {"params": [p for p in agent.parameters() if id(p) not in circuit_ids], "lr": lr_head}])
    else:
        opt = optim.Adam(agent.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss() if use_huber else nn.MSELoss()

    ck = None
    if checkpoint is not None and os.path.exists(checkpoint):
        try:
            ck = torch.load(checkpoint, weights_only=False)
        except Exception as e:
            print("(checkpoint unreadable, starting fresh: %s)" % e, flush=True)
    if ck is not None:
        agent.load_state_dict(ck["agent"])
        target_agent.load_state_dict(ck["target"])
        opt.load_state_dict(ck["opt"])
        memory = deque(ck["memory"], maxlen=replay_size)
        eps, step_count, history, start_ep = ck["eps"], ck["step_count"], list(ck["history"]), ck["next_ep"]
        _set_rng_state(env, ck["rng"])
        prior_time = ck.get("elapsed", 0.0)
        print("  [%s seed=%d] resumed at episode %d" % (label, seed, start_ep), flush=True)
    else:
        memory, eps, step_count, history, start_ep, prior_time = deque(maxlen=replay_size), eps_start, 0, [], 0, 0.0
    t0 = time.time()

    def save_checkpoint(next_ep):
        durable_write(checkpoint, lambda fh: torch.save({
            "agent": agent.state_dict(), "target": target_agent.state_dict(), "opt": opt.state_dict(),
            "memory": list(memory), "eps": eps, "step_count": step_count, "history": history, "next_ep": next_ep,
            "rng": _rng_state(env), "elapsed": prior_time + (time.time() - t0)}, fh), "wb")

    for ep in range(start_ep, episodes):
        s, _ = env.reset()
        s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        done, ep_return = False, 0.0
        while not done:
            if random.random() < eps:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    a = agent(s_t).argmax(1).item()
            ns, r, term, trunc, _ = env.step(a)
            done = term or trunc
            ep_return += r
            ns_t = torch.tensor(ns, dtype=torch.float32).unsqueeze(0)
            memory.append((s_t, a, r, ns_t, float(term if bootstrap_truncation else done)))
            s_t = ns_t
            step_count += 1

            if len(memory) >= batch_size and step_count % train_every == 0:
                sb, ab, rb, nsb, db = zip(*random.sample(memory, batch_size))
                sb, nsb = torch.cat(sb), torch.cat(nsb)
                ab = torch.tensor(ab, dtype=torch.long)
                rb = torch.tensor(rb, dtype=torch.float32)
                db = torch.tensor(db, dtype=torch.float32)
                q_curr = agent(sb).gather(1, ab.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    target = rb + gamma * target_agent(nsb).max(1)[0] * (1 - db)
                loss = loss_fn(q_curr, target)
                opt.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), grad_clip)
                opt.step()

            if target_update_steps is not None and step_count % target_update_steps == 0:
                target_agent.load_state_dict(agent.state_dict())

        history.append(ep_return)
        if target_update_steps is None and ep % target_update == 0:
            target_agent.load_state_dict(agent.state_dict())
        eps = max(eps_end, eps * eps_decay)

        if ep % 5 == 0 or ep == episodes - 1:
            recent = np.mean(history[-10:])
            print("  [%s seed=%d] episode %4d  return %6.1f  mean(10) %6.1f  eps %.2f  %6.0fs"
                  % (label, seed, ep, ep_return, recent, eps, prior_time + time.time() - t0), flush=True)
            if on_progress is not None:
                on_progress(history, count_parameters(agent))
        if snapshot_every and (ep + 1) % snapshot_every == 0:
            durable_write(snapshot_fmt % (ep + 1), lambda fh: torch.save(agent.state_dict(), fh), "wb")
        if checkpoint is not None and checkpoint_every and (ep + 1) % checkpoint_every == 0:
            save_checkpoint(ep + 1)

    env.close()
    return {"history": history, "params": count_parameters(agent), "time_s": prior_time + (time.time() - t0),
            "agent": agent}
