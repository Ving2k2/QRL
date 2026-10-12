"""Training protocols: keyword arguments of qrl.training.train (Table 3 of the paper).

Every protocol uses experience replay of 10,000 transitions (50,000 for the classical DQN with its own protocol),
epsilon-greedy exploration from 1 to 0.01, Adam, gamma = 0.99 and one update per environment step.
"""

# The protocol of the QDQN-R circuits: target copied every 10 episodes, Huber loss, batches of 32, gradient clipping
# at norm 1, epsilon x0.97 per episode. Learning rates come with the cell (qrl.cells.CELLS).
QDQNR = dict(batch_size=32, use_huber=True, grad_clip=1.0, eps_decay=0.97, target_update=10)

# Skolik et al., Quantum 6, 720 (2022), Table 2, "Cart Pole v0, optimal": circuit and input weights at 1e-3, output
# weights (the head) at 0.1, batch 16, MSE loss, epsilon x0.99, model updated and target copied every step. No
# gradient clipping is mentioned.
SKOLIK = dict(lr=1e-3, lr_head=1e-1, batch_size=16, use_huber=False, grad_clip=None, eps_decay=0.99,
              target_update_steps=1)

# The classical DQN of Skolik et al. (their Appendix B, Table 4) on CartPole-v0: MSE, no clipping, epsilon x0.99,
# model and target updated every step; batch 64 for the (20, 20) network and 16 for (64, 64).
SKOLIK_CLASSICAL = dict(lr=1e-3, lr_head=None, use_huber=False, grad_clip=None, eps_decay=0.99, target_update_steps=1)

# A classical DQN with a protocol of its own on CartPole-v1: Huber loss, batches of 64, clipping at norm 10, replay of
# 50,000, bootstrap at the time limit. Learning rate, target copy interval and epsilon decay come from a grid.
CLASSICAL_OWN = dict(lr_head=None, batch_size=64, use_huber=True, grad_clip=10.0, replay_size=50000,
                     bootstrap_truncation=True)


def skolik(head_lr=0.1, **changes):
    """The Skolik protocol with another head learning rate or other changes."""
    return dict(SKOLIK, lr_head=head_lr, **changes)
