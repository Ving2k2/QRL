"""Q-networks of the study: the QDQN-R circuits, the agent of Skolik et al., the naive hybrid QDQN and the MLPs.

Every constructor here reproduces the stored runs bit for bit, so the order in which parameters are created and
initialized must not change.
"""
import math

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

CARTPOLE_SCALES = (2.4, 2.5, 0.21, 2.5)    # c_i in tanh(s_i / c_i)
POLE_ON_Q1 = [0, 2, 1, 3]                  # rewiring: pole angle on qubit 1 instead of qubit 2
SHAM_PERM = [1, 0, 2, 3]                   # sham rewiring: cart position and velocity swapped, pole angle stays on qubit 2


# ----------------------------------------------------------------------------------------------- QDQN-R
def _cartpole_normalise(x):
    return torch.tanh(x / torch.tensor(CARTPOLE_SCALES, dtype=x.dtype, device=x.device))


def per_qubit_circuit(n_qubits, n_layers, reuploading=True):
    """R_Y(pi * lambda * x) encoding, Rot gates and a CNOT ring per layer; returns <Z_i> for every qubit."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights, input_scale):
        for layer in range(n_layers):
            if reuploading or layer == 0:
                for q in range(n_qubits):
                    qml.RY(inputs[..., q] * input_scale[layer, q] * math.pi, wires=q)
            for q in range(n_qubits):
                qml.Rot(weights[layer, q, 0], weights[layer, q, 1], weights[layer, q, 2], wires=q)
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit, {"weights": (n_layers, n_qubits, 3), "input_scale": (n_layers, n_qubits)}


def two_body_circuit(n_qubits, n_layers, reuploading=True, n_outputs=2):
    """The same layers; returns one two-body observable per action, Z0Z2 and Z1Z3 for four qubits."""
    dev = qml.device("default.qubit", wires=n_qubits)

    def observable(a):
        if n_qubits == 4 and n_outputs == 2:
            return qml.PauliZ(0) @ qml.PauliZ(2) if a == 0 else qml.PauliZ(1) @ qml.PauliZ(3)
        wires = list(range(a, n_qubits, n_outputs))
        op = qml.PauliZ(wires[0])
        for w in wires[1:]:
            op = op @ qml.PauliZ(w)
        return op

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights, input_scale):
        for layer in range(n_layers):
            if reuploading or layer == 0:
                for q in range(n_qubits):
                    qml.RY(inputs[..., q] * input_scale[layer, q] * math.pi, wires=q)
            for q in range(n_qubits):
                qml.Rot(weights[layer, q, 0], weights[layer, q, 1], weights[layer, q, 2], wires=q)
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])
        return [qml.expval(observable(a)) for a in range(n_outputs)]

    return circuit, {"weights": (n_layers, n_qubits, 3), "input_scale": (n_layers, n_qubits)}


class QDQNR(nn.Module):
    """Variational Q-network with re-uploading, trainable input scaling and a trainable output head.

    readout="per_qubit": Q(s, a) = sum_i W_ai <Z_i> + b_a, a dense head with weights from N(0, sigma^2).
    readout="two_body":  Q(s, a) = w_a <O_a> + b_a, one scalar pair per action (w_a = 30 by default);
                         with dense_head=True the two-body observables go through the dense head instead.
    initial_range: the expected bound E|Q|_max of one action at initialization; it sets sigma (dense head) or w_a
                   (scalar head). None keeps sigma = 1 and w_a = 30.
    trainable_head=False replaces the head by a fixed readout that passes <Z_a> to action a (the no-scale arm).
    input_perm: qubit q receives state dimension input_perm[q] (the rewiring test).
    """

    def __init__(self, state_dim=4, action_dim=2, n_qubits=4, n_layers=5, reuploading=True, readout="per_qubit",
                 initial_range=None, trainable_head=True, dense_head=False, input_perm=None):
        super().__init__()
        assert state_dim == n_qubits
        two_body = readout == "two_body"
        assert not dense_head or (two_body and trainable_head)
        self.n_qubits, self.n_layers, self.action_dim = n_qubits, n_layers, action_dim
        self.reuploading, self.two_body, self.dense_head = reuploading, two_body, dense_head
        self.trainable_head = trainable_head
        self.input_perm = None if input_perm is None else list(input_perm)

        if two_body:
            circuit, shapes = two_body_circuit(n_qubits, n_layers, reuploading, n_outputs=action_dim)
            self.n_obs = action_dim
        else:
            circuit, shapes = per_qubit_circuit(n_qubits, n_layers, reuploading)
            self.n_obs = n_qubits
        init = {"weights": lambda t: nn.init.uniform_(t, -0.1, 0.1), "input_scale": lambda t: nn.init.ones_(t)}
        self.vqc = qml.qnn.TorchLayer(circuit, shapes, init_method=init)

        if trainable_head and two_body and not dense_head:
            w0 = 30.0 if initial_range is None else float(initial_range)
            self.output_w = nn.Parameter(torch.full((action_dim,), w0))
            self.output_b = nn.Parameter(torch.zeros(action_dim))
        elif trainable_head:
            # A row of n_obs weights from N(0, sigma^2) reaches E|Q|_max = n_obs * sigma * sqrt(2 / pi).
            self.output_head = nn.Linear(self.n_obs, action_dim, bias=True)
            std = 1.0 if initial_range is None else float(initial_range) / (self.n_obs * math.sqrt(2.0 / math.pi))
            nn.init.normal_(self.output_head.weight, mean=0.0, std=std)
            nn.init.zeros_(self.output_head.bias)
        elif two_body and self.n_obs == action_dim:
            self.output_head = nn.Identity()
        else:
            head = nn.Linear(n_qubits, action_dim, bias=False)
            with torch.no_grad():
                head.weight.zero_()
                for a in range(action_dim):
                    head.weight[a, a % n_qubits] = 1.0
            for p in head.parameters():
                p.requires_grad_(False)
            self.output_head = head

    def forward(self, x):
        x = _cartpole_normalise(x)
        if self.input_perm is not None:
            x = x[..., self.input_perm]
        out = self.vqc(x).float()
        if self.two_body and not self.dense_head:
            return out * self.output_w + self.output_b if self.trainable_head else out
        return self.output_head(out)


# ----------------------------------------------------------------------------------------------- Skolik et al.
def skolik_circuit(n_qubits=4, n_layers=5, reuploading=True, pairs=((0, 1), (2, 3))):
    """Skolik et al., Quantum 6, 720 (2022), Fig. 1: R_X(arctan(x w_d)) encoding, R_Y and R_Z rotations, a CZ chain."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, theta, w_in):
        for layer in range(n_layers):
            if reuploading or layer == 0:
                for q in range(n_qubits):
                    qml.RX(torch.atan(inputs[..., q] * w_in[layer, q]), wires=q)
            for q in range(n_qubits):
                qml.RY(theta[layer, q, 0], wires=q)
                qml.RZ(theta[layer, q, 1], wires=q)
            for q in range(n_qubits - 1):
                qml.CZ(wires=[q, q + 1])
        return [qml.expval(qml.PauliZ(a) @ qml.PauliZ(b)) for a, b in pairs]

    return circuit, {"theta": (n_layers, n_qubits, 2), "w_in": (n_layers, n_qubits)}


class SkolikQDQN(nn.Module):
    """Q(s, a) = w_a (<O_a> + 1) / 2 on the raw CartPole state, O = Z0Z1 and Z2Z3 (Eq. 19 of Skolik et al.).

    Not stated in the paper and taken from the TensorFlow Quantum tutorial: one input weight per layer and qubit,
    angles initialized uniformly on [0, pi], input and output weights at 1.
    """

    def __init__(self, n_qubits=4, n_layers=5, reuploading=True):
        super().__init__()
        circuit, shapes = skolik_circuit(n_qubits, n_layers, reuploading)
        init = {"theta": lambda t: nn.init.uniform_(t, 0.0, math.pi), "w_in": lambda t: nn.init.ones_(t)}
        self.vqc = qml.qnn.TorchLayer(circuit, shapes, init_method=init)
        self.w_out = nn.Parameter(torch.ones(2))

    def forward(self, x):
        return self.w_out * (self.vqc(x).float() + 1.0) / 2.0


# ----------------------------------------------------------------------------------------------- naive hybrid QDQN
def strongly_entangling_circuit(n_qubits, n_layers):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs * np.pi, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit, {"weights": (n_layers, n_qubits, 3)}


class NaiveQDQN(nn.Module):
    """Classical pre-network, R_Y angle encoding, strongly entangling layers, linear head, no re-uploading."""

    def __init__(self, state_dim=4, action_dim=2, n_qubits=4, n_layers=3):
        super().__init__()
        self.pre_net = nn.Linear(state_dim, n_qubits)
        circuit, shapes = strongly_entangling_circuit(n_qubits, n_layers)
        self.vqc = qml.qnn.TorchLayer(circuit, shapes)
        self.post_net = nn.Linear(n_qubits, action_dim)

    def forward(self, x):
        return self.post_net(self.vqc(torch.tanh(self.pre_net(x))).float())


# ----------------------------------------------------------------------------------------------- Acrobot
ACROBOT_SCALES = (1.0, 1.0, 1.0, 1.0, 4 * math.pi, 4 * math.pi)


def _acrobot_normalise(x):
    return torch.tanh(x / torch.tensor(ACROBOT_SCALES, dtype=x.dtype, device=x.device))


def _batched(forward):
    def wrapped(self, x):
        squeeze = x.dim() == 1
        out = forward(self, x.unsqueeze(0) if squeeze else x)
        return out.squeeze(0) if squeeze else out
    return wrapped


class AcrobotQDQNR(nn.Module):
    """The QDQN-R circuit on six qubits with the per-qubit readout and a dense head Linear(6 -> 3)."""

    def __init__(self, n_layers=5, reuploading=True, trainable_head=True):
        super().__init__()
        circuit, shapes = per_qubit_circuit(6, n_layers, reuploading)
        init = {"weights": lambda t: nn.init.uniform_(t, -0.1, 0.1), "input_scale": lambda t: nn.init.ones_(t)}
        self.vqc = qml.qnn.TorchLayer(circuit, shapes, init_method=init)
        if trainable_head:
            self.output_head = nn.Linear(6, 3)
            nn.init.normal_(self.output_head.weight, mean=0.0, std=1.0)
            nn.init.zeros_(self.output_head.bias)
        else:
            head = nn.Linear(6, 3, bias=False)
            with torch.no_grad():
                head.weight.zero_()
                for a in range(3):
                    head.weight[a, a] = 1.0
            for p in head.parameters():
                p.requires_grad_(False)
            self.output_head = head

    @_batched
    def forward(self, x):
        q = self.vqc(_acrobot_normalise(x))
        if isinstance(q, (list, tuple)):
            q = torch.stack(q, dim=-1)
        return self.output_head(q)


class AcrobotNaiveQDQN(nn.Module):
    """Linear pre-network to six qubits, one R_Y encoding, Rot gates and a CNOT ring, <Z_0..2> as Q-values."""

    def __init__(self, n_layers=3):
        super().__init__()
        self.pre_net = nn.Linear(6, 6)
        dev = qml.device("default.qubit", wires=6)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            for q in range(6):
                qml.RY(inputs[..., q], wires=q)
            for layer in range(n_layers):
                for q in range(6):
                    qml.Rot(weights[layer, q, 0], weights[layer, q, 1], weights[layer, q, 2], wires=q)
                for q in range(6):
                    qml.CNOT(wires=[q, (q + 1) % 6])
            return [qml.expval(qml.PauliZ(i)) for i in range(3)]

        self.vqc = qml.qnn.TorchLayer(circuit, {"weights": (n_layers, 6, 3)},
                                      init_method={"weights": lambda t: nn.init.uniform_(t, -0.1, 0.1)})

    @_batched
    def forward(self, x):
        q = self.vqc(self.pre_net(x))
        if isinstance(q, (list, tuple)):
            q = torch.stack(q, dim=-1)
        return q


# ----------------------------------------------------------------------------------------------- classical networks
class MLP(nn.Module):
    """Two hidden ReLU layers of 64 units: the classical DQN of the head ablation."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, action_dim))

    def forward(self, x):
        return self.net(x)


def mlp(hidden, state_dim=4, action_dim=2):
    """Constructor of an MLP with the given hidden widths, e.g. [64, 64]."""
    def make():
        layers, prev = [], state_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        return nn.Sequential(*layers, nn.Linear(prev, action_dim))
    return make


class TanhInput(nn.Module):
    """The input normalization of the circuits, tanh(s / c), in front of an MLP."""

    def __init__(self, body):
        super().__init__()
        self.body = body

    def forward(self, x):
        return self.body(torch.tanh(x / torch.tensor(CARTPOLE_SCALES).to(x.dtype)))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
