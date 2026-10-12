"""Named configurations ("cells") with their learning rates under the QDQN-R protocol.

The names are the code labels of the paper (Table 2) and appear in every result file. A factorial cell is read as
readout / initial range / re-uploading:

    QDQN-R (full)                    Zi / 3.2 / on       QDQN-R no-reup                   Zi / 3.2 / off
    QDQN-R full hi-scale             Zi / 30  / on       QDQN-R no-reup hi-scale          Zi / 30  / off
    QDQN-R global lo-scale           ZZ / 3.2 / on       QDQN-R global no-reup lo-scale   ZZ / 3.2 / off
    QDQN-R global-obs                ZZ / 30  / on       QDQN-R global no-reup            ZZ / 30  / off

("global" denotes the two-body readout Z0Z2, Z1Z3.)
"""
import math

from .models import (AcrobotNaiveQDQN, AcrobotQDQNR, MLP, NaiveQDQN, POLE_ON_Q1, QDQNR, SHAM_PERM, SkolikQDQN)

RANGE_LO = 4 * math.sqrt(2.0 / math.pi)   # 3.19: the reach of the dense head with sigma = 1
RANGE_HI = 30.0                           # the reach of the scalar head with w_a = 30

FACTORIAL = ["QDQN-R (full)", "QDQN-R full hi-scale", "QDQN-R global lo-scale", "QDQN-R global-obs",
             "QDQN-R no-reup", "QDQN-R no-reup hi-scale", "QDQN-R global no-reup lo-scale", "QDQN-R global no-reup"]
FACTORS = {  # cell -> (readout, initial range, re-uploading)
    "QDQN-R (full)": ("Zi", "3.2", "on"), "QDQN-R full hi-scale": ("Zi", "30", "on"),
    "QDQN-R global lo-scale": ("ZZ", "3.2", "on"), "QDQN-R global-obs": ("ZZ", "30", "on"),
    "QDQN-R no-reup": ("Zi", "3.2", "off"), "QDQN-R no-reup hi-scale": ("Zi", "30", "off"),
    "QDQN-R global no-reup lo-scale": ("ZZ", "3.2", "off"), "QDQN-R global no-reup": ("ZZ", "30", "off")}


def _qdqnr(reuploading, readout, initial_range=None, **kw):
    return lambda: QDQNR(reuploading=reuploading, readout=readout, initial_range=initial_range, **kw)


# cell -> (constructor, circuit learning rate, head learning rate) under the QDQN-R protocol
CELLS = {
    "Classical DQN": (lambda: MLP(4, 2), 1e-3, None),
    "Naive QDQN": (lambda: NaiveQDQN(), 5e-3, 1e-3),
    "QDQN-R no-scale": (_qdqnr(True, "per_qubit", trainable_head=False), 5e-3, None),
    "QDQN-R (full)": (_qdqnr(True, "per_qubit"), 5e-3, 1e-2),
    "QDQN-R full hi-scale": (_qdqnr(True, "per_qubit", RANGE_HI), 5e-3, 1e-2),
    "QDQN-R global lo-scale": (_qdqnr(True, "two_body", RANGE_LO), 5e-3, 1e-2),
    "QDQN-R global-obs": (_qdqnr(True, "two_body"), 5e-3, 1e-2),
    "QDQN-R no-reup": (_qdqnr(False, "per_qubit"), 5e-3, 1e-2),
    "QDQN-R no-reup hi-scale": (_qdqnr(False, "per_qubit", RANGE_HI), 5e-3, 1e-2),
    "QDQN-R global no-reup lo-scale": (_qdqnr(False, "two_body", RANGE_LO), 5e-3, 1e-2),
    "QDQN-R global no-reup": (_qdqnr(False, "two_body"), 5e-3, 1e-2),
    # rewiring test: pole angle on qubit 1, which Z0Z2 sees without re-uploading; sham: pole angle stays on qubit 2
    "QDQN-R global-obs perm": (_qdqnr(True, "two_body", input_perm=POLE_ON_Q1), 5e-3, 1e-2),
    "QDQN-R full hi-scale perm": (_qdqnr(True, "per_qubit", RANGE_HI, input_perm=POLE_ON_Q1), 5e-3, 1e-2),
    "QDQN-R global no-reup perm": (_qdqnr(False, "two_body", input_perm=POLE_ON_Q1), 5e-3, 1e-2),
    "QDQN-R no-reup hi-scale perm": (_qdqnr(False, "per_qubit", RANGE_HI, input_perm=POLE_ON_Q1), 5e-3, 1e-2),
    "QDQN-R global no-reup lo-scale perm": (_qdqnr(False, "two_body", RANGE_LO, input_perm=POLE_ON_Q1), 5e-3, 1e-2),
    "QDQN-R global no-reup sham": (_qdqnr(False, "two_body", input_perm=SHAM_PERM), 5e-3, 1e-2),
    # two-body observables read through the dense head of the per-qubit readout (observable vs head type)
    "ZZ-dense no-reup lo-scale": (_qdqnr(False, "two_body", RANGE_LO, dense_head=True), 5e-3, 1e-2),
    "ZZ-dense no-reup": (_qdqnr(False, "two_body", RANGE_HI, dense_head=True), 5e-3, 1e-2),
    # the agent of Skolik et al. (trained under its own protocol unless stated otherwise)
    "Skolik": (lambda: SkolikQDQN(), 1e-3, 1e-1),
    # Acrobot-v1: six qubits, three actions
    "Classical DQN (Acrobot)": (lambda: MLP(6, 3), 1e-3, None),
    "Naive QDQN (Acrobot)": (lambda: AcrobotNaiveQDQN(), 5e-3, 1e-3),
    "QDQN-R (Acrobot)": (lambda: AcrobotQDQNR(), 5e-3, 1e-2),
    "QDQN-R no-reup (Acrobot)": (lambda: AcrobotQDQNR(reuploading=False), 5e-3, 1e-2),
    "QDQN-R no-scale (Acrobot)": (lambda: AcrobotQDQNR(trainable_head=False), 5e-3, None),
}


def slug(cell):
    """File-name form of a cell name: 'QDQN-R global no-reup' -> 'QDQN-R_global_no-reup'."""
    return cell.replace(" ", "_").replace("(", "").replace(")", "")
