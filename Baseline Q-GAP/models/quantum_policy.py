import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

# ==========================================
# Quantum Device
# ==========================================
dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)

# ==========================================
# Quantum Circuit
# ==========================================
@qml.qnode(
    dev,
    interface="torch"
)
def quantum_circuit(inputs, weights):

    # ======================================
    # Angle Embedding
    # ======================================
    qml.templates.AngleEmbedding(
        inputs,
        wires=range(N_QUBITS)
    )

    # ======================================
    # Entangling Layers
    # ======================================
    qml.templates.StronglyEntanglingLayers(
        weights,
        wires=range(N_QUBITS)
    )

    # ======================================
    # Quantum Measurements
    # ======================================
    return [
        qml.expval(
            qml.PauliZ(i)
        )
        for i in range(N_QUBITS)
    ]


# ==========================================
# Quantum Policy Network
# ==========================================
class QuantumPolicy(nn.Module):

    def __init__(self):

        super().__init__()

        weight_shapes = {

            "weights": (
                Q_DEPTH,
                N_QUBITS,
                3
            )
        }

        self.qlayer = qml.qnn.TorchLayer(
            quantum_circuit,
            weight_shapes
        )

    # ======================================
    # Forward Pass
    # ======================================
    def forward(self, x):

        # ==================================
        # Quantum Layer
        # ==================================
        x = self.qlayer(x)

        # ==================================
        # Convert to Valid Probabilities
        # ==================================
        x = F.softmax(
            x,
            dim=-1
        )

        return x