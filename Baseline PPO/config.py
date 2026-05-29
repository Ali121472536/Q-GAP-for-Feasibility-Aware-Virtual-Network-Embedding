import torch

# ==========================================
# DEVICE CONFIGURATION
# ==========================================
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", DEVICE)

# ==============================
# SUBSTRATE NETWORK SETTINGS
# ==============================
SUBSTRATE_NODES = 80

SUBSTRATE_LINK_PROB = 0.35

CPU_MIN = 140
CPU_MAX = 260

BW_MIN = 350
BW_MAX = 700

# ==============================
# VNR SETTINGS
# ==============================
VNR_MIN_NODES = 2
VNR_MAX_NODES = 6

VNR_CPU_MIN = 8
VNR_CPU_MAX = 20

VNR_BW_MIN = 15
VNR_BW_MAX = 50

# ==============================
# GAT SETTINGS
# ==============================
GAT_INPUT_DIM = 2

GAT_HIDDEN_DIM = 32

GAT_HEADS = 4

# ==============================
# TRAINING SETTINGS
# ==============================
EPISODES = 5000

LEARNING_RATE = 0.0003

GAMMA = 0.99

BATCH_SIZE = 64

# ==============================
# QUANTUM SETTINGS
# ==============================
N_QUBITS = 4

Q_DEPTH = 4

N_CANDIDATES = 12
# ==========================================
# REWARD WEIGHTS
# ==========================================
ALPHA = 1.0
BETA = 0.4
GAMMA_REWARD = 0.3
DELTA = 0.2

# ==========================================
# BASELINE MODE
# ==========================================
BASELINE = "PPO"