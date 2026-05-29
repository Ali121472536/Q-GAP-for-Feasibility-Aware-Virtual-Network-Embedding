import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATEncoder(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, heads):
        super(GATEncoder, self).__init__()

        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim)

    def forward(self, x, edge_index):

        x = self.gat1(x, edge_index)
        x = F.elu(x)

        x = self.gat2(x, edge_index)

        return x