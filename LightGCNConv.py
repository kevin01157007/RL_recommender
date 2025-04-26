import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj

class LightGCNConv(MessagePassing):
    def __init__(self, **kwargs):
        super().__init__(aggr='add', **kwargs)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        # Compute degree
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return 'LightGCNConv()'
