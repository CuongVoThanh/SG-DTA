import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool


class NodeGATNet(torch.nn.Module):
    def __init__(self,
                num_features_node: int = 128,
                output_dim:int = 1024,
                p: float = 0.2) -> None:
        super(NodeGATNet, self).__init__()

        self.num_features_node = num_features_node
        self.output_dim = output_dim
        self.p = p # Dropout rate

        self.gcn1 = GATConv(self.num_features_node, self.num_features_node, heads=10, dropout=self.p)
        self.gcn2 = GATConv(self.num_features_node*10, self.output_dim, dropout=self.p)
        self.fc_g1 = nn.Linear(self.output_dim, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, edge_index: torch.LongTensor, node: torch.Tensor) -> torch.Tensor:
        node = F.dropout(node, p = self.p, training = self.training)
        node = F.elu(self.gcn1(node, edge_index))
        node = F.dropout(node, p = self.p, training = self.training)

        node = self.gcn2(node, edge_index)
        node = self.relu(node)

        node = self.fc_g1(node)
        node = self.relu(node)

        return node