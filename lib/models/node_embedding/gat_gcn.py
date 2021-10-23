import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv, GATConv


class NodeGAT_GCN(torch.nn.Module):
    def __init__(self,
                num_features_node: int = 128,
                output_dim:int = 1024,
                p: float = 0.2) -> None:
        super(NodeGAT_GCN, self).__init__()

        self.num_features_node = num_features_node
        self.output_dim = output_dim
        self.p = p # Dropout Rate

        self.conv1 = GATConv(self.num_features_node, self.num_features_node, heads=10)
        self.conv2 = GCNConv(self.num_features_node*10, self.num_features_node*10)
        self.fc_g1 = nn.Linear(self.num_features_node*10, self.output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.p)

    def forward(self, edge_index: torch.LongTensor, node: torch.Tensor) -> torch.Tensor:
        node = self.conv1(node, edge_index)
        node = self.relu(node)
        node = self.conv2(node, edge_index)
        node = self.relu(node)

        node = self.fc_g1(node)

        return node
