import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv
import torch


class NodeGCNNet(torch.nn.Module):
    def __init__(self,
                num_features_node: int = 128,
                output_dim:int = 1024,
                p: float = 0.2) -> None:
        super(NodeGCNNet, self).__init__()

        self.num_features_node = num_features_node
        self.output_dim = output_dim
        self.p = p # Dropout rate

        self.conv1 = GCNConv(self.num_features_node, self.num_features_node)
        self.conv2 = GCNConv(self.num_features_node, self.num_features_node*2)
        self.conv3 = GCNConv(self.num_features_node*2, self.num_features_node*4)
        self.fc_g1 = nn.Linear(self.num_features_node*4, self.output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.p)

    def forward(self, edge_index: torch.LongTensor, node: torch.Tensor) -> torch.Tensor:
        node = self.conv1(node, edge_index)
        node = self.relu(node)

        node = self.conv2(node, edge_index)
        node = self.relu(node)

        node = self.conv3(node, edge_index)
        node = self.relu(node)
        node = self.fc_g1(node)
        node = self.dropout(node)

        return node
