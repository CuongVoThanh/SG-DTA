import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU

from torch_geometric.nn import GINConv


class NodeGINConvNet(torch.nn.Module):
    def __init__(self,
                num_features_node: int = 128,
                output_dim:int = 1024,
                p: float = 0.2,
                dim: int = 32) -> None:
        super(NodeGINConvNet, self).__init__()

        self.num_features_node = num_features_node
        self.output_dim = output_dim
        self.p = p # Dropout rate
        self.dim = dim

        self.dropout = nn.Dropout(self.p)
        self.relu = nn.ReLU()

        nn1 = Sequential(Linear(self.num_features_node, self.dim), ReLU(), Linear(self.dim, self.dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = nn.BatchNorm1d(self.dim)

        nn2 = Sequential(Linear(self.dim, self.dim), ReLU(), Linear(self.dim, self.dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = nn.BatchNorm1d(self.dim)

        nn3 = Sequential(Linear(self.dim, self.dim), ReLU(), Linear(self.dim, self.dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = nn.BatchNorm1d(self.dim)

        nn4 = Sequential(Linear(self.dim, self.dim), ReLU(), Linear(self.dim, self.dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = nn.BatchNorm1d(self.dim)

        nn5 = Sequential(Linear(self.dim, self.dim), ReLU(), Linear(self.dim, self.dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = nn.BatchNorm1d(self.dim)

        self.fc = Linear(self.dim, self.output_dim)

    def forward(self, edge_index: torch.LongTensor, node: torch.Tensor) -> torch.Tensor:
        node = F.relu(self.conv1(node, edge_index))
        node = self.bn1(node)
        node = F.relu(self.conv2(node, edge_index))
        node = self.bn2(node)
        node = F.relu(self.conv3(node, edge_index))
        node = self.bn3(node)
        node = F.relu(self.conv4(node, edge_index))
        node = self.bn4(node)
        node = F.relu(self.conv5(node, edge_index))
        node = self.bn5(node)
        
        node = F.relu(self.fc(node))

        node = F.dropout(node, p = self.p, training = self.training)

        return node