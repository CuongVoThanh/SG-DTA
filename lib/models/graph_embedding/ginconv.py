import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU

from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data.batch import Batch


class DrugGINConvNet(torch.nn.Module):
    def __init__(self,
                num_features_drug: int = 78,
                output_dim: int = 128,
                p: float = 0.2,
                dim: int = 32) -> None:
        super(DrugGINConvNet, self).__init__()

        self.num_features_drug = num_features_drug
        self.output_dim = output_dim
        self.p = p # Dropout rate
        self.dim = dim

        self.dropout = nn.Dropout(self.p)
        self.relu = nn.ReLU()

        nn1 = Sequential(Linear(self.num_features_drug, self.dim), ReLU(), Linear(self.dim, self.dim))
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

    def forward(self, data_drug: Batch) -> torch.Tensor:
        batch,edge_index, drug = data_drug.batch, data_drug.edge_index, data_drug.x
        drug = F.relu(self.conv1(drug, edge_index))
        drug = self.bn1(drug)
        drug = F.relu(self.conv2(drug, edge_index))
        drug = self.bn2(drug)
        drug = F.relu(self.conv3(drug, edge_index))
        drug = self.bn3(drug)
        drug = F.relu(self.conv4(drug, edge_index))
        drug = self.bn4(drug)
        drug = F.relu(self.conv5(drug, edge_index))
        drug = self.bn5(drug)

        drug = global_add_pool(drug, batch)
        drug = F.relu(self.fc(drug))

        drug = F.dropout(drug, p = self.p, training = self.training)

        return drug