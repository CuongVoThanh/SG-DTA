import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool
from torch_geometric.data.batch import Batch


class DrugGATNet(torch.nn.Module):
    def __init__(self,
                num_features_drug: int = 78,
                output_dim: int = 128,
                p: float = 0.2) -> None:
        super(DrugGATNet, self).__init__()

        self.num_features_drug = num_features_drug
        self.output_dim = output_dim
        self.p = p # Dropout rate

        self.gcn1 = GATConv(self.num_features_drug, self.num_features_drug, heads=10, dropout=self.p)
        self.gcn2 = GATConv(self.num_features_drug*10, self.output_dim, dropout=self.p)
        self.fc_g1 = nn.Linear(self.output_dim, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, data_drug: Batch) -> torch.Tensor:
        batch,edge_index, drug = data_drug.batch, data_drug.edge_index, data_drug.x
        drug = F.dropout(drug, p = self.p, training = self.training)
        drug = F.elu(self.gcn1(drug, edge_index))
        drug = F.dropout(drug, p = self.p, training = self.training)

        drug = self.gcn2(drug, edge_index)
        drug = self.relu(drug)
        drug = global_max_pool(drug, batch)

        drug = self.fc_g1(drug)
        drug = self.relu(drug)

        return drug