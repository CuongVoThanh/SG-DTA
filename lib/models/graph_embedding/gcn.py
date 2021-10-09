import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.data.batch import Batch


class DrugGCNNet(torch.nn.Module):
    def __init__(self,
                num_features_drug: int = 78,
                output_dim:int = 128,
                p: float = 0.2) -> None:
        super(DrugGCNNet, self).__init__()

        self.num_features_drug = num_features_drug
        self.output_dim = output_dim
        self.p = p # Dropout rate

        self.conv1 = GCNConv(self.num_features_drug, self.num_features_drug)
        self.conv2 = GCNConv(self.num_features_drug, self.num_features_drug*2)
        self.conv3 = GCNConv(self.num_features_drug*2, self.num_features_drug*4)
        self.fc_g1 = nn.Linear(self.num_features_drug*4, 1024)
        self.fc_g2 = nn.Linear(1024, self.output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.p)

    def forward(self, data_drug: Batch) -> torch.Tensor:
        batch,edge_index, drug = data_drug.batch, data_drug.edge_index, data_drug.x
        drug = self.conv1(drug, edge_index)
        drug = self.relu(drug)

        drug = self.conv2(drug, edge_index)
        drug = self.relu(drug)

        drug = self.conv3(drug, edge_index)
        drug = self.relu(drug)
        drug = global_max_pool(drug, batch)

        drug = self.fc_g1(drug)
        drug = self.relu(drug)
        drug = self.dropout(drug)
        drug = self.fc_g2(drug)
        drug = self.dropout(drug)

        return drug