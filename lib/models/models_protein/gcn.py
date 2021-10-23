import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv, global_mean_pool as gep


class ProteinGCNNet(torch.nn.Module):
    def __init__(self,
                num_features_pro: int = 54,
                output_dim:int = 128,
                p: float = 0.2) -> None:
        super(ProteinGCNNet, self).__init__()
        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GCNConv(num_features_pro, num_features_pro * 2)
        self.pro_conv3 = GCNConv(num_features_pro * 2, num_features_pro * 4)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)

    def forward(self, target: torch.LongTensor) -> torch.Tensor:
        target_x, target_edge_index, target_batch = target.x, target.edge_index, target.batch
        protein = self.pro_conv1(target_x, target_edge_index)
        protein = self.relu(protein)

        protein = self.pro_conv2(protein, target_edge_index)
        protein = self.relu(protein)

        protein = self.pro_conv3(protein, target_edge_index)
        protein = self.relu(protein)

        protein = gep(protein, target_batch)  # global pooling

        protein = self.relu(self.pro_fc_g1(protein))
        protein = self.dropout(protein)
        protein = self.pro_fc_g2(protein)
        protein = self.dropout(protein)

        return protein