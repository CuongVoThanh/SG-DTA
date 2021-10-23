import torch
import torch.nn as nn
from torch_geometric.data.batch import Batch


class DeepDTA(nn.Module):
    def __init__(self,
                model_drug: nn.Module,
                embedding_protein: nn.Module,
                n_output: int = 1,
                input_dim: int = 128,
                p: float = 0.2) -> None:
        super(DeepDTA, self).__init__()

        self.drug = model_drug
        self.proteins = embedding_protein
        self.n_output = n_output
        self.input_dim = input_dim
        self.p = p # Dropout Rate

        self.fc1 = nn.Linear(2*self.input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.p)

    def forward(self, data_drug: Batch, data_protein: torch.LongTensor) -> torch.tensor:
        drug_features = self.drug(data_drug)
        protein_features = self.proteins(data_protein)

        combination = torch.cat((drug_features, protein_features), 1)

        combination = self.fc1(combination)
        combination = self.relu(combination)
        combination = self.dropout(combination)
        combination = self.fc2(combination)
        combination = self.relu(combination)
        combination = self.dropout(combination)
        output = self.out(combination)                    

        return output
