from torch_geometric.data import Batch
import numpy as np

import torch
import torch.nn as nn


class SGDTA(torch.nn.Module):
    def __init__(self,
                model_drug: nn.Module,
                embedding_protein: nn.Module,
                node_embedding: nn.Module,
                input_dim_fc_left: int = 128,
                input_dim_fc_right: int = 1024,
                num_node: int = 510,
                num_features_node: int = 128,
                n_output: int = 1,
                p: float = 0.2,
                device: None or torch.device = None) -> None:
        super(SGDTA, self).__init__()

        self.num_node = num_node
        self.num_features_node = num_features_node

        self.drug = model_drug
        self.proteins = embedding_protein
        self.node_embedding = node_embedding
        
        self.n_output = n_output
        self.input_dim_fc_left = input_dim_fc_left
        self.input_dim_fc_right = input_dim_fc_right
        self.p = p # Dropout Rate

        self.fc_left = FC_model(input_dim=self.input_dim_fc_left)
        self.fc_right = FC_model(input_dim=self.input_dim_fc_right)

        self.out = nn.Linear(2, 1)

        self.device = device

        self.node_feature = torch.rand(self.num_node, self.num_features_node).to(device)    

    def forward(self, data_drug: Batch, data_protein: torch.LongTensor or Batch, edge_index_drug_protein: torch.tensor) -> torch.tensor:
        feature_drug = self.drug(data_drug)
        embedding_protein = self.proteins(data_protein)

        output_1 = self.fc_left(feature_drug, embedding_protein)
        
        with torch.no_grad():
            drug_id = np.array(data_drug.edge_index_drug_protein)[:,0]
            protein_id = np.array(data_drug.edge_index_drug_protein)[:,1]

            unique_drug = np.unique(drug_id)
            for i in unique_drug:
                index = np.where(drug_id==i)[0]
                self.node_feature[i]=feature_drug[index][0]
            
            unique_protein = np.unique(protein_id)
            for i in unique_protein:
                index = np.where(protein_id==i)[0]
                self.node_feature[i] = embedding_protein[index][0]
                            
        nodes = self.node_embedding(edge_index_drug_protein, self.node_feature)

        drugs = nodes[drug_id]
        proteins = nodes[protein_id]
        
        output_2 = self.fc_right(drugs, proteins)
        
        output = torch.cat((output_1, output_2), 1)

        output = self.out(output)
        return output

class FC_model(torch.nn.Module):
    def __init__(self,
                n_output: int = 1,
                input_dim: int = 128,
                p: float = 0.2) -> None: 
        super(FC_model, self).__init__()

        self.n_output = n_output
        self.input_dim = input_dim
        self.p = p # Dropout Rate

        self.fc1 = nn.Linear(2*self.input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.p)
        
    def forward(self, drug_features: torch.Tensor, protein_features: torch.Tensor) -> torch.Tensor:
        assert drug_features.shape == protein_features.shape, "drug_features shape and protein_features shape must be the same"
        combination = torch.cat((drug_features, protein_features), 1)

        combination = self.fc1(combination)
        combination = self.relu(combination)
        combination = self.dropout(combination)
        combination = self.fc2(combination)
        combination = self.relu(combination)
        combination = self.dropout(combination)
        combination = self.out(combination)
        
        return combination
