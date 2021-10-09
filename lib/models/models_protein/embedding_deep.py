import torch
import torch.nn as nn

FLATTEN_SHAPE = 96*107


class EmbeddingProteinDeep(torch.nn.Module):
    def __init__(self,
                n_filters: int = 32,
                embed_dim: int = 128,
                num_features_protein: int = 25,
                output_dim: int = 128,
                in_channels: int = 1200) -> None:
        super(EmbeddingProteinDeep, self).__init__()

        self.n_filters = n_filters
        self.embed_dim = embed_dim
        self.num_features_protein = num_features_protein
        self.output_dim = output_dim
        self.in_channels = in_channels

        self.embedding_protein = nn.Embedding(self.num_features_protein+1, self.embed_dim)
        self.conv_protein_1 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.n_filters, kernel_size=8)
        self.conv_protein_2 = nn.Conv1d(in_channels=self.n_filters, out_channels=self.n_filters*2, kernel_size=8)
        self.conv_protein_3 = nn.Conv1d(in_channels=self.n_filters*2, out_channels=self.n_filters*3, kernel_size=8)
        self.fc1_protein = nn.Linear(FLATTEN_SHAPE, self.output_dim)

        self.relu = nn.ReLU()
        
    def forward(self, target: torch.LongTensor) -> torch.Tensor:
        embedded_protein = self.embedding_protein(target.target_deep)

        conv_protein = self.conv_protein_1(embedded_protein)
        conv_protein = self.relu(conv_protein)

        conv_protein = self.conv_protein_2(conv_protein)
        conv_protein = self.relu(conv_protein)

        conv_protein = self.conv_protein_3(conv_protein)
        conv_protein = self.relu(conv_protein)

        protein = conv_protein.view(-1, FLATTEN_SHAPE)
        protein = self.fc1_protein(protein)

        return protein
