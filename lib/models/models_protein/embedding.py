import torch
import torch.nn as nn

FLATTEN_SHAPE = 32*121


class EmbeddingProtein(torch.nn.Module):
    def __init__(self,
                n_filters: int = 32,
                embed_dim: int = 128,
                num_features_protein: int = 25,
                output_dim: int = 128) -> None:
        super(EmbeddingProtein, self).__init__()

        self.n_filters = n_filters
        self.embed_dim = embed_dim
        self.num_features_protein = num_features_protein
        self.output_dim = output_dim

        self.embedding_protein = nn.Embedding(self.num_features_protein+1, self.embed_dim)
        self.conv_protein_1 = nn.Conv1d(in_channels=1000, out_channels=self.n_filters, kernel_size=8)
        self.fc1_protein = nn.Linear(FLATTEN_SHAPE, self.output_dim)
        
    def forward(self, target: torch.LongTensor) -> torch.Tensor:
        embedded_protein = self.embedding_protein(target.target)
        conv_protein = self.conv_protein_1(embedded_protein)

        protein = conv_protein.view(-1, FLATTEN_SHAPE)
        protein = self.fc1_protein(protein)

        return protein
