import torch
import torch.nn as nn
from torch_geometric.data.batch import Batch

FLATTEN_SHAPE = 96*107


class EmbeddingSmile(torch.nn.Module):
    def __init__(self,
                n_filters: int = 32,
                embed_dim: int = 128,
                num_features_smile: int = 63,
                output_dim: int = 128,
                in_channels: int = 100) -> None:
        super(EmbeddingSmile, self).__init__()

        self.n_filters = n_filters
        self.embed_dim = embed_dim
        self.num_features_smile = num_features_smile
        self.output_dim = output_dim
        self.in_channels = in_channels

        self.embedding_smile = nn.Embedding(self.num_features_smile+1, self.embed_dim)
        self.conv_smile_1 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.n_filters, kernel_size=8)
        self.conv_smile_2 = nn.Conv1d(in_channels=self.n_filters, out_channels=self.n_filters*2, kernel_size=8)
        self.conv_smile_3 = nn.Conv1d(in_channels=self.n_filters*2, out_channels=self.n_filters*3, kernel_size=8)
        self.fc1_smile = nn.Linear(FLATTEN_SHAPE, self.output_dim)

        self.relu = nn.ReLU()

    def forward(self, drug: Batch) -> torch.Tensor:
        embedded_smile = self.embedding_smile(drug.smile_deep)
        
        conv_smile = self.conv_smile_1(embedded_smile)
        conv_smile = self.relu(conv_smile)

        conv_smile = self.conv_smile_2(conv_smile)
        conv_smile = self.relu(conv_smile)

        conv_smile = self.conv_smile_3(conv_smile)
        conv_smile = self.relu(conv_smile)

        drug = conv_smile.view(-1, FLATTEN_SHAPE)
        drug = self.fc1_smile(drug)

        return drug
