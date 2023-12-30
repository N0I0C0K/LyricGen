import torch
import torch.nn as nn


class SongNet(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_size: int,
        lstm_hidden_size: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.lstm = nn.LSTM(
            embedding_size, lstm_hidden_size, num_layers=2, batch_first=True
        )
        self.h2h = nn.Linear(lstm_hidden_size, lstm_hidden_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.h2o = nn.Linear(lstm_hidden_size, num_embeddings)

    def forward(self, raw, hidden_state=None, noise: bool = False):
        embedd = self.embedding(raw)
        if noise:
            embedd = embedd + torch.rand_like(embedd)

        out, hidden_state = self.lstm(embedd, hidden_state)

        out = self.leaky_relu(self.h2h(out))

        out = self.h2o(out)

        return out, hidden_state
