import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, gru_layers):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, gru_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.gru(embeddings)
        outputs = self.linear(hiddens)
        return outputs
