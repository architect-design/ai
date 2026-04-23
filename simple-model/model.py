# model.py
import torch
import torch.nn as nn


class SpecificationLearner(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SpecificationLearner, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        out = lstm_out.reshape(lstm_out.size(0) * lstm_out.size(1), -1)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(1, batch_size, self.hidden_dim).zero_(),
                  weight.new(1, batch_size, self.hidden_dim).zero_())
        return hidden