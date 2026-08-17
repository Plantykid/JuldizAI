import torch
import torch.nn as nn

class JuldizModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(JuldizModel, self).__init__()
        # Vectorical of showing words
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # stot LSTM
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        # Exit slot
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden
