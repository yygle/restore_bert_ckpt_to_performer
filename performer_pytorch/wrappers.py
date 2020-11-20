# -*- coding:utf-8 -*-
import torch.nn as nn

class ClassificationWrapper(nn.Module):
    def __init__(self, encoder, num_labels):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder.dim, num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, lens):
        x = self.encoder(x).max(1)[0]
        x = self.fc(x)
        return self.softmax(x), self.log_softmax(x)
