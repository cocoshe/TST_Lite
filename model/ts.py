import os

import torch
import torch.nn as nn
from matplotlib import pyplot as plt


class PureTransformer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(PureTransformer, self).__init__()
        self.transformer = nn.Transformer(nhead=16, num_encoder_layers=12, dropout=dropout)

    def forward(self, x):
        return self.transformer(x)


# transformer_model = nn.Transformer(32, nhead=16, num_encoder_layers=12)
#
# src = torch.randn(35, 32, 32)
# tgt = torch.randn(20, 32, 32)
# out = transformer_model(src, tgt)
# print(out.size())

import pandas as pd
print(os.getcwd())
df = pd.read_csv('../dataset/mammography_label.csv')
features = df.iloc[:, :-1].values
labels = df.iloc[:, -1].values

transformer_model = nn.Transformer(features.shape[1], nhead=2, num_encoder_layers=12, dim_feedforward=128)
src = tgt = features
src = torch.tensor(src.reshape(src.shape[0], 1, src.shape[1]), dtype=torch.float)
tgt = torch.tensor(tgt.reshape(tgt.shape[0], 1, tgt.shape[1]), dtype=torch.float)
print(src.shape)
print(tgt.shape)
out = transformer_model(src, tgt)
for i in range(out.size(0)):
    plt.plot(out[:, i])
    plt.show()




