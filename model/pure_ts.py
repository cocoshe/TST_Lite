import torch
import torch.nn as nn


class PureTransformer(nn.Module):
    def __init__(self, d_model, nhead=1, dropout=0.1):
        super(PureTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=2, dropout=dropout, num_decoder_layers=1)
        self.model_type = 'pure_transformer'

    def forward(self, src):
        return self.transformer(src, src)


