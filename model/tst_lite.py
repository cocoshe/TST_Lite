import torch
import torch.nn as nn
from model.PositionalEncoding import PositionalEncoding


class TST_Lite(nn.Module):
    def __init__(self, feature_size=250, hidden_size=128, num_layers=1, dropout=0.1):
        super(TST_Lite, self).__init__()
        self.model_type = 'tst_lite'
        output_size = feature_size
        self.src_mask = None
        if feature_size == 1:
            feature_size = 250
            output_size = 1

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=1, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=1, dropout=dropout)
        self.transformer_decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=num_layers)

        self.decoder1 = nn.Linear(feature_size, output_size)
        self.l_relu = nn.LeakyReLU()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder1.bias.data.zero_()
        self.decoder1.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):

        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder1(output)
        # output = self.l_relu(output)
        # output = self.decoder2(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
