from __future__ import absolute_import
import torch
import torch.nn as nn
from .p_graph_conv import PGraphConv

class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = PGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x

class ResSimplePGCN(nn.Module):
    def __init__(self, adj, hidden_dim, num_layers=4):
        super(ResSimplePGCN, self).__init__()
        _gconv_layers = []
        for i in range(num_layers):
            _gconv_layers.append(_GraphConv(adj, hidden_dim, hidden_dim))
        self.gconv_layers = nn.Sequential(*_gconv_layers)

    def forward(self, x):
        out = self.gconv_layers(x)
        return out