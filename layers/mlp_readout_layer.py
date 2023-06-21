import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    MLP Layer used after graph vector representation
"""

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, dropout=0.5, L=1): # L=nb_hidden_layers
        super().__init__()
        # list_FC_layers = [nn.Linear(input_dim//2**l, input_dim//2**(l+1), bias=True) for l in range(L)]
        list_FC_layers = []
        list_FC_layers.append(nn.Linear(input_dim, 128*2, bias=True))
        list_FC_layers.append(nn.Linear(128*2, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.dropout = nn.Dropout(dropout)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = self.dropout(y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y