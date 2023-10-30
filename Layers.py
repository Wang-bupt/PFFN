
import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np
import torch.nn.init as init

class FullyConnectedOutput(torch.nn.Module):
    def __init__(self,in1,out1,out2):
        super(FullyConnectedOutput,self).__init__()
        self.fc = torch.nn.Sequential(
            
            torch.nn.Linear(in1, out1),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(out1, out2),
        )

    def forward(self, x):
        out = self.fc(x)
        return out

class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.4):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.4):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) 
        self.w_2 = nn.Linear(d_hid, d_in) 
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def __init_weights__(self):
        init.xavier_normal_(self.w_1)
        init.xavier_normal_(self.w_2)

    def forward(self, x):

        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)

        return x