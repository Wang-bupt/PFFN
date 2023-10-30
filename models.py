import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from Layers import *

class textcnn(nn.Module):
    def __init__(self):
        super(textcnn,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=200, kernel_size=(200, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=4)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
  
    def forward(self, x):
        x = self.conv1(x)

        x = self.relu(x)
        x = x.squeeze(2)
        x = self.pool1(x)
        x = self.flatten(x)

        return x
    
class myAttention(nn.Module):
    def __init__(self,dq,dk,dv,d_out):
        super(myAttention,self).__init__()
        self.w_qs = nn.Linear(dq, d_out, bias=True)
        self.w_ks = nn.Linear(dk, d_out, bias=True)
        self.w_vs = nn.Linear(dv, d_out, bias=True)
        self.attention = ScaledDotProductAttention(d_out ** 0.5)
    def forward(self,q_f,k_f,v_f):
        q = self.w_qs(q_f)
        k = self.w_ks(k_f)
        v = self.w_vs(v_f)
        q, attn = self.attention(q, k, v)

        return q, attn
    
class attentionFusion(nn.Module):
    def __init__(self, d1, d2,dout, d_hid, dropout=0.4):
        super(attentionFusion, self).__init__()
    
        self.attn1 = myAttention(d1,d2,d2,dout)
        self.attn2 = myAttention(d2,d1,d1,dout)
        self.feedforward = PositionwiseFeedForward(dout, d_hid,dropout=dropout)
        self.fc = FullyConnectedOutput(2000,1000,1)
        self.sig = nn.Sigmoid()
        
    def forward(self,f1,f2):

        q1,attn = self.attn1(f1,f2,f2)
        q2,attn = self.attn2(f2,f1,f1)
        q1 = self.feedforward(q1)
        q2 = self.feedforward(q2)
        qin = torch.concat([q1,q2],3)
        
        return self.sig(self.fc(qin))

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
