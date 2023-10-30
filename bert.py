import torch

from transformers import BertModel
from transformers import BertTokenizer
from torch import nn
import numpy as np


class BertEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        tokenizer = BertTokenizer.from_pretrained("../../bert-base-chinese")
        model = BertModel.from_pretrained("../../bert-base-chinese")
        # bert-base-uncased bert-base-chinese
        self.tokenlize = tokenizer
        self.bert = model
        
    def forward(self, x, max_length):
        x = self.tokenlize.encode(x)

        if len(x)<max_length:
            padded = np.array(x + [0] * (max_length - len(x)))
        else:
            padded = np.array([1]*max_length)
            x = x[:max_length]

        attention_mask = np.where(padded != 0, 1, 0)

        input_ids = torch.tensor(padded)

        attention_mask = torch.tensor(attention_mask).unsqueeze(0)

        input_ids=input_ids.unsqueeze(0)

        bert_outputs = self.bert(input_ids,attention_mask)[0]
        
        return bert_outputs
