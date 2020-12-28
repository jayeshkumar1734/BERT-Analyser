import torch
from transformers import BertModel
import math
import torch.nn.functional as F
from transformers import AdamW, BertConfig
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
class BertForSequenceClassification(nn.Module):
  
    def __init__(self, num_labels=2):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.size = 768
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps=1e-9
        self.h = 4
        self.first_layer=nn.Linear(768,32)
        self.q_linear = nn.Linear(32,32)
        self.v_linear = nn.Linear(32,32)
        self.k_linear = nn.Linear(32,32)
        #self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(32,32)
        self.classifier = nn.Linear(768, num_labels)
        
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.xavier_normal_(self.first_layer.weight)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        q=input_ids.float().to(device)
        v=input_ids.float().to(device)
        k=input_ids.float().to(device)
        bs = q.size(0)
        
        k = self.k_linear(k).view(bs, -1, self.h, 8)
        #print(k)
        q = self.q_linear(q).view(bs, -1, self.h, 8)
        v = self.v_linear(v).view(bs, -1, self.h, 8)
        
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        scores= torch.matmul(q, k.transpose(-2, -1))/math.sqrt(8)
        scores = torch.matmul(scores, v)
        concat = scores.transpose(1,2).contiguous().view(bs, -1, 32)
        output = self.out(concat)
        #output=self.dropout(output)
        norm = self.alpha * (pooled_output - pooled_output.mean(dim=-1, keepdim=True))/(pooled_output.std(dim=-1, keepdim=True) + self.eps) + self.bias
        logits = self.classifier(norm)
        

        return logits

    