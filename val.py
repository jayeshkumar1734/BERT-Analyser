import numpy as np
import time
from model import BertForSequenceClassification
from dataloader_training import dataloader
import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
import os.path
validation_dataloader=dataloader()
device = torch.device("cuda")
model=BertForSequenceClassification(3)
model.cuda()
def flat_accuracy(preds, labels):
  pred_flat = np.argmax(preds, axis=1).flatten()
  labels_flat=labels.flatten()
  return np.sum(pred_flat == labels_flat) / len(labels_flat)

resume_weights = './checkpoint_4.pth.tar'
if os.path.isfile(resume_weights):
  if device:
    checkpoint = torch.load(resume_weights)
   
  start_epoch = checkpoint['epoch']
  best_accuracy = checkpoint['best_accuracy']
  model.load_state_dict(checkpoint['state_dict'])
  print("=> loaded checkpoint '{}' (trained for {} epochs)",checkpoint['epoch'],best_accuracy,start_epoch)

print("Running Validation...")
#t0 = time.time()
model.eval() 
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
for batch in validation_dataloader(1):
  batch = tuple(t.to(device) for t in batch)
  b_input_ids, b_input_mask, b_labels = batch
  with torch.no_grad():  
    outputs = model(b_input_ids,token_type_ids=None,attention_mask=b_input_mask)  
  logits_val = outputs
  logits_val=F.softmax(logits_val,dim=-1)
  logits_val = logits_val.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
 
  tmp_eval_accuracy = flat_accuracy(logits_val.reshape(logits_val.shape[0],3), label_ids)
  pred_flat = np.argmax(logits_val.reshape(logits_val.shape[0],3), axis=1).flatten()
  labels_pred=label_ids.flatten()

  eval_accuracy += tmp_eval_accuracy 
  nb_eval_steps += 1 
print(pred_flat,labels_pred)
print(np.sum(pred_flat == labels_pred) / len(labels_pred))