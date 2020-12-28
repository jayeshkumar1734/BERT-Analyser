from model import BertForSequenceClassification
import torch
import math
import torch.nn.functional as F
from transformers import AdamW, BertConfig
from torch.autograd import Variable
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import os 
import os.path
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dataloader_training import dataloader
train_dataloader=dataloader()
device = torch.device("cuda")
model=BertForSequenceClassification(3)
model.cuda()
optimizer = AdamW(model.parameters(),lr = 2e-5,eps = 1e-8)
epochs =10
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps = 0,num_training_steps = total_steps)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))
def save_checkpoint(state, is_best, filename='./checkpoint_4.pth.tar'):
    
    if is_best:
        print ("=> Saving a new best")
        torch.save(state, filename)  
    else:
        print ("=> Training Loss did not improve")
resume_weights = './checkpoint_4.pth.tar'
if os.path.isfile(resume_weights):
    if device:
        checkpoint = torch.load(resume_weights)
   
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (trained for {} epochs)",checkpoint['epoch'],best_accuracy,start_epoch)



  
loss_values = []
best_accuracy=torch.FloatTensor([0.0004])
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
for epoch_i in range(0, epochs):
    if epoch_i%1==0:
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...') 
    t0 = time.time() 
    total_loss = 0 
    model.train()
 
    criterion_loss = nn.CrossEntropyLoss()
 
    for step, batch in enumerate(train_dataloader(0)): 
        if step % 50 == 0 and not step == 0:

            elapsed = format_time(time.time() - t0)


            print(' Batch {:>5,} of {:>5,}. Elapsed: {:}.'.format(step, len(train_dataloader), elapsed)) # Unpack this training batch from our dataloader. 

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device) 
        model.zero_grad() 
  
        outputs = model.forward(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
  
        loss=criterion_loss(outputs.reshape(b_labels.shape[0],3),b_labels.long().to(device))

        total_loss += loss.item()/64 
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step() 
        scheduler.step() 

    avg_train_loss = total_loss / len(train_dataloader) 
    print(avg_train_loss)
 
    acc = torch.FloatTensor([avg_train_loss])
 
    is_best = bool(acc.numpy() < best_accuracy.numpy())
 
    best_accuracy = torch.FloatTensor(min(acc.numpy(), best_accuracy.numpy()))
    save_checkpoint({
                'epoch': start_epoch + epoch_i + 1,
                'state_dict': model.state_dict(),
                'best_accuracy': best_accuracy
            }, is_best)
    loss_values.append(avg_train_loss)
print("Training complete!")