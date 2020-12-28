from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop=set(stopwords.words('english'))
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler 
from model import BertForSequenceClassification
import torch
device = torch.device("cuda")
model=BertForSequenceClassification(3)
model.cuda()
import numpy as numpy
import json
import os
import os.path
import math
import torch.nn.functional as F
import numpy as np
import pandas as pd


# string=r"#$%&'!()*+,-.:;<=>?@[\]^_\"`{|}~"

# def remove_URL(text):
#     return re.sub(r"https?://\S+|www\.\S+", "", text)

# def remove_html(text):
#     html=re.compile(r'<.*?>')
#     return html.sub(r'',text)

# def remove_punct(text):
#     table=str.maketrans('\t',' ',string)
#     return text.translate(table)


# def fix_Plan(location):
#     letters_only = re.sub("[^a-zA-Z]",  # Search for all non-letters
#                           " ",          # Replace all non-letters with spaces
#                           str(location))     # Column and row to search    

#     words = letters_only.lower().split()     
#     stops = set(stopwords.words("english"))      
#     meaningful_words = [w for w in words]      
#     return (" ".join(meaningful_words))

 

# class Prediction():



#     def processed(text):

#         text=text.apply(str)
#         text=text.apply(lambda x : x.lower())
#         text=text.apply(lambda x : remove_URL(x))
#         text=text.apply(lambda x : remove_html(x))
#         text=text.apply(lambda x : remove_punct(x))
#         num_responses = len(text)    
#         clean_Plan_responses = []
        
#         for i in range(0,num_responses):
#             clean_Plan_responses.append(fix_Plan(text[i]))

#         text_predict=[]
#         for i in range(len(clean_Plan_responses)):
#             result = re.sub(r"^\s+", "", clean_Plan_responses[i])
#             text_predict.append(re.sub(r"\s+$", "", result))

#     def data_tensor():
#         input_ids = []
#         for sent in text_predict:
#             encoded_sent = tokenizer.encode(sent,add_special_tokens = True)
#             input_ids.append(encoded_sent)
#         MAX_LEN = 32
#         input_ids = pad_sequences(input_ids_train, maxlen=MAX_LEN, dtype="long",value=0, truncating="post", padding="post")
#         attention_masks = []
#         for sent in input_ids:
#             att_mask = [int(token_id > 0) for token_id in sent]
#             attention_masks.append(att_mask)
        
#         inputs = torch.tensor(input_ids)
#         masks = torch.tensor(attention_masks)
#         data = TensorDataset(inputs,masks)
#         sampler = SequentialSampler(data)
#         loader = DataLoader(data, sampler=sampler, batch_size=inputs.shape[0])
        

#     resume_weights = './checkpoint_4.pth.tar'
#     if os.path.isfile(resume_weights):
#         if device:
#             checkpoint = torch.load(resume_weights)
       
#         start_epoch = checkpoint['epoch']
#         best_accuracy = checkpoint['best_accuracy']
#         model.load_state_dict(checkpoint['state_dict'])


#     model.eval() 
#     for batch in loader:
#         batch = tuple(t.to(device) for t in batch)
#         b_input_ids, b_input_mask, b_labels = batch
#         with torch.no_grad():  
#             outputs = model(b_input_ids,token_type_ids=None,attention_mask=b_input_mask)  
#         logits_val = outputs
#         logits_val=F.softmax(logits_val,dim=-1)
#         logits_val = logits_val.detach().cpu().numpy()
#         label_ids = b_labels.to('cpu').numpy()
#         pred_flat = np.argmax(logits_val.reshape(logits_val.shape[0],3), axis=1).flatten()

#     for i in range(len(pred_flat)):
#         if pred_flat[i]==0:
#             pred_flat[i]="neutral"
#         if pred_flat[i]==1:
#             pred_flat[i]="positive"
#         if pred_flat[i]==2:
#             pred_flat[i]=="negative"

#     predictions=json.dumps(pred_flat)
# print(predictons)


def prediction(text):

  string=r"#$%&'!()*+,-.:;<=>?@[\]^_\"`{|}~"
  #text = input().split(':')
  text_df=pd.DataFrame()
  text_df["text"]=text
  print(text_df)
  def remove_URL(text):
      return re.sub(r"https?://\S+|www\.\S+", "", text)

  def remove_html(text):
      html=re.compile(r'<.*?>')
      return html.sub(r'',text)

  def remove_punct(text):
      table=str.maketrans('\t',' ',string)
      return text.translate(table)


  def fix_Plan(location):
      letters_only = re.sub("[^a-zA-Z]",  # Search for all non-letters
                            " ",          # Replace all non-letters with spaces
                            str(location))     # Column and row to search    

      words = letters_only.lower().split()     
      stops = set(stopwords.words("english"))      
      meaningful_words = [w for w in words]      
      return (" ".join(meaningful_words))

  resume_weights = './checkpoint_4.pth.tar'
  if os.path.isfile(resume_weights):
    if device:
      checkpoint = torch.load(resume_weights)
        
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']
    model.load_state_dict(checkpoint['state_dict'])
  #print("=> loaded checkpoint '{}' (trained for {} epochs)",checkpoint['epoch'],best_accuracy,start_epoch)





  text_df["text"]=text_df["text"].apply(str)
  text_df["text"]=text_df["text"].apply(lambda x : x.lower())
  text_df["text"]=text_df["text"].apply(lambda x : remove_URL(x))
  text_df["text"]=text_df["text"].apply(lambda x : remove_html(x))
  text_df["text"]=text_df["text"].apply(lambda x : remove_punct(x))
  num_responses = text_df.shape[0]    
  clean_Plan_responses = []

  for i in range(0,num_responses):
      clean_Plan_responses.append(fix_Plan(text[i]))

  text_predict=[]
  for i in range(len(clean_Plan_responses)):
      result = re.sub(r"^\s+", "", clean_Plan_responses[i])
      text_predict.append(re.sub(r"\s+$", "", result))

  input_ids = []
  for sent in text_predict:
      encoded_sent = tokenizer.encode(sent,add_special_tokens = True,max_length=512)
      input_ids.append(encoded_sent)
  MAX_LEN = 32
  input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",value=0, truncating="post", padding="post")
  attention_masks = []
  for sent in input_ids:
      att_mask = [int(token_id > 0) for token_id in sent]
      attention_masks.append(att_mask)

  inputs = torch.tensor(input_ids)
  masks = torch.tensor(attention_masks)
  data = TensorDataset(inputs,masks)
  sampler = SequentialSampler(data)
  loader = DataLoader(data, sampler=sampler, batch_size=inputs.shape[0])
  #print("aaa")
  results=[]
  model.eval() 
  for batch in loader:
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_input_mask = batch
      with torch.no_grad():  
          outputs = model(b_input_ids,token_type_ids=None,attention_mask=b_input_mask)  
      logits_val = outputs
      logits_val=F.softmax(logits_val,dim=-1)
      logits_val = logits_val.detach().cpu().numpy()
    
      results.append(logits_val)
  results=np.array(results)
  print(results.shape)
  temp=[]
  for i in range(len(text)):
    temp.append(results[0][i])
  results=np.array(temp)
  
  return results

