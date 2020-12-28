from preprocessing import giveout_words_labels
from keras.preprocessing.sequence import pad_sequences
import torch
import tensorflow as tf
from transformers import BertTokenizer
text=giveout_words_labels()
text_train=text[0]
text_test=text[2]

if torch.cuda.is_available():
    device = torch.device("cuda") 
    print('There are %d GPU(s) available.' % torch.cuda.device_count()) 
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def tokenized():
    input_ids_train = []
    for sent in text_train:
        encoded_sent = tokenizer.encode(sent,add_special_tokens = True)
        input_ids_train.append(encoded_sent)


    input_ids_test = []
    for sent in text_test:
        encoded_sent = tokenizer.encode(sent,add_special_tokens = True)
        input_ids_test.append(encoded_sent)

    MAX_LEN = 32
    input_ids_train = pad_sequences(input_ids_train, maxlen=MAX_LEN, dtype="long",value=0, truncating="post", padding="post")
    input_ids_test = pad_sequences(input_ids_test, maxlen=MAX_LEN, dtype="long",value=0, truncating="post", padding="post")

    attention_masks_train = []
    for sent in input_ids_train:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks_train.append(att_mask)
    attention_masks_test = []
    for sent in input_ids_test:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks_test.append(att_mask)

    return input_ids_train,attention_masks_train,input_ids_test,attention_masks_test