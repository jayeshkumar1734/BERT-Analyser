from tokenizer import tokenized
from preprocessing import giveout_words_labels
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler 
labels=giveout_words_labels()
labels_train=labels[1]
labels_test=labels[3]
inputs=tokenized()
input_ids_train=inputs[0]
attention_masks_train=inputs[1]
input_ids_test=inputs[2]
attention_masks_test=inputs[3]

def dataloader():
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids_train, labels_train,random_state=2018, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks_train, labels_train,random_state=2018, test_size=0.1)

    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    test_inputs = torch.tensor(input_ids_test)
    test_labels = torch.tensor(labels_test)
    test_masks = torch.tensor(attention_masks_test)



    batch_size = 32
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=validation_inputs.shape[0])

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=test_inputs.shape[0])

    return train_dataloader,validation_dataloader,test_dataloader