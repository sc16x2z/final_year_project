# -*- coding=utf-8 -*-
# based on Tobias's blog https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/
# For running python in terminal, define project root path
import os
import sys

# Current file path
current_path = os.path.abspath(os.path.dirname(__file__))
print("Current Path: "+current_path)
# Remove the file name
root_path = os.path.split(current_path)[0]
print("Project Root Path: "+root_path)
sys.path.append(root_path)


import logging

# path
train_set_path = root_path+"/data/CCKS2017/preProcess/"

### 1. Load data with word and tag
import pandas as pd
import numpy as np
import csv

from EMR_MedicalNER.preprocess.CCKS2017_data_preprocess import WordGetter,SentenceGetter
from EMR_MedicalNER.preprocess.data_load import *
# path
data_set_root = root_path+"/data/CCKS2017/preProcess/"
## Read train and test set data
train_data = get_full(data_set_root+"sent_train_BIO_c_train.txt")
# train_data = get_full(root_path+"/smalldata.txt")
print("train set row: " + str(train_data.shape[0]))
print("train set col: " + str(train_data.shape[1]))
# validation data set
val_data = get_full(data_set_root+"/kSeg/sent_train_BIO_c_train_v_0.txt")
# val_data = get_full(root_path+"/smalldata.txt")
print("validation set row: " + str(train_data.shape[0]))
print("validation set col: " + str(train_data.shape[1]))
# test data set
test_data = get_full(data_set_root+"sent_train_BIO_c_test.txt")
print("test set row: " + str(test_data.shape[0]))
print("test set col: " + str(test_data.shape[1]))

# Represent tags
word_getter = WordGetter()
tags2idx = word_getter.category_dic
n_tags = len(tags2idx)
tag_values = list(tags2idx.keys())
tag_values.append("PAD")
tags2idx = {t: i for i, t in enumerate(tag_values)}
print('Number of tags (n_tags): '+ str(n_tags))
# tags2idx["PAD"] = tags2idx["O"] # padding with 'O': 0
# tags2idx["CLS"] = "[CLS]"
# tags2idx["SEP"] = "[SEP]"

## Get sentences
train_sent_getter  = SentenceGetter(train_data)
val_sent_getter = SentenceGetter(val_data)
test_sent_getter  = SentenceGetter(test_data)

train_sentences = [[c[0] for c in sentence] for sentence in train_sent_getter.sentences]
val_sentences = [[c[0] for c in sentence] for sentence in val_sent_getter.sentences]
test_sentences = [[c[0] for c in sentence] for sentence in test_sent_getter.sentences]

# tag for character or word in each sentence
train_tags =  [[c[1] for c in sentence] for sentence in train_sent_getter.sentences]
val_tags =  [[c[1] for c in sentence] for sentence in val_sent_getter.sentences]
test_tags =  [[c[1] for c in sentence] for sentence in test_sent_getter.sentences]


n_sent_train = len(train_sentences)
n_sent_val = len(val_sentences)
n_sent_test = len(test_sentences)

import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = BertTokenizer.from_pretrained('', do_lower_case=True)
tokenizer = BertTokenizer.from_pretrained(root_path+'/packages/chinese_roberta_wwm_large_ext_pytorch')
# Tokenize all sentences
# Get tokens and corresponding labels
tokenized_texts_and_labels_train = [
    tokenize_and_preserve_labels(tokenizer,sent, labs)
    for sent, labs in zip(train_sentences, train_tags)
]
tokenized_texts_train = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels_train]
labels_train = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels_train]

tokenized_texts_and_labels_val = [
    tokenize_and_preserve_labels(tokenizer,sent, labs)
    for sent, labs in zip(val_sentences, val_tags)
]
tokenized_texts_val = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels_val]
labels_val = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels_val]

tokenized_texts_and_labels_test = [
    tokenize_and_preserve_labels(tokenizer,sent, labs)
    for sent, labs in zip(test_sentences, test_tags)
]
tokenized_texts_test = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels_test]
labels_test = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels_test]



print("Padding Sequences ...")

max_len = 100 # max squence length
batch_size = 32

input_train = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts_train],
                          maxlen=max_len, dtype="long", value=0.0, truncating="post", padding="post")
print("input_train")
print(input_train)
input_val = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts_val],
                          maxlen=max_len, dtype="long", value=0.0, truncating="post", padding="post")
input_test = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts_test],
                          maxlen=max_len, dtype="long", value=0.0, truncating="post", padding="post")
tag_train = pad_sequences([[tags2idx.get(t) for t in tag] for tag in labels_train], maxlen=max_len,
                          value=tags2idx["PAD"], padding="post", dtype="long", truncating="post")
print("tag_train")
print(tag_train)
tag_val = pad_sequences([[tags2idx.get(t) for t in tag] for tag in labels_val], maxlen=max_len,
                          value=tags2idx["PAD"], padding="post", dtype="long", truncating="post")
tag_test = pad_sequences([[tags2idx.get(t) for t in tag] for tag in labels_test], maxlen=max_len,
                          value=tags2idx["PAD"], padding="post", dtype="long", truncating="post")

# Create masks
attention_masks_train = [[float(i!=0) for i in input_id] for input_id in input_train]
attention_masks_val = [[float(i!=0) for i in input_id] for input_id in input_val]
attention_masks_test = [[float(i!=0) for i in input_id] for input_id in input_test]
print("attention_masks_train")
print(attention_masks_train)

input_train = torch.tensor(input_train)
input_val = torch.tensor(input_val)
tag_train = torch.tensor(tag_train)
tag_val = torch.tensor(tag_val)
masks_train = torch.tensor(attention_masks_train)
masks_val = torch.tensor(attention_masks_val)

print("input_train")
print(input_train)
print("tag_train")
print(tag_train)
print("masks_train")
print(masks_train)

train_data = TensorDataset(input_train, masks_train, tag_train)
train_dataloader = DataLoader(train_data,batch_size=batch_size)
val_data = TensorDataset(input_val, masks_val, tag_val)
val_dataloader = DataLoader(val_data,batch_size=batch_size)


import transformers
from transformers import BertForTokenClassification, AdamW

# Setup the Bert model for finetuning
print("Setup Models ...")
# Setup Models
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.callbacks import EarlyStopping
# from kashgari.tasks.seq_labeling import BLSTMCRFModel
from pytorch_pretrained_bert import BertForTokenClassification

input = Input(shape=(max_len,))
# model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tags2idx))(input)
model = BertForTokenClassification.from_pretrained(
    root_path + '/packages/chinese_roberta_wwm_large_ext_pytorch', num_labels=len(tags2idx),
    # output_attentions = False,
    # output_hidden_states = False
)

FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)
# add a scheduler to linearly reduce the learning rate throughout the epochs
from transformers import get_linear_schedule_with_warmup

epochs = 3
max_grad_norm = 1.0

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs
print("Total steps:")
print(total_steps)

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)


from seqeval.metrics import f1_score
from tqdm import tqdm, trange

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

## Store the average loss after each epoch so we can plot them.
loss_values, validation_loss_values = [], []

for _ in trange(epochs, desc="Epoch"):
    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.

    # Put the model into training mode.
    model.train()
    # Reset the total loss for this epoch.
    total_loss = 0

    # Training loop
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        print('b_input_ids:')
        print(b_input_ids)
        print('b_input_mask:')
        print(b_input_mask)
        print('b_labels')
        print(b_labels)
        # Always clear any previously calculated gradients before performing a backward pass.
        model.zero_grad()
        # forward pass
        # This will return the loss (rather than the model output)
        # because we have provided the `labels`.
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
        print("outputs")
        print(outputs)
        # get the loss
        # loss = outputs[0]
        loss = outputs
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # track train loss
        total_loss += loss.item()
        # Clip the norm of the gradient
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)


    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    # Put the model into evaluation mode
    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
        # Move logits and labels to CPU
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()
        eval_accuracy += flat_accuracy(logits, label_ids)
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    validation_loss_values.append(eval_loss)
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    valid_tags = [tag_values[l_i] for l in true_labels
                                  for l_i in l if tag_values[l_i] != "PAD"]
    print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
    print()

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(loss_values, 'b-o', label="training loss")
    plt.plot(validation_loss_values, 'r-o', label="validation loss")

    # Label the plot.
    plt.title("Learning curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()