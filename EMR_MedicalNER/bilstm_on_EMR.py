# -*- coding=utf-8 -*-

'''

Coding based on tutorial: https://www.depends-on-the-definition.com/guide-to-word-vectors-with-gensim-and-keras/
'''


### Set project source folder
### Load necessary packages
import os
import sys
# Current file path
current_path = os.path.abspath(os.path.dirname(__file__))
print("Current Path: "+current_path)
# Remove the file name
root_path = os.path.split(current_path)[0]
print("Project Root Path: "+root_path)
sys.path.append(root_path)


from EMR_MedicalNER.preprocess.CCKS2017_data_preprocess import WordGetter,SentenceGetter
# from EMR_MedicalNER.preprocess.CCKS2018_data_preprocess import WordGetter,SentenceGetter
from EMR_MedicalNER.preprocess.data_load import *
from EMR_MedicalNER.preprocess.text_embedding import *


import pandas as pd
import numpy as np
import csv
import logging
import tensorflow as tf


### Global Variables
# paths
data_set_root = root_path+"/data/CCKS2017/preProcess/"
# data_set_root = root_path+"/data/CCKS2018/preProcess/"

# word2vec vector dimension
WV_DIMENSION = 100 # As defined and trained in xx_data_process.py
############################################################################################

### 1. Load data needed

## Read train and test set data
# train_data = get_full(data_set_root+"kSeg/sent_train_BIO_w_train_t_0.txt") # word
train_data = get_full(data_set_root+"kSeg/sent_train_BIO_c_train_t_0.txt") # character
print("train set row: " + str(train_data.shape[0]))
print("train set col: " + str(train_data.shape[1]))
# validation data set
# val_data = get_full(data_set_root+"kSeg/sent_train_BIO_w_train_v_0.txt") # word
val_data = get_full(data_set_root+"kSeg/sent_train_BIO_c_train_v_0.txt") # character
print("validation set row: " + str(train_data.shape[0]))
print("validation set col: " + str(train_data.shape[1]))
# test data set
# test_data = get_full(data_set_root+"sent_train_BIO_w_test.txt") # word
test_data = get_full(data_set_root+"sent_train_BIO_c_test.txt") # character
print("test set row: " + str(test_data.shape[0]))
print("test set col: " + str(test_data.shape[1]))

######################### Option 1: Random ID #######################################
token_set = set(train_data[1].values).union(set(test_data[1].values),set(val_data[1].values))
n_tokens = len(token_set)

## Load word vectors

######################### Option 2.A: Word2Vec #######################################
from gensim.models import KeyedVectors
# CCKS 2017 trained character word2vec
w2vmodel_path = root_path+'/models/word2vec/'+'w2v_sg_1_hs_0_negative_10_min_count_1_iter_5_size_100_c.model.bin'
# # CCKS 2017 trained word word2vec
# w2vmodel_path = root_path+'/models/word2vec/'+'w2v_sg_1_hs_0_negative_10_min_count_1_iter_5_size_100_w.model.bin'
# # CCKS 2018 trained character word2vec
# w2vmodel_path = root_path+'/models/word2vec/'+'w2v_sg_1_hs_0_negative_10_min_count_1_iter_5_size_100_c_ccks2018.model.bin'
# # CCKS 2018 trained word word2vec
# w2vmodel_path = root_path+'/models/word2vec/'+'w2v_sg_1_hs_0_negative_10_min_count_1_iter_5_size_100_w_ccks2018.model.bin'

w2vmodel = KeyedVectors.load_word2vec_format(w2vmodel_path, binary=True)
word_vectors = w2vmodel.wv
n_tokens = len(word_vectors.vocab)
print("Number of word vectors: {}".format(n_tokens))
vec_dimension = WV_DIMENSION

######################### Option 2.B: Chinese Wiki Corpus #################################
chinese_wv_loader = ChineseWordVectorsLoader()
# vec_count,vec_dimension,word_vectors = chinese_wv_loader.load_vec_from_file('sgns.wiki.bigram-char') # char
# vec_count,vec_dimension,word_vectors = chinese_wv_loader.load_vec_from_file('sgns.wiki.bigram') # word

######################### Option 2.C: SCWE #################################
# scwe_wv_loader = SCWELoader()
# vec_count,vec_dimension,word_vectors = scwe_wv_loader.load_scwe_char_from_file('') # char
# vec_count,vec_dimension,word_vectors = scwe_wv_loader.load_scwe_word_from_file('') # word


## Create dictionary of characters (or word)  and tags
# Represent characters/words

# tokens,tokens2idx,n_tokens = get_val2idx(train_data[1].values)     # option 1
tokens,tokens2idx,n_tokens = get_val2idx(word_vectors.vocab) # option 2
# # word embeddings                                               # option 2
# embeddings = get_word2vecEmbedding(w2vmodel_path,chars)         # option 2
print('Number of characters/words (n_tokens): '+ str(n_tokens))
tokens2idx["UNK"] = 1 # "unknown token" - is used to replace the rare words that did not fit in vocabulary.
tokens2idx["PAD"] = 0 # padding with 0


# # Represent tags
word_getter = WordGetter()
tags2idx = word_getter.category_dic
n_tags = len(tags2idx)
print('Number of tags (n_tags): '+ str(n_tags))
tags2idx["PAD"] = tags2idx["O"] # padding with 'O': 0


## Get sentences
train_sent_getter  = SentenceGetter(train_data)
val_sent_getter = SentenceGetter(val_data)
test_sent_getter  = SentenceGetter(test_data)

train_sentences = train_sent_getter.sentences
val_sentences = val_sent_getter.sentences
test_sentences = test_sent_getter.sentences

n_sent_train = len(train_sentences)
n_sent_val = len(val_sentences)
n_sent_test = len(test_sentences)

## Padding sequences
# sentence level
print("Padding Sequences ...")
X_tr = [[tokens2idx[c[0]] for c in s] for s in train_sentences]
X_v = [[tokens2idx[c[0]] for c in s] for s in val_sentences]
X_te = [[tokens2idx[c[0]] for c in s] for s in test_sentences]
y_tr = [[tags2idx[c[1]] for c in s] for s in train_sentences]
y_v = [[tags2idx[c[1]] for c in s] for s in val_sentences]
y_te = [[tags2idx[w[1]] for w in s] for s in test_sentences]

from keras.preprocessing.sequence import pad_sequences
# maximum length of sentence (characters or words in a sentence)
max_len_train = max(len(s) for s in train_sentences)
max_len_val = max(len(s) for s in val_sentences)
max_len_test = max(len(s) for s in test_sentences)
max_len = max(max_len_train,max_len_val,max_len_test)
# padding
X_tr = pad_sequences(maxlen=max_len, sequences=X_tr, padding="post", truncating='post', value=charas2idx["PAD"])
X_v = pad_sequences(maxlen=max_len, sequences=X_v, padding="post", truncating='post', value=charas2idx["PAD"])
X_te = pad_sequences(maxlen=max_len, sequences=X_te, padding="post", truncating='post', value=charas2idx["PAD"])

y_tr = pad_sequences(maxlen=max_len, sequences=y_tr, padding="post", truncating='post',value=tags2idx["PAD"])
y_v = pad_sequences(maxlen=max_len, sequences=y_v, padding="post", truncating='post',value=tags2idx["PAD"])
y_te = pad_sequences(maxlen=max_len, sequences=y_te, padding="post", truncating='post',value=tags2idx["PAD"])

from keras.utils import to_categorical
y_tr= [to_categorical(i, num_classes=n_tags) for i in y_tr]
y_v = [to_categorical(i, num_classes=n_tags) for i in y_v]
y_te = [to_categorical(i, num_classes=n_tags) for i in y_te]

## Load word vectors


# initialize the matrix with random numbers
wv_matrix = (np.random.rand(n_tokens, vec_dimension) - 0.5) / 5.0 # initialize word vector matrix
for word, i in tokens2idx.items():
    if i >= n_tokens:
        continue
    try:
        embedding_vector = word_vectors[word]
        # words not found in embedding index will be all-zeros.
        wv_matrix[i] = embedding_vector
    except:
        pass



### 2. Setup Models
print("Setup Models ...")

from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout,\
    Bidirectional,SpatialDropout1D

from keras.optimizers import Adam


## Inputs (indexs)
input = Input(shape=(max_len,),dtype='int32')

## Embedded sequences
## Uncomment the option you choose
######################################################
##  Option1: Use integer index as input
######################################################
model = Embedding(input_dim=n_tokens + 1, output_dim=20, input_length=max_len, mask_zero=True)(input)

######################################################
##  Option2: Word2Vec Embedding Layer
######################################################
#
# wv_layer = Embedding(n_tokens,
#                      WV_DIMENSION,
#                      mask_zero=False, weights=[wv_matrix],
#                      input_length=max_len,
#                      trainable=False)
# model = wv_layer(input) # embedded_sequences

# Input Dropout
model = SpatialDropout1D(0.1)(model) # embedded_sequences
model = Bidirectional(LSTM(units=100, return_sequences=True))(model) # bilstm

# Output
model = Dropout(0.2)(model)# output_drop
out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer

### 3. Build Model
model = Model(inputs=input,outputs=out)
batch_size = 32
ephochs = 20
# learning rate to be (1/x) of the original learning rate at the end of each epoch
dataset_size = train_data.shape[0]
batches_per_epoch = dataset_size/batch_size
lr_decay = (1./(1/32) -1)/batches_per_epoch
# lr_decay_ccks2018 = (1./(1/12) -1)/batches_per_epoch
model.compile(
    optimizer= Adam(lr=0.032, decay=lr_decay), # ccks2017
    # optimizer= Adam(lr=0.032, decay=lr_decay_ccks2018), #ccks2018
    loss="categorical_crossentropy", #
    metrics=["accuracy"]
)
history = model.fit(X_tr, np.array(y_tr),
                    batch_size=batch_size,
                    epochs=ephochs,
                    validation_data = (X_v, np.array(y_v)),
                    verbose=1
                    )

### Save Model
# model.save(root_path + '/models/lstm/w2v_bilstm_w_2.h5')



# history is a dictionary，kyes are val_loss,val_acc,loss,acc
hist = pd.DataFrame(history.history)
fig = plt.figure(figsize=(12,12))
# add subplots
sub_fig1 = fig.add_subplot(1,2,1) # 1 row 2 cols 1st figure
sub_fig2 = fig.add_subplot(1,2,2)
# set titles
sub_fig1.set_title('Accuracy')
sub_fig2.set_title('Loss')
# set values and labels
sub_fig1.plot(hist["accuracy"],label='acc')
sub_fig1.plot(hist["val_accuracy"], label='val_acc')
sub_fig1.legend(loc="lower right")
sub_fig2.plot(hist["loss"],label='loss')
sub_fig2.plot(hist["val_loss"],label='val_loss')
sub_fig2.legend(loc="upper right")

# show figure
plt.show()


# from keras.models import load_model
# model = load_model(root_path + '/models/lstm/w2v_bilstm_c_1.h5')
# Load test data set

score = model.evaluate(X_te, np.array(y_te), batch_size=32,verbose=1)
print("Score:")
print(model.metrics_names)
print(score)


# ## Prediction on test set
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
# print("Input:")
# print(X_te[0])
# print("Supposed output:")
# print(y_te)
# print(np.array(y_te))
test_pred = model.predict([X_te], verbose=1)
# print("Prediction result:")
# print(test_pred[0])
idx2tag = {i: w for w,i in tags2idx.items()}
tags_size = len(idx2tag)
idx2tag[tags_size] = 'O'


def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out

pred_labels = pred2label(test_pred)
test_labels = pred2label(np.array(y_te))
# print(pred_labels)
# print(test_labels)
# strict
print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))
print(classification_report(test_labels, pred_labels))
# relaxed
def relax_label(l):
    if l != 'O':
        return l[2:]
    return l
# r_test_labels = np.array(map(relax_label,list(test_labels)))
# r_pred_labels = np.array(map(relax_label,list(pred_labels)))
# print("F1-score: {:.1%}".format(f1_score(r_test_labels, r_pred_labels)))
# print(classification_report(r_test_labels, r_pred_labels))

