# -*- coding=utf-8 -*-

'''
Perform BiLSTM-CNNS-CRF
'''

import os
import sys
from sys import path
import numpy as np
import pandas as pd

# Current file path
current_path = os.path.abspath(os.path.dirname(__file__))
print("Current Path: "+current_path)
# Remove the file name
root_path = os.path.split(current_path)[0]
print("Project Root Path: "+root_path)
print(root_path)
sys.path.append(root_path)


from EMR_MedicalNER.preprocess.CCKS2017_data_preprocess import WordGetter,SentenceGetter
# from EMR_MedicalNER.preprocess.CCKS2018_data_preprocess import WordGetter,SentenceGetter
from EMR_MedicalNER.preprocess.data_load import *


# path
data_set_root = root_path+"/data/CCKS2017/preProcess/"

### 1. Load data needed

## Read train and test set data
train_data = get_full(data_set_root+"kSeg/sent_train_BIO_w_train_t_0.txt") # word
print("train set row: " + str(train_data.shape[0]))
print("train set col: " + str(train_data.shape[1]))
# validation data set
val_data = get_full(data_set_root+"kSeg/sent_train_BIO_w_train_v_0.txt") # word
print("validation set row: " + str(train_data.shape[0]))
print("validation set col: " + str(train_data.shape[1]))
# test data set
test_data = get_full(data_set_root+"sent_train_BIO_w_test.txt") # word
print("test set row: " + str(test_data.shape[0]))
print("test set col: " + str(test_data.shape[1]))

## Load word vectors
from gensim.models import KeyedVectors

# CCKS 2017 trained word word2vec
w2vmodel_path = root_path+'/models/word2vec/'+'w2v_sg_1_hs_0_negative_10_min_count_1_iter_5_size_100_w.model.bin'
# # CCKS 2018 trained word word2vec
# w2vmodel_path = root_path+'/models/word2vec/'+'w2v_sg_1_hs_0_negative_10_min_count_1_iter_5_size_100_w_ccks2018.model.bin'

w2vmodel = KeyedVectors.load_word2vec_format(w2vmodel_path, binary=True)
word_vectors = w2vmodel.wv
n_words = len(word_vectors.vocab)
print("Number of word vectors: {}".format(n_words))

# CCKS 2017 trained character word2vec
c2vmodel_path = root_path+'/models/word2vec/'+'w2v_sg_1_hs_0_negative_10_min_count_1_iter_5_c.model.bin'
# # CCKS 2018 trained character word2vec
# c2vmodel_path = root_path+'/models/word2vec/'+'w2v_sg_1_hs_0_negative_10_min_count_1_iter_5_c_ccks2018.model.bin'

c2vmodel = KeyedVectors.load_word2vec_format(w2vmodel_path, binary=True)
char_vectors = c2vmodel.wv
n_chars = len(char_vectors.vocab)
print("Number of character vectors: {}".format(n_chars))



## Create dictionary of characters (or words)  and tags
# Index characters/words
words,words2idx,n_words = get_val2idx(word_vectors.vocab)
embeddings = get_word2vecEmbedding(w2vmodel_path,words)
words2idx["UNK"] = 1 # "unknown token" - is used to replace the rare words that did not fit in vocabulary.
words2idx["PAD"] = 0 # padding with 0

chars = set([w_i for w in words for w_i in w])
n_chars = len(chars)
char2idx = {c: i + 2 for i, c in enumerate(chars)}
char2idx["UNK"] = 1
char2idx["PAD"] = 0

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

print("Number of sentences in train set (n_sent_train): " + str(n_sent_train))
print("Number of sentences in validation set (n_sent_val): " + str(n_sent_val))
print("Number of sentences in test set (n_sent_train): " + str(n_sent_test))



## Padding sequences
# sentence level
print("Padding Sequences ...")
# words
X_w_tr = [[words2idx[c[0]] for c in s] for s in train_sentences]
X_w_v = [[words2idx[c[0]] for c in s] for s in val_sentences]
X_w_te = [[words2idx[c[0]] for c in s] for s in test_sentences]

# tags
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
X_w_tr = pad_sequences(maxlen=max_len, sequences=X_w_tr, padding="post", truncating='post', value=words2idx["PAD"])
X_w_v = pad_sequences(maxlen=max_len, sequences=X_w_v, padding="post", truncating='post', value=words2idx["PAD"])
X_w_te = pad_sequences(maxlen=max_len, sequences=X_w_te, padding="post", truncating='post', value=words2idx["PAD"])

y_tr = pad_sequences(maxlen=max_len, sequences=y_tr, padding="post", truncating='post',value=tags2idx["PAD"])
y_v = pad_sequences(maxlen=max_len, sequences=y_v, padding="post", truncating='post',value=tags2idx["PAD"])
y_te = pad_sequences(maxlen=max_len, sequences=y_te, padding="post", truncating='post',value=tags2idx["PAD"])

# characters
max_len_char = 5
X_c_tr = get_chars_of_words_in_sentences(train_sentences,char2idx,max_len,max_len_char)
X_c_v = get_chars_of_words_in_sentences(val_sentences,char2idx,max_len,max_len_char)
X_c_te = get_chars_of_words_in_sentences(test_sentences,char2idx,max_len,max_len_char)


# wv_matrix = (np.random.rand(n_characters, WV_DIMENSION) - 0.5) / 5.0 # initialize word vector matrix
from keras.utils import to_categorical
y_tr= [to_categorical(i, num_classes=n_tags) for i in y_tr]
y_v = [to_categorical(i, num_classes=n_tags) for i in y_v]
y_te = [to_categorical(i, num_classes=n_tags) for i in y_te]

## Load word vectors
WV_DIMENSION = 100 # As defined and trained in xx_data_process.py
CV_DIMENSION = 100
# initialize the matrix with random numbers
wv_matrix = (np.random.rand(n_words, WV_DIMENSION) - 0.5) / 5.0
for word, i in words2idx.items():
    if i >= n_words:
        continue
    try:
        embedding_vector = word_vectors[word]
        # words not found in embedding index will be all-zeros.
        wv_matrix[i] = embedding_vector
    except:
        pass

cv_matrix = (np.random.rand(n_chars, CV_DIMENSION) - 0.5) / 5.0
for char, i in words2idx.items():
    if i >= n_chars:
        continue
    try:
        embedding_vector = char_vectors[char]
        # words not found in embedding index will be all-zeros.
        cv_matrix[i] = embedding_vector
    except:
        pass


### 2. Setup Models
print("Setup Models ...")

from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout,\
    Bidirectional,SpatialDropout1D, Conv1D, Flatten, MaxPooling1D,GlobalMaxPooling1D,concatenate

from keras_contrib.layers import CRF
from keras.optimizers import Adam


# Word Embeddings
# word_input = Input(shape=(max_len,),dtype='int32')
# wv_layer = Embedding(n_words,
#                      WV_DIMENSION,
#                      mask_zero=False, weights=[wv_matrix],
#                      input_length=max_len,
#                      trainable=False)
# word_embedding_layer = wv_layer(word_input)

word_input = Input(shape=(max_len,),dtype='int32')

wv_layer = Embedding(n_words,
                     WV_DIMENSION,
                     mask_zero=False, weights=[wv_matrix],
                     # input_length=max_len,
                     trainable=False)
word_embedding_layer = wv_layer(word_input)

# Char Embeddings
# char_input = Input(shape=(max_len,max_len_char,))
# char_embedding = TimeDistributed(Embedding(n_chars,
#                                            CV_DIMENSION,
#                                            mask_zero=False,
#                                            weights=[cv_matrix],
#                                            input_length=max_len,
#                                            trainable=True))(char_input)
char_embedding =  np.identity(len(char2idx))
char_input = Input(shape=(max_len,max_len_char))

char_embedding_layer = TimeDistributed(Embedding(input_dim=char_embedding.shape[0],
                                            output_dim=char_embedding.shape[1],
                                           mask_zero=False,
                                           weights=[char_embedding],
                                           trainable=True))(char_input)

# char CNN
# same paramter in emnlp2017-bilstm-cnn-crf
filterSize = 30
filterLength = 3
poolSize = 3
cnn = TimeDistributed(Conv1D(filters=filterSize, kernel_size=filterLength,padding='same'))(char_embedding_layer)
# cnn = TimeDistributed(LeakyReLU(alpha=self.leaky_alpha))(cnn)
# cnn = TimeDistributed(Dropout(rate=))(cnn)
# cnn = TimeDistributed(MaxPooling1D(pool_size=poolSize))(cnn)
# char_cnn = TimeDistributed(Flatten())(char_cnn)
cnn = TimeDistributed(GlobalMaxPooling1D())(cnn)

# Concatenate
rnn_input = concatenate([word_embedding_layer, cnn])
print("RNN Input shape:" + str(rnn_input))



# Input Dropout
# model = SpatialDropout1D(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True))(rnn_input)

# Output
# output_drop = Dropout(0.2)(model)
dense = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer
# dense = TimeDistributed(Dense(100, activation="softmax"))(output_drop)
crf = CRF(n_tags)  # CRF layer
# crf = CRF(n_tags,sparse_target=True)  # CRF layer
out = crf(dense)  # output
# out = CRF(n_tags)(dense)

### 3. Build Model
model = Model(inputs=[word_input,char_input ],outputs=out)
batch_size = 32
ephochs = 15
# learning rate decay
dataset_size = train_data.shape[0]
batches_per_epoch = dataset_size/batch_size
lr_decay = (1./(1/32) -1)/batches_per_epoch
model.compile(
    optimizer=Adam(lr=0.012, decay=lr_decay),
    loss=crf.loss_function,
    metrics=[crf.accuracy]
)
model.summary()
from keras.utils.vis_utils import plot_model


history = model.fit([X_w_tr,np.array(X_c_tr).reshape((len(X_c_tr), max_len, max_len_char))], np.array(y_tr),
                    batch_size=batch_size,
                    epochs=ephochs,
                    validation_data = ([X_w_v,np.array(X_c_v).reshape((len(X_c_v), max_len, max_len_char))], np.array(y_v)),
                    verbose=1,
                    )


# # history is a dictionary，keys are val_loss,val_acc,loss,acc
hist = pd.DataFrame(history.history)
fig = plt.figure(figsize=(12,12))
# add subplots
sub_fig1 = fig.add_subplot(1,2,1) # 1 row 2 cols 1st figure
sub_fig2 = fig.add_subplot(1,2,2)
# set titles
sub_fig1.set_title('Accuracy')
sub_fig2.set_title('Loss')
print(hist)
# set values and labels
sub_fig1.plot(hist["crf_viterbi_accuracy"],label='acc')
sub_fig1.plot(hist["val_crf_viterbi_accuracy"], label='val_acc')
sub_fig1.legend(loc="lower right")
sub_fig2.plot(hist["loss"],label='loss')
sub_fig2.plot(hist["val_loss"],label='val_loss')
sub_fig2.legend(loc="upper right")
plt.xlabel('epoch')
# show figure
plt.show()

score = model.evaluate([X_w_te,np.array(X_c_te).reshape((len(X_c_te), max_len, max_len_char))], np.array(y_te), batch_size=batch_size,verbose=1)
print(model.metrics_names)
print("Score:")
print(score)

# ## Prediction on test set
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
# print("Input:")
# print(X_te[0])
# print("Supposed output:")
# print(y_te)
# print(np.array(y_te))
test_pred = model.predict([X_w_te,np.array(X_c_te).reshape((len(X_c_te), max_len, max_len_char))], verbose=1)
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
print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))
print(classification_report(test_labels, pred_labels))

