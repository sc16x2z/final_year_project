# -*- coding=utf-8 -*-
'''
Load data for training and testing
'''
import os
import pandas as pd
import csv
from gensim.models import Word2Vec,KeyedVectors
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pylab import mpl

current_path = os.path.abspath(os.path.dirname(__file__))



def get_full(file):
    '''
    Separate whole data in to chunks and concatenate chunked data due to machine limitation.
    :param file: file elements are seperated by '\t'
    :return: full data in pandas DataFrame format
    '''
    _chunk_data = []
    for chunk in pd.read_csv(file, sep='\t',header=None, chunksize=1000,quoting=csv.QUOTE_NONE):
        _chunk_data.append(chunk)
    full_data = pd.concat(_chunk_data, axis= 0) # concat rows
    del _chunk_data
    return full_data


def get_val2idx(datalist):
    '''
    Values to index. Index all values and number of total values (non-duplicate)
    :param datalist: a column of data in list format. eg: DateFrame data['tags'].values
    :return: values (non-duplicate), dictionary {val:index}, number of values (non-duplicate)
    '''
    vals = list(set(datalist))
    n_vals = len(vals)
    vals2idx = {v: i for i, v in enumerate(vals)}
    # valsidx = [vi for vi in vals2idx.values()]
    return vals,vals2idx,n_vals




def get_word2vecEmbedding(model_path,words):
    '''
    Get wordembeddings by word2vec
    :param model_path:
    :return:
    '''

    w2vmodel = KeyedVectors.load_word2vec_format(model_path, binary=True)
    # vocab = []
    embedding = []
    for word in words:
        try:
            vector = w2vmodel.wv[word]
            # vocab.append(word)
            embedding.append(vector)
        except:
            print(word + "is not found in trained word vectors.")


    return np.array(embedding)


def word_counts(words):
    word_count_dic = Counter(words)
    return word_count_dic

def show_word_counts( word_count_dic):
    count_word_dic = sorted(word_count_dic.items(), key=lambda x: x[1], reverse=True)
    # label = list(map(lambda x: x[0], count_word_dic[:50]))
    value = list(map(lambda y: y[1], count_word_dic[:50]))
    print(value)
    plt.bar(x=range(len(value)), height=value, label="50 Most Common Token's Count")
    plt.show()
    return

def show_sents_len(sentences):
    plt.style.use("ggplot")
    plt.title("Length of each sentence")
    plt.hist([len(s) for s in sentences], bins=50)
    plt.show()
    print(max(len(s) for s in sentences))



def tokenize_and_preserve_labels(tokenizer,sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

def get_chars_of_words_in_sentences(sentences,char2idx,max_len,max_len_char):
    '''
    get character matrix from words in sentences as neural network input
    :param sentences: all sentences of text in 2-d arraay
    :param char2idx: dictionary of character and index
    :param max_len: maximum sentence length
    :param max_len_char: maximum character length
    :return:
    '''
    X_char = []
    for sentence in sentences:
        sent_seq = []
        for i in range(max_len):
            word_seq = []
            for j in range(max_len_char):
                try:
                    word_seq.append(char2idx.get(sentence[i][0][j]))
                except:
                    word_seq.append(char2idx.get("PAD"))
            sent_seq.append(word_seq)
        X_char.append(np.array(sent_seq))
    return X_char

# def get_chinese_embedding(embedding_model):
