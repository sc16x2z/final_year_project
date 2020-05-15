# -*- coding=utf-8 -*-

'''
Data preparation process for ../data/CCKS2018

Input: corresponding .ann file and .txtoriginal.txt pairs
.txt files -- entity  start position of this entity	end position of this entity	entity label
example:"直肠	8	10	解剖部位"
.txtoriginal.txt files  -- EMR in Chinese for one patient
example (one of the sentence in this EMR file):
"，患者3月前因“直肠癌”于在我院于全麻下行直肠癌根治术（DIXON术）"


Output: words with corresponding labels
example:
Sentence:5	无	O
Sentence:5	头	B-SIGNS
Sentence:5	晕	I-SIGNS
Sentence:5	头	B-SIGNS
Sentence:5	痛	I-SIGNS
Sentence:5	，	O

'''

### Set project source folder
### Load necessary packages
import os
import sys

current_path = os.path.abspath(os.path.dirname(__file__))
print("Current Path: "+current_path)
# Remove the file name
root_path = os.path.split(current_path)[0]
root_path = os.path.split(root_path)[0]
print("Project root path: " + root_path)
sys.path.append(root_path)

from EMR_MedicalNER.preprocess.data_load import *

import pandas as pd
import numpy as np
import math
import csv

import re
import jieba

from gensim.models import Word2Vec,KeyedVectors


#########################################################################


### Check all the labels
class LabelChecker():
    '''
    Check all the labels
    Example:
    label_checker = LabelChecker()
    labels=label_checker.check_label()
    print(labels)
    Results:
    ['解剖部位', '症状描述', '手术', '独立症状', '药物']
    '''

    def __init__(self):
        self.origin_path = root_path + "/data/CCKS2018/"
        self.labels = []

    def check_label(self):
        for root, dirs, files in os.walk(self.origin_path):
            for file in files:
                filepath = os.path.join(root, file)

                if '.txtoriginal' in filepath:
                    continue

                elif '.txt' in filepath:
                    print(filepath)
                    label_data = pd.read_csv(filepath,sep='	',header=None)
                    # add unduplicated labels
                    self.labels=list(set(self.labels+list(set(label_data[3].values))))

        return self.labels


### Get and Tag All Words in Electronic Medical Record in BIO

class WordGetter(object):
    def __init__(self):
        self.origin_path = root_path + "/data/CCKS2018/"
        self.train_set_path = root_path+"/data/CCKS2018/preProcess/"
        self.word_seg_path = root_path + "/data/CCKS2018/preProcess/wordSeg/"
        self.category_dic = {
            'O': 0,
            'I-SITE': 1,
            'B-SITE': 2,
            'I-SYMDESCRIPTION': 3,
            'B-SYMDESCRIPTION': 4,
            'I-OPERATION': 5,
            'B-OPERATION': 6,
            'I-INDEPENDENTSYM': 7,
            'B-INDEPENDENTSYM': 8,
            'I-MEDICINE': 9,
            'B-MEDICINE': 10
        }
        self.label_dic = {
            '解剖部位': 'SITE',
            '症状描述': 'SYMDESCRIPTION',
            '手术': 'OPERATION',
            '独立症状': 'INDEPENDENTSYM',
            '药物': 'MEDICINE'
        }

        self.empty_charas = "\s+"
        self.stop_chars = "[\s\+\!\/_,.\$\%\^\*(+\"\'):<>-]+|[：+——()?【】“”‘’《》！，。；=？、~@#￥%……&*（）]+|[0-9]+"

    # Segmentation of single Chinese character
    def char_seg(self):

        f = open(self.word_seg_path + "seg_c.txt", 'w+', encoding='utf-8')
        for root, dirs, files in os.walk(self.origin_path):
            for file in files:
                filepath = os.path.join(root, file)
                # Skip files that are not original EMR
                if 'original' not in filepath:
                    continue
                # Read original EMR files ('.txtoriginal.txt')
                content = open(filepath).read().strip()

                for indx, char in enumerate(content):
                    # if char not in stop word
                    if re.match(self.stop_chars, char) is None:
                        f.write(char + '\n')

        f.close()
        return

    # Segmentation of Chinese word
    def word_seg(self):

        f = open(self.word_seg_path + "seg_w.txt", 'w+', encoding='utf-8')
        for root, dirs, files in os.walk(self.origin_path):
            for file in files:
                filepath = os.path.join(root, file)
                # Skip files that are not original EMR
                if 'original' not in filepath:
                    continue
                # Read original EMR files ('.txtoriginal.txt')
                content = open(filepath).read().strip()

                # Seperate Chinese words by jieba
                seg_content = jieba.cut(content, cut_all=False)
                for word in seg_content:
                    if re.match(self.stop_chars, word) is None:
                        f.write(word + '\n')
            f.close()
            return

    # Tranfer all infomation from dataset into one file
    # in single Chinese character format
    def transfer_file_c(self):
        f = open(self.train_set_path+"sent_train_BIO_c.txt", 'w+', encoding='utf-8')
        sentence_num = 0 # sentence#, sentences are seperated by "。"

        # Relate and combin information in .ann
        # and .txtoriginal.txt files  with the same name
        for root, dirs, files in os.walk(self.origin_path):
            for file in files:
                filepath = os.path.join(root, file)
                print(filepath)
                # Skip files that are not original EMR
                if 'original' not in filepath:
                    continue
                # Find corresponding files of original EMR with labeled words
                label_filepath = filepath.replace('.txtoriginal.txt', '.txt')
                print("Check "+filepath, '\t\t', label_filepath)

                # Read original EMR files ('.txtoriginal.txt')
                content = open(filepath).read().strip()
                res_dict = {} # {postion: BIO tag}
                # Get labels from .ann file for words in EMR
                for line in open(label_filepath):
                    res = line.strip().split('\t')
                    print("RES:")
                    print(res)
                    start = int(res[1])  # start position of this Chinese word
                    end = int(res[2]) # end position of this Chinese word
                    label = res[3] # entity label
                    label_id = self.label_dic.get(label)
                    # Write BIO tags
                    for i in range(start, end): # different from CCKS2017
                        if i == start:
                            label_cate = 'B-' + label_id
                        else:
                            label_cate = 'I-' + label_id
                        res_dict[i] = label_cate

                single_file_data = ""
                # Write needed information into file
                last_char = None # a flag to avoid two consecutive '。'
                for indx, char in enumerate(content):

                    # if char not in drops
                    # if re.match(self.del_charas,char) is None:
                    if re.match(self.empty_charas, char) is None:
                        char_label = res_dict.get(indx, 'O') # 'O'for characters not stored in res_dict
                        print(char, char_label)

                        if re.match(r'。',char) is not None:
                            if re.match(r'。',last_char) is None:
                                sentence_num+=1
                        # elif (re.match(self.del_charas,char))is not None:
                        #     print("Delete character: " + str(char))
                        else:
                            # Use tab to split train set data
                            single_file_data += "Sentence:" + str(sentence_num) + '\t' + char + '\t' + char_label + '\n'
                            f.write("Sentence:" + str(sentence_num) + '\t' + char + '\t' + char_label + '\n')
                        # updata last char
                        last_char = char


                with open( self.train_set_path + "sentCharaOfEach/"+file.split('.')[0]+'.txt',
                          "w",encoding='utf-8') as wf:
                    wf.write(single_file_data)

        f.close()
        return

    # Tranfer all infomation from dataset into one file
    # in Chinese word(one word can be one character or an aggregation of two more characters) format
    def transfer_file_w(self,create_word_dic=False):
        f = open(self.train_set_path+"sent_train_BIO_w.txt", 'w+', encoding='utf-8')
        sentence_num = 0 # sentence#, sentences are seperated by "。"

        # Relate and combin information in .ann
        # and .txtoriginal.txt files  with the same name
        for root, dirs, files in os.walk(self.origin_path):
            for file in files:
                filepath = os.path.join(root, file)

                # Skip files that are not original EMR
                if 'txtoriginal' not in filepath:
                    continue
                # Find corresponding files of original EMR with labeled words
                label_filepath = filepath.replace('.txtoriginal.txt', '.txt')
                print("Check "+filepath, '\t\t', label_filepath)


                # Read original EMR files ('.txtoriginal.txt')
                content = open(filepath).read().strip()
                # Seperate Chinese words by jieba
                seg_content = jieba.cut(content,cut_all=False)
                seg_content = "#".join(seg_content)
                print(seg_content)

                res_dict = {} # {postion: BIO tag}
                # if create_word_dic is True:  # create a dictionary

                # Get labels from .ann file for words in EMR
                for line in open(label_filepath):
                    res = line.strip().split('	')
                    start = int(res[1])  # start position of this Chinese word
                    end = int(res[2]) # end position of this Chinese word
                    label = res[3] # entity label
                    label_id = self.label_dic.get(label)
                    # Write BIO tags
                    for i in range(start, end):  # different from CCKS2017
                        if i == start:
                            label_cate = 'B-' + label_id
                        else:
                            label_cate = 'I-' + label_id
                        res_dict[i] = label_cate
                print(res_dict)
                # Write needed information into file
                word = ""
                next_word_idx = [] # the index of next word
                sep_count = 0 # counter of seperator '/'

                single_file_data = "" # string that stores all processed data for one file
                last_chars = []  # a flag to avoid two consecutive '。#。#'
                for indx, char in enumerate(seg_content):
                    print(indx,char)

                    if re.match(r'。', char) is not None:
                        # "#。#。#"
                        if (re.match(r'。', seg_content[indx-2]) is None):
                            sentence_num += 1


                    else:
                        # Get word index and label
                        # if re.match(self.del_charas, char) is None:
                        if (re.match(self.empty_charas, char) is None) and (re.match('#', char) is None) :
                            word = word + char
                            next_word_idx.append(indx)

                        if ((char is "#" )):
                            sep_count += 1
                            if (word is not ""):
                                print("word: " + word +" index: " + str(next_word_idx)+
                                      " sep_count: "+ str(sep_count)+
                                      " first index: "+ str(next_word_idx[0] - sep_count + 1))
                                # 'O'for characters not stored in res_dict
                                word_label = res_dict.get(next_word_idx[0] - sep_count + 1,'O')
                                # Use tab to split train set data
                                line = ("Sentence:" + str(sentence_num) + '\t' + word + '\t' + word_label + '\n')
                                f.write("Sentence:" + str(sentence_num) + '\t' + word + '\t' + word_label + '\n')
                                single_file_data += line # add line to
                            next_word_idx.clear()  # clean word index list
                            word = ""  # clean word container


                with open( self.train_set_path + "sentWordOfEach/"+file.split('.')[0]+'.txt',
                          "w",encoding='utf-8') as wf:
                    wf.write(single_file_data)

        f.close()
        return

## Get sentences
class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        # Chinese characters with BIO tag
        self.data = data
        self.empty = False
        # List of tuples for word and corresponding tag in one sentence
        agg_word_tag_func = lambda s: [(w, t) for w, t in zip(s[1].values.tolist(),
                                                           s[2].values.tolist())]
        # List of words in one sentence
        agg_word_func = lambda s: [w for w in s[1].values.tolist()]
        self.grouped = self.data.groupby(0).apply(agg_word_tag_func)
        self.groupedword = self.data.groupby(0).apply(agg_word_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence:{}".format(self.n_sent - 1)] # sentences start from  Sentence:0
            self.n_sent += 1
            return s
        except:
            return None

    # Get one sentence by sentence number (tuple of word and tag)
    def get_sentence_by_num(self,num):
        try:
            s = self.grouped["Sentence:{}".format(num)]
            return s
        except:
            return None

    # Get one sentence by sentence number (word only)
    def get_sentence_word_by_num(self,num):
        try:
            s = self.groupedword["Sentence:{}".format(num)]
            return s
        except:
            return None

    def show_sent_num_distribution(self):
        show_sents_len(self.sentences)




## Segment whole data into training set & validation set

all_labeled_data= [
    root_path+'/data/CCKS2018/preProcess/sent_train_BIO_c.txt',
    root_path+'/data/CCKS2018/preProcess/sent_train_BIO_w.txt'
]

kfold_data= [
    root_path+'/data/CCKS2018/preProcess/sent_train_BIO_c_train.txt',
    root_path+'/data/CCKS2018/preProcess/sent_train_BIO_w_train.txt'
]

def train_test_segment(src,target_path,ratio):
    '''
    Segmentation of data set into train set and test set.
    To maintain context between sentence, split in order rather in random.
    :param src:
    :param target_path: path of segmented data
    :param ratio: ratio of train test set split
    '''
    labeled_data = get_full(src)
    # print(labeled_data[0].values)
    # Due to my machine performance, SentenceGetter can only count 7801 sentences
    # Here I use the last element of DataFrame
    num_sentence = int(labeled_data[0].values[-1].split(':')[-1]) + 1  # sentence num start from 0
    print("num_sentence: {}".format(num_sentence))

    file_name = src.split('/')[-1].split('.')[0]
    n_sent_train = math.ceil(num_sentence * ratio) # number of sentence in target train set

    _train_set = []
    _test_set = []
    for sent in range(num_sentence):
        if sent < n_sent_train:
            _train_set.append(labeled_data.loc[labeled_data[0] == "Sentence:"+str(sent)])
        else:
            _test_set.append(labeled_data.loc[labeled_data[0] == "Sentence:" + str(sent)])
    train_data = pd.concat(_train_set, axis=0)
    test_data = pd.concat(_test_set, axis=0)
    train_data.to_csv(target_path+file_name+'_train.txt', sep='\t', header=False, index=False,quoting=csv.QUOTE_NONE)
    test_data.to_csv(target_path+file_name+'_test.txt', sep='\t', header=False, index=False,quoting=csv.QUOTE_NONE)




def train_test_kfold_segment(src,target_path,k=10):
    '''
    Segmentation of data set into train set and validation set in k fold.
    To maintain context between sentence, split in order rather in random.
    :param src: source file of data set
    :param target_path: path of segmented data
    :param k: k fold
    '''
    labeled_data = get_full(src)
    num_sentence = int(labeled_data[0].values[-1].split(':')[-1]) + 1 # sentence num start from 0
    print("num_sentence: {}".format(num_sentence))
    print("num_sentence%k: {}".format(num_sentence % k))

    file_name = src.split('/')[-1].split('.')[0]

    # sub set
    _kfold_data = [0 for i in range(k)]
    for i in range(k):
        if (i < (num_sentence%k)) :
            print("number of sentences in sample%d: %d" %(i,num_sentence // k+1))
            sent_flag_s = (num_sentence // k + 1) * (i)
            sent_flag_e = (num_sentence // k + 1) * (i + 1) - 1
            print("start: %d, end: %d" %(sent_flag_s,sent_flag_e))
            _temp_data = []
            for sent in range (sent_flag_s,sent_flag_e+1):
                _temp_data.append(labeled_data.loc[labeled_data[0] == "Sentence:"+str(sent)])
            _kfold_data[i] = pd.concat(_temp_data, axis= 0)
            del _temp_data
        else:
            print("number of sentences in sample%d: %d" % (i, num_sentence // k))
            sent_flag_s = (num_sentence // k + 1) * (num_sentence%k) + \
                          (num_sentence // k) * (i - num_sentence%k)
            sent_flag_e = sent_flag_s + (num_sentence // k) - 1
            print("start: %d, end: %d" % (sent_flag_s, sent_flag_e))
            _temp_data = []
            for sent in range(sent_flag_s, sent_flag_e + 1):
                _temp_data.append(labeled_data.loc[labeled_data[0] == "Sentence:" + str(sent)])
            _kfold_data[i] = pd.concat(_temp_data, axis=0)
            del _temp_data

    for i in range(k):
        # get segmentation file path
        train_path = target_path + file_name + '_t_' + str(i) + '.txt'
        test_path = target_path + file_name + '_v_' + str(i) + '.txt'
        print(train_path)
        print(test_path)
        _test_set = []
        _test_set.append(_kfold_data[i])
        test_data = pd.concat(_test_set, axis=0)
        test_data.to_csv(test_path, sep='\t', header=False, index=False,quoting=csv.QUOTE_NONE)
        _train_set = []
        for kd in _kfold_data[:i]+_kfold_data[i+1:] :
            _train_set.append( kd)
        train_data = pd.concat(_train_set, axis=0)
        train_data.to_csv(train_path, sep='\t', header=False, index=False,quoting=csv.QUOTE_NONE)



if __name__ == '__main__':

    ## Check labels
    # label_checker = LabelChecker()
    # labels = label_checker.check_label()
    # print(labels)

    ## Process orginal data
    word_getter = WordGetter()
    # word_getter.char_seg()
    # word_getter.word_seg()
    # word_getter.transfer_file_c()
    # word_getter.transfer_file_w()

    ## Plot word numbers
    # for labeled_data_path in all_labeled_data:
    #     data = get_full(labeled_data_path)
    #     word_count_dic = word_counts(words=data[1].values)
    #     show_word_counts(word_count_dic)

    ## Plot sentence length
    # data = get_full(root_path + '/data/CCKS2018/preProcess/sent_train_BIO_c.txt')
    # sentences = SentenceGetter(data).sentences
    # show_sents_len(sentences)


    ## Seperate dataset
    # # Get train & test set
    # for i in range(len(all_labeled_data)):
    #     train_test_segment(all_labeled_data[i],target_path=root_path+'/data/CCKS2018/preProcess/',ratio=0.8)
    #
    # # Get train & validation set (for crf++) from original train set
    # for i in range(len(kfold_data)):
    #     train_test_kfold_segment(kfold_data[i],target_path=root_path+'/data/CCKS2018/preProcess/kSeg/')

    ## Train word embeddings
    model_root_w2v = root_path + "/models/word2vec/"
    # Load sentences
    # data = get_full(root_path + '/data/CCKS2018/preProcess/sent_train_BIO_c.txt')
    data = get_full(root_path + '/data/CCKS2018/preProcess/sent_train_BIO_w.txt')
    # Use the last sentence# as total number due to machine limitation on SentenceGetter.sentences
    n_sent = int(data[0].values[-1].split(':')[-1]) + 1
    sentences = [] # sentences in loaded data
    died_sentences = [] # sentences NOT in loaded data
    sent_getter = SentenceGetter(data)
    for i in range(n_sent):
        if sent_getter.get_sentence_word_by_num(i) is None:
            died_sentences.append(i)
        else:
            sentences.append(sent_getter.get_sentence_word_by_num(i))
    print(died_sentences)

    # Train word2vec model
    w2vparas = {
        'sg':1,# skip-gram
        'hs':0,# negative sampling
        'negative':10, # negative sampling
        'min_count':1,
        'iter':5,
        'size':100 # default size
        # default window = 5
    }

    model_w2v = Word2Vec(sentences=sentences,
                         sg=w2vparas['sg'],
                         hs=w2vparas['hs'],
                         negative=w2vparas['negative'],
                         min_count=w2vparas['min_count'],
                         iter=w2vparas['iter'],
                         )

    # Save word2vec model
    modelname = "w2v"
    for key,value in w2vparas.items():
        modelname += "_"+str(key)+"_"+str(value)
    # model_w2v.wv.save_word2vec_format(model_root_w2v + modelname + '_c_ccks2018.model.bin', binary=True) # character w2v
    model_w2v.wv.save_word2vec_format(model_root_w2v+modelname+'_w_ccks2018.model.bin', binary=True) # word w2v


