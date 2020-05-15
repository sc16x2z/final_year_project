# -*- coding=utf-8 -*-

'''
Competition: https://tianchi.aliyun.com/competition/entrance/231687/introduction

Data preparation process for ../data/Diabetes

Input: corresponding .ann file and .txt pairs

.ann files -- entityid  entity label    start position of this entity	end position of this entity	entity
example:
T7	Drug 3348 3352	二甲双胍
T8	Drug 3447 3448;3449 3452	二 甲双胍

.txt files  -- text of authoritative diabetes related Chinese journals of 7 years
example (one of the sentence in this .txt file):
" Vinaixa 等[13] 结
合运用核磁共振、液相色谱-质谱及气相色谱-质谱技术证实了
氟他胺、二甲双胍、匹格列酮及雌孕激素多药联合治疗多囊卵
巢综合症的有效性。 因此代谢组学的出现为药物研究提供了
新的平台。"


'''
import os
import pandas as pd
import re
import random
import math
import matplotlib.pyplot as plt

current_path = os.path.abspath(os.path.dirname(__file__))
print("Current Path: "+current_path)
# Remove the file name
root_path = os.path.split(current_path)[0]
root_path = os.path.split(root_path)[0]
print("Project root path: " + root_path)

# whole trainning set path
train_set_path = root_path+"/data/Diabetes/preProcess/"

from EMR_MedicalNER.preprocess.data_load import *

## Check all the labels
class LabelChecker():
    '''
    Example:
    label_checker = LabelChecker()
    labels=label_checker.check_label()
    print(labels)
    Results:
    ['Amount_Drug', 'SideEff', 'Frequency', 'Reason', 'Test_Value', 'SideEff-Drug', 'Disease', 'Treatment_Disease', 'Method', 'Anatomy_Disease', 'Operation', 'Test_Disease', 'Anatomy', 'Method_Drug', 'Frequency_Drug', 'Symptom_Disease', 'Duration', 'Amount', 'Test', 'Symptom', 'Level', 'Drug_Disease', 'Treatment', 'Duration_Drug', 'Drug']

    '''

    def __init__(self):
        self.origin_path = root_path + "/data/Diabetes/"
        self.labels = []

    def check_label(self):
        for root, dirs, files in os.walk(self.origin_path):
            for file in files:
                filepath = os.path.join(root, file)
                # print(filepath)
                if '.ann' not in filepath:
                    continue
                else:
                    label_data = pd.read_csv(filepath,sep='\t',header=None)
                    temp_labels = []
                    for data in label_data[1]:
                        data = data.split(' ')
                        # print(data[0])
                        temp_labels.append(data[0])

                    self.labels = list(set(self.labels + list(set(temp_labels))))
                    # add unduplicated labels
                    # self.labels=list(set(self.labels+list(set(label_data[1].values))))
                    # print(self.labels)
        return self.labels


# following data preprocess is based on
# https://tianchi.aliyun.com/notebook-ai/detail?postId=34280

def preprocess():
    origin_data_dir = root_path+"/data/Diabetes/"
    proc_data_dir = train_set_path

    file_list = os.listdir(origin_data_dir)

    text_file_list = [x for x in file_list if x.endswith("txt")]

    total_str = ""

    sent_num = 0
    total_wf = open(proc_data_dir + "sent_train_BIO.txt", "w+", encoding='utf-8')

    # Read all ann and corresponding txt file
    for textfile in text_file_list:

        with open(origin_data_dir + textfile, 'r') as rftxt:
            txt_str = rftxt.read()

        with open(origin_data_dir + textfile.split("txt")[0] + "ann", "r") as rfann:
            ann_str = rfann.read()

        dict_ann = {}
        for line in ann_str.split("\n"):
            if len(line) == 0:
                continue
            try:
                entity_str = re.split(r"\t", line)[-1]
                attrib = re.split(r"\s", re.split(r"\t", line)[1])[0]
                indx_start = re.split(r"\s", re.split(r"\t", line)[1])[1]
                indx_end = re.split(r"\s", re.split(r"\t", line)[1])[-1]

                assert re.sub(r"\s|\n", "", txt_str[int(indx_start):int(indx_end)]) == re.sub(r"\n|\s", "", entity_str)
            except Exception as e:
                print("line goes hoo {}".format(line))
                continue
            for iinnd in range(int(indx_start), int(indx_end)):
                if iinnd == int(indx_start):
                    dict_ann[iinnd] = "B-" + attrib
                # elif iinnd == int(indx_end) - 1:
                #     dict_ann[iinnd] = "E-" + attrib
                else:
                    dict_ann[iinnd] = "I-" + attrib

            proc_str = ""

            list_ind_dict_keys = dict_ann.keys()
            for ind in range(len(txt_str)):
                # count sentence number
                if re.match(r"。", txt_str[ind]):
                    sent_num += 1

                if re.match(r"\s|\t|\n", txt_str[ind]):
                    continue
                elif ind not in list_ind_dict_keys:
                    line2add = "Sentence:"+str(sent_num)+"\t"+txt_str[ind] + "\t" + "O" + "\n"
                    # line2add = txt_str[ind] + "\t" + "O" + "\n"
                else:
                    line2add = "Sentence:" + str(sent_num) + "\t" + txt_str[ind] + "\t" + dict_ann[ind] + "\n"
                    # line2add = txt_str[ind] + "\t" + dict_ann[ind] + "\n"
                print(line2add)
                proc_str += line2add
                total_wf.write(line2add)

            # Change all continuous '\n' to only one '\n'
            proc_str_single = re.sub(r"\n{2,}", "\n", proc_str)
            proc_str_single = re.sub(r"(?<=。\tO|；\tO|;\tO)\n", "\n\n", proc_str_single)

            total_str += proc_str_single
            '''
            # write tag results into corresponding txt file
            with open(proc_data_dir + textfile,
                      "w") as wf:
                wf.write(proc_str_single)
            '''

        with open(proc_data_dir+"sent_train_BIO.txt",
                  "w") as twf:
            twf.write(total_str)

        print("Finished.")

if __name__ == '__main__':

    ## Check labels
    # label_checker = LabelChecker()
    # labels = label_checker.check_label()
    # print(labels)
    ## Write character and label
    # preprocess()



    ## Plot word numbers

    data = get_full(root_path + '/data/Diabetes/preProcess/sent_train_BIO.txt')
    word_count_dic = word_counts(words=data[1].values)
    show_word_counts(word_count_dic)

    ## Plot sentence length
    # data = get_full(root_path + '/data/Diabetes/preProcess/sent_train_BIO_c.txt')
    # sentences = SentenceGetter(data).sentences
    # show_sents_len(sentences)

    ## Process orginal data
    # word_getter = WordGetter()
    # word_getter.char_seg()
    # word_getter.word_seg()
    # word_getter.transfer_file_c()
    # word_getter.transfer_file_w()


    ## Seperate dataset
    # Get train & test set
    # for i in range(len(all_labeled_data)):
    #     train_test_segment(all_labeled_data[i],target_path=root_path+'/data/CCKS2017/preProcess/',ratio=0.8)
    #
    # # Get train & validation set (for crf++) from original train set
    # for i in range(len(kfold_data)):
    #     train_test_kfold_segment(kfold_data[i],target_path=root_path+'/data/CCKS2017/preProcess/kSeg/')

    ## Get word segmentation
    # word_seg_getter = WordSegmentationGetter()
    # word_seg_getter.single_word_seg()
    # word_seg_getter.chinese_words_seg()


    ## Train word embeddings
    model_root_w2v = root_path + "/models/word2vec/"
    # Load sentences
    data = get_full(root_path + '/data/CCKS2017/preProcess/sent_train_BIO_w.txt')
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

    ## Train word2vec model
    w2vparas = {
        'sg':1,# skip-gram
        'hs':0,# negative sampling
        'negative':10, # negative sampling
        'min_count':2,
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

    ## Save word2vec model
    modelname = "w2v"
    for key,value in w2vparas.items():
        modelname += "_"+str(key)+"_"+str(value)
    # model_w2v.wv.save_word2vec_format(model_root_w2v + modelname + '_c.model.bin', binary=True) # character w2v
    model_w2v.wv.save_word2vec_format(model_root_w2v+modelname+'_w.model.bin', binary=True) # word w2v


