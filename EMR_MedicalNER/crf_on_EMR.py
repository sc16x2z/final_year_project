# -*- coding=utf-8 -*-

'''
Perform CRF by crf++
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


from EMR_MedicalNER.preprocess.CCKS2017_data_preprocess import WordGetter
# from EMR_MedicalNER.preprocess.CCKS2018_data_preprocess import WordGetter
from EMR_MedicalNER.preprocess.data_load import get_full


### Create Model By crf++ template
# path of crf++ toolkit
crfpp_package_path = root_path+"/packages/crfpp/.libs/"
# tags
word_getter = WordGetter()
tags2idx= word_getter.category_dic
tags = [t for t in tags2idx.keys()]

labels = word_getter.label_dic.values()


def set_train_paras(f_start,f_end, c_start,c_end,c_num,l=1):
    '''
    set crf++'s parameters -f -c -a
    :param f_start: start value of -f
    :param f_end: end value of -f
    :param f_num: the number of values for parameter -f
    :param c_start:
    :param c_end:
    :param c_num:
    :param l: flag for paramter -a, 0 for CRF-L1, 1 for CRF-L2, default CRF-L2
    :return:
    '''
    paras= {
        '-f': [f for f in range(f_start,f_end+1)],
        '-c': np.linspace(start=c_start,stop=c_end,num=c_num),
        '-a': ['CRF-L1', 'CRF-L2', ][l],
        # only one cpu
    }
    return paras


# train crfpp on k fold train - validation sets
# crf_learn template_file train_file model_file
def train_crfpp(paras,template_path, training_set_path, model_root_path):
    models = []
    k = training_set_path.split('.')[0][-1]
    for f in paras['-f']:
        for c in paras['-c']:
            crf_train_cmd = "crf_learn" + " -f "+ str(f) + " -c "+ str(c) + " -a "+ paras['-a']
            # print("Running command: " + crf_train_cmd)
            cmd = crfpp_package_path + \
                  crf_train_cmd + " "+\
                  template_path + " "+\
                  training_set_path + " "+\
                  model_root_path + "model-f_" + str(f) + "-c_"+str(c) + "-a_"+paras['-a']+"_"+k
            models.append(model_root_path + "model-f_" + str(f) + "-c_"+str(c) + "-a_"+paras['-a']+"_"+k)
            print("Running command: " + cmd)
            # execution
            os.system(cmd)
    return models




###  Evaluate model performance

def get_prediction_data(prediction_result_path):
    data = get_full(prediction_result_path)
    return data[2]

# prediction result of model
def cal_result(test_set_path,result_root_path, model):
    serial_num = model.split('/')[-1]
    result = result_root_path+"crfpp_result_" + serial_num+".txt"
    # crf_test -m dg_model dg_test.txt -o dg_result.txt"
    # command for prediction
    # "-v -n -o" + \
    cmd = "crf_test -m " + \
          model + " " + \
          test_set_path + " "+ \
          "> "+result

    print(cmd)

    os.system(cmd)

    return result



# TP
def cal_tp(data, tag):
    TP = 0
    for i in range(len(data)):
        actual_value = data.iloc[i][2]
        predicted_value = data.iloc[i][3]
        if actual_value == tag and actual_value == predicted_value:
            TP += 1
    return TP


# FN
def cal_fn(data, tag):
    FN = 0
    for i in range(len(data)):
        actual_value = data.iloc[i][2]
        predicted_value = data.iloc[i][3]
        if actual_value == tag and actual_value != predicted_value:
            FN += 1
    return FN


# FP
def cal_fp(data, tag):
    FP = 0
    for i in range(len(data)):
        actual_value = data.iloc[i][2]
        predicted_value = data.iloc[i][3]
        if predicted_value == tag and actual_value != predicted_value:
            FP += 1
    return FP


# TN
def cal_tn(data, tag):
    TN = 0
    for i in range(len(data)):
        actual_value = data.iloc[i][2]
        predicted_value = data.iloc[i][3]
        if predicted_value != tag and actual_value != tag:
            TN += 1
    return TN





# TP
def cal_relaxed_tp(data, label):
    TP = 0
    for i in range(len(data)):
        actual_value = data.iloc[i][2]
        if actual_value != '0':
            actual_value = actual_value[2:] # delete 'B-''I-'
        predicted_value = data.iloc[i][3]
        if predicted_value != '0':
            predicted_value = predicted_value[2:]
        if actual_value == label and actual_value == predicted_value:
            TP += 1
    return TP


# FN
def cal_relaxed_fn(data, label):
    FN = 0
    for i in range(len(data)):
        actual_value = data.iloc[i][2]
        if actual_value != '0':
            actual_value = actual_value[2:]  # delete 'B-''I-'
        predicted_value = data.iloc[i][3]
        if predicted_value != '0':
            predicted_value = predicted_value[2:]
        if actual_value == label and actual_value != predicted_value:
            FN += 1
    return FN


# FP
def cal_relaxed_fp(data, label):
    FP = 0
    for i in range(len(data)):
        actual_value = data.iloc[i][2]
        if actual_value != '0':
            actual_value = actual_value[2:]  # delete 'B-''I-'
        predicted_value = data.iloc[i][3]
        if predicted_value != '0':
            predicted_value = predicted_value[2:]
        if predicted_value == label and actual_value != predicted_value:
            FP += 1
    return FP


# TN
def cal_relaxed_tn(data, label):
    TN = 0
    for i in range(len(data)):
        actual_value = data.iloc[i][2]
        if actual_value != '0':
            actual_value = actual_value[2:]  # delete 'B-''I-'
        predicted_value = data.iloc[i][3]
        if predicted_value != '0':
            predicted_value = predicted_value[2:]
        if predicted_value != label and actual_value != label:
            TN += 1
    return TN



def cal_tag(result_path):
    '''
    Calculate evaluation for each tag separately
    :param result_path: output file path
    :return: dictionary of evaluation results for each tag
    '''
    evaluation_result = {}
    data = get_full(result_path)

    for tag in tags:
        TP = cal_tp(data, tag)
        FN = cal_fn(data, tag)
        FP = cal_fp(data, tag)
        TN = cal_tn(data, tag)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        accuracy = (TP + TN) / (TP + TN + FN + FP)
        f1 = 2 * precision * recall / (precision + recall)
        print('tag: '+ tag)
        print('precision: '+str(precision) + '; recall: ' + str(recall) + '; accurary: '+ str(accuracy) + '; f1: '+ str(f1))
        evaluation_result[tag] = {'precision':precision, 'recall':recall, 'accuracy':accuracy, 'f1':f1}

    return evaluation_result

def cal_label(result_path):
    '''
    Calculate evaluation on whole label set. Results for each label.
    :param result_path: output file path
    :return:
    '''

    data = get_full(result_path)
    evaluation_result = {}
    for label in labels:
        tags = ['B-'+label, 'I-'+label]
        TP = cal_tp(data, tags[0]) + cal_tp(data,tags[1])
        FN = cal_fn(data, tags[0]) + cal_fn(data,tags[1])
        FP = cal_fp(data, tags[0]) + cal_fp(data, tags[1])
        TN = cal_tn(data, tags[0]) + cal_tn(data, tags[1])
        print(tags)
        print(TP,FN,FP,TN)
        if TP+FP != 0:
            precision = TP / (TP + FP)
        else:
            precision = 0
        if TP+FN != 0:
            recall = TP / (TP + FN)
        else:
            recall = 0
        if (TP + TN + FN + FP)!=0:
            accuracy = (TP + TN) / (TP + TN + FN + FP)
        else:
            accuracy = 0
        if (precision + recall) != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        print(label)
        print('precision: ' + str(precision) + '; recall: ' + str(recall) + '; accurary: ' + str(accuracy) + '; f1: ' + str(
        f1))
        evaluation_result[label] = {'precision': precision, 'recall': recall, 'accuracy': accuracy, 'f1': f1}
    return evaluation_result



def cal_total(result_path):
    '''
    Calculate evaluation on whole set
    :param result_path: output file path
    :return:
    '''

    data = get_full(result_path)
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    for tag in tags:
        _TP = cal_tp(data, tag)
        _FN = cal_fn(data, tag)
        _FP = cal_fp(data, tag)
        _TN = cal_tn(data, tag)
        TP += _TP
        FN += _FN
        FP += _FP
        TN += _TN

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FN + FP)
    f1 = 2 * precision * recall / (precision + recall)
    print('precision: ' + str(precision) + '; recall: ' + str(recall) + '; accurary: ' + str(accuracy) + '; f1: ' + str(
        f1))
    evaluation_result = {'precision': precision, 'recall': recall, 'accuracy': accuracy, 'f1': f1}
    return evaluation_result


def train_on_k_fold(paras,template_path,data_root,model_root,k=10, isword=False ):
    '''
    :param paras: parameters' dictionary. {'crfpp_para_name' : list of possible values}
    :param template_path: template file of crfpp
    :param data_root: path that stores train validation set
    :param model_root_path: path to store models that program will train
    :param k: k fold
    :param isword: whether use data in word format. True: word format False: character format
    :return:
    '''

    evaluation_results = {}
    # training on k folds data sets
    for k in range(0, k):

        if isword is False:
            training_set_path = data_root + "sent_train_BIO_c_train_t_"+str(k)+".txt"
            val_set_path = data_root + "sent_train_BIO_c_train_v_" +str(k)+".txt"
        else:
            training_set_path = data_root + "sent_train_BIO_w_train_t_" + str(k) + ".txt"
            val_set_path = data_root + "sent_train_BIO_w_train_v_" + str(k) + ".txt"
        try:
            # Train k fold models
            models = train_crfpp(paras=paras,
                                 template_path=template_path,
                                 training_set_path=training_set_path,
                                 model_root_path=model_root)
        except:
            print("Do not have all required file for validation. Please check the value k")

def result_on_k_fold(paras,model_root,result_root,val_set_path,k=10):
    results = []
    models=[]
    for i in range(k):
        model = model_root + "model-f_" + str(paras['-f'][0]) + "-c_" + str(paras['-c'][0]) + "-a_" + paras['-a'] + "_" + str(i)
        models.append(model)
    for model in models:
        result_path= cal_result(test_set_path=val_set_path,
                                result_root_path=result_root,
                                model=model
                               )
        results.append(result_path)
    return results


# Validation
def score_label_on_k_fold(paras, result_root_path, k=10):
    all_scores = {}
    for f in paras['-f']:
        for c in paras['-c']:
            scores = {}
            for i in range(0, k):
                result = result_root_path + "crfpp_result_model-f_" + str(f) + "-c_" + str(c) + "-a_" + paras[
                    '-a'] + "_" + str(i) + ".txt"
                scores[k] = cal_label(result)

            # score
            print("Scores for parameters " + "-f " + str(f) + "-c " + str(c) + "-a " + paras['-a'] + ":")
            score = {}
            for label in labels:
                _sum_precision = 0
                _sum_recall = 0
                _sum_acc = 0
                _sum_f1 = 0
                for i in range(0, k):
                    _sum_precision += float(scores[k][label]['precision'])
                    _sum_recall += float(scores[k][label]['recall'])
                    _sum_acc += float(scores[k][label]['accuracy'])
                    _sum_f1 += float(scores[k][label]['f1'])
                score[label] = {}
                score[label]['precision'] = _sum_precision / k
                score[label]['recall'] = _sum_recall / k
                score[label]['accuracy'] = _sum_acc / k
                score[label]['f1'] = _sum_f1 / k
            all_scores["f_" + str(f) + "c_" + str(c) + "a_" + paras['-a']] = score
    return all_scores


def score_total_on_k_fold(paras,result_root_path, k=10):
    all_scores = {}
    for f in paras['-f']:
        for c in paras['-c']:
            scores = {}
            for i in range(0,k):
                result = result_root_path +"crfpp_result_model-f_" + str(f) + "-c_" + str(c) + "-a_" + paras['-a'] + "_" + str(i)+".txt"
                scores[k] = cal_tag(result)

            # score
            print("Scores for parameters "+"-f " + str(f) + "-c " + str(c) + "-a " + paras['-a']+":")
            score = {}
            for tag in tags:
                _sum_precision=0
                _sum_recall=0
                _sum_acc=0
                _sum_f1=0
                for i in range(0,k):
                    _sum_precision += float(scores[k][tag]['precision'])
                    _sum_recall += float(scores[k][tag]['recall'])
                    _sum_acc += float(scores[k][tag]['accuracy'])
                    _sum_f1 += float(scores[k][tag]['f1'])
                score[tag] = {}
                score[tag]['precision'] = _sum_precision / k
                score [tag]['recall'] = _sum_recall / k
                score [tag]['accuracy'] = _sum_acc / k
                score [tag]['f1'] = _sum_f1 / k
            all_scores["f_" + str(f) + "c_" + str(c) + "a_" + paras['-a']] = score
    return all_scores

def cal_relaxed_label(result_path):
    '''
    Calculate evaluation on whole label set. Results for each label.
    :param result_path: output file path
    :return:
    '''

    data = get_full(result_path)
    evaluation_result = {}
    for label in labels:
        TP = cal_relaxed_tp(data, label)
        FN = cal_relaxed_fn(data, label)
        FP = cal_relaxed_fp(data, label)
        TN = cal_relaxed_tn(data, label)

        if TP+FP != 0:
            precision = TP / (TP + FP)
        else:
            precision = 0
        if TP+FN != 0:
            recall = TP / (TP + FN)
        else:
            recall = 0
        if (TP + TN + FN + FP)!=0:
            accuracy = (TP + TN) / (TP + TN + FN + FP)
        else:
            accuracy = 0
        if (precision + recall) != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        print(label)
        print('precision: ' + str(precision) + '; recall: ' + str(recall) + '; accurary: ' + str(accuracy) + '; f1: ' + str(
        f1))
        evaluation_result[label] = {'precision': precision, 'recall': recall, 'accuracy': accuracy, 'f1': f1}
    return evaluation_result

def relaxed_score_total_on_k_fold(paras,result_root_path, k=10):
    all_scores = {}
    for f in paras['-f']:
        for c in paras['-c']:
            scores = {}
            for i in range(0,k):
                result = result_root_path +"crfpp_result_model-f_" + str(f) + "-c_" + str(c) + "-a_" + paras['-a'] + "_" + str(i)+".txt"
                scores[k] = cal_relaxed_label(result)

            # score
            print("Scores for parameters "+"-f " + str(f) + "-c " + str(c) + "-a " + paras['-a']+":")
            score = {}
            for label in labels:
                _sum_precision=0
                _sum_recall=0
                _sum_acc=0
                _sum_f1=0
                for i in range(0,k):
                    _sum_precision += float(scores[k][label]['precision'])
                    _sum_recall += float(scores[k][label]['recall'])
                    _sum_acc += float(scores[k][label]['accuracy'])
                    _sum_f1 += float(scores[k][label]['f1'])
                score[label] = {}
                score[label]['precision'] = _sum_precision / k
                score [label]['recall'] = _sum_recall / k
                score [label]['accuracy'] = _sum_acc / k
                score [label]['f1'] = _sum_f1 / k
            all_scores["f_" + str(f) + "c_" + str(c) + "a_" + paras['-a']] = score
    return all_scores


if __name__ == "__main__":



    # Set of parameters
    paras = set_train_paras(1, 1, 1, 1, 1) # here is only one parater list -f 1 -c 1 l2
    print("All parameters you chose:")
    for key, values in paras.items():
        print(key, values)

    print("CRF++ package path: " + crfpp_package_path)
    k=10

    data_root = root_path + "/data/CCKS2017/preProcess/kSeg/"
    # data_root = root_path + "/data/CCKS2018/preProcess/kSeg/"

    model_root = root_path + "/models/crf++/kfold/2017-char/"
    # model_root = root_path + "/models/crf++/kfold/2018-char/"
    # model_root = root_path + "/models/crf++/kfold/2017-word/"
    # model_root = root_path + "/models/crf++/kfold/2018-word/"


    template_file = root_path + "/models/crf++/templates/template_word_windowsize_3"

    test_set_path = root_path + "/data/CCKS2017/preProcess/sent_train_BIO_c_test.txt"
    # test_set_path = root_path + "/data/CCKS2017/preProcess/sent_train_BIO_w_test.txt"
    # test_set_path = root_path + "/data/CCKS2018/preProcess/sent_train_BIO_c_test.txt"
    # test_set_path = root_path + "/data/CCKS2018/preProcess/sent_train_BIO_w_test.txt"

    test_result_root = root_path + "/results/crf++/2017-char/"
    # test_result_root = root_path + "/results/crf++/2018-char/"
    # test_result_root = root_path + "/results/crf++/2017-word/"
    # test_result_root = root_path + "/results/crf++/2017-word/"



    # train all models
    train_on_k_fold(paras=paras,
                    template_path=template_file,
                    data_root=data_root,
                    model_root=model_root,
                    isword=False) # input is character
                    # isword=True) # input is word
    
    # get results
    result_on_k_fold(paras=paras,
                     model_root=model_root,
                     result_root=test_result_root,
                     val_set_path=test_set_path,
                     )
    # report
    print("all label scores:")
    all_label_scores = score_label_on_k_fold(paras, test_result_root)
    print("====================")
    print(all_label_scores)
    print("====================")
    print("\nall relaxed label scores:")
    print("====================")
    print(relaxed_score_total_on_k_fold(paras, test_result_root))
    print("====================")
















