# -*- coding=utf-8 -*-

import os
import sys

# Current file path
current_path = os.path.abspath(os.path.dirname(__file__))
# Remove the file name
root_path = os.path.split(current_path)[0]
root_path = os.path.split(root_path)[0]

sys.path.append(root_path)

class SCWELoader():
    def __init__(self):
        self.package_root = root_path + "/packages/SCWE-master/"
        self.scwe_char = {}
        self.scwe_word = {}
        self.N_char = 0 # number of characters
        self.M_char = 0 # character embedding dimension
        self.N_word = 0 # number of words
        self.M_word = 0 # word embedding dimension

    # ./scwe -train seg_w.txt -output-word chinese_scwe_word.txt -output-char chinese_scwe_char.txt"
    def load_scwe_word_from_file(self,word_file):
        with open(word_file, 'r') as f:
            first_line = next(f).split() # N M  information
            self.N_word = int(first_line[0])
            self.M_word = int(first_line[1])
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                word = line[0]
                word_embedding = line[1:]
                print(word)
                print(word_embedding)
                self.scwe_word[word] = word_embedding
        return self.N_word, self.M_word, self.scwe_word


    def load_scwe_char_from_file(self,char_file):
        with open(char_file, 'r') as f:
            first_line = next(f).split() # N M  information
            self.N_char = int(first_line[0])
            self.M_char = int(first_line[1])

            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                char = line[0]
                # numi denote the i-th meanings' embedding of character
                # b m e are position of words
                numi = line[1]
                char_embedding = line[2:]
                print(char)
                print(numi)
                print(char_embedding)
                self.scwe_char[char] = {numi:char_embedding}
        return self.N_char, self.M_char, self.scwe_char

# load pretrained word embeddings of https://github.com/Embedding/Chinese-Word-Vectors
class ChineseWordVectorsLoader():
    def __init__(self):
        self.vector_root = root_path + "/packages/Chinese-Word-Vectors/"
        self.evaluation_root = root_path + "/packages/Chinese-Word-Vectors-master/"

    def load_vec_from_file(self,word_filename):
        vec_dic = {}
        word_file = self.vector_root + word_filename
        print("Reading "+ word_file + "...")
        with open(word_file, 'r') as f:
            first_line = next(f).split() # N M  information
            count = int(first_line[0]) # count of embeddings / tokens
            print("Tokens Count: " + str(count))
            dimension = int(first_line[1]) # dimension of embedding
            print("Embedding Dimension: " + str(dimension))
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                word = line[0]
                word_embedding = line[1:]
                print(word)
                print(word_embedding)
                vec_dic[word] = word_embedding
        return count,dimension,vec_dic



if __name__ == "__main__":
    scwe_word = root_path + '/data/CCKS2017/preProcess/wordSeg/chinese_scwe_word.txt'
    scwe_char = root_path + '/data/CCKS2017/preProcess/wordSeg/chinese_scwe_char.txt'

    # scweloader = SCWELoader()
    # corpus = root_path + '/data/preProcess/wordSeg/seg_w.txt'
    #
    # scwe_word_vec = scweloader.load_scwe_word_from_file(scwe_word)
    # for word, wv in scwe_word_vec.items():
    #     print(word,wv)
    # scweloader.load_scwe_char_from_file(scwe_char)

    chinese_wv_loader = ChineseWordVectorsLoader()
    token_count, token_dimension,wv_dic = chinese_wv_loader.load_vec_from_file('sgns.wiki.word')



