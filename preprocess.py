'''
title: Preprocess for fake news data
author: Yi-Zhan Xu
date: 2018/12/03
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import jieba.posseg as pseg
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from collections import Counter
import copy, os
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

class Preprocess(object):
    def __init__(self, csv, nrows=1688):
        self.csv = csv
        self.nrows = nrows
        self.df = pd.read_csv(csv, nrows=nrows, encoding="utf-8").dropna().reset_index(drop=True)
        if csv == "dataset/train.csv":
            try:
                os.remove("dataset/D2V.model")
            except:
                pass
            self.train_A_Dict, self.train_B_Dict, self.title_unique = self.title_Dict()
            self.train_A_Flag, self.train_B_Flag = self.title_Flag()
            self.train_A_D2V, self.train_B_D2V = self.title_D2V()
            self.train_B_Keyword = self.title_Keyword()
            self.labelOriginal = self.label_Original()
            self.labelRelation, self.related_idx = self.label_Relation()
            self.labelClaim, self.train_A_Flag_Claim, self.train_B_Flag_Claim, self.train_A_D2V_Claim, self.train_B_D2V_Claim = self.label_Claim()
        elif csv == "dataset/validate.csv":
            self.train_A_Dict, self.train_B_Dict, self.test_A_Dict, self.test_B_Dict, self.title_unique = self.title_Dict()
            self.test_A_Flag, self.test_B_Flag = self.title_Flag()
            self.test_A_D2V, self.test_B_D2V = self.title_D2V()
            self.test_B_Keyword = self.title_Keyword()
            self.labelOriginal = self.label_Original()
        else: # test.csv
            self.submit = pd.read_csv("dataset/sample_submission.csv")
            self.train_A_Dict, self.train_B_Dict, self.test_A_Dict, self.test_B_Dict, self.title_unique = self.title_Dict()
            self.test_A_Flag, self.test_B_Flag = self.title_Flag()
            self.test_A_D2V, self.test_B_D2V = self.title_D2V()
            self.test_B_Keyword = self.title_Keyword()


    def title_Dict(self):
        df = pd.read_csv("dataset/train.csv", nrows=self.nrows, encoding="utf-8").dropna().reset_index(drop=True)
        train_A_Dict, train_B_Dict = {}, {}
        A_words, A_flags, B_words, B_flags = [], [], [], []

        for i in range(len(df)):          
            jieba_res = pseg.cut(df["title1_zh"][i])
            words_tmp, flags_tmp = [], []
            for word, flag in jieba_res:
                words_tmp.append(word)
                flags_tmp.append(flag)
            A_words.append(words_tmp)
            A_flags.append(flags_tmp)

            jieba_res = pseg.cut(df["title2_zh"][i])
            words_tmp, flags_tmp = [], []
            for word, flag in jieba_res:
                words_tmp.append(word)
                flags_tmp.append(flag)
            B_words.append(words_tmp)
            B_flags.append(flags_tmp)

        train_A_Dict["title"], train_A_Dict["words"], train_A_Dict["flags"] = df["title1_zh"].values, A_words, A_flags
        train_B_Dict["title"], train_B_Dict["words"], train_B_Dict["flags"] = df["title2_zh"].values, B_words, B_flags
        title_unique = np.unique(np.append(np.concatenate(train_A_Dict["flags"]), np.concatenate(train_B_Dict["flags"])))

        if self.csv != "dataset/train.csv":
            test_A_Dict, test_B_Dict = {}, {}
            A_words, A_flags, B_words, B_flags = [], [], [], []
            for i in range(len(self.df)):          
                jieba_res = pseg.cut(self.df["title1_zh"][i])
                words_tmp, flags_tmp = [], []
                for word, flag in jieba_res:
                    if flag in title_unique:
                        words_tmp.append(word)
                        flags_tmp.append(flag)
                A_words.append(words_tmp)
                A_flags.append(flags_tmp)

                jieba_res = pseg.cut(self.df["title2_zh"][i])
                words_tmp, flags_tmp = [], []
                for word, flag in jieba_res:
                    if flag in title_unique:
                        words_tmp.append(word)
                        flags_tmp.append(flag)
                B_words.append(words_tmp)
                B_flags.append(flags_tmp)

            test_A_Dict["title"], test_A_Dict["words"], test_A_Dict["flags"] = self.df["title1_zh"].values, A_words, A_flags
            test_B_Dict["title"], test_B_Dict["words"], test_B_Dict["flags"] = self.df["title2_zh"].values, B_words, B_flags

            return train_A_Dict, train_B_Dict, test_A_Dict, test_B_Dict, title_unique

        return train_A_Dict, train_B_Dict, title_unique

    def title_Flag(self):
        if self.csv == "dataset/train.csv":
            A_distribution, B_distribution = [], []
            for i in range(self.df.shape[0]):
                A_flag, B_flag = {}, {}
                for j in self.title_unique:
                    A_flag[j], B_flag[j] = 0, 0
                A_freq = dict(Counter(self.train_A_Dict["flags"][i]).most_common())
                B_freq = dict(Counter(self.train_B_Dict["flags"][i]).most_common())

                for j in A_freq.items():
                    if j[0] in self.title_unique:
                        A_flag[j[0]] = j[1]
                for j in B_freq.items():
                    if j[0] in self.title_unique:
                        B_flag[j[0]] = j[1]
                A_distribution.append(list(A_flag.values()))
                B_distribution.append(list(B_flag.values()))

            train_A_Flag = np.asarray(A_distribution)
            train_B_Flag = np.asarray(B_distribution)
            return train_A_Flag, train_B_Flag

        else:
            A_distribution, B_distribution = [], []
            for i in range(self.df.shape[0]):
                A_flag, B_flag = {}, {}
                for j in self.title_unique:
                    A_flag[j], B_flag[j] = 0, 0
                A_freq = dict(Counter(self.test_A_Dict["flags"][i]).most_common())
                B_freq = dict(Counter(self.test_B_Dict["flags"][i]).most_common())

                for j in A_freq.items():
                    if j[0] in self.title_unique:
                        A_flag[j[0]] = j[1]
                for j in B_freq.items():
                    if j[0] in self.title_unique:
                        B_flag[j[0]] = j[1]
                A_distribution.append(list(A_flag.values()))
                B_distribution.append(list(B_flag.values()))

            test_A_Flag = np.asarray(A_distribution)
            test_B_Flag = np.asarray(B_distribution)
            return test_A_Flag, test_B_Flag

    def title_D2V(self):
        # Must delete D2V.model before execute
        if not os.path.exists("dataset/D2V.model"):
            train_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(np.append(self.train_A_Dict["title"], self.train_B_Dict["title"]))]
            model = Doc2Vec(train_documents, vector_size=100, window=2, min_count=1, workers=4)
            model.save("dataset/D2V.model")
            A_vec, B_vec = [], []
            for i, j in zip(range(len(self.train_A_Dict["title"])), range(len(self.train_A_Dict["title"]),len(self.train_A_Dict["title"])+len(self.train_B_Dict["title"]))):
                A_vec.append(model.docvecs[i])
                B_vec.append(model.docvecs[j])
            train_A_D2V = np.asarray(A_vec)
            train_B_D2V = np.asarray(B_vec)
            return train_A_D2V, train_B_D2V

        if self.csv != "dataset/train.csv": 
            mymodel = Doc2Vec.load("dataset/D2V.model")
            test_A_D2V = np.asarray([mymodel.infer_vector(i) for i in self.test_A_Dict["title"]])
            test_B_D2V = np.asarray([mymodel.infer_vector(i) for i in self.test_B_Dict["title"]])
            return test_A_D2V, test_B_D2V

    def title_Keyword(self):
        if self.csv == "dataset/train.csv":
            words_length = [len(i) for i in self.train_B_Dict["words"]]
            max_length = np.max(words_length)

            disagreed_idx = np.where(self.df["label"].values == "disagreed")[0]
            count = Counter(np.concatenate(np.asarray(self.train_B_Dict["words"])[disagreed_idx])).most_common()[0:max_length]
            global KEY_UNIQUE
            KEY_UNIQUE = [i[0] for i in count]
            train_B_distribution = []
            for i in range(self.df.shape[0]):
                train_B_key = {}
                for j in KEY_UNIQUE:
                    train_B_key[j] = 0
                train_B_freq = dict(Counter(self.train_B_Dict["words"][i]).most_common())

                for j in train_B_freq.items():
                    if j[0] in train_B_key:
                        train_B_key[j[0]] = j[1]

                train_B_distribution.append(list(train_B_key.values()))
            train_B_Keyword = np.asarray(train_B_distribution)

        else:
            test_B_distribution = []
            for i in range(self.df.shape[0]):
                test_B_key = {}
                for j in KEY_UNIQUE:
                    test_B_key[j] = 0
                test_B_freq = dict(Counter(self.test_B_Dict["words"][i]).most_common())

                for j in test_B_freq.items():
                    if j[0] in test_B_key:
                        test_B_key[j[0]] = j[1]

                test_B_distribution.append(list(test_B_key.values()))
            test_B_Keyword = np.asarray(test_B_distribution)
            return test_B_Keyword

        return train_B_Keyword
    '''
    def title_W2V(self):
        # if not os.path.exists("dataset/W2V.model"):
        model = Word2Vec(word_filter, min_count=1)
        model.save("dataset/W2V.model")
        train_B_outter_word = []
        for i in range(len(self.train_B_Dict["words"])):
            train_B_inner_word = []
            for j in self.train_B_Dict["words"][i]:
                try:
                    train_B_inner_word.append(model[j])
                except:
                    train_B_inner_word.append(np.zeros(100))
            if len(train_B_inner_word) != max_length:
                train_B_inner_word_zeros = np.vstack((train_B_inner_word, [np.zeros(100) for i in range(max_length - len(train_B_inner_word))]))
            train_B_outter_word.append(train_B_inner_word_zeros)
        train_B_W2V = np.asarray(train_B_outter_word)
        return train_B_W2V

        if self.csv != "dataset/train.csv": 
            mymodel = Word2Vec.load("dataset/W2V.model")
            test_B_outter_word = []
            for i in range(len(self.test_B_Dict["words"])):
                test_B_inner_word = []
                for j in self.test_B_Dict["words"][i]:
                    try:
                        test_B_inner_word.append(mymodel[j])
                    except:
                        test_B_inner_word.append(np.zeros(100))
                if len(test_B_inner_word) != max_length:
                    test_B_inner_word_zeros = np.vstack((test_B_inner_word, [np.zeros(100) for i in range(max_length - len(test_B_inner_word))]))
                test_B_outter_word.append(test_B_inner_word_zeros)
            test_B_W2V = np.asarray(test_B_outter_word)
            return test_B_W2V
    '''
    def label_Original(self): # merge two models
        labelOriginal = LabelEncoder().fit_transform(self.df["label"].values).reshape(-1,1)
        # onehot = OneHotEncoder().fit_transform(y_label).toarray()
        return labelOriginal

    def label_Relation(self): # model 2
        related_idx = np.where(self.df["label"].values != "unrelated")[0]
        df_copy = copy.deepcopy(self.df)
        df_copy.loc[related_idx, "label"] = "related"
        labelRelation = LabelEncoder().fit_transform(df_copy["label"].values).reshape(-1,1)
        return labelRelation, related_idx 

    def label_Claim(self): # model 1
        unrelated_idx = np.where(self.df["label"].values == "unrelated")[0]
        y = np.delete(self.df["label"].values, unrelated_idx)
        labelClaim = LabelEncoder().fit_transform(y).reshape(-1,1)
        if self.csv == "dataset/train.csv":
            train_A_Flag_Claim, train_B_Flag_Claim = np.delete(self.train_A_Flag, unrelated_idx, axis=0), np.delete(self.train_B_Flag, unrelated_idx, axis=0)
            train_A_D2V_Claim, train_B_D2V_Claim = np.delete(self.train_A_D2V, unrelated_idx, axis=0), np.delete(self.train_B_D2V, unrelated_idx, axis=0)
            return labelClaim, train_A_Flag_Claim, train_B_Flag_Claim, train_A_D2V_Claim, train_B_D2V_Claim
        else:
            return labelClaim
