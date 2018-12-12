from preprocess import Preprocess
import pickle
import numpy as np
import pandas as pd
import sys, io

sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

if __name__ == "__main__":
    with open("dataset/train.pkl", "wb") as f:
        train = pickle.load(f)
    with open("dataset/test.pkl", "wb") as f:
        test = pickle.load(f)
        
    print(train.train_A_D2V.shape, train.train_B_D2V.shape)
    print(train.train_A_Flag.shape, train.train_B_Flag.shape)
    print(train.train_B_Keyword.shape)
    print(train.train_A_D2V_Claim.shape, train.train_B_D2V_Claim.shape)
    print(train.train_A_Flag_Claim.shape, train.train_B_Flag_Claim.shape)
    print(train.labelOriginal.shape, train.labelRelation.shape, train.labelRelation.shape)

    print(test.test_A_D2V.shape, validate.test_B_D2V.shape)
    print(test.test_A_Flag.shape, validate.test_B_Flag.shape)
    print(test.test_B_Keyword.shape)
