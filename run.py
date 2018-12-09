from preprocess import Preprocess
import pickle
from preprocess import Preprocess
import numpy as np
import pandas as pd

if __name__ == "__main__":
    train = Preprocess("dataset/train.csv")
    print(train.train_A_D2V.shape, train.train_B_D2V.shape)
    print(train.train_A_Flag.shape, train.train_B_Flag.shape)
    print(train.train_A_D2V_Claim.shape, train.train_B_D2V_Claim.shape)
    print(train.train_A_Flag_Claim.shape, train.train_B_Flag_Claim.shape)
    print(train.labelOriginal.shape, train.labelRelation.shape, train.labelRelation.shape)

    validate = Preprocess("dataset/validate.csv")
    print(validate.test_A_D2V.shape, validate.test_B_D2V.shape)
    print(validate.test_A_Flag.shape, validate.test_B_Flag.shape)
    print(validate.labelOriginal.shape)

    test = Preprocess("dataset/test.csv")
    print(validate.test_A_D2V.shape, validate.test_B_D2V.shape)
    print(validate.test_A_Flag.shape, validate.test_B_Flag.shape)
    
    # with open("dataset/train.pkl", "wb") as f:
    #     pickle.dump(train, f, pickle.HIGHEST_PROTOCOL)
    # with open("dataset/validate.pkl", "wb") as f:
    #     pickle.dump(validate, f, pickle.HIGHEST_PROTOCOL)
    # with open("dataset/test.pkl", "wb") as f:
    #     pickle.dump(test, f, pickle.HIGHEST_PROTOCOL)

