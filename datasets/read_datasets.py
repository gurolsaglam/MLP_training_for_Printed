# Script created by Gurol Saglam (or Guerol Saglam, guerol.saglam@kit.edu, gurol.saglam@ozu.edu.tr)
# Python native libraries
import os

# Python public packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# from tensorflow.keras.utils import to_categorical
import numpy as np

# Our packages
# None for this script

BASE_DIR = "./datasets/"
DIVISION_PERCENTAGE = 0.7  # 70 percent training, 30 percent testing subsets.


class Dataset(object):
    def __init__(self, fname, csv_separator=","):
        self.fname = fname
        data_train_file = BASE_DIR + "data/" + fname + "/" + fname + "_train.csv"
        data_test_file = BASE_DIR + "data/" + fname + "/" + fname + "_test.csv"

        # check if we already have the dataset divided into subsets.
        if not (os.path.exists(data_train_file) and os.path.exists(data_test_file)):
            # if we don't have the subsets, read from "data_raw" folder, divide into subsets, save to "data" folder.
            # Then parse to the fields. check if dataset is already separated into train and test. If not; read and
            # separate by DIVISION_PERCENTAGE, if so; read both.
            data_train = []
            data_test = []
            if len(os.listdir(BASE_DIR + "data_raw/" + fname + "/")) == 1:
                data_raw_file = BASE_DIR + "data_raw/" + fname + "/" + fname + ".csv"
                data = pd.read_csv(data_raw_file, sep=csv_separator)
                data_train, data_test = train_test_split(data, test_size=(1 - DIVISION_PERCENTAGE), random_state=42)
            else:
                data_raw_train_file = BASE_DIR + "data_raw/" + fname + "/" + fname + "_train.csv"
                data_raw_test_file = BASE_DIR + "data_raw/" + fname + "/" + fname + "_test.csv"
                data_train = pd.read_csv(data_raw_train_file, sep=csv_separator)
                data_test = pd.read_csv(data_raw_test_file, sep=csv_separator)

            # if the dataset folder in "data" is not created, create it.
            if not (os.path.exists(BASE_DIR + "data/" + fname + "/")):
                os.makedirs(BASE_DIR + "data/" + fname + "/")
            # create the csv files in "data"
            data_train.to_csv(data_train_file, sep=csv_separator, index=False)
            data_test.to_csv(data_test_file, sep=csv_separator, index=False)

        # read the subsets and parse to the fields.
        data_train = pd.read_csv(data_train_file, sep=csv_separator)
        data_test = pd.read_csv(data_test_file, sep=csv_separator)

        self.Ytrain = data_train["Y"]
        self.Xtrain = data_train.drop("Y", axis=1)
        self.Ytest = data_test["Y"]
        self.Xtest = data_test.drop("Y", axis=1)

        # with LabelEncoder, relabel samples to categorical.
        dataY = pd.concat([self.Ytrain, self.Ytest])
        le = LabelEncoder()
        le.fit(dataY)
<<<<<<< HEAD
        self.Ytrain = le.transform(self.Ytrain)
        self.Ytest = le.transform(self.Ytest)
=======
        self.Ytrain = pd.Series(le.transform(self.Ytrain))
        self.Ytest = pd.Series(le.transform(self.Ytest))
        #We had "to_categorical" in the old scripts, but removing this improves accuracy.
>>>>>>> 7d18a966b4058d8fbe182d3df66a2fc9057d6bb4
        # self.Ytrain = to_categorical(self.Ytrain, num_classes=None)
        # self.Ytest = to_categorical(self.Ytest, num_classes=None)

    def getXtrain(self):
        return self.Xtrain

    def getYtrain(self):
        return self.Ytrain

    def getXtest(self):
        return self.Xtest

    def getYtest(self):
        return self.Ytest

    def rescale_features(self, feature_range):
        Xall = np.concatenate((self.Xtrain, self.Xtest))
        scaler = MinMaxScaler(feature_range=feature_range)
        scaler.fit(Xall)
<<<<<<< HEAD
        self.Xtest = scaler.transform(self.Xtest)
        self.Xtrain = scaler.transform(self.Xtrain)
        return self.Xtrain, self.Xtest
=======
        self.Xtrain = pd.DataFrame(scaler.transform(self.Xtrain.values), columns = self.Xtrain.columns)
        self.Xtest = pd.DataFrame(scaler.transform(self.Xtest.values), columns = self.Xtest.columns)
        return self.Xtrain, self.Xtest
    
    def quantize_features(self, input_bitwidth):
        Xall = np.concatenate((self.Xtrain, self.Xtest))
        #get the new range, e.g. if input_bitwidth is 4, then new range will be (0,15), 15 included.
        max_range = (2**input_bitwidth) - 1
        new_feature_range = (0, np.max(Xall)*max_range)
        scaler = MinMaxScaler(feature_range = new_feature_range)
        scaler.fit(Xall)
        self.Xtrain = pd.DataFrame(scaler.transform(self.Xtrain.values), columns = self.Xtrain.columns)
        self.Xtest = pd.DataFrame(scaler.transform(self.Xtest.values), columns = self.Xtest.columns)
        #convert to integer to quantize, then divide by max resolution to get the values between 0 and 1, all quantized to number of bits.
        self.Xtrain = np.rint(self.Xtrain)
        self.Xtest = np.rint(self.Xtest)
        self.Xtrain = np.divide(self.Xtrain, max_range+1)
        self.Xtest = np.divide(self.Xtest, max_range+1)
        return self.Xtrain, self.Xtest
    
    def binarize_labels(self):
        #This function is more useful for QAT.
        dataY = pd.concat([self.Ytrain, self.Ytest])
        dataY = to_categorical(dataY, num_classes=None)
        self.Ytrain = dataY[0:self.Ytrain.shape[0],:]
        self.Ytest = dataY[self.Ytrain.shape[0]:,:]
>>>>>>> 7d18a966b4058d8fbe182d3df66a2fc9057d6bb4
