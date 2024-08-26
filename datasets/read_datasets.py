# Script created by Gurol Saglam (or Guerol Saglam, guerol.saglam@kit.edu, gurol.saglam@ozu.edu.tr)
# Python native libraries
import os

# Python public packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import pandas as pd
import random

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
                # splitting the dataset without caring about the labels sometimes put all samples of a certain label into the testing dataset, thus the network not learning anything for that label.
                # so we split the dataset label by label.
                # data_train, data_test = train_test_split(data, test_size=(1 - DIVISION_PERCENTAGE), random_state=42) #OLD LINE
                labels = data["Y"].unique()
                data_train = pd.DataFrame()
                data_test = pd.DataFrame()
                for label in labels:
                    tempdata = data[data["Y"] == label]
                    temp_train, temp_test = train_test_split(tempdata, test_size=(1 - DIVISION_PERCENTAGE), random_state=42)
                    data_train = pd.concat([data_train, temp_train])
                    data_test = pd.concat([data_test, temp_test])
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
        
        # if fname == "arrhythmia":
            # dataset = pd.concat([data_train, data_test])
            # selected_features = self.select_features(dataset)
            # self.Xtrain = data_train[selected_features]
            # self.Xtest = data_test[selected_features]

        # with LabelEncoder, relabel samples to categorical.
        dataY = pd.concat([self.Ytrain, self.Ytest])
        le = LabelEncoder()
        le.fit(dataY)
        self.Ytrain = le.transform(self.Ytrain)
        self.Ytest = le.transform(self.Ytest)
        # self.Ytrain = to_categorical(self.Ytrain, num_classes=None)
        # self.Ytest = to_categorical(self.Ytest, num_classes=None)
        self.Ytrain = pd.DataFrame(self.Ytrain, columns = ["Y"])
        self.Ytest = pd.DataFrame(self.Ytest, columns = ["Y"])
    
    def getXtrain(self):
        return self.Xtrain
    
    def getYtrain(self):
        return self.Ytrain
    
    def getXtest(self):
        return self.Xtest
    
    def getYtest(self):
        return self.Ytest
    
    def plot_score(self, score):
        plt.figure(figsize=(16,8))
        plt.plot(score)
        plt.xlabel('# of feature')
        plt.ylabel('score')
        plt.show()
        
    def add_max_score_to_list(self, temp_scores, current_score, selected_indices, selected_indices_list):
        max_score_index = np.argmax(np.array(temp_scores))
        current_score.append(temp_scores[max_score_index])
        selected_indices.add(max_score_index)
        selected_indices_list.append(max_score_index)
    
    def select_features(self, data):
        Ytrain = data["Y"]
        Xtrain = data.drop("Y", axis=1)
        
        X = Xtrain
        y = Ytrain
        y = pd.Series(y, name='y')
        data = pd.concat([X, y], axis=1)
        # Correlation Matrix
        corr = data.corr()
        corr_ = corr.iloc[:,-1]

        boolean_array = (corr_ > 0.05).to_numpy()
        selected_indices = np.argwhere(corr_ > 0.05)[:-1, -1]
        features = X.columns
        # selected_features = features[selected_indices]
        selected_features = []

        for i in range(len(boolean_array[:-1])):
            if boolean_array[i] == False:
                X = X.drop(features[i], axis=1)
                
        num_features = len(X.columns)
        features = X.columns
        
        start_feature_index = random.randint(0, num_features-1)
        selected_indices = set()
        selected_indices_list = []

        selected_indices.add(start_feature_index)
        selected_indices_list.append(start_feature_index)

        mi_scores = [mutual_info_classif(X.to_numpy()[:,i].reshape(-1,1), y) for i in range(num_features)]
        mi_score_matrix = np.zeros((num_features, num_features))
        current_score = []
        
        k = 40
        for _ in range(k-1):
            temp_scores = []
            for i in range(num_features):
                if i in selected_indices:
                    temp_scores.append(-float('inf'))
                else:
                    score = mi_scores[i][0]
                    diff = 0
                    for j in selected_indices:
                        if j > i:
                            if mi_score_matrix[i][j] == 0:
                                mi_score_matrix[i][j] = np.corrcoef(X.iloc[:,i], X.iloc[:,j])[0, 1]
                            diff += mi_score_matrix[i][j]
                        else:
                            if mi_score_matrix[j][i] == 0:
                                mi_score_matrix[j][i] = np.corrcoef(X.iloc[:,i], X.iloc[:,j])[0, 1]
                            diff += mi_score_matrix[j][i]
                    temp_scores.append(score - diff/len(selected_indices))
            self.add_max_score_to_list(temp_scores, current_score, selected_indices, selected_indices_list)
        # self.plot_score(current_score) 
        return features[selected_indices_list]
    
    
    def rescale_features(self, feature_range=(0,1)):
        Xall = pd.concat([self.Xtrain, self.Xtest])
        scaler = MinMaxScaler(feature_range = feature_range)
        scaler.fit(Xall.values)
        self.Xtrain = pd.DataFrame(scaler.transform(self.Xtrain.values), columns = self.Xtrain.columns)
        self.Xtest = pd.DataFrame(scaler.transform(self.Xtest.values), columns = self.Xtest.columns)
        return self.Xtrain, self.Xtest
    
    def quantize_features(self, input_bitwidth=4):
        Xall = np.concatenate((self.Xtrain, self.Xtest))
        # get the new range, e.g. if input_bitwidth is 4, then new range will be (0,15), 15 included.
        max_range = (2**input_bitwidth) - 1
        new_feature_range = (0, np.max(Xall)*max_range)
        scaler = MinMaxScaler(feature_range = new_feature_range)
        scaler.fit(Xall)
        self.Xtrain = pd.DataFrame(scaler.transform(self.Xtrain.values), columns = self.Xtrain.columns)
        self.Xtest = pd.DataFrame(scaler.transform(self.Xtest.values), columns = self.Xtest.columns)
        # convert to integer to quantize, then divide by max resolution to get the values between 0 and 1, all quantized to number of bits.
        self.Xtrain = np.rint(self.Xtrain)
        self.Xtest = np.rint(self.Xtest)
        self.Xtrain = np.divide(self.Xtrain, max_range+1)
        self.Xtest = np.divide(self.Xtest, max_range+1)
        return self.Xtrain, self.Xtest
    
    def bitseparate_features(self, input_bitwidth=4):
        # get the max range from input_bitwidth which by default is 4 bits, multiply the quantized features (they have to be quantized using quantize_features method.
        max_range = (2**input_bitwidth)
        self.Xtrain = np.multiply(self.Xtrain, max_range)
        self.Xtest = np.multiply(self.Xtest, max_range)
        # now separate each feature into their bits
        dfTrainTemp = pd.DataFrame()
        dfTestTemp = pd.DataFrame()
        for column in self.Xtrain.columns.values:
            cols = []
            for i in range(input_bitwidth-1,-1,-1):
                cols.append(column + str(i))
            column_values = self.Xtrain[column].astype(int).values
            dfTrainTemp[cols] = pd.DataFrame((((column_values[:,None] & ((2**(input_bitwidth-1)) >> np.arange(input_bitwidth)))) > 0).astype(int))
            column_values = self.Xtest[column].astype(int).values
            dfTestTemp[cols] = pd.DataFrame((((column_values[:,None] & ((2**(input_bitwidth-1)) >> np.arange(input_bitwidth)))) > 0).astype(int))
        self.Xtrain = dfTrainTemp
        self.Xtest = dfTestTemp
        return self.Xtrain, self.Xtest
    
    def binarize_labels(self):
        # This function is more useful for QAT.
        dataY = pd.concat([self.Ytrain, self.Ytest])
        dataY = to_categorical(dataY, num_classes=None)
        self.Ytrain = dataY[0:self.Ytrain.shape[0],:]
        self.Ytest = dataY[self.Ytrain.shape[0]:,:]
