#Script modified by Gurol Saglam (or Guerol Saglam, guerol.saglam@kit.edu, gurol.saglam@ozu.edu.tr)
#Python native libraries
import sys
import multiprocessing as mp
import time

#Python public packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from tensorflow.keras.utils import to_categorical
from collections import defaultdict
import csv
from itertools import zip_longest
from tensorflow.keras.models import save_model
from joblib import dump, load
from sklearn.metrics import accuracy_score
import numpy as np
import torch

#Our packages
from datasets.datasets import RedWine, WhiteWine, Cardio, HAC, Pendigits, Arrhythmia, GasID, Breast_Cancer, Vertebral_Column_2C, Vertebral_Column_3C, Mammographic, Balance_Scale, Seeds, Dermatology, Parkinsons, LungCancer, SPECTF, EpilepticSeizure, EpilepticSeizure5Class, HAR, SportsActivities, MexDataPM, RealDispIdeal, MuscleActivity, ActPhysiological
from algos import DecisionTree, SVM, MLP, LogReg, RandomForest

if __name__ == "__main__":
    data = sys.argv[1]
    algo = sys.argv[2]
    name = data+"."+algo

    if algo in ['MLP']:
      algo = MLP
    elif algo in ['DecisionTree']:
      algo = DecisionTree
    elif algo in ['RandomForest']:
      algo = RandomForest
    elif algo in ['SVM']:
      algo = SVM
    elif algo in ['LogReg']:
      algo = LogReg
    else: assert(False)


    if data in ['Acute']:
      data = torch.load("./datasets/data/Dataset_acuteinflammation.ds")
    elif data in ['Balance_Scale']:
      data = torch.load("./datasets/data/Dataset_balancescale.ds")
    elif data in ['Breast_Cancer']:
      data = torch.load("./datasets/data/Dataset_breastcancerwisc.ds")
    elif data in ['Cardio']:
      data = torch.load("./datasets/data/Dataset_cardiotocography3clases.ds")
    elif data in ['Energy1']:
      data = torch.load("./datasets/data/Dataset_energyy1.ds")
    elif data in ['Energy2']:
      data = torch.load("./datasets/data/Dataset_energyy2.ds")
    elif data in ['Iris']:
      data = torch.load("./datasets/data/Dataset_iris.ds")
    elif data in ['Mammographic']:
      data = torch.load("./datasets/data/Dataset_mammographic.ds")
    elif data in ['Pendigits']:
      data = torch.load("./datasets/data/Dataset_pendigits.ds")
    elif data in ['Seeds']:
      data = torch.load("./datasets/data/Dataset_seeds.ds")
    elif data in ['Tic_tac_toe']:
      data = torch.load("./datasets/data/Dataset_tictactoe.ds")
    elif data in ['Vertebral_Column_2C']:
      data = torch.load("./datasets/data/Dataset_vertebralcolumn2clases.ds")
    elif data in ['Vertebral_Column_3C']:
      data = torch.load("./datasets/data/Dataset_vertebralcolumn3clases.ds")


    X_train = np.array(data["X_train"])
    X_test = np.array(data["X_test"])
    y_train = np.array(data["y_train"])
    y_test = np.array(data["y_test"])

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    y_train = to_categorical(y_train, num_classes=None)
    y_test = to_categorical(y_test, num_classes=None)

    # print(X_train[0:5])
    # print(X_test[0:5])

    input_s = 4 #4
    norm = 2**input_s
    max_range = (norm-1)

    scaler = MinMaxScaler(feature_range=(0,max_range))
    scaler.fit(X_train)


    #print(scaler.transform(X_train)[0])
    X_test = scaler.transform(X_test)
    X_train = scaler.transform(X_train)


    X_train = np.round(X_train)
    X_train = np.divide(X_train, norm)
    X_test = np.round(X_test)
    X_test = np.divide(X_test, norm)

    # print(X_train[0:5])
    # print(X_train2[0:5])
    # print(X_test[0:5])
    # sys.exit(0)

    hidden_layer_sizes = 3

    for i in range(0, 20):
        ml = algo(X_train, X_test, y_train, y_test, name+'.'+str(i), hidden_layer_sizes=hidden_layer_sizes)
        #svm = algo(X_train, X_test, y_train, y_test)
        
        model, params, acc = ml.evalAlgo()



