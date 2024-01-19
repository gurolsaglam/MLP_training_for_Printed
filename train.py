# Script modified by Gurol Saglam (or Guerol Saglam, guerol.saglam@kit.edu, gurol.saglam@ozu.edu.tr)
# It would be great if any developers could also add their name and contacts here:

# Python native libraries
import sys
import os
from scipy.stats import reciprocal, uniform, expon

import torch

# Python public packages
import numpy as np
import pandas as pd
from joblib import dump

# Our packages
from sklearn.preprocessing import StandardScaler

from datasets.read_datasets import Dataset
from algos import *
from algos_pytorch.mlp import MLP, Algo

# from datasets.datasets import SportsActivities, MexDataPM, RealDispIdeal, MuscleActivity, ActPhysiological

if __name__ == "__main__":
    # get parameters from the command line, the current usage is:
    # "main.py DATASET_NAME ALGO_NAME RANGE" where:
    # DATASET_NAME should be the same name as the folder name of the dataset,
    # ALGO_NAME should be the possible algorithms we have implemented in the algos folder,
    # RANGE should be the min,max range with comma inbetween, no spaces, e.g. 0,1.
    dataset_name = sys.argv[1]
    algo_name = sys.argv[2]
    feature_range = (int(sys.argv[3].split(",")[0]), int(sys.argv[3].split(",")[1]))

    # create the Dataset object for the specific dataset. csv_separator is "," by default in the parameters,
    # but it is also explicitly given here for the case that the user needs to change it to some other separator.
    dataset = Dataset(dataset_name, csv_separator=",")

    # the dataset is already divided into train and test subsets. Labels are already in unique & categorical format.
    # print(dataset.Xtrain)
    # print(dataset.Ytrain)
    # print(dataset.Xtest)
    # print(dataset.Ytest)

    # rescale the input to the desired range.
    # dataset.rescale_features(feature_range)
    scaler = StandardScaler()

    X_train, y_train = dataset.Xtrain.values, dataset.Ytrain
    X_train, y_train = X_train.astype(np.float32), y_train.astype(np.int64)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    X_test, y_test = dataset.Xtest.values, dataset.Ytest
    X_test, y_test = X_test.astype(np.float32), y_test.astype(np.int64)
    X_test = scaler.transform(X_test)

    dim_input = X_train.shape[1]
    dim_output = max(set(y_train)) + 1
    # choose the algorithm we want with the input parameter provided by the user.
    if algo_name in ['MLP']:
        net = MLP(topology=(dim_input, 10, 20, dim_output),
                  )
        algo = Algo(net,
                    X_train, X_test,
                    y_train, y_test,
                    'hi', task='classification',
                    max_epochs=20,
                    lr=0.1,
                    optimizer=torch.optim.Adam,
                    momentum=1,
                    nesterov=True,)
        # algo = MLP
    elif algo_name in ['DecisionTree']:
        algo = DecisionTree
    elif algo_name in ['RandomForest']:
        algo = RandomForest
    elif algo_name in ['SVM']:
        algo = SVM
    elif algo_name in ['LogReg']:
        algo = LogReg
    else:
        assert False

    # check if the dump directory exists, create if not.
    trained_models = "trained_models/"
    algorithm_dump_folder = trained_models + algo_name + "/"
    dataset_dump_folder = algorithm_dump_folder + dataset_name + "/"
    results_dump_file = dataset_dump_folder + "results_table.xlsx"
    if not os.path.exists(dataset_dump_folder):
        os.makedirs(dataset_dump_folder)
        # also add an Excel sheet for result dumping.
        df = pd.DataFrame(
            columns=["dataset", "algo", "feature_range", "hidden_layer_sizes", "index", "joblib_filename", "params",
                     "accuracy"])
        df.to_excel(results_dump_file, index=False)

    # train the same algorithm 20 times.
    hidden_layer_sizes = 3
    for i in range(0, 1):
        # create the algorithm.
        name = dataset_name + "_" + str(feature_range) + "_" + str(hidden_layer_sizes) + "_" + str(i)
        param_dists = {'lr': uniform(0.001, 0.1),
                       'module__topology': [(dim_input, 64, dim_output),
                                            (dim_input, 64, 32, dim_output),
                                            (dim_input, 64, 32, 16, dim_output)],
                       'module__activation': ['identity', 'tanh', 'relu'],
                       'module__dropout': uniform(0, 0.5),
                       'optimizer': [torch.optim.LBFGS, torch.optim.SGD, torch.optim.Adam],
                       'optimizer__momentum': uniform(0, 1),
                       'optimizer__nesterov': [False, True],
                       # 'optimizer__beta_1': np.linspace(0.8, 0.99, 5),  # Sample 5 values between 0.8 and 0.99
                       # 'optimizer__beta_2': np.linspace(0.9, 0.999, 5)  # Sample 5 values between 0.9 and 0.999
                       # 'optimizer__eps ': uniform(0, 0.999),
                       }
        model, params, acc = algo.evalAlgo(param_dists)
        # dump the information of the algorithm into a joblib file.
        dump(model, dataset_dump_folder + name + ".joblib")
        # append the result to the existing Excel sheet.
        df = pd.read_excel(results_dump_file)
        df.loc[len(df.index)] = [dataset_name, algo_name, feature_range, hidden_layer_sizes, str(i),
                                 dataset_dump_folder + name, params, acc]
        df.to_excel(results_dump_file, index=False)
        print(params, acc)
