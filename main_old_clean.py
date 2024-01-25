# Script modified by Gurol Saglam (or Guerol Saglam, guerol.saglam@kit.edu, gurol.saglam@ozu.edu.tr)
# It would be great if any developers could also add their name and contacts here:

# Python native libraries
import sys
import os

# Python public packages
import pandas as pd
from joblib import dump

# Our packages
from datasets.read_datasets import Dataset
from algos import *

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
    dataset.rescale_features(feature_range)
    
    # modify the labels into binary vector form from integer vector (one-hot coding). (do this only if the dataset has 2 unique labels.
    dataY = pd.concat([dataset.Ytrain, dataset.Ytest])
    if (len(dataY.unique()) == 2):
        dataset.binarize_labels()
    
    # choose the algorithm we want with the input parameter provided by the user.
    if algo_name in ['MLP']:
        algo = MLP
    elif algo_name in ['DecisionTree']:
        algo = DecisionTree
    elif algo_name in ['RandomForest']:
        algo = RandomForest
    elif algo_name in ['SVM']:
        algo = SVM
    elif algo_name in ['LogReg']:
        algo = LogReg
    else:
        assert(False)
    
    # check if the dump directory exists, create if not.
    trained_models = "trained_models/"
    algorithm_dump_folder = trained_models + algo_name + "/"
    dataset_dump_folder = algorithm_dump_folder + dataset_name + "/"
    results_dump_file = dataset_dump_folder + "results_table.xlsx"
    if (not os.path.exists(dataset_dump_folder)):
        os.makedirs(dataset_dump_folder)
        # also add an Excel sheet for result dumping.
        df = pd.DataFrame(columns=["dataset", "algo", "feature_range", "hidden_layer_sizes",
                                   "index", "joblib_filename", "params", "accuracy"])
        df.to_excel(results_dump_file, index=False)
    
    # train the same algorithm 20 times.
    hidden_layer_sizes = 3
    for i in range(0, 20):
        # create the algorithm.
        name = dataset_name + "_" + str(feature_range) + "_" + str(hidden_layer_sizes) + "_" + str(i)
        ml = algo(dataset.Xtrain, dataset.Xtest, dataset.Ytrain, dataset.Ytest, name,
                  hidden_layer_sizes=hidden_layer_sizes)
        # learn and evaluate the algorithm.
        model, params, acc = ml.evalAlgo()
        # dump the information of the algorithm into a joblib file.
        dump(model, dataset_dump_folder + name + ".joblib")
        # append the result to the existing Excel sheet.
        df = pd.read_excel(results_dump_file)
        df.loc[len(df.index)] = [dataset_name, algo_name, feature_range, hidden_layer_sizes, str(i),
                                 dataset_dump_folder + name, params, acc]
        df.to_excel(results_dump_file, index=False)
