#Script created by Gurol Saglam (or Guerol Saglam, guerol.saglam@kit.edu, gurol.saglam@ozu.edu.tr)

#Python native libraries
import sys
import os

#Python public packages
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from hls4ml.utils import config_from_keras_model
from joblib import load

#Our packages
from datasets.read_datasets import Dataset
from qat_blocks.fxpconverter import *
from mlp_blocks.MLP_Conventional import *

def getFeatsfromModel(joblib_filename, input_bitwidth, weight_bitwidth, X_test, Y_test):
    #load the scikit model.
    loaded_model = load(joblib_filename)
    loaded_model.module_.to("cpu")
    
    #get weights and biases
    model = loaded_model.module_
    
    #get model configurations.
    fxp_inputs = [fxpconverter(0, 0, input_bitwidth)]
    # fxp_qrelu = [fxpconverter(0, int_part_relu, frac_part_relu), fxpconverter(0, 6, 26)] #useful for qkeras models that have qrelu already.
    fxp_qrelu = [fxpconverter(0, 6, 26), fxpconverter(0, 6, 26)]
    
    #I use the same representation for all the weights --> suboptimal
    w_int = get_width(get_maxabs(model.getWeights()))
    w_frac = weight_bitwidth - 1 - w_int
    fxp_weights = [fxpconverter(1, w_int, w_frac), fxpconverter(1, w_int, w_frac)]
    
    # trunc 1st relu
    L0trunc = 0
    trlst = [0,L0trunc]
    fxp_biases=[]
    for i, c in enumerate(model.getBiases()):
        b_int = get_width(get_maxabs(c))
        fxp_biases.append(fxpconverter(1, b_int, fxp_inputs[0].frac+(i+1)*fxp_weights[i].frac-trlst[i]))
    
    coefficients = model.getWeights()
    intercepts = model.getBiases()
    
    return coefficients, intercepts, fxp_inputs, fxp_qrelu, fxp_weights, fxp_biases

def quantize_model(dataset_name, joblib_filename, input_bitwidth, feature_range, weight_bitwidth, bias_bitwidth, relu_bitwidth):
    #print("Dataset is: " + dataset_name)
    
    #create the Dataset object for the specific dataset.
    #csv_separator is "," by default in the parameters, but it is also explicitly given here for the case that the user needs to change it to some other separator.
    dataset = Dataset(dataset_name, csv_separator=",")
    
    #rescale the input to the desired range.
    dataset.rescale_features(feature_range)
    
    #rescale Xtrain and Xtest into the number of bits we want.
    dataset.quantize_features(input_bitwidth)
    #WHEN FEATURES ARE BINARIZED AND SEPARATED:
    # dataset.bitseparate_features(input_bitwidth)
    
    #modify the labels into binary vector form from integer vector (one-hot coding).
    dataset.binarize_labels()
    
    #set the Y_train and Y_test in non-one-hot-coded format for some operations
    Y_train = [np.argmax(y, axis=None, out=None) for y in dataset.Ytrain[:]]
    Y_test = [np.argmax(y, axis=None, out=None) for y in dataset.Ytest[:]]
    
    #get model features.
    coefficients, intercepts, fxp_inputs, fxp_qrelu, fxp_weights, fxp_biases = getFeatsfromModel("./" + joblib_filename, input_bitwidth, weight_bitwidth, dataset.Xtest, dataset.Ytest)
    
    #initialize the MLP model.
    last_layer = "relu"
    qmlp = MLP_Conventional(coefficients, intercepts, fxp_inputs, fxp_qrelu, fxp_weights, fxp_biases, 
                            min(Y_test), len(np.unique(Y_train)), last_layer)
    
    ##Get integer model accuracy
    acc_int = qmlp.get_accuracy(dataset.Xtest.values, Y_test)
    
    return acc_int

if __name__ == "__main__":
    #get parameters from the command line, the current usage is "main.py DATASET_NAME INPUT_BWIDTH RANGE FOLDERNAME"
    #DATASET_NAME should be the same name as the folder name of the dataset,
    #INPUT_BWIDTH is the number of bits wanted for the inputs,
    #RANGE should be the min,max range with comma inbetween, no spaces, e.g. 0,1.
    #FOLDERNAME should be the folder of the trained models such as "trained_models".
    data = sys.argv[1]
    input_bitwidth = int(sys.argv[2])
    feature_range = (int(sys.argv[3].split(",")[0]), int(sys.argv[3].split(",")[1]))
    trained_models = sys.argv[4] #"trained_models"
    weight_bitwidth = 8
    bias_bitwidth = 8
    relu_bitwidth = 8
    
    algo_name = "MLP"
    
    dfs_all = []
    algorithm_dump_folder = trained_models + "/" + algo_name + "/"
    #get what datasets the folder has,
    datasets = [d for d in os.listdir(algorithm_dump_folder) if (len(d.split("."))==1)]
    datasets = sorted(datasets)
    if not (data == "all") and data in datasets:
        datasets = [data]
    
    columns = []
    summary_results = []
    #for each dataset:
    for dataset_name in datasets:
        dataset_dump_folder = algorithm_dump_folder + dataset_name + "/"
        results_dump_file = dataset_dump_folder + "results_table.xlsx"
        #read excel
        df = pd.read_excel(results_dump_file)
        #add a new column for the new location of the trained model.
        joblib_filenames = list(df.joblib_filename)
        
        #for each model saved in each joblib:
        quantized_accuracies = []
        accuracy_drops = []
        for joblib_filename in joblib_filenames:
            #quantize the model and get accuracy
            quantized_accuracy = quantize_model(dataset_name, joblib_filename+".joblib", input_bitwidth, feature_range, weight_bitwidth, bias_bitwidth, relu_bitwidth)
            original_accuracy = df[df.joblib_filename == joblib_filename]["accuracy"].values[0]
            accuracy_drop = original_accuracy - quantized_accuracy
            quantized_accuracies.append(quantized_accuracy)
            accuracy_drops.append(accuracy_drop)
        df["fxp_accuracy"] = quantized_accuracies
        df["orig-fxp-drop"] = accuracy_drops
        df.to_excel(results_dump_file, index=False)
        
        #from the original and quantized models, choose the best one. #TODO
        columns = list(df.columns)
        dfs_all.append(df)
    
    df_all = pd.concat(dfs_all, ignore_index=True)
    # df_all.to_excel(trained_models + "/summary_results_table_all.xlsx", index=False)
    df_all.to_excel(trained_models + "/summary_fxp_all.xlsx", index=False)
    
    datasets = np.unique(df_all["dataset"].values)
    results = []
    for dataset in datasets:
        df = df_all[df_all.dataset == dataset]
        avg_acc = df["accuracy"].mean()
        df_sub = df[df.accuracy >= avg_acc]
        #df_sub = df_sub[df_sub.fxp_accuracy >= avg_acc]
        #df_res = df_sub[df_sub["orig-fxp-drop"] == df_sub["orig-fxp-drop"].min()]
        df_res = df_sub[df_sub["fxp_accuracy"] == df_sub["fxp_accuracy"].max()]
        if (len(df_res.index) > 1):
            #df_res = df_res[df_res["fxp_accuracy"] == df_res["fxp_accuracy"].max()]
            df_res = df_res[df_res["orig-fxp-drop"] == df_res["orig-fxp-drop"].min()]
            if (len(df_res.index) > 1):
                df_res = df_res[df_res["joblib_filename"] == df_res.iloc[0]["joblib_filename"]]
        results.append(df_res)
    df_last = pd.concat(results, ignore_index=True)
    # df_last.to_excel(trained_models + "/summary_results_table.xlsx", index=False)
    df_last.to_excel(trained_models + "/summary_fxp.xlsx", index=False)
    
    
    
