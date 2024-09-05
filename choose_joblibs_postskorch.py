#Script created by Gurol Saglam (or Guerol Saglam, guerol.saglam@kit.edu, gurol.saglam@ozu.edu.tr)
#It would be great if any developers could also add their name and contacts here:

#Python native libraries
import sys
import os

#Python public packages
import pandas as pd
import numpy as np

#Our packages
from qat_blocks.fxpquantizer import quantize_model

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
            quantized_accuracy, model = quantize_model(dataset_name, "./"+joblib_filename+".joblib", feature_range, input_bitwidth, weight_bitwidth, bias_bitwidth, relu_bitwidth, "fxp")
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
    
    
    
