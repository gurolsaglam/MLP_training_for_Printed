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
            quantized_accuracy, model = quantize_model(dataset_name, "./"+joblib_filename+".joblib", feature_range, input_bitwidth, weight_bitwidth, bias_bitwidth, relu_bitwidth, "fxp_po2")
            original_accuracy = df[df.joblib_filename == joblib_filename]["accuracy"].values[0]
            accuracy_drop = original_accuracy - quantized_accuracy
            quantized_accuracies.append(quantized_accuracy)
            accuracy_drops.append(accuracy_drop)
        df["fxppo2_accuracy_qkeras"] = quantized_accuracies
        df["orig-fxppo2-drop_qkeras"] = accuracy_drops
        df.to_excel(results_dump_file, index=False)
        
        #from the original and quantized models, choose the best one. #TODO
        columns = list(df.columns)
        dfs_all.append(df)
    
    df_all = pd.concat(dfs_all, ignore_index=True)
    # df_all.to_excel(trained_models + "/summary_results_table_all.xlsx", index=False)
    df_all.to_excel(trained_models + "/summary_fxppo2_all_qkeras.xlsx", index=False)
    
    # df_all = pd.read_excel(trained_models + "/summary_fxppo2_all_qkeras.xlsx")
    
    datasets = np.unique(df_all["dataset"].values)
    results = []
    for dataset_name in datasets:
        #get the models of the specific dataset.
        df = df_all[df_all.dataset == dataset_name]
        
        #find the average accuracy of the floating point models and drop the models below the accuracy.
        avg_acc = df["accuracy"].mean()
        df_sub = df[df.accuracy >= avg_acc]
        #df_sub = df_sub[df_sub.fxp_accuracy >= avg_acc]
        #df_res = df_sub[df_sub["orig-fxp-drop"] == df_sub["orig-fxp-drop"].min()]
        
        #find the max fxppo2 accuracy and drop the models if their fxppo2 accuracy is more than 2% lower than the max accuracy.
        max_acc = df_sub["fxppo2_accuracy_qkeras"].max()
        df_sub = df_sub[df_sub["fxppo2_accuracy_qkeras"] >= (max_acc-0.02)]
        
        #find the minimal absolute accuracy difference
        diffs = df_sub["orig-fxppo2-drop_qkeras"].values
        diffs = np.abs(diffs)
        df_sub["abs-orig-fxppo2-drop_qkeras"] = diffs
        df_sub = df_sub[df_sub["abs-orig-fxppo2-drop_qkeras"] < (df_sub["abs-orig-fxppo2-drop_qkeras"].min()+0.01)]
        # # min_abs_drop_index = np.argmin(diffs)
        
        #get the model with minimal absolute accuracy difference
        # # df_res = df_sub.iloc[[min_abs_drop_index]]
        df_res = df_sub
        if (len(df_res.index) > 1):
            df_res = df_res[df_res["fxppo2_accuracy_qkeras"] == df_res["fxppo2_accuracy_qkeras"].max()]
            if (len(df_res.index) > 1):
                df_res = df_res[df_res["joblib_filename"] == df_res.iloc[0]["joblib_filename"]]
        results.append(df_res)
        
        #THIS IS THE OLD MAX MODEL FINDER
        # df_res = df_sub[df_sub["fxppo2_accuracy_qkeras"] == max_acc]
        # if (len(df_res.index) > 1):
            # # df_res = df_res[df_res["fxp_accuracy"] == df_res["fxp_accuracy"].max()]
            # df_res = df_res[df_res["orig-fxppo2-drop_qkeras"] == df_res["orig-fxppo2-drop_qkeras"].min()]
            # if (len(df_res.index) > 1):
                # df_res = df_res[df_res["joblib_filename"] == df_res.iloc[0]["joblib_filename"]]
        # # results.append(df_res)
    df_last = pd.concat(results, ignore_index=True)
    # df_last.to_excel(trained_models + "/summary_results_table.xlsx", index=False)
    df_last.to_excel(trained_models + "/summary_fxppo2_qkeras.xlsx", index=False)
    
    #Recreate the fxppo2_qkeras models (load the floatingpoint model and quantize), and then save the models.
    qat_models = "postqat/"
    for dataset_name in datasets:
        model_save_folder = qat_models + "MLP/" + dataset_name + "/"
        if not os.path.exists(model_save_folder):
            os.makedirs(model_save_folder)
        model_info = df_last[df_last.dataset == dataset_name]
        joblib_filename = model_info["joblib_filename"].values[0]
        quantized_accuracy, model = quantize_model(dataset_name, "./"+joblib_filename+".joblib", feature_range, input_bitwidth, weight_bitwidth, bias_bitwidth, relu_bitwidth, "fxp_po2")
        model.save(model_save_folder + dataset_name + ".keras")
    
    
    
