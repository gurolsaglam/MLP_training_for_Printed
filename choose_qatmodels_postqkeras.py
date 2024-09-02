#Script created by Gurol Saglam (or Guerol Saglam, guerol.saglam@kit.edu, gurol.saglam@ozu.edu.tr)

#Python native libraries
import os
import sys

#Python public packages
import pandas as pd

#Our packages

#USAGE:
# python choose_qatmodels.py FOLDERNAME
#Where FOLDERNAME is the name of the folder that has the recordings of QAT and the best model found after QAT.
# python choose_qatmodels.py postqat_signed

algo_name = "MLP"
qat_models = str(sys.argv[1]) + "/"
algorithm_dump_folder = qat_models + algo_name + "/"

datasets = os.listdir(algorithm_dump_folder)
datasets = sorted(datasets)
columns = []
summary_results = []
for dataset_name in datasets:
    dataset_dump_folder = algorithm_dump_folder + dataset_name + "/"
    results_dump_file = dataset_dump_folder + "sum_epoch_accuracies.xlsx"
    #read excel and find the max accuracy
    df = pd.read_excel(results_dump_file)
    columns = list(df.columns)
    columns.append("dataset")
    maxdesign = df[df.accuracy == df.accuracy.max()]
    maxdesign = maxdesign.iloc[0]
    maxdesign["dataset"] = dataset_name
    summary_results.append(maxdesign)
    
dfNew = pd.DataFrame(columns=columns)
for entry in summary_results:
    dfNew.loc[len(dfNew.index)] = entry
    dfNew.to_excel(algorithm_dump_folder + "summary_results_table.xlsx", index=False)
