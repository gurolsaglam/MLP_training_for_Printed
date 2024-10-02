#Script by Gurol Saglam (or Guerol Saglam, guerol.saglam@kit.edu, gurol.saglam@ozu.edu.tr)
#Python native libraries
import os
import sys

#Python public packages
import numpy as np
import pandas as pd
# from joblib import load
from tensorflow.keras.models import load_model
from hls4ml.utils import config_from_keras_model
from scipy import stats
from keras import backend as K

#Our packages
from qat_blocks.fxpquantizer import get_quantized_dataset, get_fxp_configs
from mlp_blocks.MLP_RISCV_Bespoke_po2 import *


def getModelValues(model):
    ## get model configurations
    intercepts = []
    coefficients = []
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        if layer_type == "QDense":
            bias = layer.get_weights()[1].tolist()
            weight = layer.get_weights()[0].tolist()
            intercepts.append(bias)
            coefficients.append(weight)
    return coefficients, intercepts

def getGranularities(model, input_bitwidth, weight_bitwidth, bias_bitwidth, relu_bitwidth, df_dataset):
    #get configurations.
    fxp_inputs = [fxpconverter(0, 0, input_bitwidth)]
    
    quantized_relu = df_dataset["quantized_relu"].values[0]
    int_part_relu = int(quantized_relu.split(",")[1][:-1])
    total_relu = int((quantized_relu.split("(")[1]).split(",")[0])
    frac_part_relu = total_relu - int_part_relu
    
    # quantized_relu = model.layers[1].quantizer
    # int_part_relu = quantized_relu.integer.numpy()[0]
    # frac_part_relu = quantized_relu.bits - int_part_relu
    
    # fxp_qrelu = [fxpconverter(0, int_part_relu, frac_part_relu), fxpconverter(0, 6, 26)] #useful for qkeras models that have qrelu already. OLD, NOW WE LET MLP OBJECT FIND THE OUTPUT BIT PRECISION
    fxp_qrelu = [fxpconverter(0, int_part_relu, frac_part_relu)] #useful for qkeras models that have qrelu already.
    
    coefficients, intercepts = getModelValues(model)
    #I use the same representation for all the weights --> suboptimal
    w_int = get_width(get_maxabs(coefficients))
    w_frac = weight_bitwidth - 1 - w_int
    fxp_weights = [fxpconverter(1, w_int, w_frac), fxpconverter(1, w_int, w_frac)]
    
    #make a temporary array for all inputs of layers for bias max bitwidth calculation
    fxp_inputs_all = [fxpconverter(0, 0, input_bitwidth), fxpconverter(0, 0, relu_bitwidth)]
    # trunc 1st relu
    L0trunc = 0
    trlst = [0,L0trunc]
    fxp_biases=[]
    for i, c in enumerate(intercepts):
        b_int = get_width(get_maxabs(c))
        # fxp_biases.append(fxpconverter(1, b_int, fxp_inputs[0].frac+(i+1)*fxp_weights[i].frac-trlst[i]))
        fxp_biases.append(fxpconverter(1, b_int, fxp_inputs_all[i].frac+fxp_weights[i].frac))
    
    return coefficients, intercepts, fxp_inputs, fxp_qrelu, fxp_weights, fxp_biases

def create_asms(dataset_name, df_dataset, model_save_folder, asm_folder, feature_range, input_bitwidth, weight_bitwidth, bias_bitwidth, relu_bitwidth, fxp_type="fxp"):
    dataset = get_quantized_dataset(dataset_name, feature_range, input_bitwidth)
    
    #set the Y_train and Y_test in non-one-hot-coded format for some operations.
    Y_train = [np.argmax(y, axis=None, out=None) for y in dataset.Ytrain[:]]
    Y_test = [np.argmax(y, axis=None, out=None) for y in dataset.Ytest[:]]
    
    
    # load model
    model_path =  model_save_folder + dataset_name + "/" + dataset_name + ".keras"
    model = load_model(model_path)
    ## loaded model accuracy
    # loaded_accuracy = model.evaluate(dataset.Xtest, dataset.Ytest)
    # print(f"Accuracy of loaded model is {(loaded_accuracy[1]*100)}")
    
    #get model features.
    coefficients, intercepts, fxp_inputs, fxp_qrelu, fxp_weights, fxp_biases = getGranularities(model, input_bitwidth, weight_bitwidth, bias_bitwidth, relu_bitwidth, df_dataset)
    
    # print(coefficients)
    # print(intercepts)
    # print(fxp_inputs)
    # print(fxp_qrelu)
    # print(fxp_weights)
    # print(fxp_biases)
    
    ##Initialize our integer model
    last_layer = "relu"
    qmlp = MLP_RISCV_Bespoke_po2(coefficients, intercepts, fxp_inputs, fxp_qrelu, fxp_weights, fxp_biases, min(Y_train), len(np.unique(Y_train)), last_layer)
    
    ##Get integer model accuracy
    acc_int = qmlp.get_accuracy(dataset.Xtest.values, Y_test)
    print(dataset_name)
    print(acc_int)
    print(qmlp.coefs)
    print(qmlp.intercept)
    
    ##Set parameters
    # sum_relu_size = [[(32,6), relu_size], [(32,6), (32,6)]] #UNUSED
    
    f = open(asm_folder + str(dataset_name) + ".s","w")
    qmlp.write_asm(f, dataset_name)
    f.close()
    
    return 0

if __name__ == "__main__":
    #get parameters from the command line, the current usage is "main.py DATASET_NAME"
    #DATASET_NAME should be the same name as the folder name of the dataset.
    data = sys.argv[1]
    # input_bitwidth = int(sys.argv[2])
    input_bitwidth = 4
    # feature_range = (int(sys.argv[3].split(",")[0]), int(sys.argv[3].split(",")[1]))
    feature_range = (0,1)
    weight_bitwidth = 8
    bias_bitwidth = 8
    relu_bitwidth = 8
    
    algo_name = "MLP"
    
    trained_models = "./trained_models/"
    df_all = pd.read_excel(trained_models + "summary_fxppo2_qkeras.xlsx")
    
    qat_models = "./postqat/"
    
    model_save_folder = qat_models + algo_name + "/"
    #get what datasets the folder has,
    datasets = [d for d in os.listdir(model_save_folder)]
    datasets = sorted(datasets)
    if not (data == "all") and data in datasets:
        datasets = [data]
    
    asm_folder = "./asms/"
    if not os.path.exists(asm_folder):
        os.makedirs(asm_folder)
    for dataset_name in datasets:
        df_dataset = df_all[df_all.dataset == dataset_name]
        create_asms(dataset_name, df_dataset, model_save_folder, asm_folder, feature_range, input_bitwidth, weight_bitwidth, bias_bitwidth, relu_bitwidth, fxp_type="fxppo2")
    
    