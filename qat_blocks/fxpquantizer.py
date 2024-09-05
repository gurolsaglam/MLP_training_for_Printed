#Script created by Gurol Saglam (or Guerol Saglam, guerol.saglam@kit.edu, gurol.saglam@ozu.edu.tr)
#It would be great if any developers could also add their name and contacts here:

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
import qat_blocks.blackbox as bb

def get_quantized_dataset(dataset_name, feature_range, input_bitwidth):
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
    return dataset

#loads a joblib (Skorch NeuralNetClassifier object that holds a PyTorch MLP object)
def get_model(joblib_filename):
    #load the scikit model.
    loaded_model = load(joblib_filename)
    loaded_model.module_.to("cpu")
    
    #get model, weights and biases
    model = loaded_model.module_
    coefficients = model.getWeights()
    intercepts = model.getBiases()
    return model, coefficients, intercepts

def get_fxp_configs(coefficients, intercepts, input_bitwidth, weight_bitwidth):
    #get configurations.
    fxp_inputs = [fxpconverter(0, 0, input_bitwidth)]
    # fxp_qrelu = [fxpconverter(0, int_part_relu, frac_part_relu), fxpconverter(0, 6, 26)] #useful for qkeras models that have qrelu already.
    fxp_qrelu = [fxpconverter(0, 6, 26), fxpconverter(0, 6, 26)]
    
    #I use the same representation for all the weights --> suboptimal
    w_int = get_width(get_maxabs(coefficients))
    w_frac = weight_bitwidth - 1 - w_int
    fxp_weights = [fxpconverter(1, w_int, w_frac), fxpconverter(1, w_int, w_frac)]
    
    # trunc 1st relu
    L0trunc = 0
    trlst = [0,L0trunc]
    fxp_biases=[]
    for i, c in enumerate(intercepts):
        b_int = get_width(get_maxabs(c))
        fxp_biases.append(fxpconverter(1, b_int, fxp_inputs[0].frac+(i+1)*fxp_weights[i].frac-trlst[i]))
    
    return fxp_inputs, fxp_qrelu, fxp_weights, fxp_biases

#gets weights and biases of a QKeras MLP that is loaded
def get_model_values(model):
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

def get_model_po2(loaded_model, fxp_weights, fxp_biases, relu_bitwidth, dataset):
    #weight and bias bit sizes
    hl_weight_size = (fxp_weights[0].get_width(), fxp_weights[0].int) #(total bits, decimal part bits)
    hl_bias_size = (fxp_biases[0].get_width(), fxp_biases[0].int) #(total bits, decimal part bits)
    ol_weight_size = (fxp_weights[0].get_width(), fxp_weights[1].int) #(total bits, decimal part bits)
    ol_bias_size = (fxp_biases[0].get_width(), fxp_biases[1].int) #(total bits, decimal part bits)
    
    relu_size = (relu_bitwidth, None) #(total bits, decimal part bits) #None in the second part because we have made qrelu adaptive.
    
    #training parameters
    epochs = 100
    optimizer_lr = 0.001
    #pruning parameters
    sparsity = 0
    pr_begin_step=300
    pruning_params = [sparsity, pr_begin_step]
    signed = 1
    
    weight_bias_size=[ [hl_weight_size, hl_bias_size], [ol_weight_size, ol_bias_size] ]
    model, accuracy, weights = bb.blackbox(loaded_model, weight_bias_size, relu_size, 
                    dataset.Xtrain, dataset.Ytrain, dataset.Xtest, dataset.Ytest, 
                    epochs, optimizer_lr, pruning_params, signed)
    
    coefficients, intercepts = get_model_values(model)
    
    return model, coefficients, intercepts

#fxp_type refers to the fixed point type that we want to quantize the model to. The default is "fxp" for fixed point weight values, "fxp_po2" for fixed point+power of 2 weights.
def quantize_model(dataset_name, joblib_filename, feature_range, input_bitwidth, weight_bitwidth, bias_bitwidth, relu_bitwidth, fxp_type="fxp"):
    dataset = get_quantized_dataset(dataset_name, feature_range, input_bitwidth)
    
    #set the Y_train and Y_test in non-one-hot-coded format for some operations.
    Y_train = [np.argmax(y, axis=None, out=None) for y in dataset.Ytrain[:]]
    Y_test = [np.argmax(y, axis=None, out=None) for y in dataset.Ytest[:]]
    
    #get the model.
    model, coefficients, intercepts = get_model(joblib_filename)
    
    #get model features.
    fxp_inputs, fxp_qrelu, fxp_weights, fxp_biases = get_fxp_configs(coefficients, intercepts, input_bitwidth, weight_bitwidth)
    
    if (fxp_type == "fxp_po2"):
        model, coefficients, intercepts = get_model_po2(model, fxp_weights, fxp_biases, relu_bitwidth, dataset)
    
    #initialize the MLP model.
    last_layer = "relu"
    qmlp = MLP_Conventional(coefficients, intercepts, fxp_inputs, fxp_qrelu, fxp_weights, fxp_biases, 
                            min(Y_test), len(np.unique(Y_train)), last_layer)
    
    ##Get integer model accuracy
    acc_int = qmlp.get_accuracy(dataset.Xtest.values, Y_test)
    
    return acc_int, model
