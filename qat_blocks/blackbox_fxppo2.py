#Script by Gurol Saglam (or Guerol Saglam, guerol.saglam@kit.edu, gurol.saglam@ozu.edu.tr)
#Python native libraries

#Python public packages
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from qkeras.qlayers import QDense, QActivation, QAdaptiveActivation
from qkeras.quantizers import quantized_po2, quantized_relu
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule, prune, pruning_callbacks
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from keras.constraints import NonNeg
from qkeras.utils import model_save_quantized_weights

#Our packages
from qat_blocks.callbacks import all_callbacks_wosavepoint

#set seed for randomness
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

def blackbox(loaded_model, weight_bias_size, relu_size, 
                X_train, Y_train, X_test, Y_test, 
                epochs, optimizer_lr, pruning_params, signed):
    
    hidden_layer_sizes = loaded_model.getHiddenLayerTopology()
    if type(hidden_layer_sizes) == int:
        hidden_layer_sizes = np.array([hidden_layer_sizes])
    input_layer = X_train.shape[1]
    hidden_layer = hidden_layer_sizes[0]
    output_layer = 1
    if (len(Y_train.shape) == 2):
        output_layer = Y_train.shape[1]
    
    #separate the parameters for the hidden layer and output layer
    hl_kq_maxv=2**(weight_bias_size[0][0][1])
    hl_bq_maxv=2**(weight_bias_size[0][1][1])
    hl_kreg=0.0001
    
    ol_kq_maxv=2**(weight_bias_size[1][0][1])
    ol_bq_maxv=2**(weight_bias_size[1][1][1])
    ol_kreg=0.0001
    
    
    #Setup the MLP Network in QKeras
    model = Sequential()
    model.add(Input(shape=(input_layer,)))
    
    model.add(QDense(hidden_layer, name = 'fc1', 
                kernel_quantizer = quantized_po2(weight_bias_size[0][0][0],
                                                max_value=hl_kq_maxv,
                                                use_stochastic_rounding=False,
                                                quadratic_approximation=False),
                bias_quantizer = quantized_po2(weight_bias_size[0][1][0],
                                                max_value=hl_bq_maxv,
                                                use_stochastic_rounding=False,
                                                quadratic_approximation=False),
                kernel_initializer = 'lecun_uniform', kernel_regularizer=l1(hl_kreg), 
                kernel_constraint = None if (signed==1) else NonNeg())) #signed/unsigned weights and biases
    
    #this adaptive activation layer of qrelu was added by Gurol, to find the best qrelu precision.
    model.add(QAdaptiveActivation(activation="quantized_relu", total_bits=relu_size[0], name='relu1', quantization_delay=5))#, po2_rounding=True))
    #model.add(QActivation(activation=quantized_relu(relu_size[0], relu_size[1], use_stochastic_rounding=False), name='relu1'))
    
    model.add(QDense(output_layer, name = 'output',
                kernel_quantizer = quantized_po2(weight_bias_size[1][0][0],
                                                max_value=ol_kq_maxv,
                                                use_stochastic_rounding=False,
                                                quadratic_approximation=False), 
                bias_quantizer = quantized_po2(weight_bias_size[1][1][0],
                                                max_value=ol_bq_maxv,
                                                use_stochastic_rounding=False,
                                                quadratic_approximation=False),
                kernel_initializer = 'lecun_uniform', kernel_regularizer=l1(ol_kreg), 
                kernel_constraint = None if (signed==1) else NonNeg())) #signed/unsigned weights and biases
    #this relu was added by Gurol, qkeras accuracy and our mlp accuracy matches exactly with this, and the qkeras model is more accurate.
    model.add(Activation(activation='relu', name='reluout'))
    model.add(Activation(activation='softmax', name='softmax'))
    
    
    #Pruning
    #Sparsity, previous script used 0 for this
    sparsity = pruning_params[0]
    sparsity_val = float(sparsity / 10)
    begin_step = pruning_params[1]
    pruning_params = {"pruning_schedule" : pruning_schedule.ConstantSparsity(sparsity_val, begin_step=begin_step, frequency=100)}
    # for layer in model.layers:
        # if not (layer.name == "relu1"):
            # layer = prune.prune_low_magnitude(layer, **pruning_params)
    #model = prune.prune_low_magnitude(model, **pruning_params)
    
    #Set the weights from loaded model to the QKeras model
    wb1=[loaded_model.getWeights()[0], loaded_model.getBiases()[0]]
    wb2=[loaded_model.getWeights()[1], loaded_model.getBiases()[1]]
    
    model.layers[0].set_weights(wb1)
    model.layers[2].set_weights(wb2)
    
    
    #Train the new model
    adam = Adam(learning_rate=optimizer_lr)
    model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
    
    callbacks = all_callbacks_wosavepoint(outputDir = "")
    # callbacks = all_callbacks_wosavepoint(outputDir = "./postqat/" + dataset_name + "_classification_prune")
    callbacks.callbacks.append(pruning_callbacks.UpdatePruningStep())
    
    history = model.fit(X_train, Y_train, batch_size=1,
              epochs=epochs,validation_split=0.2, verbose=0, shuffle=True,
              callbacks = callbacks.callbacks)
    model_save_quantized_weights(model)
    
    # model = strip_pruning(model)
    model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
    
    accuracy = model.evaluate(X_test,Y_test)
    # print(model.get_weights())
    print(" ACCURACY IS "+str(accuracy[1]) )
    # print(model.summary())
    # print(model.layers[1].get_config())
    # print(model.layers[1].get_quantization_config())

    return model, accuracy[1], model.get_weights()
