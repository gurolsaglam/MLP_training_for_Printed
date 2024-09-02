This repository is intended for a clean framework for training, quantizing and carrying quantization aware training (QAT) machine learning models.

### OLD CODE
The old code is retrieved from Georgios Armeniakos. It was using Scikit-learn for the whole framework.
main_old_clean.py is the main script for the old framework, which still exists within the "algos" folder.

### NEW CODE
- main_example.py is for the new user to start implementing their own version of a framework with the dataset input already set up.

- main.py is the main script for the current framework which uses Skorch and Pytorch. You can use this by entering:
python main.py DATASET_NAME ALGO_NAME RANGE
the parameters are explained in the script. If using linux, then "python" keyword should be "python3".

  After training, the framework will have created some joblibs under "trained_models" folder (or the folder you want to use), 
    when you load the joblibs, if you want to get the weights and biases, use "loaded_model.module_" 
    and DO NOT USE "loaded_model.module". The original randomly-initialized pytorch MLP will be kept under "loaded_module.module" 
    and the actual trained model will be kept under "loaded_module.module_".

- choose_joblibs_postskorch.py is the script for the current framework that quantizes the weights and biases or the joblibs for the dataset(s) under the folder FOLDERNAME, the float weights and biases are quantized to integers (without re-training), their accuracies are calculated and recorded in the results tables under each folder, and then, the joblib model with the highest accuracy after quantization is chosen to be used. The list of models chosen will be in the folder FOLDERNAME, in the file "summary_fxp.xlsx". You can use this by entering:
choose_joblibs_postskorch.py DATASET_NAME INPUT_BWIDTH RANGE FOLDERNAME

- quantize_joblibs_postskorch_po2.py is the script to take the chosen joblib file with the trained MLP and quantize it to power-of-2 weights and biases. The script uses the "quantize_model" function in the folder "qat_blocks", and after quantization, it re-trains the MLP using QAT (Quantization Aware Training) so that the accuracy drop is minimized. You can use this by entering:
quantize_joblibs_postskorch_po2.py DATASET_NAME INPUT_BWIDTH RANGE FOLDERNAME

- choose_qatmodels_postqkeras.py is the script that will search and find the best quantization result. The previous script quantizes the network in different bit precision and records the accuracy at each increment, which enables the user to find and choose the specific model to use. You can use this by entering:
choose_qatmodels.py FOLDERNAME

