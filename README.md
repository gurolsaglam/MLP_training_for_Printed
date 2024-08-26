This repository is intended for a clean framework for training, quantizing and carrying quantization aware training (QAT) machine learning models.

The old code is retrieved from Georgios Armeniakos. It was using Scikit-learn for the whole framework.
main_old_clean.py is the main script for the old framework, which still exists within the "algos" folder.

main_example.py is for the new user to start implementing their own version of a framework with the dataset input already set up.

main.py is the main script for the current framework which uses Skorch and Pytorch. You can use this by entering:
python main.py DATASET_NAME ALGO_NAME RANGE
the parameters are explained in the script. If using linux, then "python" keyword should be "python3".

After training, the framework will have created some joblibs under "trained_models" folder (or the folder you want to use), 
  when you load the joblibs, if you want to get the weights and biases, use "loaded_model.module_" 
  and DO NOT USE "loaded_model.module". The original randomly-initialized pytorch MLP will be kept under "loaded_module.module" 
  and the actual trained model will be kept under "loaded_module.module_".

