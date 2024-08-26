#Script created by Gurol Saglam (or Guerol Saglam, guerol.saglam@kit.edu, gurol.saglam@ozu.edu.tr)
#It would be great if any developers could also add their name and contacts here:

#Python native libraries
import sys
# import os

#Python public packages
# import pandas as pd


#Our packages
from datasets.read_datasets import Dataset

if __name__ == "__main__":
    #get parameters from the command line, the current usage is "main.py DATASET_NAME" DATASET_NAME should be the same name as the folder name of the dataset.
    dataset_name = sys.argv[1]
    
    #create the Dataset object for the specific dataset.
    #csv_separator is "," by default in the parameters, but it is also explicitly given here for the case that the user needs to change it to some other separator.
    dataset = Dataset(dataset_name, csv_separator=",")
    
    #the dataset is already divided into train and test subsets. two ways to access:
    
    #number1:
    # print(dataset.Xtrain)
    # print(dataset.Ytrain)
    # print(dataset.Xtest)
    # print(dataset.Ytest)
    
    #number2:
    # print(dataset.getXtrain())
    # print(dataset.getYtrain())
    # print(dataset.getXtest())
    # print(dataset.getYtest())
