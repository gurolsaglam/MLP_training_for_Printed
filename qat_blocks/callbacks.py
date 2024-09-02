'''
Created on 7 Apr 2017

@author: jkiesele
@editor: gurolsaglam
'''
from __future__ import print_function


from tensorflow.keras.callbacks import Callback, EarlyStopping,History,ModelCheckpoint,TensorBoard,ReduceLROnPlateau
# loss per epoch
from time import time
from pdb import set_trace
import json
import pandas as pd

class newline_callbacks_begin(Callback):
    
    def __init__(self,outputDir):
        self.outputDir=outputDir
        self.loss=[]
        self.val_loss=[]
        self.full_logs=[]
        
    def on_epoch_end(self,epoch, epoch_logs={}):
        import os
        lossfile=os.path.join( self.outputDir, 'losses.log')
        #print('\n***callbacks***\nsaving losses to '+lossfile)
        self.loss.append(epoch_logs.get('loss'))
        self.val_loss.append(epoch_logs.get('val_loss'))
        f = open(lossfile, 'w')
        for i in range(len(self.loss)):
            f.write(str(self.loss[i]))
            f.write(" ")
            f.write(str(self.val_loss[i]))
            f.write("\n")
        f.close()    
        normed = {}
        for vv in epoch_logs:
            normed[vv] = float(epoch_logs[vv])
        self.full_logs.append(normed)
        lossfile=os.path.join( self.outputDir, 'full_info.log')
        with open(lossfile, 'w') as out:
            out.write(json.dumps(self.full_logs))
        
class newline_callbacks_end(Callback):
    def on_epoch_end(self,epoch, epoch_logs={}):
        a=1
        
        
class Losstimer(Callback):
    def __init__(self, every = 5):
        self.points = []
        self.every = every

    def on_train_begin(self, logs):
        self.start = time()

    def on_batch_end(self, batch, logs):
        if (batch % self.every) != 0: return
        elapsed = time() - self.start
        cop = {}
        for i, j in logs.items():
            cop[i] = float(j)
        cop['elapsed'] = elapsed
        self.points.append(cop)
        
class all_callbacks(object):
    def __init__(self,
                 stop_patience=10,
                 lr_factor=0.5,
                 lr_patience=1,
                 lr_epsilon=0.001,
                 lr_cooldown=4,
                 lr_minimum=1e-5,
                 outputDir=''):
        

        
        self.nl_begin=newline_callbacks_begin(outputDir)
        self.nl_end=newline_callbacks_end()
        
        self.stopping = EarlyStopping(monitor='val_loss', 
                                      patience=stop_patience, 
                                      verbose=0, mode='min')
        
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=lr_patience, 
                                mode='min', verbose=0, epsilon=lr_epsilon,
                                 cooldown=lr_cooldown, min_lr=lr_minimum)

        self.modelbestcheck=ModelCheckpoint(outputDir+"/KERAS_check_best_model.h5", 
                                        monitor='val_loss', verbose=0, 
                                        save_best_only=True)

        self.modelbestcheckweights=ModelCheckpoint(outputDir+"/KERAS_check_best_model_weights.h5", 
                                            monitor='val_loss', verbose=0, 
                                            save_best_only=True,save_weights_only=True)
                
        self.modelcheckperiod=ModelCheckpoint(outputDir+"/KERAS_check_model_epoch{epoch:02d}.h5", verbose=0,period=10)
        
        self.modelcheck=ModelCheckpoint(outputDir+"/KERAS_check_model_last.h5", verbose=0)

        self.modelcheckweights=ModelCheckpoint(outputDir+"/KERAS_check_model_last_weights.h5", verbose=0,save_weights_only=True)
        
        self.tb = TensorBoard(log_dir=outputDir+'/logs')
  
        self.history=History()
        self.timer = Losstimer()
        
        self.callbacks=[
            self.nl_begin, 
            self.modelbestcheck,self.modelbestcheckweights, self.modelcheck,self.modelcheckweights,self.modelcheckperiod,
            self.reduce_lr, self.stopping, self.nl_end, self.tb, self.history,
            self.timer
        ]

class newline_callbacks_begin_wosavepoint(Callback): #same as newline_callbacks_begin, but no savepoints, so the callback does not save the model while training.
    
    def __init__(self,outputDir):
        self.outputDir=outputDir
        self.loss=[]
        self.val_loss=[]
        self.full_logs=[]
        
    def on_epoch_end(self,epoch, epoch_logs={}):
        import os
        self.loss.append(epoch_logs.get('loss'))
        self.val_loss.append(epoch_logs.get('val_loss'))
        normed = {}
        for vv in epoch_logs:
            normed[vv] = float(epoch_logs[vv])
        self.full_logs.append(normed)

class all_callbacks_wosavepoint(object): #same as all_callbacks, but no savepoints, so the callback does not save the model while training.
    def __init__(self,
                 stop_patience=10,
                 lr_factor=0.5,
                 lr_patience=1,
                 lr_epsilon=0.001,
                 lr_cooldown=4,
                 lr_minimum=1e-5,
                 outputDir=''):
        

        
        self.nl_begin=newline_callbacks_begin_wosavepoint(outputDir)
        self.nl_end=newline_callbacks_end()
        
        self.stopping = EarlyStopping(monitor='val_loss', 
                                      patience=stop_patience, 
                                      verbose=0, mode='min')
        
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=lr_patience, 
                                mode='min', verbose=0, epsilon=lr_epsilon,
                                 cooldown=lr_cooldown, min_lr=lr_minimum)
        self.history=History()
        self.accuracylogger = AccuracyLogger()
        self.timer = Losstimer()
        
        self.callbacks=[
            self.nl_begin, 
            self.reduce_lr, self.stopping, self.nl_end, self.history, self.accuracylogger, 
            self.timer
        ]

class AccuracyLogger(Callback):
    "A Logger that log average performance per `display` steps."

    def __init__(self):
        self.step = 0
        self.display = 1
        self.metric_cache = {}

    def on_epoch_end(self, epoch, logs={}):
        self.step += 1
        self.metric_cache[epoch] = logs["accuracy"]
    
    def on_train_end(self, epoch, logs={}):
        df = pd.DataFrame(columns = ["epoch", "accuracy"])
        df["epoch"] = self.metric_cache.keys()
        df["accuracy"] = self.metric_cache.values()
        df.to_excel("epoch_accuracies.xlsx", index=False)
        self.metric_cache.clear()
