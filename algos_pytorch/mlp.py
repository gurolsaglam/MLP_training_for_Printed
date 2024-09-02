import torch
import torch.nn as nn
import torch.nn.functional as F

from skorch import NeuralNetClassifier, NeuralNetRegressor

from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.dummy import DummyClassifier

import numpy as np

from joblib import dump


class MLP(nn.Module):
    def __init__(self, topology, activation='relu', max_iter=100, dropout=0):
        super(MLP, self).__init__()
        self.topology = topology
        self.max_iter = max_iter
        self.blocks = nn.ModuleList()
        self.activation_block = nn.ReLU()

        for i in range(len(topology)):
            self.blocks.append(nn.Linear(topology[i], topology[i + 1]))
            if activation == 'identity':
                self.blocks.append(nn.Identity())
            elif activation == 'tanh':
                self.blocks.append(nn.Tanh())
            else:
                self.blocks.append(nn.ReLU())
            self.blocks.append(nn.Dropout(p=dropout))
            if (i + 1) == (len(topology) - 1):
                break
        self.blocks.append(nn.ReLU())

        self.model = nn.Sequential(*self.blocks)

    def forward(self, x):
        x = self.model(x)
        out = F.softmax(x, dim=-1)
        return out
    
    def getWeights(self):
        weights = []
        for block in self.blocks:
            if isinstance(block, nn.modules.linear.Linear):
                weights.append(np.transpose(np.array(block.weight.data)))
        return weights
    
    def setWeights(self, weights):
        i = 0
        for block in self.blocks:
            if isinstance(block, nn.modules.linear.Linear):
                block.weight.data = torch.tensor(np.transpose(weights[i]), dtype=torch.float32)
                i = i + 1
    
    def getBiases(self):
        biases = []
        for block in self.blocks:
            if isinstance(block, nn.modules.linear.Linear):
                biases.append(np.array(block.bias.data))
        return biases
    
    def setBiases(self, biases):
        i = 0
        for block in self.blocks:
            if isinstance(block, nn.modules.linear.Linear):
                block.bias.data = torch.tensor(biases[i], dtype=torch.float32)
                i = i + 1
    
    def getHiddenLayerTopology(self):
        temp = []
        for i in range(1, len(self.topology)-1):
            temp.append(self.topology[i])
        return temp


class Algo:
    def __init__(self, model,
                 X_train, X_test,
                 y_train, y_test,
                 name,
                 task,
                 max_epochs,
                 lr,
                 optimizer,
                 momentum,
                 nesterov,
                 verbose=0):
        self.model = NeuralNetClassifier(model,
                                         max_epochs=max_epochs,
                                         lr=lr,
                                         optimizer=optimizer,
                                         optimizer__momentum=momentum,
                                         optimizer__nesterov=nesterov,
                                         verbose=verbose,
                                         ) if task == "classification" else \
            NeuralNetRegressor(model,
                               max_epochs=max_epochs,
                               lr=lr,
                               optimizer=optimizer,
                               optimizer__momentum=momentum,
                               optimizer__nesterov=nesterov,
                               verbose=verbose,
                               )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.name = name
        self.task = task
        self.nJobs = 40

    def evalAlgo(self, param_dists):
        model, params, accuracy = None, None, None
        if self.task == 'regression':
            model, params, accuracy = self.searchAndEvalRegressor(param_dists)

        elif self.task == 'classification':
            model, params, accuracy = self.searchAndEvalClassifier(param_dists)

        return model, params, accuracy

    def evalRegressor(self):
        np.set_printoptions(threshold=np.inf)
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)

        print("mean squared error: ", mean_squared_error(y_pred, self.y_test))
        print("accuracy score: ", accuracy_score(y_pred, self.y_test))
        print(classification_report(self.y_test, y_pred))

        return y_pred

    def evalClassifier(self):
        np.set_printoptions(threshold=np.inf)
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)

        print("mean squared error: ", mean_squared_error(y_pred, self.y_test))
        print("accuracy score: ", accuracy_score(y_pred, self.y_test))
        print(classification_report(self.y_test, y_pred))

        return y_pred

    def searchAndEvalClassifier(self, param_dists):
        # gs = RandomizedSearchCV(self.model, param_dists, random_state=0, cv=5, n_iter=600, n_jobs=self.nJobs)
        gs = RandomizedSearchCV(self.model, param_dists, random_state=0, cv=3, n_iter=600, verbose=0)
        # gs = GridSearchCV(self.model, param_dists, scoring='f1_micro', cv=3)
        # classifier = GridSearchCV(dt, param_dists, scoring='f1_micro', cv=5)

        gs.fit(self.X_train, self.y_train)
        # print('------------ Best -------------')
        # print(gs.best_score_, gs.best_params_)
        bmodel = gs.best_estimator_
        pred = bmodel.predict(self.X_test)
        accuracy = accuracy_score(pred, self.y_test)

        dummy = DummyClassifier(strategy='most_frequent').fit(self.X_train, self.y_train)
        # print("Baseline_Accuracy: {}".format(accuracy_score(dummy.predict(self.X_test), self.y_test)))
        # print('**** Classifier ****')

        # print("accuracy score: ", accuracy)
        return bmodel, gs.best_params_, accuracy

    def searchAndEvalRegressor(self, param_dists):
        # gs = RandomizedSearchCV(self.model, param_dists, random_state=0, cv=5, n_iter=500, n_jobs=self.nJobs)
        gs = RandomizedSearchCV(self.model, param_dists, random_state=0, cv=3, n_iter=600)
        gs.fit(self.X_train, self.y_train)
        print("best parameters found: ", gs.best_params_)
        pred = gs.predict(self.X_test)
        print("mean squared error: ", mean_squared_error(pred, self.y_test))
        # self.reg_to_class(pred)
        # print('**** Regressor ****')
        bmodel = gs.best_estimator_
        # print('weights: ', bmodel.coefs_)
        # print('intercepts: ', bmodel.intercepts_)
        # print('act: ', bmodel.activation)
        # print('outact: ', bmodel.out_activation_)
        dump(bmodel, self.name + '_reg.joblib')
        # print("####")
        # pred=bmodel.predict(self.X_test)
        # self.reg_to_class(pred)
        # print("####")
        return bmodel


if __name__ == '__main__':
    input_ = torch.rand(100, 2)
    net = MLP([2, 10, 5, 20])
    out_ = net(input_)
    print(out_.shape)
