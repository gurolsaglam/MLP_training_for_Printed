import torch
import torch.nn as nn
import torch.nn.functional as F

from skorch import NeuralNetClassifier, NeuralNetRegressor

from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.dummy import DummyClassifier

import numpy as np


class MLP(nn.Module):
    def __init__(self, topology, activation='relu', max_iter=100):
        super(MLP, self).__init__()
        self.topology = topology
        self.max_iter = max_iter
        self.blocks = nn.ModuleList()
        self.activation_block = F.relu if activation == 'relu' else F.tanh

        for i in range(len(topology)):
            self.blocks.append(nn.Linear(topology[i], topology[i + 1]))
            if (i + 1) == (len(topology) - 1):
                break

        self.model = nn.Sequential(self.blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
            x = self.activation_block(x)
        out = F.softmax(x, dim=-1)
        return out


class Algo:
    def __init__(self, model, X_train, X_test, y_train, y_test, name, task, max_epochs):
        self.model = NeuralNetClassifier(model,
                                         max_epochs=max_epochs,
                                         lr=0.1,
                                         ) if task == "classification" else \
            NeuralNetRegressor(model,
                               max_epochs=max_epochs,
                               lr=0.1,
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
            pass
            # print("MLPRegressor Regressor w/ Random Parameter Search")
            # mlprS = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter)
            # param_dists = dict(
            # #activation = ['identity', 'logistic', 'tanh', 'relu'],
            # solver = ['lbfgs', 'sgd', 'adam'],
            # learning_rate = ['constant', 'invscaling', 'adaptive'],
            # momentum = uniform(0,1),
            # nesterovs_momentum = [True, False],
            # validation_fraction = uniform(0,1),
            # beta_1 = uniform(0,0.999),
            # beta_2 = uniform(0,0.999),
            # epsilon = uniform(0,0.999)
            # )
            # bmodel=self.searchAndEvalRegressor(mlprS,param_dists)
            # print('**** Regressor ****')
            # #print('weights: ', bmodel.coefs_)
            # #print('intercepts: ', bmodel.intercepts_)
            # #print('act: ', bmodel.activation)
            # #print('outact: ', bmodel.out_activation_)

            # print("")
            # print("MLPRegressor w/o Random Parameter Search")
            # mlpr0 = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter)
            # self.evalRegressor(mlpr0)

            # print("")
        elif self.task == 'classification':
            # param_dists = dict(
            #     # activation = ['identity', 'logistic', 'tanh', 'relu'],
            #     solver=['lbfgs', 'sgd', 'adam'],
            #     learning_rate=['constant', 'invscaling', 'adaptive'],
            #     momentum=uniform(0, 1),
            #     nesterovs_momentum=[True, False],
            #     validation_fraction=uniform(0, 1),
            #     beta_1=uniform(0, 0.999),
            #     beta_2=uniform(0, 0.999),
            #     epsilon=uniform(0, 0.999)
            # )

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
        # gs = RandomizedSearchCV(self.model, param_dists, random_state=0, cv=5, n_iter=600)
        gs = GridSearchCV(self.model, param_dists, scoring='f1_micro', cv=3)
        # classifier = GridSearchCV(dt, param_dists, scoring='f1_micro', cv=5)

        gs.fit(self.X_train, self.y_train)
        print('------------ Best -------------')
        print(gs.best_score_, gs.best_params_)
        bmodel = gs.best_estimator_

        dummy = DummyClassifier(strategy='most_frequent').fit(self.X_train, self.y_train)
        print("Baseline_Accuracy: {}".format(accuracy_score(dummy.predict(self.X_test), self.y_test)))
        # print('**** Classifier ****')

        pred = bmodel.predict(self.X_test)
        accuracy = accuracy_score(pred, self.y_test)
        print("accuracy score: ", accuracy)
        return bmodel, gs.best_params_, accuracy


if __name__ == '__main__':
    input_ = torch.rand(100, 2)
    net = MLP([2, 10, 5, 20])
    out_ = net(input_)
    print(out_.shape)
