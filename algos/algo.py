import sys

import numpy as np

np.set_printoptions(threshold=sys.maxsize)
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.dummy import DummyClassifier

from joblib import dump


class Algo(object):
    def __init__(self, X_train, X_test, y_train, y_test, name):
        self.name = name
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.njobs = 40

    def reg_to_class(self, reg_preds):
        new_preds = []
        for reg_pred in reg_preds:
            new_preds.append(min(max(self.y_train), max(min(self.y_train), round(reg_pred))))
        new_preds = np.array(new_preds)

        print("new mean squared error: ", mean_squared_error(new_preds, self.y_test))
        print("new accuracy score: ", accuracy_score(new_preds, self.y_test))
        print(classification_report(self.y_test, new_preds))

    def evalClassifier(self, clf):
        np.set_printoptions(threshold=np.inf)
        search = clf.fit(self.X_train, self.y_train)
        pred = clf.predict(self.X_test)

        print("mean squared error: ", mean_squared_error(pred, self.y_test))
        print("accuracy score: ", accuracy_score(pred, self.y_test))
        print(classification_report(self.y_test, pred))
        bmodel = clf
        return pred

    def searchAndEvalClassifier(self, clf, param_dists):
        classifier = RandomizedSearchCV(clf, param_dists, random_state=0, cv=5, n_iter=600, n_jobs=self.njobs)
        # classifier = GridSearchCV(dt, param_dists, scoring='f1_micro', cv=5)

        search = classifier.fit(self.X_train, self.y_train)
        pred = classifier.predict(self.X_test)

        print("best parameters found: ", search.best_params_)
        print("mean squared error: ", mean_squared_error(pred, self.y_test))
        print("accuracy score: ", accuracy_score(pred, self.y_test))
        print(classification_report(self.y_test, pred))

        dummy = DummyClassifier(strategy='most_frequent').fit(self.X_train, self.y_train)
        print("Baseline_Accuracy: {}".format(accuracy_score(dummy.predict(self.X_test), self.y_test)))
        # print('**** Classifier ****')
        bmodel = classifier.best_estimator_

        pred = bmodel.predict(self.X_test)
        accuracy = accuracy_score(pred, self.y_test)
        print("accuracy score: ", accuracy)
        return bmodel, search.best_params_, accuracy

    # TODO: clean
    def evalRegressor(self, reg):
        np.set_printoptions(threshold=np.inf)
        # print("X_test after scale:",self.X_test)
        search = reg.fit(self.X_train, self.y_train)
        # print('Intercept: ')
        # print(reg.intercept_)
        pred = reg.predict(self.X_test)
        # print("Pred:",pred)
        print("mean squared error: ", mean_squared_error(pred, self.y_test))
        self.reg_to_class(pred)
        bmodel = reg
        dump(bmodel, self.name + '_reg_nosearch.joblib')
        return pred

    # TODO: clean
    def searchAndEvalRegressor(self, reg, param_dists):
        classifier = RandomizedSearchCV(reg, param_dists, random_state=0, cv=5, n_iter=500, n_jobs=self.njobs)
        search = classifier.fit(self.X_train, self.y_train)
        print("best parameters found: ", search.best_params_)
        pred = classifier.predict(self.X_test)
        print("mean squared error: ", mean_squared_error(pred, self.y_test))
        self.reg_to_class(pred)
        # print('**** Regressor ****')
        bmodel = classifier.best_estimator_
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
