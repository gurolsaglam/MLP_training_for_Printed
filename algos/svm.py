from sklearn.svm import SVC, SVR
from algos.algo import Algo
from scipy.stats import reciprocal, uniform, expon
import numpy as np
class SVM(Algo):
    def __init__(self, X_train, X_test, y_train, y_test, name, kernel='linear'):
        self.kernel = kernel
        self.max_iter=-1
        #self.decision_function_shape= decision_function_shape
        super(SVM, self).__init__(X_train, X_test, y_train, y_test, name)

    def evalAlgo(self):

        print("")
        print("SVM Classifier w/o Random Parameter Search")
        svm = SVC(kernel=self.kernel, max_iter=self.max_iter)
        self.evalClassifier(svm)

        print("SVM Classifier w/ Random Parameter Search")
        svm = SVC(kernel=self.kernel, max_iter=self.max_iter)
        param_dists = dict(
            C = expon(scale=100),
            shrinking = [True, False]
        )
        if self.kernel == 'poly': param_dists.degree = range(2,5)
        if self.kernel in ['rbf', 'poly', 'sigmoid']: param_dists.gamma = ['scale', 'auto']
        class_preds = self.searchAndEvalClassifier(svm, param_dists)
    
        print("")
        print("SVM Regressor w/o Random Parameter Search")
        svm = SVR(kernel=self.kernel, max_iter=self.max_iter)
        self.evalRegressor(svm)
        print("")
        print("SVM Regressor w/ Random Parameter Search")
        svm = SVR(kernel=self.kernel, max_iter=self.max_iter)
        param_dists = dict(
            C = expon(scale=100),
            shrinking = [True, False],
            epsilon = uniform(0.01, 0.2)
        )
        if self.kernel == 'poly': param_dists.degree = range(2,5)
        if self.kernel in ['rbf', 'poly', 'sigmoid']: param_dists.gamma = ['scale', 'auto']
        reg_preds = self.searchAndEvalRegressor(svm, param_dists)

