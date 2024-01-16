from sklearn.linear_model import LogisticRegression
from algos.algo import Algo
from scipy.stats import reciprocal, uniform, expon

class LogReg(Algo):
    def __init__(self, X_train, X_test, y_train, y_test, max_iter=100):
        self.max_iter = max_iter
        super(LogReg, self).__init__(X_train, X_test, y_train, y_test)

    def evalAlgo(self):
        print("")
        print("Logistic Regression Multinomial w/o Random Parameter Search")
        #lr = LogisticRegression(max_iter=self.max_iter, multi_class='multinomial', random_state=0)
        lr = LogisticRegression(max_iter=self.max_iter, solver='newton-cg', multi_class='multinomial', random_state=0)
        self.evalRegressor(lr)

        print("")
        print("Logistic Regression w/ Random Parameter Search")
        lr = LogisticRegression(max_iter=self.max_iter, multi_class='multinomial', random_state=0)
        param_dists = dict( 
            C = expon(scale=100),
            #penalty = ['l1', 'l2', 'elasticnet', 'none'], # not supported for newton-cg
            penalty = ['l2'],
            #dual = [True, False], # not supported for sag
            dual = [False],
            fit_intercept = [True, False],
            intercept_scaling = uniform(0.1, 10),
            solver = ['newton-cg', 'lbfgs', 'sag', 'saga'],
            #l1_ratio = uniform(0, 1)
        )
        self.searchAndEvalRegressor(lr, param_dists)

        print("")
        print("Logistic Classification OvR w/o Random Parameter Search")
        lr = LogisticRegression(max_iter=self.max_iter, multi_class='ovr', random_state=0)
        self.evalClassifier(lr)

        print("Logistic Classification w/ Random Parameter Search")
        lr = LogisticRegression(max_iter=self.max_iter, multi_class='ovr', random_state=0)
        param_dists = dict( 
            C = expon(scale=100),
            penalty = ['l2'], # not supported for newton-cg
            #dual = [True, False], # not supported for other than l2
            dual = [False],
            fit_intercept = [True, False],
            intercept_scaling = uniform(0.1, 10),
            solver = ['newton-cg', 'lbfgs', 'sag', 'saga'],
            #l1_ratio = uniform(0, 1)
        )
        self.searchAndEvalClassifier(lr, param_dists)


