from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from algos.algo import Algo
from scipy.stats import reciprocal, uniform, expon

class RandomForest(Algo):
    def __init__(self, X_train, X_test, y_train, y_test, max_depth=10):
        self.max_depth = max_depth
        super(RandomForest, self).__init__(X_train, X_test, y_train, y_test)

    def evalAlgo(self):
        print("")
        print("Random Forest Regressor w/o Random Parameter Search")
        dt = RandomForestRegressor(max_depth=self.max_depth, random_state=0)
        self.evalRegressor(dt)

        print("")
        print("Random Forest Regressor w/ Random Parameter Search")
        dt = RandomForestRegressor(max_depth=self.max_depth, random_state=0)
        param_dists = dict( 
            criterion = ['mse', 'friedman_mse', 'mae'],
            min_samples_split = range(2, 10),
            min_samples_leaf = range(1, 10),
            min_impurity_decrease = reciprocal(1e-20, 1e-10),
            max_features = ['auto', 'sqrt', 'log2'],
            bootstrap = [True, False],
            #oob_score = [True, False]
        )
        self.searchAndEvalRegressor(dt, param_dists)

        print("")
        print("Random Forest w/o Random Parameter Search")
        dt = RandomForestClassifier(max_depth=self.max_depth, random_state=0)
        self.evalClassifier(dt)

        print("Random Forest w/ Random Parameter Search")
        dt = RandomForestClassifier(max_depth=self.max_depth, random_state=0)
        param_dists = dict( 
            criterion = ['gini', 'entropy'],
            min_samples_split = range(2, 10),
            min_samples_leaf = range(1, 10),
            min_impurity_decrease = reciprocal(1e-20, 1e-10),
            max_features = ['auto', 'sqrt', 'log2'],
            bootstrap = [True, False],
            #oob_score = [True, False]
        )
        self.searchAndEvalClassifier(dt, param_dists)

