from algos.algo import Algo
from sklearn.neural_network import MLPRegressor, MLPClassifier
from scipy.stats import reciprocal, uniform, expon


class MLP(Algo):
    def __init__(self, X_train, X_test, y_train, y_test, name, hidden_layer_sizes=12, activation='relu', max_iter=100):
        # redwine=3;
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.activation = activation
        super(MLP, self).__init__(X_train, X_test, y_train, y_test, name)

    def evalAlgo(self):
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

        print("MLP w/o Random Parameter Search")
        mlpc0 = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter)
        self.evalClassifier(mlpc0)
        print("MLP w/ Random Parameter Search")
        mlpcS = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter)
        param_dists = dict(
            # activation = ['identity', 'logistic', 'tanh', 'relu'],
            solver=['lbfgs', 'sgd', 'adam'],
            learning_rate=['constant', 'invscaling', 'adaptive'],
            momentum=uniform(0, 1),
            nesterovs_momentum=[True, False],
            validation_fraction=uniform(0, 1),
            beta_1=uniform(0, 0.999),
            beta_2=uniform(0, 0.999),
            epsilon=uniform(0, 0.999)
        )
        bmodel, params, accuracy = self.searchAndEvalClassifier(mlpcS, param_dists)
        return bmodel, params, accuracy
