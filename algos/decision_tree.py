from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from scipy.stats import reciprocal
from algos.algo import Algo
from sklearn.tree import export_graphviz
from sklearn import tree
import graphviz
import pydotplus

class DecisionTree(Algo):
    def __init__(self, X_train, X_test, y_train, y_test, max_depth=8):
        self.max_depth = max_depth
        super(DecisionTree, self).__init__(X_train, X_test, y_train, y_test)

    def evalAlgo(self):
        #print("")
        #print("Decision Tree Regressor w/o Random Parameter Search")
        #dt = DecisionTreeRegressor(max_depth=self.max_depth, random_state=0)
        #self.evalRegressor(dt)

        #print("")
        #print("Decision Tree Regressor w/ Random Parameter Search")
        #dt = DecisionTreeRegressor(max_depth=self.max_depth, random_state=0)
        #param_dists = dict( 
        #    criterion = ['mse', 'friedman_mse', 'mae'],
        #    splitter = ['best', 'random'],
        #    min_samples_split = range(2, 10),
        #    min_samples_leaf = range(1, 10),
        #    min_impurity_decrease = reciprocal(1e-20, 1e-10),
        #    max_features = ['auto', 'sqrt', 'log2']
        #)
        #self.searchAndEvalRegressor(dt, param_dists)

        print("")
        print("Decision Tree w/o Random Parameter Search")
        dt = DecisionTreeClassifier(max_depth=self.max_depth, random_state=0)
        self.evalClassifier(dt)

        print("Decision Tree w/ Random Parameter Search")
        dt = DecisionTreeClassifier(max_depth=self.max_depth, random_state=0)
        param_dists = dict( 
            criterion = ['gini', 'entropy'],
            splitter = ['best', 'random'],
            min_samples_split = range(2, 10),
            min_samples_leaf = range(1, 10),
            min_impurity_decrease = reciprocal(1e-20, 1e-10),
            max_features = ['auto', 'sqrt', 'log2']
        )
        self.searchAndEvalClassifier(dt, param_dists)
        #tree.plot_tree(dt.fit(self.Xtrain,self.y_train))
        '''
        dt.fit(self.X_train,self.y_train)
        export_graphviz(dt, out_file="mytree.dot")
        with open("mytree.dot") as f:
            dot_graph = f.read()
        #graph = graphviz.Source(dot_graph)
        graph = pydotplus.graph_from_dot_data(dot_graph)
        graph.write_png('my.png')
        #print("Hey number of nodes are {}".format(dt.tree_.node_count))
        '''
