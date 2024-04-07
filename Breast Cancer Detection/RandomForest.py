from DecisionTree import DecisionTree
import numpy as np

class RandomForest:

    def __init__(self, No_trees = 10, min_samples_needed_for_split = 2, max_depth = 100, No_features = None):
        self.No_trees = No_trees
        self.min_samples_needed_for_split = min_samples_needed_for_split
        self.max_depth = max_depth
        self.No_features = No_features
        self.trees = []
    
    def fit(self, x, y):
        self.trees = []
        for _ in range(self.No_trees):
            # tree is nothing but a DecisionTree classifier which was implemented in DecisionTree.py
            tree = DecisionTree(
                max_depth = self.max_depth, 
                min_samples_needed_for_split = self.min_samples_needed_for_split, 
                No_features = self.No_features
                )
            x_samples, y_samples = self.bootStrapSamples(x, y) # boot strap samples
            tree.fit(x_samples, y_samples)
            self.trees.append(tree)
    
    def bootStrapSamples(self, x, y):
        idxs = np.random.choice(x.shape[0], x.shape[0], replace = True) # Some samples are dropped randomly for every tree
        x_samples, y_samples = x[idxs], y[idxs]
        return x_samples, y_samples
    
    def predict(self, x):
        tree_predictions = []
        # the input data is given to each tree one after the other
        # and the predictions from each tree are stored in a array
        for tree in self.trees:
            predictions = tree.predict(x)
            tree_predictions.append(predictions)
        
        tree_predictions = np.array(tree_predictions)
        # print(tree_predictions)
        tree_predictions = np.swapaxes(tree_predictions, 0, 1)
        # print(tree_predictions)
        predictions = []
        for tree_pred in tree_predictions:
            y_predictions = tree.most_common_label(tree_pred)
            predictions.append(y_predictions)
        return np.array(predictions)