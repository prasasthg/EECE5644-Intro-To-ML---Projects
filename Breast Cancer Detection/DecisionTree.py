import numpy as np
from collections import Counter

class Node:
    
    def __init__(self, feature = None, threshold = None, left = None, right = None, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:

    def __init__(self, max_depth = 100, min_samples_needed_for_split = 2, No_features = None):
        self.min_samples_needed_for_split = min_samples_needed_for_split
        self.max_depth = max_depth
        self.No_features = No_features
        self.root = None

    def  fit(self, x, y):
        self.root = self.grow_tree(x, y)
    
    def predict(self, x):
        predictions = []
        for each_sample in x:
            predcition = self.traverseTree(each_sample, self.root)
            predictions.append(predcition)
        return np.array(predictions)

    def grow_tree(self, x, y, depth = 0):

        #  Total number of samples and features in the dataset
        samples, features = x.shape
        labels = np.unique(y)
        n_lables = len(labels)

        # stopping criteria
        if (depth > self.max_depth or n_lables == 1 or samples < self.min_samples_needed_for_split):
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)

        # Taking required number of features for splitting the data
        if not self.No_features:
            self.No_features = x.shape[1]
        else:
             self.No_features = min(x.shape[1], self.No_features)
        
        # Getting the feature indexes randomly without replacement
        feature_idxs = np.random.choice(features, self.No_features, replace = False)

        #  Computing the best criteria (best feature and best threshold) for splitting among all the selected features
        best_feat, best_threshold = self.best_criteria(x, y, feature_idxs)

        # Splitting based on the best feature and best threshold
        left_idxs, right_idxs = self.split(best_threshold, x[:, best_feat])

        # Recursively calling the grow tree for left and right trees
        left = self.grow_tree(x[left_idxs, :], y[left_idxs], depth+1)
        right = self.grow_tree(x[right_idxs, :], y[right_idxs], depth+1)

        return Node(best_feat, best_threshold, left, right)
    
    def best_criteria(self, x, y, feature_idxs):
        best_gain = -1
        best_feature_idx = None
        best_threshold = None
        for ftr_idx in feature_idxs:
            x_column = x[:, ftr_idx]    # Considering all the thresholds / values in that feature i.e.. one entire column.
            thresholds = np.unique(x_column)
            for thr in thresholds:  # Looping throgh each threshold and calculating gain by splittig data on that threshold
                gain = self.information_gain(thr, x_column, y)

                if (gain > best_gain):
                    best_gain = gain
                    best_feature_idx = ftr_idx
                    best_threshold = thr

        return best_feature_idx, best_threshold

    def information_gain(self, thr, x_column, y):
        entropy_parent = self.entropy(y)
        left_idxs, right_idxs = self.split(thr, x_column)
        n = len(y) # Total number of samples
        n_L = len(left_idxs) # Number of samples in the left child
        n_R = len(right_idxs) # Number of samples in the right child

        if n_L == 0 or n_R == 0:
            return 0
        
        entropy_leftChild = self.entropy(y[left_idxs])
        entropy_rightChild = self.entropy(y[right_idxs])
        entropy_Child = (n_L/n) * entropy_leftChild + (n_R/n) * entropy_rightChild
        # gain is difference in loss before vs. after split
        gain = entropy_parent - entropy_Child
        return gain

    def split(self, thr, x_column):
        left_idxs = np.argwhere(x_column <= thr).flatten()
        right_idxs = np.argwhere(x_column > thr).flatten()
        return left_idxs, right_idxs
        
    def traverseTree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverseTree(x, node.left)
        return self.traverseTree(x, node.right)

    def most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])