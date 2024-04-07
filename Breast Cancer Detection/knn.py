import numpy as np
from collections import Counter

def euclidean_distance(sample1, sample2):
    return np.sqrt(np.sum( ( sample1 - sample2) ** 2 ))

class KNN:
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        
        predictions = []
        for sample in X:
            # call _prediction helper funtion on each sample of the test data
            prediction = self._predict(sample)
            predictions.append(prediction)
        return np.array(predictions)
    
    def _predict(self, sample):
        
        distances = []
        # Calculate the distance between the sample and 
        # each sample in training data 
        for x in self.X_train:
            distance = euclidean_distance(sample, x)
            distances.append(distance)
        
        # sort the distances and extract the indexes 
        # of samples which are nearer to the test sample
        idxs = np.argsort(distances)[: self.k]

        # check the labels for those k nearest labels
        k_Nearest_Labels = []
        for i in idxs:
            labels = self.y_train[i]
            k_Nearest_Labels.append(labels)
        
        # check for the most common label among these k_Nearest_Labels
        most_common = Counter(k_Nearest_Labels).most_common(1)
        return most_common[0][0]