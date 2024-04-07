import numpy as np

class NaiveBayes:
    def __init__(self, alpha=1e-9):
        self.alpha = alpha
    
    def fit(self, X, y):
        No_samples, No_features = X.shape
        self.classes = np.unique(y)
        No_classes = len(self.classes)

        # Compute Mean, Variance, Priors
        self.mean = np.zeros((No_classes, No_features), dtype = np.float64)
        self.variance = np.zeros((No_classes, No_features), dtype = np.float64)
        self.priors = np.zeros((No_classes), dtype = np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis = 0)
            self.variance[idx, :] = X_c.var(axis = 0)
            self.priors[idx] = X_c.shape[0] / float(No_samples)

    def predict(self, X):
        y_predictions = []
        for sample in X:
            prediction = self.pred(sample)
            y_predictions.append(prediction)
        return np.array(y_predictions)

    def pred(self, sample):
        posteriors = []

        # Calculate posterios probability for each class
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self.pdf(idx, sample)))
            posterior = prior + posterior
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)] # Returning the class with highest posterior probability

    def pdf(self, idx, sample):
        mean = self.mean[idx]
        var = self.variance[idx]
        numerator = np.exp( -((sample - mean) ** 2) / (2 * var) )
        denominator = np.sqrt( 2 * np.pi * var )
        return numerator / denominator

    def pdf(self, idx, sample):
        mean = self.mean[idx]
        var = self.variance[idx]
        # alpha is Small positive value for Laplace smoothing
        numerator = np.exp( -((sample - mean) ** 2) / (2 * var + self.alpha) )
        denominator = np.sqrt( 2 * np.pi * (var + self.alpha) )
        return numerator / denominator
