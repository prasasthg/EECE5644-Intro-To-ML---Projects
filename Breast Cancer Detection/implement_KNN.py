from knn import KNN
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
np.random.seed(18)

if __name__ == "__main__":
    # Imports
    from matplotlib.colors import ListedColormap
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # K-Fold validation to find the best parameters
    num_folds = 10
    kf = KFold(n_splits=num_folds)
    best_k = None
    maxValidationAccuracy = float("-inf") # Assining a large value to maxValidationAccuracy (- infinity)
    # alphas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    average_accuracy = []

    for k in range(1, 11):
        fold_loss = []
        fold_accuracy = []
        print('---------------------------------------------------------------------')
        print(f"Number of Nearest Neighbours considered : {k} ")
        print('---------------------------------------------------------------------')
        model = KNN(k = k)

        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            # print(f'Fold {fold + 1}')
            # Split data into train and test sets for the current fold
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)    
            y_pred = model.predict(X_test)
            acc = accuracy(y_test, y_pred)
            print("Accuracy:", acc)
            fold_accuracy.append(acc)
        
        # Calculate the mean error across all folds for the current number of perceptrons
        mean_accuracy = np.mean(fold_accuracy)
        print(f'Mean Accuracy for {k} nearest neighbours after 10 fold validation : {mean_accuracy:.4f}')
        average_accuracy.append(mean_accuracy)    

        if (maxValidationAccuracy < mean_accuracy):
            maxValidationAccuracy = max(mean_accuracy, maxValidationAccuracy)
            best_k = k 

    print(f'Best nearest number of neighbours to consider = {best_k} : VALIDATION ACCURACY = {maxValidationAccuracy:.4f}')

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2 )
    clf = KNN( k = best_k )
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("KNN classification accuracy", accuracy(y_test, predictions))