from RandomForest import RandomForest
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import numpy as np
np.random.seed(42)

if __name__ == "__main__":
    # Imports
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    # K-Fold validation to find the best parameters
    num_folds = 10
    kf = KFold(n_splits=num_folds)

    maxValidationAccuracy = float("-inf") # Assining a large value to maxValidationAccuracy (- infinity)
    average_accuracy = []
    optimal_depth = 5

    for trees in range(2, 11):
        fold_loss = []
        fold_accuracy = []
        print('---------------------------------------------------------------------')
        print(f"Number of trees : {trees} ")
        print('---------------------------------------------------------------------')
        model = RandomForest(No_trees = trees, max_depth = optimal_depth)

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
        print(f'Mean accuracy for {trees} Trees : {mean_accuracy:.4f}')
        average_accuracy.append(mean_accuracy)    

        if (maxValidationAccuracy < mean_accuracy):
            maxValidationAccuracy = max(mean_accuracy, maxValidationAccuracy)
            Optimal_No_trees = trees 

    print(f'Optimal Number of trees = {Optimal_No_trees} : VALIDATION ACCURACY = {maxValidationAccuracy:.4f}')

    # Plot between depth and average accuarcy after K-Fold Validation
    import matplotlib.pyplot as plt
    # Plot the accuracies against the depths
    plt.plot(range(2, 11), average_accuracy, 'o-')
    plt.xlabel('Number of Trees')
    plt.ylabel('Average Accuracy')
    plt.title('Accuracy vs. Number of Trees')
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    forest = RandomForest(No_trees = Optimal_No_trees, max_depth = optimal_depth)
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print("Accuracy:", acc)
    ConfusionMatrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(ConfusionMatrix)
    error = (1 - (ConfusionMatrix[0][0] + ConfusionMatrix[1][1]) / X_test.shape[0])
    print("Error:")
    print(error)
