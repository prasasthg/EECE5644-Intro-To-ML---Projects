# from FILENAME import CLASSNAME
from DecisionTree import DecisionTree
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
    Optimal_treeDepth = 0
    average_accuracy = []

    for depth in range(1, 11):
        fold_loss = []
        fold_accuracy = []
        print('---------------------------------------------------------------------')
        print(f"Depth of the tree : {depth} ")
        print('---------------------------------------------------------------------')
        model = DecisionTree(max_depth = depth)

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
        print(f'Mean accuracy for Depth {depth} : {mean_accuracy:.4f}')
        average_accuracy.append(mean_accuracy)    

        if (maxValidationAccuracy < mean_accuracy):
            maxValidationAccuracy = max(mean_accuracy, maxValidationAccuracy)
            Optimal_treeDepth = depth 

    print(f'Optimal tree depth = {Optimal_treeDepth} : VALIDATION ACCURACY = {maxValidationAccuracy:.4f}')

    # Plot between depth and average accuarcy after K-Fold Validation
    import matplotlib.pyplot as plt
    # Plot the accuracies against the depths
    plt.plot(range(1, 11), average_accuracy, 'o-')
    plt.xlabel('Depth of Tree')
    plt.ylabel('Average Accuracy')
    plt.title('Accuracy vs. Depth of Tree')
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Building the model with optimal tree dpeth i.e.. best model
    tree = DecisionTree(max_depth = Optimal_treeDepth)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print("Accuracy:", acc)
    ConfusionMatrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(ConfusionMatrix)
    error = (1 - (ConfusionMatrix[0][0] + ConfusionMatrix[1][1]) / X_test.shape[0])
    print("Error:")
    print(error)