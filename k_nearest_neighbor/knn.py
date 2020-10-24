import numpy as np

def euclidean_distance(x1, x2):
    # calculating the euclidean distance for our dataset
    return np.sqrt(np.sum((x1-x2)**2))

def most_common(word_list):
    # calculates the most common element in a list
    return max(set(word_list), key = word_list.count)

def accuracy(y_true, y_pred):
    # accuracy is calculated by counting how many right labels we got
    # and dividing it by the total labels
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

class KNN:

    def __init__(self, k=3):
        # setting the default k to 3
        self.k = k

    def fit_model(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # running prediction on every row and returning it as numpy array
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, row):
        # compute distance between every column’s value
        distances = [euclidean_distance(row, x_train) for x_train in self.X_train]
        # sorting all the distances, then getting the k nearest samples
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # majority vote, most common class label
        return most_common(k_nearest_labels)
