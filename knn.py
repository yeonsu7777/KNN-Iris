import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # calculate the distances using euclidean_distance function
        distances = [euclidean_distance(x, x_train) for x_train in self.train_X]
        # argsort returns an array of indices of the same shape as arr
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.train_y[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)  #Counter.most_common returns the most common element and its number
        return most_common[0][0]    #only need the most common element


class weightedKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # calculate the distances using euclidean_distance function
        distances = [euclidean_distance(x, x_train) for x_train in self.train_X]
        k_indices = np.argsort(distances)[:self.k]  # returns smallest distance's index number in arr
        k_nearest_labels = [self.train_y[i] for i in k_indices]
        weights = [1 / distances[i] for i in k_indices]  # weight = 1/distance
        sum_of_weights = [0] * self.train_y.size # actually need only 3 but make array size big enough
        for i in range(len(k_nearest_labels)):
            sum_of_weights[k_nearest_labels[i]] += weights[i]

        return np.argmax(sum_of_weights)
