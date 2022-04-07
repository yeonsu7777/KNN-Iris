import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target
y_name = iris.target_names

to_delete = [i for i in range(1, 150) if i % 15 == 14]
train_X = np.delete(X, to_delete, axis=0)
test_X = X[14::15]
train_y = np.delete(y, to_delete, axis=0)
test_y = y[14::15]


# training KNN majority vote model
from knn import KNN

model_majority = KNN(k=3)  # can change k into 5, 7, or 9
model_majority.fit(train_X, train_y)
result_majority = model_majority.predict(test_X)

for i in range(len(result_majority)):
    print("Test Data Index: ", i, "Computed class: ", y_name[result_majority[i]], ", True class: ", y_name[test_y[i]])

acc = np.sum(result_majority == test_y) / len(test_y) * 100
print(acc)

from knn import weightedKNN

model_weighted = weightedKNN(k=3)   # can change k into 5, 7, or 9
model_weighted.fit(train_X, train_y)
result_weighted = model_weighted.predict(test_X)

for i in range(len(result_weighted)):
    print("Test Data Index: ", i, "Computed class: ", y_name[result_weighted[i]], ", True class: ", y_name[test_y[i]])

acc = np.sum(result_weighted == test_y) / len(test_y) * 100
print("accuracy: ", acc)
