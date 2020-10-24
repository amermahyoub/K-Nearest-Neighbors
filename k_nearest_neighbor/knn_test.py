import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

from knn import KNN, accuracy

# loading our dataset from sklearn's library
iris = datasets.load_iris()
X, y = iris.data, iris.target

# dividing our dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inspect data

print(X_train.shape)
print(X_train[0])

print(y_train.shape)
print(y_train)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()

# fitting and testing our own knn model
k = 7
model = KNN(k=k)
model.fit_model(X_train, y_train)
predictions = model.predict(X_test)
print(predictions)
print("Our KNN classification accuracy", accuracy(y_test, predictions))

# sklearn's knn model
clf = KNeighborsClassifier(7)
clf.fit(X_train, y_train)
clf_predictions = clf.predict(X_test)
print("sklearn's KNN classification accuracy", accuracy(y_test, clf_predictions))