import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

lr_model = LogisticRegression()

# fit the model
lr_model.fit(X, y)

# make predictions
y_pred = lr_model.predict(X)
print("Prediction on training set:", y_pred)

# calculate accuracy
print("Accuracy on training set:", lr_model.score(X, y))