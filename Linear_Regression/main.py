import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

# print(diabetes.keys())
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])

# print(diabetes.data)

diabetes_X = diabetes.data[:, np.newaxis, 0]
diabetes_Y = diabetes.target
# print(diabetes_X)

diabetes_X_train = diabetes_X[-30:]
diabetes_Y_train = diabetes_Y[-30:]

diabetes_X_test = diabetes_X[:20]
diabetes_Y_test = diabetes_Y[:20]

# print(diabetes_X_train)
# print(diabetes_X_test)

model = linear_model.LinearRegression()

model.fit(diabetes_X_train, diabetes_Y_train)

diabetes_Y_predict = model.predict(diabetes_X_test)

print("Mean Squared Error is ", mean_squared_error(diabetes_Y_test, diabetes_Y_predict))

print("Weights ", model.coef_)
print("Intercepts ", model.intercept_)

# Mean Squared Error is  4207.271844639647
# Weights  [367.01062321]
# Intercepts  134.47955279902

plt.scatter(diabetes_X_test, diabetes_Y_test)
plt.plot(diabetes_X_test, diabetes_Y_predict)
plt.show()


