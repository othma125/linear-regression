#!/usr/bin/env python
import pandas as pd
import numpy as np
advertising = pd.read_csv("advertising.csv")

import seaborn as sns
# sns.pairplot(advertising, x_vars=['TV'], y_vars='Sales', height=7, aspect=0.7, kind='reg')

X = advertising['TV']
y = advertising['Sales']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 , random_state=100)

X_train = X_train[:, np.newaxis]
X_test = X_test[:, np.newaxis]

from sklearn.linear_model import LinearRegression   
lr = LinearRegression()
lr.fit(X_train, y_train)

print(lr.intercept_)
print(lr.coef_)

y_pred = lr.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)

# import matplotlib.pyplot as plt

# plt.scatter(X_test, y_test)
# plt.plot(X_test, y_pred, color='red')
# plt.show()
