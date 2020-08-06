# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import timeit as timeit

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("power-plant.csv")
df_describe = df.describe()
df_head = df.head()
df_tail = df.tail()
df_corr = df.corr()

X = df.iloc[:, 0:4]
y = df.iloc[:, -1]

standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf = MLPRegressor(hidden_layer_sizes=(50,), max_iter=10000, solver='lbfgs', activation='logistic', verbose=True)
start_time = timeit.default_timer()
clf.fit(X_train, y_train)
end_time = timeit.default_timer()
elapsed_time = end_time - start_time
y_pred = clf.predict(X_test)
print(y_test)
print(np.reshape(y_pred, (-1, 1)))
print("r2: %.10f" % r2_score(y_test, y_pred))
print("MSE: %.5f" % mean_squared_error(y_test, y_pred))

'''
scaled logistic lbfgs=0.95 MSE=14.68
logistic lbfgs=-0.00000
scaled relu adam=0.9406 MSE=16.76 loss=9.046
relu adam=0.9134
scaled relu lbfgs=0.9429 MSE=15.92
relu lbjgs=0.9149
scaled logistic adam=0.94156 MSE=16.75 loss=8.82
logistic adam=0.93
'''
