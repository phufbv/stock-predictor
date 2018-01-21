import quandl
import pandas as pd
import numpy as np
import datetime

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm

import warnings
warnings.filterwarnings("ignore")#, category=DeprecationWarning)


# Get share data
df = quandl.get("WIKI/AMZN")

df = df[['Adj. Close']]

forecast_out = int(30) # predicting 30 days into future
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out) #  label column with data shifted 30 units up

print(df.tail(50))


# Setup input/output arrays
X = np.array(df.drop(['Prediction'], 1))
X = preprocessing.scale(X)
X_forecast = X[-forecast_out:] # set X_forecast equal to last 30
X = X[:-forecast_out] # remove last 30 from X

y = np.array(df['Prediction'])
y = y[:-forecast_out]


# Do linear regression
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

clf = LinearRegression()
clf.fit(X_train,y_train) # training

confidence = clf.score(X_test, y_test) # testing fit to training data - get r^2 (coefficient of determination) reading
print("\nconfidence: ", confidence)


# Extrapolate to forecast future prices
forecast_prediction = clf.predict(X_forecast)
print(forecast_prediction)
