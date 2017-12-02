import requests, pandas as pd, numpy as np
from pandas import DataFrame
from io import StringIO
import time, json
from datetime import date
import statsmodels
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

sheet = pd.read_csv('godiva_data.csv')


#read in original sales data
target_series = sheet.loc[:, ['VIP_sales']].values.ravel().astype(float)


#eliminate nan values
target_series = target_series[~np.isnan(target_series)]
print(target_series)
target_series=np.log10(target_series)
print(target_series)
size = int(len(target_series) - 12)
train, test = target_series[0:size], target_series[size:len(target_series)]
history = [x for x in train]
predictions = list()

print('Printing Predicted vs Expected Values...')
print('\n')
for t in range(len(test)):
    model = ARIMA(history, order=(1,0,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(float(10**yhat))
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (10**yhat, 10**obs))

test=[10**x for x in test]

error = mean_squared_error(test, predictions)

print('\n')
print('Printing Mean Squared Error of Predictions...')
print('Test MSE: %.6f' % error)


fig, ax = plt.subplots()

ax.scatter(range(0,len(target_series)),[10**x for x in target_series],label="Real Data")
ax.plot(range(size,len(target_series)),predictions,label="Prediction")

ax.legend()
plt.show()