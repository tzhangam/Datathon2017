from statsmodels.graphics.tsaplots import *
from sklearn import linear_model
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import f_regression as fg, mutual_info_regression as mg
from math import log10
from math import log
from statsmodels.tsa.stattools import adfuller

def print_adf_result(result):

	print('ADF Statistic: %f' % result[0])
	print('p-value: %f' % result[1])
	print('Critical Values:')
	for key, value in result[4].items():
		print('\t%s: %.3f' % (key, value))

def get_diff_list(list,interval):
	l=[]
	if interval>0:
		for i in range(0,len(list)-interval):
			l.append(list[i+interval]-list[i])
	return l;

def get_log_diff_list(list,interval):
	l=[]
	print(list)
	if interval>0:
		for i in range(0,len(list)-interval):
			l.append(log10(list[i+interval])-log10(list[i]))
	return l;


sheet = pd.read_csv('godiva_data.csv')


#read in original sales data
target = sheet.loc[:, ['VIP_sales']].values.ravel().astype(float)


#eliminate nan values
target = target[~np.isnan(target)]

target=np.log10(target)

result=adfuller(target)

# print("Before difference")

print_adf_result(result)


plot_acf(target)

plt.show()

plot_pacf(target)

plt.show()
# for interval in range(1,20):
# 	t=get_log_diff_list(target,interval)

# 	result=adfuller(t)

# 	print("After difference "+str(interval))

# 	print_adf_result(result)

# 	plot_acf(t)

# 	plt.show()

# 	plot_pacf(t)

# 	plt.show()