from sklearn import linear_model
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import f_regression as fg, mutual_info_regression as mg

sheet = pd.read_csv('godiva_data.csv')

interval_length=5
#   Read in data and target

target = sheet.loc[10:10+interval_length-1, ['sold_count_Bar_selftreat']].values.ravel().astype(float)


# target = sheet.loc[0:150, ['Date']].values.ravel()

print("target:",target)

data = np.array([])

count=0
# for i in range(0, 21, 1):
for i in range(0,22-interval_length):
    data = np.append(data, sheet.loc[i:i+interval_length-1, ['VIP_sales']].values.ravel().astype(float))
    # data = np.append(data, sheet.loc[i:i+150, ['Date']].values.ravel())
    count=count+1

# print(count)

# print(data)
# cut a 1d array in to 20 lists, each with size 151. Each list is seen as a row in matrix. The matrix is a 20 row-151 col matrix
data = data.reshape(count, interval_length)

# print(data)
# transpose the matrix, to be a 151 row-20 col matrix. Each col serves as a data series.
data=data.transpose()

print("data.size()",len(data),data)
# target = target.transpose()
# print(target)
# print(target.ravel())
# f_test, _ = fg(X=data, y=target.ravel())
# f_test /= np.max(f_test)

mi = mg(X=data, y=target)
mi /= np.max(mi)

# for i in range(len(mi)):
#     print("Data series ", str(i), "f_relation=", f_test[i],"mutual info ratio=", mi[i])

ax = plt.subplot()
ax.set_xlabel("Month")
ax.set_ylabel("Mutual information ratio")
ax.plot(range(1,len(mi)+1), mi)
plt.title("Mutual information of data from Jan")
plt.show()

