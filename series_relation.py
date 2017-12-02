from param_prepare import *
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import f_regression as fg, mutual_info_regression as mg

sheet = pd.read_excel('StockPriceExercise.xlsx', sheet_name='Sheet1')

#   Read in data and target

target = sheet.loc[0:150, ['KONAMI']].values.ravel()

# target = sheet.loc[0:150, ['Date']].values.ravel()


data = np.array([])

count=0
for i in range(0, 630, 30):
    data = np.append(data, sheet.loc[i:i+150, ['KONAMI']].values.ravel())
    # data = np.append(data, sheet.loc[i:i+150, ['Date']].values.ravel())
    count = count+1

# cut a 1d array in to 20 lists, each with size 151. Each list is seen as a row in matrix. The matrix is a 20 row-151 col matrix
data = data.reshape(count, 151)

# transpose the matrix, to be a 151 row-20 col matrix. Each col serves as a data series.
data = data.transpose()
print(data)
f_test, _ = fg(X=data, y=target)
f_test /= np.max(f_test)

mi = mg(X=data, y=target)
mi /= np.max(mi)

for i in range(len(mi)):
    print("Data series ", str(i), "f_relation=", f_test[i], "mutual info ratio=", mi[i])

ax = plt.subplot()
ax.set_xlabel("Months ago")
ax.set_ylabel("Mutual information ratio")
ax.plot(range(len(mi)), mi)
plt.title("Mutual information of data from n months ago")
plt.show()

