from sklearn import linear_model
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
sheet=pd.read_excel('StockPriceExercise.xlsx',sheet_name='Sheet1')

# iris=load_iris()

# print(type(iris))

# print(sheet['Date'].values.tolist())
# print(sheet['Date'].values)
# plt.scatter(sheet['Date'].values.tolist(),sheet['NINTENDO'].values.tolist())
# plt.show()
# print(sheet.loc[1:,['Date','KONAMI','EA']])

# print(sheet.loc[1:,['NINTENDO']])
# model=linear_model.LogisticRegression()
model=xgb.XGBRegressor()

data=sheet.loc[15:,['KONAMI','EA']].values
labels=sheet.loc[15:,['NINTENDO']].values.astype(int)


model.fit(X=data,y=labels)

for i in range(0,15):
	print(i,model.predict([sheet.loc[i,['KONAMI','EA']]]))


for i in range(7001,7015):
	print(i,model.predict([sheet.loc[i,['KONAMI','EA']]]))

xgb.plot_importance(model).figure.savefig("importance.png")

