from sklearn import linear_model
from param_prepare import *
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing
from sklearn.model_selection import learning_curve
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_predict


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
    return;


sheet = pd.read_excel('StockPriceExercise.xlsx', sheet_name='Sheet1')

#   Read in data and target
data = sheet.loc[:, ['Date', 'KONAMI', 'EA']].values
target = sheet.loc[:, ['NINTENDO']].values.astype(int)

#   Standardization
data = preprocessing.scale(data)

# Parameter Tuning
#  First, list the parameters you want to tune, and their corresponding value range.
para_dict = xgb_params()
model = xgb.XGBRegressor()
model = RandomizedSearchCV(model, param_distributions=para_dict, n_iter=5)
# report(model.cv_results_)

# Training
prediction = cross_val_predict(model, X=data, y=target.ravel(), cv=10)
train_sizes, train_scores, valid_scores = learning_curve(model, X=data, y=target.ravel(), cv=10)


# Plot result
ax = plt.subplot()
ax.scatter(sheet['Date'].values.tolist(), prediction.tolist(), edgecolors=(0, 0, 0), label="predicted price ")

ax.plot(sheet['Date'].values.tolist(), sheet['NINTENDO'].values.tolist(), 'k--', lw=2, label="actual price")
ax.set_xlabel('Date')
ax.set_ylabel('Price')

plt.legend()
# plt.savefig("Prediction.png")
plt.show()
plt.figure()
fig = plt.subplot()
fig.plot(train_scores[0], lw=2, label="Train Score")
fig.plot(valid_scores[0], lw=2, label="Validation Score")
plt.legend()
plt.savefig("Learning Curve.png")
# xgb.plot_importance(model).figure.savefig("importance.png")
