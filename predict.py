'''
predict: verify model's performance on unseen data
'''

import pandas as pd
from sklearn.metrics import mean_absolute_error
from skops.io import load

## load prepared data
X_test = pd.read_csv('data/feat_eng/X_test.csv')
y_test = pd.read_csv('data/feat_eng/y_test.csv')

## load baseline and model
baseline = load("data/modelling/baseline.skops", trusted=[])
model    = load("data/modelling/model.skops", trusted=[])

## see how baseline perform
baseline_pred = baseline.predict(X_test)
baseline_score = mean_absolute_error(y_test, baseline_pred)

print(f'Dummy Regression MAE: {baseline_score:.2f}')

## see how model perform
model_pred = model.predict(X_test)
model_score = mean_absolute_error(y_test, model_pred)

print(f'GBT MAE             : {model_score:.2f}')