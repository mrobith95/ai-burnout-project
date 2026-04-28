'''
modelling
'''

import pandas as pd
import os
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from skops.io import dump, get_untrusted_types

## load prepared data
X_train = pd.read_csv('data/feat_eng/X_train.csv')
y_train = pd.read_csv('data/feat_eng/y_train.csv')

## make y into 1d array
y_train = y_train.to_numpy().ravel()

## define cv split, set random_state for replication
kf = KFold(shuffle=True, random_state=300)
kf.split(X_train,y_train)

## consider baseline: mean
baseline = DummyRegressor()

## fit baseline
baseline.fit(X_train, y_train)

## now the main part: GradientBoostingReggresor
model = GradientBoostingRegressor(loss='absolute_error', ## use robust loss function
                                  ##learning_rate=0.1,
                                  n_estimators=1000, ##gradient boosting tend to yield better result when using more estimators
                                  ##max_depth=3,
                                  random_state=300, ##for replication
                                  ##n_iter_no_change=10 ##allow early stopping. no additional estimator fitted when there are no improvement after 10 iteration. The rest of early stopping settings are follow default settings.
                                  )

# set search space: here we look for learning_rate and max_speth
param = {
    'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.12, 0.15, 0.2],
    'max_depth': [1, 2, 3, 4, 5]
}

# do grid search
res = GridSearchCV(model, ## model to optimize 
                   param, ## search space
                   scoring='neg_mean_absolute_error', ## metric
                   n_jobs=-1, ## use all resource
                   cv=kf, ##use kf defined earlier
                   verbose=2 ##print cv result
                   )

res.fit(X_train, y_train)

## obtain best model and it's best learning rate and max_depth
best_model = res.best_estimator_
print(f'best learning_rate : {res.best_params_['learning_rate']}')
print(f'best max_depth     : {res.best_params_['max_depth']}')

## save both models
if not os.path.exists('data/modelling'):
    os.makedirs('data/modelling')

dump(baseline, 'data/modelling/baseline.skops')
dump(best_model, 'data/modelling/model.skops')

untrust_base  = get_untrusted_types(file="data/modelling/baseline.skops")
untrust_model = get_untrusted_types(file="data/modelling/model.skops")

print(untrust_base)
print(untrust_model)


