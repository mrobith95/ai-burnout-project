'''
read data and do preliminary checks
'''

import pandas as pd
import os
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/source/data.csv') ## read data

## do something on some columns
data.set_index('employee_id', inplace=True) ## set id as index
## drop unused columns for this task
data.drop(['job_satisfaction_1_5', 'attrition_risk'],
          axis=1,
          inplace=True)

## check for NaNs
nan_col = data.columns[data.isna().any()].tolist()

## print something to remember about NaNs
if len(nan_col)==0:
    print('No NaNs detected')
else:
    print(f'NaNs detected on : {nan_col}')

## classify target, numericals, categoricals, and ordinals
target = 'burnout_score'
numerik = list(data.select_dtypes(include='number').columns)
kategori = list(data.select_dtypes(include='str').columns)

if target in numerik:
    numerik.remove(target)

if target in kategori:
    kategori.remove(target)

## check uniques on categorical data
for i in kategori:
    isi = data[i].unique()
    print(f"{i}: {isi}, len: {len(isi)}")
## NOTE: Categoricals seems nice

## separate feature and target
X = data.copy().drop(target, axis=1)
print(X.head())

y = data[target].copy()
print(y.head())

## split data into training and test data
## we plan to use 75% train 25% test (default train_test_split)
## random_state set as int for replication
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=300,
                                                    shuffle=True)

print(len(X_train))
print(len(X_test))

## save preparation dataset
if not os.path.exists('data/prep'):
    os.makedirs('data/prep')

X_train.to_csv('data/prep/X_train.csv', index=False)
X_test.to_csv('data/prep/X_test.csv', index=False)
y_train.to_csv('data/prep/y_train.csv', index=False)
y_test.to_csv('data/prep/y_test.csv', index=False)

