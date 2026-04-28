'''
feature engineering
'''

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os

## load prepared data
X_train = pd.read_csv('data/preprocess/X_train.csv')
X_test = pd.read_csv('data/preprocess/X_test.csv')
y_train = pd.read_csv('data/preprocess/y_train.csv')
y_test = pd.read_csv('data/preprocess/y_test.csv')

## get all categorical columns
train_cat = list(X_train.select_dtypes(include='str').columns)
X_train_c = X_train[train_cat]
X_test_c = X_test[train_cat]

print(X_train_c.head())

## set up encoder
ohe = OneHotEncoder(drop='if_binary', ## drop 1 category if it only has 2 categories
                    handle_unknown='ignore', ## ignore unknown categories (encode as 0)
                    sparse_output=False ## let the output dense for easier look-up
                    )
ohe.set_output(transform='pandas') ## set transform's output as pandas

## encode training data
ohe.fit(X_train_c)
X_train_enc = ohe.transform(X_train_c)
new_feats = list(X_train_enc.columns) ## get the name of new columns

## replace unencoded categories
X_train.drop(train_cat, axis=1, inplace=True)
X_train[new_feats] = X_train_enc

## do the same with test data
X_test_enc = ohe.transform(X_test_c)
X_test.drop(train_cat, axis=1, inplace=True)
X_test[new_feats] = X_test_enc

## save dataset
if not os.path.exists('data/feat_eng'):
    os.makedirs('data/feat_eng')

X_train.to_csv('data/feat_eng/X_train.csv', index=False)
X_test.to_csv('data/feat_eng/X_test.csv', index=False)
y_train.to_csv('data/feat_eng/y_train.csv', index=False)
y_test.to_csv('data/feat_eng/y_test.csv', index=False)