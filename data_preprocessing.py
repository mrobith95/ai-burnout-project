'''
data preprocessing: remove probable outlier and correlated feature
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

## load prepared data
X_train = pd.read_csv('data/prep/X_train.csv')
X_test = pd.read_csv('data/prep/X_test.csv')
y_train = pd.read_csv('data/prep/y_train.csv')
y_test = pd.read_csv('data/prep/y_test.csv')

print(len(X_train))

## here we only focus on numericals
numerik = list(X_train.select_dtypes(include='number').columns)

## pick only numericals 
X_train_n = X_train[numerik]

## get upper and lower bound
qs = X_train_n.quantile([0.25, 0.75])
iqr = qs.loc[0.75] - qs.loc[0.25]
upper = qs.loc[0.75] + (1.5 * iqr)
lower = qs.loc[0.25] - (1.5 * iqr)

## pick data that is inlier, and filter accordingly
inlier = (X_train_n<=upper) & (X_train_n>=lower)
# X_train_n = X_train_n[inlier.all(axis=1)]
X_train = X_train[inlier.all(axis=1)]
y_train = y_train[inlier.all(axis=1)]

print(len(X_train))

## compute correlation
corr = X_train.corr(method='spearman', numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool)) ## masking

sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1)

plt.title("Correlation Heatmap")
plt.show()

## NOTES: max correlation reached between ai_tools_used_per_day and productivity_score at 0.31
##        this is above other correlation that is typically less than 0.05.
##        thus here we remove productivity_score, since other feature is easy to explain later
X_train.drop(['productivity_score'], axis=1, inplace=True)

## do the same with X_test
X_test.drop(['productivity_score'], axis=1, inplace=True)

## save dataset
if not os.path.exists('data/preprocess'):
    os.makedirs('data/preprocess')

X_train.to_csv('data/preprocess/X_train.csv', index=False)
X_test.to_csv('data/preprocess/X_test.csv', index=False)
y_train.to_csv('data/preprocess/y_train.csv', index=False)
y_test.to_csv('data/preprocess/y_test.csv', index=False)