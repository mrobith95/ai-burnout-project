'''
shap_explain: make explainations using SHAP
'''

import pandas as pd
from skops.io import load
import shap

## load prepared data
X_test = pd.read_csv('data/feat_eng/X_test.csv')
y_test = pd.read_csv('data/feat_eng/y_test.csv')

## load model
model = load("data/modelling/model.skops", trusted=[])

## make SHAP explanation
background = shap.maskers.Independent(X_test, max_samples=100)
explainer = shap.Explainer(model, background)
shap_values = explainer(X_test)

## show base values and rough explanation
print(shap_values.base_values[0])
shap.plots.bar(shap_values)

shap.plots.scatter(shap_values[:, "ai_replaces_my_tasks_pct"])
shap.plots.scatter(shap_values[:, "hours_with_ai_assistance_daily"])