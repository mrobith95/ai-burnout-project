import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from pickle import dump, load
import os
from sklearn.model_selection import train_test_split

df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    'nudratabbas/ai-worker-burnout-and-attrition-risk-dataset',
    'ai_worker_burnout_attrition_2026.csv'
)

## saving data
if not os.path.exists('data/raw'):
    os.makedirs('data/raw')
if not os.path.exists('data/source'):
    os.makedirs('data/source')

df.to_csv('data/raw/raw_data.csv')

df = df.drop_duplicates() ## drop duplicates

df.to_csv('data/source/data.csv', index=False)