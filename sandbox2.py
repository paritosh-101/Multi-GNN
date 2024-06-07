import numpy as np
from snapml import GraphFeaturePreprocessor
import pandas as pd

df = pd.read_csv('HI_Small_formatted_transactions.csv')
df.head()

# detach and store the last column as the target variable
y = df['Is Laundering']
X = df.drop(['Is Laundering'], axis=1)

X.head()

# dataframe to numpy array
X = X.to_numpy()
y = y.to_numpy()

print(X.shape)
print(X[0:5])

gfp = GraphFeaturePreprocessor()

params = {
    'num_threads': 18,
    'time_window': -1,
    'max_no_edges': -1,
    'vertex_stats': True,
    'vertex_stats_cols': [3, 4],
    'vertex_stats_feats': [0, 1, 2, 3, 4, 8, 9, 10],
    'fan': True,
    'fan_tw': 86410,
    'fan_bins': [1, 2, 3, 4, 5],
    'degree': True,
    'degree_tw': 86410,
    'degree_bins': [1, 2, 3, 4, 5],
    'scatter-gather': True,
    'scatter-gather_tw': 21610,
    'scatter-gather_bins': [1, 2, 3, 4, 5],
    'temp-cycle': True,
    'temp-cycle_tw': 86410,
    'temp-cycle_bins': [1, 2, 3, 4, 5],
    'lc-cycle': True,
    'lc-cycle_tw': 86410,
    'lc-cycle_bins': [1, 2, 3, 4, 5],
    'lc-cycle_len': 10
}

gfp.get_params()

gfp.set_params(params)

# Fit and transform the edge list to generate graph-based features
features_out = gfp.fit_transform(X)

print(features_out.shape)