
# Grid Search for Algorithm Tuning
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import pandas as pd
import feature_extraction
import feature_extraction3
import get_data
from sklearn.neural_network import MLPClassifier
from scipy.stats import uniform as sp_rand

df_features = feature_extraction.get_audio_data()[:5240]
df_labels = get_data.get_labels()
df_features_validate = feature_extraction.get_audio_data()[5240:5539]
df_labels_validate = get_data.get_validation_labels()
df_test_features = feature_extraction3.get_titles_and_tags()[5539:]


neural_network = MLPClassifier()
grid = GridSearchCV(neural_network, {
    #"activation": ['identity', 'logistic', 'tanh', 'relu'],
    #alpha" : [0.001, 0.002, 0.003],
    #"solver" : ['adam']
    #"learning_rate" : ['constant', 'invscaling', 'adaptive']
    "batch_size" : [10, 20, 40, 80, 160, 320, 640, 1280, 2560],
    #"max_iter" : [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2]
    #"early_stopping" : [True, False]
    })

grid.fit(df_features, df_labels)
df_grid = pd.DataFrame(grid.cv_results_)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #print(df_grid[['params', 'mean_test_score', 'rank_test_score']])
    print(df_grid[['params','mean_test_score']])

