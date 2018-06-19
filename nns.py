import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

# from sklearn.svm import SVC
from sklearn.pipeline import *

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import *

from sklearn.preprocessing import *

# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import *

from sklearn.model_selection import *

from scipy import stats

import os

# np.random.seed(42)


def percentileFilter(X, low=.05, high=.95):
    quant_df = X.quantile([low, high])
    X = X.apply(lambda x: x[(x >= quant_df.loc[low, x.name]) &
                            (x <= quant_df.loc[high, x.name])], axis=0)
    X.dropna(inplace=True)
    return X

# Utility function to report best scores


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


DATA_PATH = os.path.expanduser(
    "~/.kaggle/competitions/music-information-retrievel-3rd-edition/")
df = pd.read_csv(DATA_PATH + 'genresTrain.csv')
test = pd.read_csv(DATA_PATH + 'genresTest2.csv')
print(df.GENRE.unique())
# filtro usando Z-score
df_n = df[(np.abs(stats.zscore(df.drop(['GENRE'], axis=1))) < 3).all(axis=1)]
y = df_n.GENRE
X = df_n.drop(['GENRE'], axis=1)
# print(X.shape)

tree = ExtraTreesClassifier()
tree = tree.fit(X, y)
smodel = SelectFromModel(tree, prefit=True)
X_new = smodel.transform(X)
print(X_new.shape)

nn_pipe = [('reduce_dim', SelectFromModel(tree, prefit=True)),
           ('clf', MLPClassifier(hidden_layer_sizes=(128, 128)))]

param_distributions = {
    'learning_rate_init': stats.uniform(0.001, 0.05),
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'momentum': stats.uniform(0.8, .15),
    'activation': ['identity', 'logistic', 'tanh', 'relu']
}

# {'activation': 'relu',
#  'learning_rate': 'constant',
#  'learning_rate_init': 0.015561457009902097,
#  'momentum': 0.89177793420835694}


# X_new = robust_scale(X_new)
nn = MLPClassifier(hidden_layer_sizes=(128, 128))
rs = RandomizedSearchCV(nn, param_distributions, n_iter=100)
# print(cross_val_score(nn, X_new, y, cv=5))
rs.fit(X_new, y)

report(rs.cv_results_)
# rs.estimator.pre
pred = rs.predict(smodel.transform(test))

# pred = nn.predict(test)


g = {'Blues': 1, 'Classical': 2, 'Jazz': 3, 'Metal': 4, 'Pop': 5, 'Rock': 6}
pred_int = [g[ea] for ea in pred]
plt.hist(pred_int)
plt.show()

preddf = pd.DataFrame(pred_int[:len(pred)], columns=['"Genres"'])
preddf.index = np.arange(1, len(preddf) + 1)
preddf.to_csv('submission.csv', index_label='"Id"', quoting=csv.QUOTE_NONE)
# # #


# # print(df.describe())
