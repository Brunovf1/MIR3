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

from sklearn.model_selection import cross_val_score

from scipy import stats

import os


def pertileFilter(X, low=.05, high=.95):
    quant_df = X.quantile([low, high])
    X = X.apply(lambda x: x[(x >= quant_df.loc[low, x.name]) &
                            (x <= quant_df.loc[high, x.name])], axis=0)
    X.dropna(inplace=True)
    return X


DATA_PATH = os.path.expanduser(
    "~/.kaggle/competitions/music-information-retrievel-3rd-edition/")
df = pd.read_csv(DATA_PATH + 'genresTrain.csv')
test = pd.read_csv(DATA_PATH + 'genresTest2.csv')
print(df.GENRE.unique())
# filtro usando Z-score
df = df[(np.abs(stats.zscore(df.drop(['GENRE'], axis=1))) < 3).all(axis=1)]
y = df.GENRE
X = df.drop(['GENRE'], axis=1)
print(X.shape)

# kbest = SelectKBest(f_classif, k=100)

# X_new = kbest.fit_transform(X, y)


tree = ExtraTreesClassifier()
tree = tree.fit(X, y)
smodel = SelectFromModel(tree, prefit=True)
X_new = smodel.transform(X)

# X_new = robust_scale(X_new)

nn = MLPClassifier(hidden_layer_sizes=(128, 128), )

# X_new = X

print(cross_val_score(nn, X_new, y, cv=5))

nn.fit(X_new, y)

# pred = nn.predict(kbest.transform(test))
pred = nn.predict(smodel.transform(test))

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
