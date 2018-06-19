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

np.random.seed(42)


def percentileFilter(X, low=.05, high=.95):
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
# print(X.shape)

clf1 = RandomForestClassifier(n_estimators=30, bootstrap=False,
                              min_samples_leaf=1, min_samples_split=3,
                              criterion='gini', max_features=10)
clf2 = RandomForestClassifier(n_estimators=20, bootstrap=False,
                              min_samples_leaf=1, min_samples_split=3,
                              criterion='gini', max_features=10)
clf3 = RandomForestClassifier()
clf4 = RandomForestClassifier()
# clf5 = MLPClassifier(hidden_layer_sizes=(128, 128))

nn_pipe = [('reduce_dim', SelectFromModel(ExtraTreesClassifier())),
           ('clf', MLPClassifier(hidden_layer_sizes=(128, 128)))]
clf5 = Pipeline(nn_pipe)

clf = VotingClassifier(
    estimators=[('RF30', clf1), ('RF20', clf2), ('nn128,128', clf5)])

print(cross_val_score(clf, X, y, cv=5))
clf.fit(X, y)

pred = clf.predict(test)

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
