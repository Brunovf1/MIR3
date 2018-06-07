import numpy as np
import pandas as pd
import csv

from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score

import os

DATA_PATH = os.path.expanduser(
    "~/.kaggle/competitions/music-information-retrievel-3rd-edition/")
DATA_FILES = os.listdir(DATA_PATH)

print(DATA_FILES)
df = pd.read_csv(DATA_PATH + DATA_FILES[2])
print(df.GENRE.unique())
y = df.GENRE.values
X = df.drop(['GENRE'], axis=1).values
test = pd.read_csv(DATA_PATH + DATA_FILES[0])

kbest = SelectKBest(mutual_info_classif, k=100)

X_new = kbest.fit_transform(X, y)

# for nn in range(3, 30):
#     knn = KNN(n_neighbors=nn)
#     # knn.fit(X, y)
#     scores = cross_val_score(knn, X_new, y, cv=5)
#     print('scores de ' + str(nn))
#     print(sum(scores) / 5)

knn = KNN(n_neighbors=8)
knn.fit(X_new, y)

pred = knn.predict(kbest.transform(test))

print(pred)
pred_int = []
for ea in pred:
    if ea == "Pop":
        pred_int.append(5)
    elif ea == "Blues":
        pred_int.append(1)
    elif ea == "Jazz":
        pred_int.append(3)
    elif ea == "Classical":
        pred_int.append(2)
    elif ea == "Rock":
        pred_int.append(6)
    elif ea == "Metal":
        pred_int.append(4)

# print(pred_int)

preddf = pd.DataFrame(pred_int[:len(pred)], columns=['"Genres"'])
preddf.index = np.arange(1, len(preddf) + 1)
preddf.to_csv('submission.csv', index_label='"Id"', quoting=csv.QUOTE_NONE)
# # #


# # print(df.describe())
