# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 19:04:59 2020

@author: ww
"""
import numpy as np
import pandas as pd

from joblib import dump, load
from itertools import permutations

from sklearn.cluster import KMeans, Birch
from sklearn.metrics import accuracy_score, adjusted_rand_score

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from visualize import visualize


def right_permutation(y_perm, pred_perm):
    y_perm, pred_perm = pd.DataFrame(y_perm)[:50], pd.DataFrame(pred_perm)[:50]
    accuracy_list = []
    perms = []
    for i in permutations(range(3)):
        perms.append(list(i))
        y_perm_i = y_perm.replace([0, 1, 2], list(i))
        accuracy = accuracy_score(y_perm_i, pred_perm)
        accuracy_list.append(accuracy)
    right_perm_num = np.argmax(accuracy_list)
    print('accuracy =', max(accuracy_list))
    return perms[right_perm_num]


datas = pd.read_csv('csv_files/train17023_dataset.csv')[:500]
X, y = datas.iloc[:, :-1], datas.iloc[:, -1]
X = scaler.fit_transform(X)


#clust = KMeans(n_clusters=3, n_init=10, random_state=1, max_iter=300)
clust = Birch(n_clusters=3, threshold=0.6, branching_factor=45)

clust.fit(X)
pred = clust.labels_

r_perm = right_permutation(y, pred)
print('Rand score =', adjusted_rand_score(y, pred))
#visualize(X, pred, r_perm, num=500)


#############################
def make_submission(clust, scaler, r_perm):
    datas_test = pd.read_csv('csv_files/sample_submission.csv')
    X_test = pd.read_csv('csv_files/test_features.csv')
    X_test = scaler.transform(X_test)

    pred_test = clust.predict(X_test)
    pred_test = np.array(pd.DataFrame(pred_test).replace(r_perm, [0, 1, 2]))

    datas_test.loc[:, ('target')] = pred_test
    datas_test.to_csv('csv_files/submission_Bicrh_clust_500_2.csv', index=False)
    return datas_test

make_submission(clust, scaler, r_perm)



