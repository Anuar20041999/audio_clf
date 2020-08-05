# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 19:04:59 2020

@author: ww
"""
import numpy as np
import pandas as pd

from joblib import dump, load

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


datas = pd.read_csv('train3600_dataset.csv')
features, labels = datas.iloc[:, :-1], datas.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

clf = GradientBoostingClassifier(learning_rate=0.2, n_estimators=150, random_state=1)
clf.fit(X_train, y_train)


test_datas = pd.read_csv('sample_submission.csv')
test_features = pd.read_csv('test_features.csv')

pred = clf.predict(test_features)
test_datas.loc[:, ('target')] = pred

test_datas.to_csv('submission3600.csv', index=False)