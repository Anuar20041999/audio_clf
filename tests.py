# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd

from joblib import dump, load

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split


datas = pd.read_csv('train3600_dataset.csv')
features, labels = datas.iloc[:, :-1], datas.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)


for i in range(1, 10):
    clf = GradientBoostingClassifier(learning_rate=0.1*i, n_estimators=150, random_state=1)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(score)



#dump(clf, 'model10000.joblib')