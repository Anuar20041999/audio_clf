# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 13:32:04 2020

@author: ww
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
#from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

def visualize(X, y, r_perm=[1,2,3], num=500):
    X, y = X[:num], y[:num]
    X = scaler.fit_transform(X)
    y = np.array(pd.DataFrame(y).replace(r_perm, [0, 1, 2]))

    dim_reduction = TSNE(n_components=2, random_state=2)
    X_embedded = dim_reduction.fit_transform(X)

    classes = ['Speech', 'Music', 'Noise']

    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='plasma')
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)

