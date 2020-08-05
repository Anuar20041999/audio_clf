# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from make_dataset import make_dataset_train


datas = pd.read_csv('train.csv', index_col=0)
datas = datas.sort_values('wav_path')
datas = datas[:17023]
datas = datas[0:3600]

dataset = make_dataset_train(datas, path_pref='',
                                save_path='train3600_dataset.csv')