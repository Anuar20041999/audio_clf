# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from make_dataset import make_dataset_train


datas = pd.read_csv('csv_files/train.csv', index_col=0)
#datas = datas.sort_values('wav_path')
datas = datas[:15]


dataset = make_dataset_train(datas, path_pref='csv_files/',
                                save_path='csv_files/train17023_dataset.csv')
