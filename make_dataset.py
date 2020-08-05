# -*- coding: utf-8 -*-

import pandas as pd
from tqdm import tqdm

from get_base_features import get_base_features


def make_dataset_train(datas, path_pref='', save_path=None):
    dataset = []
    for num in tqdm(range(len(datas))):
        path = path_pref + datas.iloc[num]['wav_path']
        feature_list = get_base_features(path)

        target = datas.iloc[num]['target']
        feature_list.append(target)

        dataset.append(feature_list)
    if save_path is not None:
        pd.DataFrame(dataset).to_csv(save_path, index=False)
    return dataset


def make_dataset_test(datas, path_pref='', save_path=None):
    dataset = []
    for num in tqdm(range(len(datas))):
        path = path_pref + datas.iloc[num]['wav_path']
        dataset.append(get_base_features(path))
    if save_path is not None:
        pd.DataFrame(dataset).to_csv(save_path, index=False)
    return dataset