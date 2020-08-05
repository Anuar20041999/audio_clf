# -*- coding: utf-8 -*-
"""
http://zabaykin.ru/?p=705
https://github.com/subho406/Audio-Feature-Extraction-using-Librosa/blob/master/Song%20Analysis.ipynb
"""
import librosa
import numpy as np
import librosa.display
import scipy

from scipy.io import wavfile

def get_base_features(wav_path):
    ff_list = []

    sr, y = wavfile.read(wav_path)
    y = y.astype(float)

    y_harmonic, y_percussive = librosa.effects.hpss(y)

    tempo, beat_frames = librosa.beat.beat_track(y=y_harmonic, sr=sr)
    chroma = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)
    mfccs = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=13)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y_harmonic, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zrate = librosa.feature.zero_crossing_rate(y_harmonic)

    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    for i in range(0, 12):
        ff_list.append(chroma_mean[i])
    for i in range(0, 12):
        ff_list.append(chroma_std[i])

    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    for i in range(0, 13):
        ff_list.append(mfccs_mean[i])
    for i in range(0, 13):
        ff_list.append(mfccs_std[i])

    cent_mean = np.mean(cent)
    cent_std = np.std(cent)
    cent_skew = scipy.stats.skew(cent, axis=1)[0]

    contrast_mean = np.mean(contrast,axis=1)
    contrast_std = np.std(contrast,axis=1)

    rolloff_mean=np.mean(rolloff)
    rolloff_std=np.std(rolloff)

    data = np.concatenate(([cent_mean, cent_std, cent_skew],
                           contrast_mean, contrast_std,
                           [rolloff_mean, rolloff_std, rolloff_std]), axis=0)
    ff_list += list(data)

    zrate_mean = np.mean(zrate)
    zrate_std = np.std(zrate)
    zrate_skew = scipy.stats.skew(zrate,axis=1)[0]

    ff_list += [zrate_mean, zrate_std, zrate_skew]

    ff_list.append(tempo)

    return ff_list
