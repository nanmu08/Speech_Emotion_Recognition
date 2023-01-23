import librosa
from librosa import display
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import librosa
import time

import pandas as pd

emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fear',
    '07': 'disgust',
    '08': 'surprised'
}

# defined RAVDESS emotions to test on RAVDESS dataset
ravdess_emotions = ['neutral', 'calm', 'angry', 'happy', 'disgust', 'sad', 'fear', 'surprised']
observed_emotions = ['sad', 'angry', 'happy', 'disgust', 'surprised', 'neutral', 'calm', 'fear']


def extract_feature(file_name, mfcc):
    X, sample_rate = librosa.load(os.path.join(file_name), res_type='kaiser_fast')
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = mfccs
    return result


def dataset_options():
    # choose datasets
    ravdess = True
    tess = False
    ravdess_speech = False
    ravdess_song = False
    data = {'ravdess': ravdess, 'ravdess_speech': ravdess_speech, 'ravdess_song': ravdess_song}
    print(data)
    return data


def load_dataset():
    x, y = [], []

    # feature to extract
    mfcc = True

    data = dataset_options()
    paths = []
    if data['ravdess']:
        paths.append(".\emotiondataset\Actor_*\*.wav")

    for path in paths:
        for file in glob.glob(path):
            file_name = os.path.basename(file)
            emotion = emotions[
                file_name.split("-")[2]]
            if emotion not in observed_emotions:
                continue
            feature = extract_feature(file, mfcc)
            x.append(feature)
            y.append(emotion)

    return {"X": x, "y": y}

start_time = time.time()
Trial_dict = load_dataset()

print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))
X = pd.DataFrame(Trial_dict["X"])
y = pd.DataFrame(Trial_dict["y"])
print(X.shape)
print(y.shape)

#renaming the label column to emotion
y=y.rename(columns= {0: 'emotion'})

#concatinating the attributes and label into a single dataframe
data = pd.concat([X, y], axis =1)
data.head()

#reindexing to shuffle the data at random
data = data.reindex(np.random.permutation(data.index))
# Storing shuffled ravdess data to avoid loading again
data.to_csv("RAVDESS_MFCC_ObservedML.csv")
