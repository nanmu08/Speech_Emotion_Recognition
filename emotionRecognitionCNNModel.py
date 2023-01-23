#this part is used to train and test model
import os
import random
import sys

## Package
import glob
import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import plotly.tools as tls
import seaborn as sns
import scipy.io.wavfile
import tensorflow as tf

## Keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras.callbacks import  History, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import model_from_json
## Sklearn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

## Rest
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm

input_duration=3

data4model = pd.read_csv('./RAVDESS_FEATURE_ObservedCNN.csv')
data4model = data4model.drop('Unnamed: 0',axis=1)

#   Split train and train validation

X = data4model.drop(['label'], axis=1)
y = data4model.label
xxx = StratifiedShuffleSplit(1, test_size=0.2, random_state=12)
for train_index, test_index in xxx.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

print(y_train.value_counts())
y_test.value_counts()
X_train.isna().sum().sum()

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
lb = LabelEncoder()
y_train = to_categorical(lb.fit_transform(y_train))
y_test = to_categorical(lb.fit_transform(y_test))

print(y_train.shape)
print(X_train.shape)

x_traincnn = np.expand_dims(X_train, axis=2)
x_testcnn = np.expand_dims(X_test, axis=2)

print(x_traincnn.shape)

# Set up Keras util functions

from tensorflow.keras import backend as K

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fscore(y_true, y_pred):
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0.0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f_score = 2 * (p * r) / (p + r + K.epsilon())
    return f_score

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

# CNN model
model = Sequential()
model.add(Conv1D(256, 8, padding='same',input_shape=(X_train.shape[1],1)))
model.add(Activation('relu'))
model.add(Conv1D(256, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
model.add(Flatten())
# Edit according to target class no.
#model.add(Dense(5))
model.add(Dense(10))
model.add(Activation('softmax'))
opt = tf.keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.000001, nesterov=False)

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', fscore])

# Model Training begin
#when need test the model in file, just comment model traning segment (model traning segment begin to end)
#######################################################################################

lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=0.000001)

mcp_save = ModelCheckpoint('modelCNN/CNNmodel.h5', save_best_only=True, monitor='val_loss', mode='min')
cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=400,
                     validation_data=(x_testcnn, y_test), callbacks=[mcp_save, lr_reduce])

# Plotting the Train Valid Loss Graph
plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Saving the model.json

import json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
#######################################################################################
# Model Training end



######################################################################################
# loading json and creating model

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("./modelCNN/CNNmodel.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
score = loaded_model.evaluate(x_testcnn, y_test, verbose=0)
print("loaded model test accuracy")
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

#read final test data to test model
data3_df = pd.read_csv('./data3_df4testCNN.csv')
data3_df = data3_df.drop('Unnamed: 0',axis=1)
df3 = pd.read_csv('./df3CNN.csv')
de3 = df3.drop('Unnamed: 0',axis=1)

data_test = pd.DataFrame(columns=['feature'])
for i in tqdm(range(len(data3_df))):
    X, sample_rate = librosa.load(data3_df.path[i], res_type='kaiser_fast', duration=input_duration, sr=22050 * 2,
                                  offset=0.5)


    result = np.array([])  # this line is new add more features extration to test
    sample_rate = np.array(sample_rate)
    # mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40), axis=0)
    # result=np.hstack((result, mfccs))
    mfcc = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally
    # below is new add in melspectrogram and mel to test
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally
    # Chroma_stft
    stft = np.abs(librosa.stft(X))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally
    feature = result

    data_test.loc[i] = [feature]

test_valid = pd.DataFrame(data_test['feature'].values.tolist())
labels = data3_df.label
newdf3 = pd.concat([df3, labels], axis=1)
rnewdf3 = newdf3.rename(index=str, columns={"0": "label"})
len(rnewdf3)
rnewdf3 = rnewdf3.fillna(0)
test_valid = np.array(test_valid)
test_valid_lb = np.array(data3_df.label)
lb3 = LabelEncoder()
test_valid_lb = to_categorical(lb3.fit_transform(test_valid_lb))
test_valid = np.expand_dims(test_valid, axis=2)
print(data3_df.label)

preds = loaded_model.predict(test_valid,
                         batch_size=16,
                         verbose=1)

preds1=preds.argmax(axis=1)
abc = preds1.astype(int).flatten()
predictions = (lb3.inverse_transform((abc)))
preddf = pd.DataFrame({'predictedvalues': predictions})

actual=test_valid_lb.argmax(axis=1)
abc123 = actual.astype(int).flatten()
actualvalues = (lb.inverse_transform((abc123)))
actualdf = pd.DataFrame({'actualvalues': actualvalues})

finaldf = actualdf.join(preddf)
finaldf[1:10]

finaldf.to_csv('CNNPredictions.csv', index=False)


def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


y_true = finaldf.actualvalues
y_pred = finaldf.predictedvalues
accuracy_score(y_true, y_pred)*100
print("accuracy for final test")
print(f1_score(y_true, y_pred, average='macro') *100)

c = confusion_matrix(y_true, y_pred)
print(c)
# Visualize Confusion Matrix


class_names = ['female_angry', 'female_calm', 'female_fearful', 'female_happy', 'female_sad', 'male_angry', 'male_calm', 'male_fearful', 'male_happy', 'male_sad']

print("final test result")
print_confusion_matrix(c, class_names)

datalive, sampling_ratelive = librosa.load('23angre.wav')
X, sample_rate = librosa.load('23angre.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)

result=np.array([]) #this line is new add more features extration to test
sample_rate = np.array(sample_rate)
#mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40), axis=0)
#result=np.hstack((result, mfccs))
mfcc = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate).T, axis=0)
result = np.hstack((result, mfcc)) # stacking horizontally
# below is new add  melspectrogram and mel to test
mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
result = np.hstack((result, mel)) # stacking horizontally
# Chroma_stft
stft = np.abs(librosa.stft(X))
chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
result = np.hstack((result, chroma_stft)) # stacking horizontally

featurelive = result
livedf2 = featurelive
livedf2= pd.DataFrame(data=livedf2)
livedf2 = livedf2.stack().to_frame().T

livetwodim= np.expand_dims(livedf2, axis=2)
livepreds = loaded_model.predict(livetwodim,
                         batch_size=16,
                         verbose=1)
livepreds1=livepreds.argmax(axis=1)
liveabc = livepreds1.astype(int).flatten()
livepredictions = (lb.inverse_transform((liveabc)))
print(livepredictions)