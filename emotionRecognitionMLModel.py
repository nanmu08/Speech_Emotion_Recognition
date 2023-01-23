import librosa
from librosa import display
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import os, glob, pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

#SVM MODEL
data = pd.read_csv('./RAVDESS_MFCC_ObservedML.csv')
print(data.head())
print(data.shape)

#dropping the column Unnamed: 0
data = data.drop('Unnamed: 0', axis=1)
print(data.columns)

#separating features and target outputs
X = data.drop('emotion', axis = 1).values
y = data['emotion'].values
print(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
print(X_train.shape)
print(X_test.shape)

#svclassifier = SVC(kernel = 'linear')
svclassifier = SVC(kernel = 'rbf', C = 10, gamma = 0.0001 )

svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print("SVM Model with rbf kernel test result:")
print(classification_report(y_test,y_pred))
print("\n")
acc = float(accuracy_score(y_test,y_pred))*100
print("----accuracy score %s ----" % acc)
print("\n")
cm = confusion_matrix(y_test,y_pred)
df_cm = pd.DataFrame(cm)
sn.heatmap(df_cm, annot=True, fmt='')
plt.show()
print("\n")
train_acc = float(svclassifier.score(X_train, y_train)*100)
print("----train accuracy score %s ----" % train_acc)
print("\n")
test_acc = float(svclassifier.score(X_test, y_test)*100)
print("----test accuracy score %s ----" % test_acc)
print("\n")

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

svc_scaled = pipeline.fit(X_train, y_train)

# unscaled data
svc_unscaled = SVC(kernel='linear').fit(X_train, y_train)

# Compute and print metrics
print("below is SVM scaled result\n")
print('Accuracy with Scaling: {}'.format(svc_scaled.score(X_test, y_test)))
print("\n")
print('Accuracy without Scaling: {}'.format(svc_unscaled.score(X_test, y_test)))
print("\n")
train_acc = float(svc_scaled.score(X_train, y_train)*100)
print("----train accuracy score %s ----\n" % train_acc)

test_acc = float(svc_scaled.score(X_test, y_test)*100)
print("----test accuracy score %s ----" % test_acc)

scaled_predictions = svc_scaled.predict(X_test)

print(classification_report(y_test,scaled_predictions))

acc = float(accuracy_score(y_test,scaled_predictions))*100
print("----accuracy score %s ----" % acc)

cm = confusion_matrix(y_test,scaled_predictions)
df_cm = pd.DataFrame(cm)
sn.heatmap(df_cm, annot=True, fmt='')
plt.show()

# no. of folds cv = 5
cv_results = cross_val_score(svc_scaled, X_train, y_train, cv = 5)
print(cv_results)


#MLP Model

# Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(solver='adam',alpha=0.01, batch_size=128, epsilon=1e-08, hidden_layer_sizes=(200,), learning_rate='constant', max_iter=500)

# Train the MLP model
model.fit(X_train,y_train)

# Predict the test set
y_pred=model.predict(X_test)
# Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
# Print the accuracy
print("below is MLP Model result\n")
print("Accuracy: {:.2f}%".format(accuracy*100))
print(classification_report(y_test,y_pred))
matrix = confusion_matrix(y_test,y_pred)
print (matrix)