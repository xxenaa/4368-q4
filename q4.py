import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score
from statistics import mean
from sklearn.neural_network import MLPClassifier


data=pd.read_csv("heart_failure_clinical_records_dataset.csv")
D = data.values
x = D[:,0:12]
y = D[:,12]

clf = svm.SVC(kernel='linear')
scores = cross_val_score(clf, x,y, cv=10)
average = mean(scores)
print("Accuracy:",average)


clf2=svm.SVC(kernel='rbf')
scores = cross_val_score(clf2, x,y, cv=10)
average = mean(scores)
print("Accuracy:",average)


clf3= MLPClassifier(activation='relu')
scores = cross_val_score(clf3, x,y, cv=10)
average = mean(scores)
print("Accuracy:",average)

clf4= MLPClassifier(activation='tanh')
scores = cross_val_score(clf4, x,y, cv=10)
average = mean(scores)
print("Accuracy:",average)
