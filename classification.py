import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

#read csv file
data = pd.read_csv('data.csv')
#print data
print(data.head(569))

#remove id from list of features
data.drop('id', axis=1, inplace=True)

#select list of samples and list of features
#rows: samples
#columns: list of features except for the first one which is our labels
X = data.iloc[:,1:31].values

#select 'diagnosis as label 'M' or 'B'
y = data['diagnosis']

#select training set and test set
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.20, random_state=27)
print(X_train)  
print(y_train)

#Decision Tree Model
DT_model = DecisionTreeClassifier()
DT_model.fit(X_train, y_train)
DT_prediction = DT_model.predict(X_test)

#print Accuracy, Confusion Matrix & Report
print('Accuracy:', sklearn.metrics.accuracy_score(DT_prediction, y_test))
print('Confusion Matrix:')
print(sklearn.metrics.confusion_matrix(DT_prediction, y_test))
print(sklearn.metrics.classification_report(DT_prediction, y_test))

#SVC Model
SVC_model = SVC()
SVC_model.fit(X_train, y_train)
SVC_prediction = SVC_model.predict(X_test)

#print Accuracy, Confusion Matrix & Report
print('Accuracy:', sklearn.metrics.accuracy_score(SVC_prediction, y_test))
print('Confusion Matrix:')
print(sklearn.metrics.confusion_matrix(SVC_prediction, y_test))
print(sklearn.metrics.classification_report(SVC_prediction, y_test))

#finding k for KNN:
#create x-axis which is k = number of neighbors
neighbors = []

#array for result of classification with each k
KNN_model = []

#perform KNN for k = 2 to 9
for i in range(2, 10):
    neighbors.append(i)
    KNN_model.append(KNeighborsClassifier(n_neighbors=i))

KNN_prediction = []
for j in KNN_model:
    j.fit(X_train, y_train)
    KNN_prediction.append(j.predict(X_test))

#compare accuracy of each k    
accuracy = []
for z in KNN_prediction:
    accuracy.append(sklearn.metrics.accuracy_score(z, y_test))

#draw chart for accuracy of each k
plt.plot(neighbors, accuracy)
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.show()

#Kth Nearest Neighbor Model
KNN_model = KNeighborsClassifier(n_neighbors=3)
KNN_model.fit(X_train, y_train)
KNN_prediction = KNN_model.predict(X_test)

#print Accuracy, Confusion Matrix & Report
print('Accuracy:', sklearn.metrics.accuracy_score(KNN_prediction, y_test))
print('Confusion Matrix:')
print(sklearn.metrics.confusion_matrix(KNN_prediction, y_test))
print(sklearn.metrics.classification_report(KNN_prediction, y_test))

#Naive Bayes Model
NB_model = GaussianNB()
NB_model.fit(X_train, y_train)
NB_prediction = NB_model.predict(X_test)

#print Accuracy, Confusion Matrix & Report
print('Accuracy:', sklearn.metrics.accuracy_score(NB_prediction, y_test))
print('Confusion Matrix:')
print(sklearn.metrics.confusion_matrix(NB_prediction, y_test))
print(sklearn.metrics.classification_report(NB_prediction, y_test))

#compare accuracy of each algorithm
accuracies = []
accuracies.append(sklearn.metrics.accuracy_score(DT_prediction, y_test))
accuracies.append(sklearn.metrics.accuracy_score(SVC_prediction, y_test))
accuracies.append(sklearn.metrics.accuracy_score(KNN_prediction, y_test))
accuracies.append(sklearn.metrics.accuracy_score(NB_prediction, y_test))

algorithms = ['DT', 'SVC', 'KNN', 'NB']

#draw chart to show accuracy of algorithms in python implementation
plt.plot(algorithms, accuracies, 'o')
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.show()

#compare accuracy of each algorithm
accuracies = []
accuracies.append(0.935)
accuracies.append(0.848)
accuracies.append(0.957)
accuracies.append(0.983)

algorithms = ['DT', 'SVC', 'KNN', 'NB']

#draw chart to show accuracy of algorithms in orange implementation
plt.plot(algorithms, accuracies, 'o')
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.show()