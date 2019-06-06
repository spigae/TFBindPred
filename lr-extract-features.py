# Python version
import sys
# scipy
import scipy
# numpy
import numpy as np
# matplotlib
import matplotlib
# pandas
#import pandas
import pandas as pd
# scikit-learn
import sklearn

# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#
fout1 = open('features.dat', 'w')
fout2 = open('accuracy.dat', 'w')
#
# Loading dataset
# training data set
df1 = pandas.read_csv('train-set.dat', sep='\s+', header=None)
array = df1.values
X = array[:,0:896]
Y = array[:,896]
#
# test data set
df2 = pandas.read_csv('test-set.dat', sep='\s+', header=None)
array = df2.values
X1 = array[:,0:896]
Y1 = array[:,896]
#
#print(' ')
#
seed = 10
scoring = 'accuracy'
#
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
#
lr = LogisticRegression(C=1000.00000, tol=0.00010)
lr.fit(X, Y)
#
predictions = lr.predict(X1)
#
# calculating the weights of the features
weights = lr.coef_
abs_weights = np.abs(weights)
#
# calculating sum
tot=np.sum(abs_weights)
#
# calculation the weights as percentage
perc=abs_weights/tot
#
# writing the weights as percentages..
for j in range(perc.shape[1]):
    fout1.write("%s %8.6f \n" % (j+1,perc[:,j]))
#
#
acc=accuracy_score(Y1, predictions)
tp=confusion_matrix(Y1, predictions)[0][0]
tn=confusion_matrix(Y1, predictions)[1][1]
fp=confusion_matrix(Y1, predictions)[0][1]
fn=confusion_matrix(Y1, predictions)[1][0]
fout2.write("%s %s %s %s %s \n" % (acc,tp,tn,fp,fn))
