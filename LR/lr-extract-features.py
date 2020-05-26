# Python version
import sys
# Load libraries
# time
import time
start_time = time.clock()
# numpy
import numpy as np
# pandas
import pandas
# scikit-learn
import sklearn
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
#
fout1 = open('features.dat', 'w')
fout2 = open('accuracy.dat', 'w')
#
print ' '
print ' Importing train and test set'
print ' '
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
#
seed = 10
scoring = 'accuracy'
#
print ' '
print ' Calling Logistic Regression'
print ' '
#
lr = LogisticRegression(C=1000.00000, tol=0.00010)
lr.fit(X, Y)
#
print ' '
print ' Calculating predictions'
print ' '
#
predictions = lr.predict(X1)
#
print ' '
print ' Calculating weights of the features'
print ' '
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
print ' '
print ' Writing weights of the features'
print ' and confusion matrix'
print ' '
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
#
#
print ' '
print ' Duration calculation:'
ts = time.clock() - start_time
tm = ts/60
t = "%8.3f %s %8.3f %s" % (ts,"s",tm,"m")
print(t)
