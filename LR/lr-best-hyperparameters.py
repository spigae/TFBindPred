# Python version
import sys
# Load libraries
# time
import time
start_time = time.clock()
# numpy
import numpy
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
# Loading dataset
# training data set
#
print ' '
print ' Importing training set'
print ' '
#
df1 = pandas.read_csv('train-set.dat', sep='\s+', header=None)
array = df1.values
X = array[:,0:896]
Y = array[:,896]
# creating the new training and the validation set
seed = 10
validation_size = 0.2
# Spot Check Algorithms
models = []
# setting statistical learning algorithm
#
print ' '
print ' Calling Logistic Regression'
print ' '
#
models.append(('LR',LogisticRegression(C=1000.00000, tol=0.00010)))
scoring = 'accuracy'
splits=10
#
print ' '
print ' Running Cross Validation with ',splits,' kfolds '
print ' '
#
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=splits, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print ' '
    print ' Results (Accuracy)'
    print ' Mean +/- Standard Deviation'
    msg = "%8.3f %8.3f" % (cv_results.mean(), cv_results.std())
    print(msg)
#
print ' '
print ' Duration calculation:'
ts = time.clock() - start_time
tm = ts/60
t = "%8.3f %s %8.3f %s" % (ts,"s",tm,"m")
print(t)
