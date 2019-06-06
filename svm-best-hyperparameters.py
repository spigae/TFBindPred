# Python version
import sys
# scipy
import scipy
# numpy
import numpy
# matplotlib
import matplotlib
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
# Loading dataset
# training data set
df1 = pandas.read_csv('train-set.dat', sep='\s+', header=None)
array = df1.values
X = array[:,0:896]
Y = array[:,896]
# creating the new training and the validation set
seed = 10
validation_size = 0.2
# Spot Check Algorithms
models = []
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
# setting statistical learning algorithm
models.append(('SVM',SVC(kernel='linear', C=1000.00000, tol=0.00010)))
scoring = 'accuracy'
splits=10
#
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=splits, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%8.3f %8.3f" % (cv_results.mean(), cv_results.std())
    print(msg)

                            
