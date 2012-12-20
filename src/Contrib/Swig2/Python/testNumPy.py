#!/usr/bin/python

import Ravl
import numpy as np
import os
import sklearn as sk

from sklearn import lda
from sklearn import qda
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

# make vector
v = Ravl.VectorC(10)
v.Fill(1.0)
b = v.AsNumPy()
print v
print b

# make matrix
m = Ravl.MatrixC(5, 5)
m.Fill(3.4)
print m
nm = m.AsNumPy()
print nm

# load dataset
irisDataSet = Ravl.LoadDataSet("{0}/share/Ravl/PatternRec/iris.csv".format(os.environ["PROJECT_OUT"]))
# Note this plot will just show first two features
Ravl.Plot(irisDataSet)

X = irisDataSet.Sample1().AsNumPy()
print X
y = irisDataSet.Sample2().AsNumPy()
print y

# train classifier
clf = svm.LinearSVC()
y_pred = clf.fit(X, y).predict(X)
#y_proba_pred1 = clf.predict_proba(X)

print y_pred
#print y_proba_pred1

for 





