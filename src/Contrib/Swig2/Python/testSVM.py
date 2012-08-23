#!/usr/bin/python
import Ravl
import os

Ravl.SysLogOpen("testSVM",True,True,True,-1,True)



# we can also load one
wdbcDataSet = Ravl.LoadDataSet("{0}/share/Ravl/PatternRec/wdbc.csv".format(os.environ["PROJECT_OUT"]))

# Note this plot will just show first two features
Ravl.Plot(wdbcDataSet)

normFunc = wdbcDataSet.Sample1().Normalise(Ravl.DATASET_NORMALISE_MEAN);
Ravl.Plot(wdbcDataSet, "Data Normalised by Mean and Cov")

# should always shuffle....(well nearly)
wdbcDataSet.Shuffle()

# this will split the data set into two parts
trainDataSet = wdbcDataSet.ExtractSample(0.5)

# Lets just run a simple KNN first
design = Ravl.DesignKNearestNeighbourC(3)
classifier = design.Apply(trainDataSet)
error = Ravl.ErrorC()
print "Error with KNN {0}".format(error.Error(classifier, wdbcDataSet))
    
# What about a Linear SVM
design = Ravl.DesignSvmSmoC(Ravl.LinearKernelC(0.01))
classifier = design.Apply(trainDataSet)
print "Error with Linear SVM {0}".format(error.Error(classifier, wdbcDataSet))
    
# What about a Quadratic SVM
design = Ravl.DesignSvmSmoC(Ravl.QuadraticKernelC(0.01))
classifier = design.Apply(trainDataSet)
print "Error with Quadratic SVM {0}".format(error.Error(classifier, wdbcDataSet))
   
# What about a RBF SVM
design = Ravl.DesignSvmSmoC(Ravl.RBFKernelC(1.0))
classifier = design.Apply(trainDataSet)
print "Error with RBF SVM {0}".format(error.Error(classifier, wdbcDataSet))
 
# What about a RBF SVM
design = Ravl.DesignSvmSmoC(Ravl.PolynomialKernelC(3.0, 1.0, 0.0))
classifier = design.Apply(trainDataSet)
print "Error with 3rd Polynomial SVM {0}".format(error.Error(classifier, wdbcDataSet))

# What about a Chi2 SVM?
design = Ravl.DesignSvmSmoC(Ravl.Chi2KernelC(1.0))
classifier = design.Apply(trainDataSet)
print "Error with Chi2 SVM {0}".format(error.Error(classifier, wdbcDataSet))
   


