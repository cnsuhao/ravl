#!/usr/bin/python
import Ravl
import os

Ravl.SysLogOpen("testPatternRec",True,True,True,-1,True)

# this is one way to quickly generate a data set
dataSet = Ravl.CreateDataSet(2, 2)
Ravl.Plot(dataSet)

# we can also load one
irisDataSet = Ravl.LoadDataSet("{0}/share/Ravl/PatternRec/iris.csv".format(os.environ["PROJECT_OUT"]))

# Note this plot will just show first two features
Ravl.Plot(irisDataSet)

# should always shuffle....(well nearly)
irisDataSet.Shuffle()

# this will split the data set into two parts
trainDataSet = irisDataSet.ExtractSample(0.5)

# Note that this will just show first two features
Ravl.Plot(trainDataSet, "Training Data Set")
Ravl.Plot(irisDataSet, "Testing Data Set")

if True:
    print "Designing Classifier!"
    #design = Ravl.DesignClassifierNeuralNetwork2C(3, 128, False, 0, 0.5, 100, 10, True)
    #design = Ravl.DesignClassifierLogisticRegressionC(2.0)
    #designFunc = Ravl.DesignFuncLSQC(1, False)
    #design = Ravl.DesignDiscriminantFunctionC(designFunc)
    design = Ravl.DesignKNearestNeighbourC(3)
    classifier = design.Apply(trainDataSet)
    #f = classifier.Discriminant()
    #gnuplot.Plot(f)
    error = Ravl.ErrorC()
    print "Error with KNN {0}".format(error.Error(classifier, irisDataSet))
    
    # What about Logistic Regression?
    design = Ravl.DesignClassifierLogisticRegressionC(0.0)
    classifier = design.Apply(trainDataSet)
    print "Error with Logistic Regression {0}".format(error.Error(classifier, irisDataSet))
    
    # What about a simple linear classifier
    designFunc = Ravl.DesignFuncLSQC(1, False)
    design = Ravl.DesignDiscriminantFunctionC(designFunc)
    classifier = design.Apply(trainDataSet)
    print "Error with Linear LSQ Classifier {0}".format(error.Error(classifier, irisDataSet))
    
     # What about a quadratic classifier
    designFunc = Ravl.DesignFuncLSQC(2, False)
    design = Ravl.DesignDiscriminantFunctionC(designFunc)
    classifier = design.Apply(trainDataSet)
    print "Error with Quadratic LSQ Classifier {0}".format(error.Error(classifier, irisDataSet))
    
    # What about a neural network
    design = Ravl.DesignClassifierNeuralNetwork2C(3, 8, False, 0, 0.0001, 5000, 0, False)
    classifier = design.Apply(trainDataSet)
    print "Error with Neural Network {0}".format(error.Error(classifier, irisDataSet))
    
     # What about a Linear SVM
    design = Ravl.DesignSvmSmoC(Ravl.LinearKernelC(0.01))
    classifier = design.Apply(trainDataSet)
    print "Error with Linear SVM {0}".format(error.Error(classifier, irisDataSet))
    
    # What about a Quadratic SVM
    design = Ravl.DesignSvmSmoC(Ravl.QuadraticKernelC(0.01), 10, 10, 0.00000001, 0.0000000001)
    classifier = design.Apply(trainDataSet)
    print "Error with Quadratic SVM {0}".format(error.Error(classifier, irisDataSet))
   
    # What about a Quadratic SVM
    design = Ravl.DesignSvmSmoC(Ravl.RBFKernelC(0.01))
    classifier = design.Apply(trainDataSet)
    print "Error with RBF SVM {0}".format(error.Error(classifier, irisDataSet))
    
    
    # we can also save the classifier at any time
    Ravl.Save("classifier.abs", classifier)



