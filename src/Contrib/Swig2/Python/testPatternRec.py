#!/usr/bin/python
import Ravl
import os

Ravl.SysLogOpen("testPatternRec",True,True,True,-1,True)

# this is one way to quickly generate a data set
dataSet = Ravl.CreateDataSet(2, 2)
Ravl.Plot(dataSet)
Ravl.PlotFile(dataSet, "dataset.png", "Automatically generated data set")

# Lets normalise by mean and cov
normFunc = dataSet.Sample1().Normalise(Ravl.DATASET_NORMALISE_SCALE);
Ravl.Plot(dataSet, "Data Scaled")

# we can also load one
irisDataSet = Ravl.LoadDataSet("{0}/share/Ravl/PatternRec/iris.csv".format(os.environ["PROJECT_OUT"]))
# Note this plot will just show first two features
Ravl.Plot(irisDataSet)

# Now lets scale it
normFunc = irisDataSet.Sample1().Normalise(Ravl.DATASET_NORMALISE_MEAN);
Ravl.Plot(irisDataSet, "Iris Data Normalised by Mean and Cov")


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
    plotClassifier = Ravl.GnuPlot2dC("Linear Classifier")
    plotClassifier.Plot(classifier, trainDataSet) 
    print "Error with Linear LSQ Classifier {0}".format(error.Error(classifier, irisDataSet))
    
     # What about a quadratic classifier
    designFunc = Ravl.DesignFuncLSQC(2, False)
    design = Ravl.DesignDiscriminantFunctionC(designFunc)
    classifier = design.Apply(trainDataSet)
    print "Error with Quadratic LSQ Classifier {0}".format(error.Error(classifier, irisDataSet))
    
    # What about a neural network
    design = Ravl.DesignClassifierNeuralNetwork2C(3, 8, False, 0, 0.0001, 5000, 0, False)
    classifier = design.Apply(trainDataSet)
    print "Error with Neural Network {0}".format(error.Error(classifier, dataSet))
    
    # we can also save the classifier at any time
    Ravl.Save("classifier.abs", classifier)


