#!/usr/bin/python
import Ravl
import os

Ravl.SysLogOpen("testSVM",True,True,True,-1,True)



# we can also load one
wdbcDataSet = Ravl.LoadDataSet("{0}/share/Ravl/PatternRec/wdbc.csv".format(os.environ["PROJECT_OUT"]))
normFunc = wdbcDataSet.Sample1().Normalise(Ravl.DATASET_NORMALISE_MEAN);

# should always shuffle....(well nearly)
wdbcDataSet.Shuffle()

# this will split the data set into two parts
trainDataSet = wdbcDataSet.ExtractSample(0.5)

# Lets just run a simple KNN first
#design = Ravl.DesignKNearestNeighbourC(3)
#design = Ravl.DesignClassifierLogisticRegressionC(0.0)
#design = Ravl.DesignDiscriminantFunctionC(Ravl.DesignFuncLSQC(3, True))
#design = Ravl.DesignClassifierNeuralNetwork2C(3, 5, False, 10.0)
#design = Ravl.DesignSvmSmoC(Ravl.LinearKernelC(1.0))
classifier = Ravl.ClassifierC()
   
# OK lets do some feature selection and just get the first two features so we 
# can see the classification decision that has been made
deltaError = 0.001
maxFeatures = 2
threads = 4
featureSelector = Ravl.FeatureSelectPlusLMinusRC(2, 1, deltaError, maxFeatures, threads)
features = featureSelector.SelectFeatures(design, trainDataSet, wdbcDataSet, classifier)

wdbcDataSet = Ravl.LoadDataSet("{0}/share/Ravl/PatternRec/wdbc.csv".format(os.environ["PROJECT_OUT"]))
normFunc = wdbcDataSet.Sample1().Normalise(Ravl.DATASET_NORMALISE_MEAN);
selectedDataSet = Ravl.DataSetVectorLabelC(Ravl.SampleVectorC(wdbcDataSet.Sample1(), features), wdbcDataSet.Sample2())
plot = Ravl.GnuPlot2dC("Best 2 Features and Classifier")
plot.SetXLabel("Feature {0}".format(features[0].V()))
plot.SetYLabel("Feature {0}".format(features[1].V()))
plot.Plot(classifier, selectedDataSet)

# And lets do it again with more features
deltaError = 0.001
maxFeatures = 20
threads = 4
featureSelector = Ravl.FeatureSelectPlusLMinusRC(2, 1, deltaError, maxFeatures, threads)
features = featureSelector.SelectFeatures(design, trainDataSet, wdbcDataSet, classifier)






