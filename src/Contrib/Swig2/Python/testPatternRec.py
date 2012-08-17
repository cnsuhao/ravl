#!/usr/bin/python
import Ravl

Ravl.SysLogOpen("testPatternRec",True,True,True,-1,True)

dataSet = Ravl.CreateDataSet(2, 2)
normFunc = Ravl.FunctionC()

gnuplot = Ravl.GnuPlot2dC("My Data")
gnuplot.Plot(dataSet)



if True:
    print "Designing Classifier!"
    #design = Ravl.DesignClassifierNeuralNetwork2C(3, 128, False, 0, 0.5, 100, 10, True)
    #design = Ravl.DesignClassifierLogisticRegressionC(2.0)
    designFunc = Ravl.DesignFuncLSQC(1, False)
    design = Ravl.DesignDiscriminantFunctionC(designFunc)
    classifier = design.Apply(dataSet.Sample1(), dataSet.Sample2())
    
    f = classifier.Discriminant()
    print f
    #gnuplot.PlotFunction(f)
    error = Ravl.ErrorC()
    print "Error {0}".format(error.Error(classifier, dataSet))
    Ravl.Save("pythonClassifier.abs", classifier)



