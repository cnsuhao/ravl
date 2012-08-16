#!/usr/bin/python
import Ravl

Ravl.SysLogOpen("testPatternRec",True,True,True,-1,True)

dataSet = Ravl.DataSetVectorLabelC()
normFunc = Ravl.FunctionC()




Ravl.LoadDataSetVectorLabel("m2vts_auto_lbp.abs", dataSet)
dataSetSmall = dataSet.ExtractPerLabel(100)

features = Ravl.SArray1dIndexC(2);
features[0] = Ravl.IndexC(100);
features[1] = Ravl.IndexC(100);
dataSetF = Ravl.DataSetVectorLabelC(Ravl.SampleVectorC(dataSetSmall.Sample1(), features), dataSetSmall.Sample2())
print dataSetF


gnuplot = Ravl.GnuPlot2dC("Some Features")
gnuplot.ScatterPlot(dataSetF)

if True:
    print "Designing Classifier!"
    #design = Ravl.DesignClassifierNeuralNetwork2C(3, 128, False, 0, 0.5, 100, 10, True)
    #design = Ravl.DesignClassifierLogisticRegressionC(2.0)
    designFunc = Ravl.DesignFuncLSQC(1, False)
    design = Ravl.DesignDiscriminantFunctionC(designFunc)
    classifier = design.Apply(dataSetF.Sample1(), dataSetF.Sample2())
    
    f = classifier.Discriminant()
    print f
    gnuplot.PlotFunction(f)
    error = Ravl.ErrorC()
    print "Error {0}".format(error.Error(classifier, dataSetF))
    Ravl.Save("pythonClassifier.abs", classifier)



