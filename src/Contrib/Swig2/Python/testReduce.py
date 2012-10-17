#!/usr/bin/python
import Ravl
import os

Ravl.SysLogOpen("testReduce",True,True,True,-1,True)


# we can also load one
irisDataSet = Ravl.LoadDataSet("{0}/share/Ravl/PatternRec/iris.csv".format(os.environ["PROJECT_OUT"]))
# Note this plot will just show first two features
Ravl.Plot(irisDataSet, "First 2-dimensions of original data set")

# Do PCA
pca = Ravl.DesignFuncPCAC(2);
pcaFunc = pca.Apply(irisDataSet.Sample1())
pcaDataSet = Ravl.DataSetVectorLabelC(Ravl.SampleVectorC(pcaFunc.Apply(irisDataSet.Sample1())), irisDataSet.Sample2())
Ravl.Plot(pcaDataSet, "PCA of data set")

# Do LDA
lda = Ravl.DesignFuncLDAC(2);
ldaFunc = lda.Apply(irisDataSet)
ldaDataSet = Ravl.DataSetVectorLabelC(Ravl.SampleVectorC(ldaFunc.Apply(irisDataSet.Sample1())), irisDataSet.Sample2())
Ravl.Plot(ldaDataSet, "LDA of data set")
