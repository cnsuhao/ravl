#include "Ravl/Option.hh"
#include "Ravl/XMLFactory.hh"
#include "Ravl/OS/SysLog.hh"
#include "Ravl/Resource.hh"
#include "Ravl/PatternRec/DesignClassifierSupervised.hh"
#include "Ravl/PatternRec/DataSetVectorLabel.hh"
#include "Ravl/PatternRec/ClassifierPreprocess.hh"
#include "Ravl/PatternRec/Error.hh"
#include "Ravl/PatternRec/DataSet2Iter.hh"
#include "Ravl/IO.hh"
#include "Ravl/OS/Filename.hh"

using namespace RavlN;

// Simple program which performs leave one out

int main(int nargs, char **argv) {

  OptionC opts(nargs, argv);

  StringC installDir = opts.String("i", PROJECT_OUT, "Install directory.");
  RavlN::SetResourceRoot(installDir);
  StringC configFile = opts.String("c", RavlN::Resource("Ravl/PatternRec", "classifier.xml"),
      "Classifier config file.");
  StringC classifierType = opts.String("classifier", "KNN", "The type of classifier to train [KNN|GMM|SVM].");
  StringC dsetFile = opts.String("dset", "", "The dataset to train on!");
  bool noNormaliseSample = opts.Boolean("noNormalise", false, "Do not normalise the sample to unit mean/var");
  FilenameC classifierOutFile = opts.String("o", "classifier.strm", "Save classifier to this file.");
  //bool verbose = opts.Boolean("v", false, "Verbose mode.");
  opts.Check();

  try {
    XMLFactoryC::RefT mainFactory = new XMLFactoryC(configFile);
    XMLFactoryContextC context(*mainFactory);

    // Get classifier designer
    SysLog(SYSLOG_INFO,"Initialising classifier '%s'", classifierType.data());
    DesignClassifierSupervisedC design;
    if (!context.UseComponent(classifierType, design, true)) {
      SysLog(SYSLOG_ERR,"No '%s' component in XML config", classifierType.data());
      return 1;
    }

    // Get dataset
    SysLog(SYSLOG_INFO,"Loading dataset from file '%s'", dsetFile.data());
    // FIXME: Want to use Load/Save instead
    DataSetVectorLabelC trainingDataSet;
    IStreamC is(dsetFile);
    if (!is.good()) {
      SysLog(SYSLOG_ERR,"Trouble loading dataset from file!");
      return 1;
    }
    is >> trainingDataSet;

    // Lets compute mean and variance of dataset and normalise input
    FuncMeanProjectionC func;
    if (noNormaliseSample) {
      SysLog(SYSLOG_INFO,"You are not normalising your sample!  I hope you know what you are doing.");
    } else {
      // FIXME: Sometimes you want to normalise on a class, rather than the whole sample
      SysLog(SYSLOG_INFO,"Normalising the whole sample!");
      MeanCovarianceC meanCovariance = trainingDataSet.Sample1().MeanCovariance();
      func = trainingDataSet.Sample1().NormalisationFunction(meanCovariance);
      trainingDataSet.Sample1().Normalise(meanCovariance);
    }

    // FIXME: Also having unequal samples can bias training as well

    // Train classifier
    SysLog(SYSLOG_INFO,"Training the classifier");
    ClassifierC classifier = design.Apply(trainingDataSet.Sample1(), trainingDataSet.Sample2());
    SysLog(SYSLOG_INFO," - finished");

    // Lets get error on training data set - even though highly biased
    ErrorC error;
    RealT pmc = error.Error(classifier, trainingDataSet);
    SysLog(SYSLOG_INFO,"The (biased) probability of miss-classification is %0.4f ", pmc);

    // If we have normalised the sample we need to make sure
    // all input data to classifier is normalised by same stats
    if(!noNormaliseSample) {
      SysLog(SYSLOG_INFO,"Making classifier with preprocessing step!");
      classifier = ClassifierPreprocessC(func, classifier);
    }

    // And save the classifier
    SysLog(SYSLOG_INFO,"Saving classifier to '%s'", classifierOutFile.data());
    if (!Save(classifierOutFile, classifier)) {
      SysLog(SYSLOG_ERR,"Trouble saving classifier");
      return 1;
    }

  } catch (const RavlN::ExceptionC &exc) {
    SysLog(SYSLOG_ERR,"Exception:%s", exc.Text());
  } catch (...) {
    SysLog(SYSLOG_ERR,"Unknown exception");
  }
}
