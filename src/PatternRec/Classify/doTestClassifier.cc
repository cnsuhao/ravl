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
#include "Ravl/PatternRec/DataSetIO.hh"

using namespace RavlN;

// Simple program which performs leave one out

int main(int nargs, char **argv) {

  OptionC opts(nargs, argv);

  StringC installDir = opts.String("i", PROJECT_OUT, "Install directory.");
  RavlN::SetResourceRoot(installDir);
  StringC classifierFile = opts.String("classifier", "", "Load the trained classifier from this file.");
  StringC testDataSetFile = opts.String("dset", "", "The dataset to perform the test on!");
  //bool verbose = opts.Boolean("v", false, "Verbose mode.");
  opts.Check();

  try {

    SysLogOpen("doTestClassifier");

    // And save the classifier
    ClassifierC classifier;
    SysLog(SYSLOG_INFO,"Loading classifier from '%s'", classifierFile.data());
    if (!Load(classifierFile, classifier)) {
      SysLog(SYSLOG_ERR,"Trouble loading classifier");
      return 1;
    }

    // Get dataset
    SysLog(SYSLOG_INFO,"Loading dataset from file '%s'", testDataSetFile.data());
    // FIXME: Still want to use Load/Save instead
    DataSetVectorLabelC testDataSet;
    if (!LoadDataSetVectorLabel(testDataSetFile, testDataSet)) {
      SysLog(SYSLOG_ERR,"Trouble loading dataset from file '%s'", testDataSetFile.data());
      return 1;
    }

    // Lets get error on the test data set
    ErrorC error;
    RealT pmc = error.Error(classifier, testDataSet);
    SysLog(SYSLOG_INFO,"The probability of miss-classification is %0.4f ", pmc);

  } catch (const RavlN::ExceptionC &exc) {
    SysLog(SYSLOG_ERR,"Exception:%s", exc.Text());
  } catch (...) {
    SysLog(SYSLOG_ERR,"Unknown exception");
  }
}
